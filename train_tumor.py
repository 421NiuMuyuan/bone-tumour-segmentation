# train_tumor.py

import torch
import os
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset_tumor import TumorSegmentationDataset
from unet_smp import get_model
from losses import CombinedLoss
import config_tumor as cfg


def get_transform(train=True):
    """数据增强和预处理管道"""
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=25,
                p=0.5,
                border_mode=0
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
            ], p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])


def calculate_class_weights(dataset):
    """计算类别权重以处理不平衡数据"""
    class_counts = torch.zeros(cfg.NUM_CLASSES)

    print("计算类别权重...")
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        for c in range(cfg.NUM_CLASSES):
            class_counts[c] += (mask == c).sum().item()

    # 计算权重 (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (cfg.NUM_CLASSES * class_counts)

    print(f"类别像素数: {class_counts}")
    print(f"类别权重: {class_weights}")

    return class_weights


def train():
    print("=== Week 2: 肿瘤三分类训练 ===")

    # 数据集准备
    full_dataset = TumorSegmentationDataset(transform=get_transform(train=True), only_positive=False)

    if len(full_dataset) == 0:
        print("❌ 未找到有效数据样本，请检查数据路径和文件结构")
        return

    # 计算类别权重
    class_weights = calculate_class_weights(full_dataset)

    # 数据集拆分
    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # 验证集使用不同的transform
    val_dataset.dataset.transform = get_transform(train=False)

    print(f"训练集: {n_train}, 验证集: {n_val}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 模型初始化
    model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    # 损失函数 (加权CrossEntropy + FocalTversky)
    criterion = CombinedLoss(w_ce=1.0, w_tv=1.0)

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"开始训练，目标epochs: {cfg.NUM_EPOCHS}")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{cfg.NUM_EPOCHS} [Train]")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(cfg.DEVICE, non_blocking=True)
            masks = masks.to(cfg.DEVICE, non_blocking=True)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_batches += 1

            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss / train_batches:.4f}'
            })

        avg_train_loss = train_loss / train_batches

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{cfg.NUM_EPOCHS} [Val]  ")
            for images, masks in val_pbar:
                images = images.to(cfg.DEVICE, non_blocking=True)
                masks = masks.to(cfg.DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_batches += 1

                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / val_batches

        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={current_lr:.2e}")

        # ========== 早停和模型保存 ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'class_weights': class_weights
            }, cfg.MODEL_NAME)
            print(f"✅ 新的最佳模型已保存: {cfg.MODEL_NAME} (val_loss: {best_val_loss:.4f})")

        else:
            patience_counter += 1
            print(f"⏳ 验证损失未改善 ({patience_counter}/{cfg.ES_PATIENCE})")

            if patience_counter >= cfg.ES_PATIENCE:
                print(f"🛑 早停触发 at epoch {epoch}")
                break

    print(f"🎉 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存位置: {cfg.MODEL_NAME}")


if __name__ == "__main__":
    train()