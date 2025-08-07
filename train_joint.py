# train_joint_improved.py
# 改进版关节分割训练，专门处理极度不平衡数据

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np

from dataset_joint import JointSegmentationDataset
from unet_smp import get_model
import config_joint as cfg


# 改进的Focal Loss，更好处理不平衡数据
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # 修复维度问题：targets需要flatten并且alpha需要在正确设备上
                targets_flat = targets.flatten()
                alpha_t = self.alpha[targets_flat].view_as(targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_transform(train=True):
    """数据增强管道 - 对关节分割更激进的增强"""
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # 添加垂直翻转
            A.RandomRotate90(p=0.3),  # 90度旋转
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.3,
                rotate_limit=45,
                p=0.7,
                border_mode=0
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
            ], p=0.5),
            # 专门针对小目标的增强
            A.OneOf([
                A.RandomCrop(height=400, width=400, p=1.0),
                A.CenterCrop(height=400, width=400, p=1.0),
            ], p=0.3),
            A.Resize(512, 512),  # 确保最终尺寸
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])


def calculate_sample_weights(dataset):
    """计算样本权重，用于平衡采样"""
    weights = []

    print("计算样本权重...")
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        has_joint = (mask == 1).any().item()

        if has_joint:
            weights.append(10.0)  # 阳性样本权重10倍
        else:
            weights.append(1.0)  # 阴性样本正常权重

    positive_count = sum(1 for w in weights if w > 1.0)
    print(f"阳性样本: {positive_count}, 权重: 10.0")
    print(f"阴性样本: {len(weights) - positive_count}, 权重: 1.0")

    return torch.tensor(weights, dtype=torch.float)


def train():
    print("=== Week 3: 关节二分类训练 (改进版) ===")

    # 数据集准备
    full_dataset = JointSegmentationDataset(transform=get_transform(train=True), only_positive=False)

    if len(full_dataset) == 0:
        print("❌ 未找到有效数据样本，请检查数据路径和文件结构")
        return

    # 计算样本权重用于平衡采样
    sample_weights = calculate_sample_weights(full_dataset)

    # 数据集拆分
    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # 为训练集创建加权采样器
    train_indices = train_dataset.indices
    train_sample_weights = sample_weights[train_indices]
    weighted_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_indices) * 2,  # 增加采样数量
        replacement=True
    )

    print(f"训练集: {n_train}, 验证集: {n_val}")
    print(f"使用加权采样，每epoch采样: {len(train_indices) * 2} 个样本")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=weighted_sampler,  # 使用加权采样
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 模型初始化
    model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    # 改进的损失函数：Focal Loss + 强类别权重
    class_weights = torch.tensor([1.0, 50.0]).to(cfg.DEVICE)  # 关节类权重50倍
    focal_loss = ImprovedFocalLoss(alpha=class_weights, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def combined_loss(pred, target):
        return 0.7 * focal_loss(pred, target) + 0.3 * ce_loss(pred, target)

    # 优化器和学习率调度器 - 降低学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True, min_lr=1e-6
    )

    # 早停机制
    best_val_loss = float('inf')
    best_iou = 0.0
    patience_counter = 0

    print(f"开始训练，目标epochs: {cfg.NUM_EPOCHS}")
    print("使用改进策略: Focal Loss + 类别权重 + 加权采样")

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
            loss = combined_loss(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # 计算IoU指标
        total_intersection = 0
        total_union = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{cfg.NUM_EPOCHS} [Val]  ")
            for images, masks in val_pbar:
                images = images.to(cfg.DEVICE, non_blocking=True)
                masks = masks.to(cfg.DEVICE, non_blocking=True)

                outputs = model(images)
                loss = combined_loss(outputs, masks)

                val_loss += loss.item()
                val_batches += 1

                # 计算IoU (关节类别)
                preds = outputs.argmax(dim=1)
                intersection = ((preds == 1) & (masks == 1)).sum().item()
                union = ((preds == 1) | (masks == 1)).sum().item()

                total_intersection += intersection
                total_union += union

                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / val_batches
        val_iou = total_intersection / total_union if total_union > 0 else 0.0

        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"val_IoU={val_iou:.4f}, lr={current_lr:.2e}")

        # ========== 模型保存策略 ==========
        # 同时考虑loss和IoU
        save_model = False
        if val_iou > best_iou:
            best_iou = val_iou
            save_model = True
            patience_counter = 0
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model = True
            patience_counter = 0
        else:
            patience_counter += 1

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_iou': best_iou,
                'class_weights': class_weights
            }, cfg.MODEL_NAME)
            print(f"✅ 新的最佳模型已保存 (IoU: {best_iou:.4f}, Loss: {best_val_loss:.4f})")
        else:
            print(f"⏳ 性能未改善 ({patience_counter}/{cfg.ES_PATIENCE})")

        if patience_counter >= cfg.ES_PATIENCE:
            print(f"🛑 早停触发 at epoch {epoch}")
            break

    print(f"🎉 训练完成! 最佳IoU: {best_iou:.4f}, 最佳损失: {best_val_loss:.4f}")
    print(f"最佳模型保存位置: {cfg.MODEL_NAME}")


if __name__ == "__main__":
    train()