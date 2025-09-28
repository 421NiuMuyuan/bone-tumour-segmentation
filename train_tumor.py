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
    """Augmentation & preprocessing pipeline"""
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
    """Compute class weights to handle imbalance"""
    class_counts = torch.zeros(cfg.NUM_CLASSES)

    print("Computing class weights...")
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        for c in range(cfg.NUM_CLASSES):
            class_counts[c] += (mask == c).sum().item()

    # Inverse-frequency weights
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (cfg.NUM_CLASSES * class_counts)

    print(f"Class pixel counts: {class_counts}")
    print(f"Class weights:      {class_weights}")

    return class_weights


def train():
    print("=== Week 2: Tumor 3-class Training ===")

    # Dataset
    full_dataset = TumorSegmentationDataset(transform=get_transform(train=True), only_positive=False)

    if len(full_dataset) == 0:
        print("❌ No valid samples found. Check dataset paths and structure.")
        return

    # Class weights
    class_weights = calculate_class_weights(full_dataset)

    # Split dataset
    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Validation uses different (eval) transforms
    val_dataset.dataset.transform = get_transform(train=False)

    print(f"Train: {n_train}, Val: {n_val}")

    # Dataloaders
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

    # Model
    model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    # Loss (weighted CrossEntropy + FocalTversky inside CombinedLoss)
    criterion = CombinedLoss(w_ce=1.0, w_tv=1.0)

    # Optimizer & LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Start training, target epochs: {cfg.NUM_EPOCHS}")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # ========== Train ==========
        model.train()
        train_loss = 0.0
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{cfg.NUM_EPOCHS} [Train]")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(cfg.DEVICE, non_blocking=True)
            masks = masks.to(cfg.DEVICE, non_blocking=True)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item()
            train_batches += 1

            # Progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss / train_batches:.4f}'
            })

        avg_train_loss = train_loss / train_batches

        # ========== Validate ==========
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

        # LR scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, lr={current_lr:.2e}")

        # ========== Early stopping & checkpointing ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'class_weights': class_weights
            }, cfg.MODEL_NAME)
            print(f"✅ New best model saved: {cfg.MODEL_NAME} (val_loss: {best_val_loss:.4f})")

        else:
            patience_counter += 1
            print(f"⏳ No improvement in val loss ({patience_counter}/{cfg.ES_PATIENCE})")

            if patience_counter >= cfg.ES_PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print(f"🎉 Training finished! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {cfg.MODEL_NAME}")


if __name__ == "__main__":
    train()
