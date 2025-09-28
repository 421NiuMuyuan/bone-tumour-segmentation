# train_joint_improved.py
# Improved joint segmentation training for extreme class imbalance

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


# Improved Focal Loss for better handling of imbalance
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
                # Fix shape/device: flatten targets and index alpha on the correct device
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
    """Augmentation pipeline — more aggressive for joint segmentation"""
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),           # add vertical flip
            A.RandomRotate90(p=0.3),         # rotate by 90°
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
            # Small-object targeted crops
            A.OneOf([
                A.RandomCrop(height=400, width=400, p=1.0),
                A.CenterCrop(height=400, width=400, p=1.0),
            ], p=0.3),
            A.Resize(512, 512),  # ensure final size
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
    """Compute per-sample weights for balanced sampling"""
    weights = []

    print("Computing sample weights...")
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        has_joint = (mask == 1).any().item()

        if has_joint:
            weights.append(10.0)  # positives weighted 10x
        else:
            weights.append(1.0)   # negatives normal weight

    positive_count = sum(1 for w in weights if w > 1.0)
    print(f"Positive samples: {positive_count}, weight: 10.0")
    print(f"Negative samples: {len(weights) - positive_count}, weight: 1.0")

    return torch.tensor(weights, dtype=torch.float)


def train():
    print("=== Week 3: Joint Binary Segmentation Training (Improved) ===")

    # Dataset
    full_dataset = JointSegmentationDataset(transform=get_transform(train=True), only_positive=False)

    if len(full_dataset) == 0:
        print("❌ No valid samples found. Check dataset paths and structure.")
        return

    # Sample weights for balanced sampling
    sample_weights = calculate_sample_weights(full_dataset)

    # Split dataset
    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Weighted sampler for the training split
    train_indices = train_dataset.indices
    train_sample_weights = sample_weights[train_indices]
    weighted_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_indices) * 2,  # oversample per epoch
        replacement=True
    )

    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Using weighted sampling, samples per epoch: {len(train_indices) * 2}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=weighted_sampler,  # use weighted sampler
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

    # Model
    model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    # Loss: Focal + strong class weights
    class_weights = torch.tensor([1.0, 50.0]).to(cfg.DEVICE)  # joint class 50x weight
    focal_loss = ImprovedFocalLoss(alpha=class_weights, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def combined_loss(pred, target):
        return 0.7 * focal_loss(pred, target) + 0.3 * ce_loss(pred, target)

    # Optimizer & LR scheduler — lower LR
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True, min_lr=1e-6
    )

    # Early stopping
    best_val_loss = float('inf')
    best_iou = 0.0
    patience_counter = 0

    print(f"Start training, target epochs: {cfg.NUM_EPOCHS}")
    print("Using improved strategy: Focal Loss + class weights + weighted sampling")

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
            loss = combined_loss(outputs, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # IoU for the joint class
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

                # IoU (joint class)
                preds = outputs.argmax(dim=1)
                intersection = ((preds == 1) & (masks == 1)).sum().item()
                union = ((preds == 1) | (masks == 1)).sum().item()

                total_intersection += intersection
                total_union += union

                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / val_batches
        val_iou = total_intersection / total_union if total_union > 0 else 0.0

        # LR schedule
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"val_IoU={val_iou:.4f}, lr={current_lr:.2e}")

        # ========== Checkpointing ==========
        # Consider both loss and IoU
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
            print(f"✅ New best model saved (IoU: {best_iou:.4f}, Loss: {best_val_loss:.4f})")
        else:
            print(f"⏳ No improvement ({patience_counter}/{cfg.ES_PATIENCE})")

        if patience_counter >= cfg.ES_PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    print(f"🎉 Training finished! Best IoU: {best_iou:.4f}, Best loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {cfg.MODEL_NAME}")


if __name__ == "__main__":
    train()
