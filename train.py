# train.py

import torch, os
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset    import FemurSegmentationDataset
from unet_smp   import get_model
from losses     import CombinedLoss
import config

def get_transform(train=True):
    if train:
        return A.Compose([
            A.Resize(512,512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize((0.5,)*3, (0.5,)*3),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512,512),
            A.Normalize((0.5,)*3, (0.5,)*3),
            ToTensorV2()
        ])

def train():
    # 数据集 & 拆分
    full_ds = FemurSegmentationDataset(transform=get_transform(train=True))
    n_val = int(0.2 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型 / 损失 / 优化 / 调度 / 早停
    model     = get_model(config.NUM_CLASSES).to(config.DEVICE)
    criterion = CombinedLoss(1.0, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    best_val, stalled = float("inf"), 0

    for epoch in range(1, config.NUM_EPOCHS+1):
        # 训练
        model.train()
        tl, vc = 0, 0
        for imgs, masks in tqdm(train_loader, desc=f"Train {epoch}"):
            imgs  = imgs.to(config.DEVICE)
            masks = masks.to(config.DEVICE).long()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)

        # 验证
        model.eval()
        vl = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(config.DEVICE)
                masks = masks.to(config.DEVICE).long()
                preds = model(imgs)
                vl += criterion(preds, masks).item()
        vl /= len(val_loader)

        scheduler.step(vl)
        print(f"Epoch {epoch}: train_loss={tl:.4f}  val_loss={vl:.4f}")

        # EarlyStopping
        if vl < best_val:
            best_val, stalled = vl, 0
            torch.save(model.state_dict(), "best_unet_smp.pth")
        else:
            stalled += 1
            if stalled >= config.ES_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Finished. Best val loss:", best_val)

if __name__=="__main__":
    train()
