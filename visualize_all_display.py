import torch
import matplotlib.pyplot as plt
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from dataset import FemurSegmentationDataset
from unet import UNet
import config

# Only Resize + Normalize; no random augmentation
transform = Compose([
    Resize(512, 512),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Build dataset
ds = FemurSegmentationDataset(transform=transform)
print(f">>> Total samples: {len(ds)}")

# Split validation set
n_val = int(0.2 * len(ds))
n_train = len(ds) - n_val
_, val_ds = random_split(ds, [n_train, n_val])
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# Load best model
model = UNet(in_channels=3, out_classes=config.NUM_CLASSES).to(config.DEVICE)
state = torch.load("best_unet.pth", map_location=config.DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

# Compute Pixel Accuracy and per-class IoU
def compute_metrics(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(preds.flatten(), gts.flatten()):
        cm[g, p] += 1
    acc = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0
    ious = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return acc, ious

all_p, all_g = [], []
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs = imgs.to(config.DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_p.append(preds)
        all_g.append(masks.numpy())
all_p = np.concatenate(all_p)
all_g = np.concatenate(all_g)
acc, ious = compute_metrics(all_p, all_g, config.NUM_CLASSES)
print(f"[Bone Segmentation] Val Pixel Acc = {acc:.4f}")
for i, iou in enumerate(ious):
    print(f"  Class {i} IoU = {iou:.4f}")

# Visualize first N samples
for idx in range(min(4, len(val_ds))):
    img, mask = val_ds[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(config.DEVICE))
        pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy()
    im_np = img.permute(1, 2, 0).cpu().numpy()
    im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax_in, ax_gt, ax_pr = axes
    ax_in.imshow(im_np); ax_in.set_title(f"Sample {idx} – Input"); ax_in.axis("off")
    cax_gt = ax_gt.imshow(mask.numpy(), cmap='nipy_spectral', vmin=0, vmax=config.NUM_CLASSES-1)
    ax_gt.set_title("Ground Truth"); ax_gt.axis("off")
    fig.colorbar(cax_gt, ax=ax_gt, ticks=range(config.NUM_CLASSES), fraction=0.046, pad=0.04)
    cax_pr = ax_pr.imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=config.NUM_CLASSES-1)
    ax_pr.set_title("Prediction"); ax_pr.axis("off")
    fig.colorbar(cax_pr, ax=ax_pr, ticks=range(config.NUM_CLASSES), fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
