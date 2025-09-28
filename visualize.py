# visualize_all_display.py

import torch
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from dataset import FemurSegmentationDataset
from unet import UNet
import config

# Only Resize + Normalize, no random augmentation
transform = Compose([
    Resize(512, 512),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Build dataset
ds = FemurSegmentationDataset(transform=transform)
print(f">>> Total samples: {len(ds)}")

# Load best model
model = UNet(in_channels=3, out_classes=config.NUM_CLASSES).to(config.DEVICE)
state = torch.load("best_unet.pth", map_location=config.DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

# Iterate all samples and display
for idx in range(len(ds)):
    img, mask = ds[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(config.DEVICE))
        pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy()

    # Normalize original image to [0, 1] for display
    im_np = img.permute(1, 2, 0).cpu().numpy()
    im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax_in, ax_gt, ax_pr = axes

    ax_in.imshow(im_np)
    ax_in.set_title(f"Sample {idx} – Input")
    ax_in.axis("off")

    cax_gt = ax_gt.imshow(
        mask.cpu().numpy(),
        cmap="nipy_spectral",
        vmin=0, vmax=config.NUM_CLASSES - 1
    )
    ax_gt.set_title("Ground Truth")
    ax_gt.axis("off")
    fig.colorbar(
        cax_gt, ax=ax_gt,
        ticks=range(config.NUM_CLASSES),
        fraction=0.046, pad=0.04
    )

    cax_pr = ax_pr.imshow(
        pred_mask,
        cmap="nipy_spectral",
        vmin=0, vmax=config.NUM_CLASSES - 1
    )
    ax_pr.set_title("Prediction")
    ax_pr.axis("off")
    fig.colorbar(
        cax_pr, ax=ax_pr,
        ticks=range(config.NUM_CLASSES),
        fraction=0.046, pad=0.04
    )

    plt.tight_layout()
    plt.show()  # show current sample window
