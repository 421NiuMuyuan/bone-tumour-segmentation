# visualize_readable.py

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from dataset_tumor import TumorSegmentationDataset
from unet_smp import get_model
import config_tumor as cfg


def create_readable_colormap():
    """Create an easy-to-read colormap"""
    # Distinct colors (RGB)
    colors = [
        [0.0, 0.0, 0.0],  # 0: Black - Background
        [1.0, 0.0, 0.0],  # 1: Red   - Surface Tumor
        [0.0, 0.0, 1.0],  # 2: Blue  - In-Bone Tumor
    ]
    return ListedColormap(colors)


def create_overlay_visualization(image, mask, alpha=0.6):
    """Create an overlay of the original image and the mask"""
    # Ensure image is RGB in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Build color mask
    colored_mask = np.zeros((*mask.shape, 3))

    # Surface tumor -> red
    surface_tumor = (mask == 1)
    colored_mask[surface_tumor] = [1.0, 0.0, 0.0]

    # In-bone tumor -> blue
    inbone_tumor = (mask == 2)
    colored_mask[inbone_tumor] = [0.0, 0.0, 1.0]

    # Alpha blend
    overlay = image.copy()
    tumor_regions = (mask > 0)
    overlay[tumor_regions] = (1 - alpha) * image[tumor_regions] + alpha * colored_mask[tumor_regions]

    return overlay


def analyze_tumor_statistics(mask, image_name=""):
    """Compute per-image tumor statistics"""
    total_pixels = mask.size
    background_pixels = (mask == 0).sum()
    surface_pixels = (mask == 1).sum()
    inbone_pixels = (mask == 2).sum()

    stats = {
        'image_name': image_name,
        'total_pixels': total_pixels,
        'background': {
            'pixels': background_pixels,
            'percentage': background_pixels / total_pixels * 100
        },
        'surface_tumor': {
            'pixels': surface_pixels,
            'percentage': surface_pixels / total_pixels * 100
        },
        'inbone_tumor': {
            'pixels': inbone_pixels,
            'percentage': inbone_pixels / total_pixels * 100
        },
        'has_surface': surface_pixels > 0,
        'has_inbone': inbone_pixels > 0,
        'has_tumor': (surface_pixels + inbone_pixels) > 0
    }

    return stats


def visualize_tumor_groundtruth(n_samples=8, save_images=True):
    """Visualize Week 2 tumor ground truth with a human-friendly view"""
    print("=== Week 2: Tumor Ground Truth Visualization & Validation ===")

    # Preprocessing
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # Dataset
    dataset = TumorSegmentationDataset(transform=transform, only_positive=False)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ Empty dataset. Please check your data paths.")
        return

    # Colormap
    tumor_cmap = create_readable_colormap()

    # Split positive/negative indices
    positive_indices = []
    negative_indices = []

    for i in range(len(dataset)):
        _, mask = dataset[i]
        if (mask > 0).any():
            positive_indices.append(i)
        else:
            negative_indices.append(i)

    print(f"Positive samples (has tumor): {len(positive_indices)}")
    print(f"Negative samples (no tumor):  {len(negative_indices)}")

    # Select samples: prefer positives
    selected_indices = []
    if positive_indices:
        random.shuffle(positive_indices)
        selected_indices.extend(positive_indices[:min(n_samples - 2, len(positive_indices))])

    if negative_indices and len(selected_indices) < n_samples:
        random.shuffle(negative_indices)
        needed = n_samples - len(selected_indices)
        selected_indices.extend(negative_indices[:min(needed, len(negative_indices))])

    # Layout: Original, GT, Overlay, Stats
    n_cols = 4
    n_rows = len(selected_indices)

    fig = plt.figure(figsize=(16, 4 * n_rows))

    all_stats = []

    for row_idx, idx in enumerate(selected_indices):
        img, gt_mask = dataset[idx]

        # Denormalize for display
        img_display = img.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display + 1.0) / 2.0  # [-1,1] -> [0,1]
        img_display = np.clip(img_display, 0, 1)

        gt_mask_np = gt_mask.cpu().numpy()

        # Stats
        stats = analyze_tumor_statistics(gt_mask_np, f"Sample_{idx}")
        all_stats.append(stats)

        # Overlay
        overlay = create_overlay_visualization(img_display, gt_mask_np)

        # Original
        ax1 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 1)
        ax1.imshow(img_display)
        ax1.set_title(f'Sample {idx}\nOriginal X-ray', fontsize=10, fontweight='bold')
        ax1.axis('off')

        # Ground truth
        ax2 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 2)
        im2 = ax2.imshow(gt_mask_np, cmap=tumor_cmap, vmin=0, vmax=2)
        ax2.set_title('Ground Truth\n(Black=BG, Red=Surface, Blue=In-Bone)',
                      fontsize=10, fontweight='bold')
        ax2.axis('off')

        # Overlay
        ax3 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 3)
        ax3.imshow(overlay)
        ax3.set_title('Overlay Visualization\n(Tumor regions highlighted)',
                      fontsize=10, fontweight='bold')
        ax3.axis('off')

        # Stats panel
        ax4 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 4)
        ax4.axis('off')

        stats_text = f"""Sample Stats:

Total pixels: {stats['total_pixels']:,}

Background: {stats['background']['pixels']:,} 
({stats['background']['percentage']:.1f}%)

Surface tumor: {stats['surface_tumor']['pixels']:,}
({stats['surface_tumor']['percentage']:.1f}%)

In-bone tumor: {stats['inbone_tumor']['pixels']:,}
({stats['inbone_tumor']['percentage']:.1f}%)

Tumor presence:
✅ Surface: {'Yes' if stats['has_surface'] else 'No'}
✅ In-bone: {'Yes' if stats['has_inbone'] else 'No'}
✅ Any:     {'Yes' if stats['has_tumor'] else 'No'}
        """

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax4.set_title('Statistics', fontsize=10, fontweight='bold')

        # Console logs
        print(f"\n--- Sample {idx} ---")
        print("Tumor type: ", end="")
        if stats['has_surface'] and stats['has_inbone']:
            print("Surface + In-bone")
        elif stats['has_surface']:
            print("Surface only")
        elif stats['has_inbone']:
            print("In-bone only")
        else:
            print("None")

        print(f"Surface tumor pixels: {stats['surface_tumor']['pixels']:,} "
              f"({stats['surface_tumor']['percentage']:.2f}%)")
        print(f"In-bone tumor pixels: {stats['inbone_tumor']['pixels']:,} "
              f"({stats['inbone_tumor']['percentage']:.2f}%)")

    # Global colorbar
    if n_rows > 0:
        cbar = fig.colorbar(im2, ax=fig.get_axes(), orientation='horizontal',
                            fraction=0.02, pad=0.02, aspect=50)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Background', 'Surface Tumor', 'In-Bone Tumor'])

    plt.tight_layout()
    plt.suptitle('Week 2: Tumor Ground Truth Validation (Enhanced Visualization)',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_images:
        plt.savefig('tumor_groundtruth_validation.png', dpi=150, bbox_inches='tight')
        print("\n✅ Visualization saved: tumor_groundtruth_validation.png")

    plt.show()

    # Dataset-wide stats
    print("\n" + "=" * 60)
    print("📊 Dataset Summary")
    print("=" * 60)

    total_samples = len(all_stats)
    samples_with_surface = sum(1 for s in all_stats if s['has_surface'])
    samples_with_inbone = sum(1 for s in all_stats if s['has_inbone'])
    samples_with_any_tumor = sum(1 for s in all_stats if s['has_tumor'])

    print(f"Total samples:            {total_samples}")
    print(f"Samples w/ surface tumor: {samples_with_surface} "
          f"({samples_with_surface / total_samples * 100:.1f}%)")
    print(f"Samples w/ in-bone tumor: {samples_with_inbone} "
          f"({samples_with_inbone / total_samples * 100:.1f}%)")
    print(f"Samples w/ any tumor:     {samples_with_any_tumor} "
          f"({samples_with_any_tumor / total_samples * 100:.1f}%)")

    return all_stats


def validate_tumor_model_predictions(n_samples=4):
    """Validate predictions of the trained tumor model"""
    print("\n" + "=" * 60)
    print("🤖 Tumor Model Prediction Validation")
    print("=" * 60)

    # Preprocessing
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # Dataset
    dataset = TumorSegmentationDataset(transform=transform, only_positive=False)

    # Load model
    try:
        model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded: {cfg.MODEL_NAME}")
    except FileNotFoundError:
        print(f"❌ Model file not found: {cfg.MODEL_NAME}")
        print("Please run train_tumor.py first.")
        return

    # Pick samples
    indices = list(range(min(len(dataset), 20)))
    random.shuffle(indices)
    selected_indices = indices[:n_samples]

    # Colormap
    tumor_cmap = create_readable_colormap()

    fig = plt.figure(figsize=(20, 5 * n_samples))

    with torch.no_grad():
        for row_idx, idx in enumerate(selected_indices):
            img, gt_mask = dataset[idx]

            # Inference
            img_batch = img.unsqueeze(0).to(cfg.DEVICE)
            output = model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            # Denormalize for display
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display + 1.0) / 2.0
            img_display = np.clip(img_display, 0, 1)

            gt_mask_np = gt_mask.cpu().numpy()

            # IoU (foreground union)
            intersection = ((gt_mask_np > 0) & (pred_mask > 0)).sum()
            union = ((gt_mask_np > 0) | (pred_mask > 0)).sum()
            iou = intersection / union if union > 0 else 1.0

            # Overlays
            gt_overlay = create_overlay_visualization(img_display, gt_mask_np)
            pred_overlay = create_overlay_visualization(img_display, pred_mask)

            # Columns: Original, GT, GT Overlay, Pred, Pred Overlay
            n_cols = 5

            # Original
            ax1 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 1)
            ax1.imshow(img_display)
            ax1.set_title(f'Sample {idx}\nOriginal', fontsize=10, fontweight='bold')
            ax1.axis('off')

            # GT mask
            ax2 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 2)
            ax2.imshow(gt_mask_np, cmap=tumor_cmap, vmin=0, vmax=2)
            ax2.set_title('Ground Truth', fontsize=10, fontweight='bold')
            ax2.axis('off')

            # GT overlay
            ax3 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 3)
            ax3.imshow(gt_overlay)
            ax3.set_title('GT Overlay', fontsize=10, fontweight='bold')
            ax3.axis('off')

            # Prediction mask
            ax4 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 4)
            ax4.imshow(pred_mask, cmap=tumor_cmap, vmin=0, vmax=2)
            ax4.set_title(f'Prediction\nIoU: {iou:.3f}', fontsize=10, fontweight='bold')
            ax4.axis('off')

            # Prediction overlay
            ax5 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 5)
            ax5.imshow(pred_overlay)
            ax5.set_title('Pred Overlay', fontsize=10, fontweight='bold')
            ax5.axis('off')

            # Console compare
            gt_stats = analyze_tumor_statistics(gt_mask_np)
            pred_stats = analyze_tumor_statistics(pred_mask)

            print(f"\nSample {idx} comparison:")
            print(f"  GT     - Surface: {gt_stats['surface_tumor']['pixels']:4d}, "
                  f"In-bone: {gt_stats['inbone_tumor']['pixels']:4d}")
            print(f"  Pred   - Surface: {pred_stats['surface_tumor']['pixels']:4d}, "
                  f"In-bone: {pred_stats['inbone_tumor']['pixels']:4d}")
            print(f"  IoU: {iou:.3f}")

    plt.tight_layout()
    plt.suptitle('Week 2: Model Prediction vs Ground Truth Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('tumor_prediction_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Prediction comparison saved: tumor_prediction_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Validate Ground Truth
    print("🔍 Start validating Week 2 tumor ground truth...")
    stats = visualize_tumor_groundtruth(n_samples=6)

    # Validate model predictions (if model exists)
    print("\n🤖 Validate model predictions...")
    validate_tumor_model_predictions(n_samples=4)
