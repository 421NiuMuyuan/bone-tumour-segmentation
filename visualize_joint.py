# visualize_joint.py

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from dataset_joint import JointSegmentationDataset
from unet_smp import get_model
import config_joint as cfg


def get_eval_transform():
    """Preprocessing for validation/evaluation"""
    return Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def denormalize_image(img_tensor):
    """Denormalize to [0, 1] for display"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np + 1.0) / 2.0
    img_np = np.clip(img_np, 0, 1)
    return img_np


def calculate_binary_metrics(pred_mask, gt_mask):
    """Compute binary classification metrics"""
    # Convert tensors to numpy
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # TP, TN, FP, FN
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # IoU for the joint class
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 1.0

    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou': iou,
        'dice': dice,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def visualize_samples(n_samples=6):
    """Visualize joint segmentation results"""
    print("=== Week 3: Joint binary segmentation visualization ===")

    # Dataset
    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ Empty dataset. Please check data paths.")
        return

    # Load model
    model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    try:
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded: {cfg.MODEL_NAME}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Model loaded (legacy format): {cfg.MODEL_NAME}")
    except FileNotFoundError:
        print(f"❌ Model file not found: {cfg.MODEL_NAME}")
        print("Please run train_joint.py first.")
        return

    model.eval()

    # Randomly pick samples
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:n_samples]

    # Figure
    fig = plt.figure(figsize=(15, 4 * n_samples))

    all_metrics = []

    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            img, gt_mask = dataset[idx]

            # Inference
            img_batch = img.unsqueeze(0).to(cfg.DEVICE)
            output = model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = output.argmax(dim=1).squeeze().cpu()

            # Denormalize for display
            img_display = denormalize_image(img)

            # Metrics
            metrics = calculate_binary_metrics(pred_mask, gt_mask)
            all_metrics.append(metrics)

            # 4-panel: input, GT, prediction, probability map
            ax1 = plt.subplot(n_samples, 4, 4 * i + 1)
            ax1.imshow(img_display)
            ax1.set_title(f'Sample {idx}: Input X-ray', fontsize=10, fontweight='bold')
            ax1.axis('off')

            ax2 = plt.subplot(n_samples, 4, 4 * i + 2)
            im2 = ax2.imshow(gt_mask.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax2.set_title('Ground Truth', fontsize=10, fontweight='bold')
            ax2.axis('off')

            ax3 = plt.subplot(n_samples, 4, 4 * i + 3)
            im3 = ax3.imshow(pred_mask.numpy(), cmap='gray', vmin=0, vmax=1)
            ax3.set_title(f'Prediction\nIoU: {metrics["iou"]:.3f}', fontsize=10, fontweight='bold')
            ax3.axis('off')

            ax4 = plt.subplot(n_samples, 4, 4 * i + 4)
            joint_prob = pred_probs[0, 1].cpu().numpy()  # probability for joint class
            im4 = ax4.imshow(joint_prob, cmap='hot', vmin=0, vmax=1)
            ax4.set_title(f'Joint Probability\nF1: {metrics["f1_score"]:.3f}', fontsize=10, fontweight='bold')
            ax4.axis('off')

            # Add colorbar only for the first row
            if i == 0:
                plt.colorbar(im4, ax=ax4, orientation='horizontal',
                             fraction=0.05, pad=0.1, aspect=30)

            # Print metrics
            print(f"\n--- Sample {idx} Metrics ---")
            print(f"Accuracy:  {metrics['accuracy']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall:    {metrics['recall']:.3f}")
            print(f"F1-Score:  {metrics['f1_score']:.3f}")
            print(f"IoU:       {metrics['iou']:.3f}")
            print(f"Dice:      {metrics['dice']:.3f}")

    plt.tight_layout()
    plt.suptitle('Week 3: Joint Segmentation Results (0=Background, 1=Joint)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.show()

    # Aggregate metrics
    print("\n=== Overall Averages ===")
    avg_metrics = {}
    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'iou', 'dice']:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        print(f"{key.capitalize():10}: {avg_metrics[key]:.3f}")

    # Save figure
    output_file = "joint_segmentation_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")


def analyze_dataset_distribution():
    """Analyze joint distribution in the dataset"""
    print("\n=== Dataset Joint Distribution Analysis ===")

    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    background_pixels = 0
    joint_pixels = 0
    positive_samples = 0
    total_pixels = 0

    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask_np = mask.cpu().numpy()

        bg_count = (mask_np == 0).sum()
        joint_count = (mask_np == 1).sum()

        background_pixels += bg_count
        joint_pixels += joint_count
        total_pixels += mask_np.size

        if joint_count > 0:
            positive_samples += 1

    print(f"Total samples: {len(dataset)}")
    print(f"Samples with joint: {positive_samples} ({positive_samples / len(dataset) * 100:.1f}%)")
    print(f"Background pixels: {background_pixels:,} ({background_pixels / total_pixels * 100:.1f}%)")
    print(f"Joint pixels:      {joint_pixels:,} ({joint_pixels / total_pixels * 100:.1f}%)")
    print(f"Imbalance ratio:   {background_pixels / joint_pixels:.1f}:1 (Background:Joint)")


if __name__ == "__main__":
    # Analyze dataset distribution
    analyze_dataset_distribution()

    # Visualize results
    visualize_samples(n_samples=6)
