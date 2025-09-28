# diagnose_joint_model.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from dataset_joint import JointSegmentationDataset
from unet_smp import get_model
import config_joint as cfg

# Configure matplotlib to support Unicode (so emojis/symbols display correctly)
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


def get_eval_transform():
    """Preprocessing for evaluation"""
    return Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def analyze_joint_distribution():
    """Detailed analysis of joint data distribution"""
    print("=" * 60)
    print("🔍 Detailed analysis of joint data distribution")
    print("=" * 60)

    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    positive_samples = []
    negative_samples = []

    print("Analyzing all samples...")
    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask_np = mask.cpu().numpy()
        joint_pixels = (mask_np == 1).sum()

        if joint_pixels > 0:
            positive_samples.append((i, joint_pixels))
        else:
            negative_samples.append(i)

    print(f"\n📊 Sample distribution:")
    print(f"Positive (with joint): {len(positive_samples)}")
    print(f"Negative (no joint):   {len(negative_samples)}")

    if positive_samples:
        print(f"\n🔍 Positive sample details:")
        positive_samples.sort(key=lambda x: x[1], reverse=True)  # sort by pixel count
        for i, (idx, pixels) in enumerate(positive_samples[:10]):
            print(f"  Sample {idx}: {pixels:,} joint pixels")

    return positive_samples, negative_samples


def test_model_predictions():
    """Test model prediction behavior"""
    print("\n" + "=" * 60)
    print("🤖 Testing model prediction behavior")
    print("=" * 60)

    # Load dataset and model
    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    try:
        model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded: {cfg.MODEL_NAME}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Analyze predictions for samples
    all_predictions = []
    all_gt_labels = []

    print("\nAnalyzing model predictions...")
    with torch.no_grad():
        for i in range(min(50, len(dataset))):  # analyze first 50 samples
            img, gt_mask = dataset[i]

            # Inference
            img_batch = img.unsqueeze(0).to(cfg.DEVICE)
            output = model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            # Stats
            gt_has_joint = (gt_mask == 1).any().item()
            pred_has_joint = (pred_mask == 1).any()

            max_joint_prob = pred_probs[0, 1].max().item()  # max prob for the joint class

            all_predictions.append({
                'sample_idx': i,
                'gt_has_joint': gt_has_joint,
                'pred_has_joint': pred_has_joint,
                'max_joint_prob': max_joint_prob,
                'gt_joint_pixels': (gt_mask == 1).sum().item(),
                'pred_joint_pixels': (pred_mask == 1).sum()
            })

            all_gt_labels.append(1 if gt_has_joint else 0)

    # Summary
    print(f"\n📊 Prediction statistics (first {len(all_predictions)} samples):")

    # Predicted class distribution
    pred_joint_count = sum(1 for p in all_predictions if p['pred_has_joint'])
    pred_bg_count = len(all_predictions) - pred_joint_count

    print("Prediction distribution:")
    print(f"  Predicted joint:    {pred_joint_count}")
    print(f"  Predicted background: {pred_bg_count}")

    # Probability distribution
    joint_probs = [p['max_joint_prob'] for p in all_predictions]
    print(f"\nJoint probability stats:")
    print(f"  Max:    {max(joint_probs):.4f}")
    print(f"  Mean:   {np.mean(joint_probs):.4f}")
    print(f"  > 0.5:  {sum(1 for p in joint_probs if p > 0.5)}")
    print(f"  > 0.1:  {sum(1 for p in joint_probs if p > 0.1)}")

    # Group by GT
    positive_preds = [p for p in all_predictions if p['gt_has_joint']]
    negative_preds = [p for p in all_predictions if not p['gt_has_joint']]

    print(f"\n🔴 Positive (GT has joint, {len(positive_preds)}):")
    if positive_preds:
        pos_probs = [p['max_joint_prob'] for p in positive_preds]
        print(f"  Mean joint prob: {np.mean(pos_probs):.4f}")
        print(f"  Max joint prob:  {max(pos_probs):.4f}")
        print(f"  Correctly predicted joint: {sum(1 for p in positive_preds if p['pred_has_joint'])}")

    print(f"\n⚪ Negative (GT no joint, {len(negative_preds)}):")
    if negative_preds:
        neg_probs = [p['max_joint_prob'] for p in negative_preds]
        print(f"  Mean joint prob: {np.mean(neg_probs):.4f}")
        print(f"  Max joint prob:  {max(neg_probs):.4f}")
        print(f"  Correctly predicted background: {sum(1 for p in negative_preds if not p['pred_has_joint'])}")

    return all_predictions


def visualize_positive_samples():
    """Visualize positive samples (with joint)"""
    print("\n" + "=" * 60)
    print("🎨 Visualizing positive samples")
    print("=" * 60)

    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    # Find all positive indices
    positive_indices = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        if (mask == 1).any():
            positive_indices.append(i)

    print(f"Found {len(positive_indices)} positive samples")

    if len(positive_indices) == 0:
        print("❌ No positive samples found")
        return

    # Load model
    try:
        model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Visualize up to 4 positives
    n_samples = min(4, len(positive_indices))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(positive_indices[:n_samples]):
            img, gt_mask = dataset[idx]

            # Inference
            img_batch = img.unsqueeze(0).to(cfg.DEVICE)
            output = model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            # Denormalize image for display
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display + 1.0) / 2.0
            img_display = np.clip(img_display, 0, 1)

            gt_mask_np = gt_mask.cpu().numpy()
            joint_prob = pred_probs[0, 1].cpu().numpy()

            # Metrics
            gt_joint_pixels = (gt_mask_np == 1).sum()
            pred_joint_pixels = (pred_mask == 1).sum()
            max_prob = joint_prob.max()

            # Plot
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title(f'Sample {idx}\nOriginal')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'GT: {gt_joint_pixels} pixels')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Pred: {pred_joint_pixels} pixels')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(joint_prob, cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Joint Prob\nMax: {max_prob:.3f}')
            axes[i, 3].axis('off')

            print(f"Sample {idx}: GT={gt_joint_pixels}, Pred={pred_joint_pixels}, MaxProb={max_prob:.3f}")

    plt.tight_layout()
    plt.suptitle('Joint Segmentation: Positive Samples Analysis', fontsize=14, y=0.98)
    plt.savefig('joint_positive_samples_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: joint_positive_samples_analysis.png")


def suggest_solutions():
    """Suggest potential fixes"""
    print("\n" + "=" * 60)
    print("💡 Diagnosis and suggested fixes")
    print("=" * 60)

    print("Possible issues:")
    print("1. Extremely imbalanced data (e.g., ~782.8:1) causing model collapse")
    print("2. Learning rate too high, converging to a lazy strategy")
    print("3. Improper loss weighting")
    print("4. Insufficient training epochs")

    print("\nSuggested fixes:")
    print("1. Use stronger class weights when retraining:")
    print("   - Increase joint-class weight to 50–100")
    print("2. Lower the learning rate:")
    print("   - LR = 5e-4 or 1e-4")
    print("3. Add Focal Loss:")
    print("   - Better handles class imbalance")
    print("4. Data augmentation:")
    print("   - Over-sample positive samples")
    print("5. Training strategy tweak:")
    print("   - Warm-up or pre-train on positive-only batches first")


if __name__ == "__main__":
    # Analyze data distribution
    positive_samples, negative_samples = analyze_joint_distribution()

    # Test model predictions
    predictions = test_model_predictions()

    # Visualize positive samples
    visualize_positive_samples()

    # Suggestions
    suggest_solutions()
