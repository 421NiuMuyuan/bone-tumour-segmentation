# fixed_evaluation.py

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Required modules
from dataset import FemurSegmentationDataset
from dataset_joint import JointSegmentationDataset
from dataset_tumor import TumorSegmentationDataset
from unet_smp import get_model

import config
import config_joint as cfg_joint
import config_tumor as cfg_tumor

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


class FixedModelEvaluator:
    """Fixed model evaluator - resolves matplotlib compatibility issues"""

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Matplotlib style - compatibility settings
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.weight'] = 'normal'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['savefig.bbox'] = 'tight'

        self.transform = Compose([
            Resize(512, 512),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def denormalize_image(self, img_tensor):
        """Denormalize image for visualization"""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1)
        return img_np

    def calculate_sample_metrics(self, pred_mask, gt_mask, num_classes):
        """Compute metrics for a single sample"""
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()

        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        # Overall accuracy
        accuracy = (pred_flat == gt_flat).mean()

        # IoU and Dice per class
        ious = []
        dices = []

        for class_id in range(num_classes):
            # Compute TP, FP, FN
            tp = ((pred_flat == class_id) & (gt_flat == class_id)).sum()
            fp = ((pred_flat == class_id) & (gt_flat != class_id)).sum()
            fn = ((pred_flat != class_id) & (gt_flat == class_id)).sum()

            # IoU and Dice
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

            ious.append(iou)
            dices.append(dice)

        # Foreground mIoU (exclude background)
        if num_classes > 2:
            foreground_miou = np.mean(ious[1:])
            foreground_mdice = np.mean(dices[1:])
        else:
            foreground_miou = ious[1] if num_classes == 2 else np.mean(ious)
            foreground_mdice = dices[1] if num_classes == 2 else np.mean(dices)

        return {
            'accuracy': accuracy,
            'miou_all': np.mean(ious),
            'miou_foreground': foreground_miou,
            'mdice_all': np.mean(dices),
            'mdice_foreground': foreground_mdice,
            'per_class_iou': ious,
            'per_class_dice': dices
        }

    def evaluate_all_samples(self, model, dataset, num_classes, model_name):
        """Evaluate all samples and collect results"""
        print(f"Evaluating {model_name} on {len(dataset)} samples...")

        model.eval()
        all_results = []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
                try:
                    img, gt_mask = dataset[idx]

                    # Inference
                    img_batch = img.unsqueeze(0).to(self.device)
                    output = model(img_batch)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_mask = output.argmax(dim=1).squeeze().cpu()

                    # Metrics
                    metrics = self.calculate_sample_metrics(pred_mask, gt_mask, num_classes)

                    # Save result
                    result = {
                        'idx': idx,
                        'image': self.denormalize_image(img),
                        'gt_mask': gt_mask.cpu().numpy(),
                        'pred_mask': pred_mask.numpy(),
                        'pred_probs': pred_probs.squeeze().cpu().numpy(),
                        'metrics': metrics,
                        'score': metrics['miou_foreground'] * 0.6 + metrics['accuracy'] * 0.4
                    }

                    all_results.append(result)

                except Exception as e:
                    print(f"Warning: sample {idx} failed: {e}")
                    continue

        print(f"Successfully evaluated {len(all_results)} samples")
        return all_results

    def get_best_samples(self, all_results, top_k=6):
        """Select top-K samples by score"""
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        best_samples = sorted_results[:top_k]

        print(f"Picked top {len(best_samples)} best-performing samples")
        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']
            print(
                f"  #{i + 1}: sample {sample['idx']}, FG mIoU={metrics['miou_foreground']:.3f}, Acc={metrics['accuracy']:.3f}")

        return best_samples

    def calculate_overall_statistics(self, all_results):
        """Compute overall statistics over all samples"""
        if not all_results:
            return {}

        metrics_keys = ['accuracy', 'miou_all', 'miou_foreground', 'mdice_all', 'mdice_foreground']
        stats = {}

        for key in metrics_keys:
            values = [r['metrics'][key] for r in all_results]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }

        return stats

    def visualize_best_results(self, best_samples, class_names, stats, model_type):
        """Visualize best results (fixed)"""
        n_samples = len(best_samples)

        if model_type == 'multiclass':
            self._visualize_multiclass_fixed(best_samples, class_names, stats)
        elif model_type == 'tumor':
            self._visualize_tumor_fixed(best_samples, class_names, stats)
        elif model_type == 'joint':
            self._visualize_joint_fixed(best_samples, class_names, stats)

    def _visualize_multiclass_fixed(self, best_samples, class_names, stats):
        """Visualization for multi-class segmentation (fixed)"""
        n_samples = len(best_samples)

        # Color map
        colors = ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292', '#AED581']
        cmap = ListedColormap(colors[:len(class_names)])

        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']

            # Original image
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'Sample {sample["idx"]}\nOriginal X-ray', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(sample['gt_mask'], cmap=cmap, vmin=0, vmax=len(class_names) - 1)
            axes[i, 1].set_title('Ground Truth', fontweight='bold')
            axes[i, 1].axis('off')

            # Prediction
            axes[i, 2].imshow(sample['pred_mask'], cmap=cmap, vmin=0, vmax=len(class_names) - 1)
            axes[i, 2].set_title(f'Prediction\nmIoU: {metrics["miou_foreground"]:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # Confidence
            max_prob = np.max(sample['pred_probs'], axis=0)
            im = axes[i, 3].imshow(max_prob, cmap='viridis', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Prediction Confidence\nAccuracy: {metrics["accuracy"]:.3f}', fontweight='bold')
            axes[i, 3].axis('off')

            # Metrics text
            axes[i, 4].axis('off')
            metrics_text = "Per-class IoU:\n"
            for j, class_name in enumerate(class_names):
                iou = metrics['per_class_iou'][j]
                metrics_text += f"{class_name[:8]}: {iou:.3f}\n"

            axes[i, 4].text(0.05, 0.95, metrics_text, transform=axes[i, 4].transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'Best Results - Multi-class Bone Segmentation\nMean FG mIoU: {stats["miou_foreground"]["mean"]:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig('fixed_multiclass_results.png', dpi=150, bbox_inches='tight')
        plt.close()  # release memory
        print("Saved: fixed_multiclass_results.png")

    def _visualize_tumor_fixed(self, best_samples, class_names, stats):
        """Visualization for tumor segmentation (fixed)"""
        n_samples = len(best_samples)

        tumor_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        tumor_cmap = ListedColormap(tumor_colors)

        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']

            # Original image
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'Sample {sample["idx"]}\nOriginal X-ray', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(sample['gt_mask'], cmap=tumor_cmap, vmin=0, vmax=2)
            axes[i, 1].set_title('Ground Truth\n(red=surface, blue=in-bone)', fontweight='bold')
            axes[i, 1].axis('off')

            # Prediction
            axes[i, 2].imshow(sample['pred_mask'], cmap=tumor_cmap, vmin=0, vmax=2)
            axes[i, 2].set_title(f'Prediction\nmIoU: {metrics["miou_foreground"]:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # Surface tumor probability
            if len(sample['pred_probs'].shape) == 3 and sample['pred_probs'].shape[0] > 1:
                surf_prob = sample['pred_probs'][1]
                axes[i, 3].imshow(surf_prob, cmap='Reds', vmin=0, vmax=1)
                axes[i, 3].set_title('Surface Tumor Probability', fontweight='bold')
            else:
                axes[i, 3].text(0.5, 0.5, 'No probability data', ha='center', va='center', transform=axes[i, 3].transAxes)
                axes[i, 3].set_title('Probability Map', fontweight='bold')
            axes[i, 3].axis('off')

            # Metrics text
            axes[i, 4].axis('off')
            metrics_text = f"""Tumor Segmentation Metrics:

FG mIoU: {metrics['miou_foreground']:.3f}
Overall Accuracy: {metrics['accuracy']:.3f}
Overall mIoU: {metrics['miou_all']:.3f}

Per-class IoU:
Background: {metrics['per_class_iou'][0]:.3f}
Surface: {metrics['per_class_iou'][1]:.3f}
In-bone: {metrics['per_class_iou'][2]:.3f}
            """

            axes[i, 4].text(0.05, 0.95, metrics_text, transform=axes[i, 4].transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'Best Results - Tumor Segmentation\nMean FG mIoU: {stats["miou_foreground"]["mean"]:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig('fixed_tumor_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: fixed_tumor_results.png")

    def _visualize_joint_fixed(self, best_samples, class_names, stats):
        """Visualization for joint segmentation (fixed)"""
        n_samples = len(best_samples)

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']

            # Original image
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'Sample {sample["idx"]}\nOriginal X-ray', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(sample['gt_mask'], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth\n(white=joint)', fontweight='bold')
            axes[i, 1].axis('off')

            # Prediction
            axes[i, 2].imshow(sample['pred_mask'], cmap='gray', vmin=0, vmax=1)
            joint_iou = metrics['per_class_iou'][1] if len(metrics['per_class_iou']) > 1 else 0
            axes[i, 2].set_title(f'Prediction\nJoint IoU: {joint_iou:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # Metrics text
            axes[i, 3].axis('off')
            metrics_text = f"""Joint Segmentation Metrics:

Joint IoU: {joint_iou:.3f}
Joint Dice: {metrics['per_class_dice'][1] if len(metrics['per_class_dice']) > 1 else 0:.3f}
Overall Accuracy: {metrics['accuracy']:.3f}
FG mIoU: {metrics['miou_foreground']:.3f}

Score: {sample['score']:.3f}
            """

            axes[i, 3].text(0.05, 0.95, metrics_text, transform=axes[i, 3].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'Best Results - Joint Segmentation\nMean Joint IoU: {stats["miou_foreground"]["mean"]:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig('fixed_joint_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: fixed_joint_results.png")

    def create_fixed_performance_summary(self, all_model_results):
        """Create a fixed, simplified performance summary across models"""
        if len(all_model_results) < 2:
            return

        print("\n" + "=" * 80)
        print("Overall model performance comparison")
        print("=" * 80)

        # Prepare data
        model_data = []
        for result in all_model_results:
            if result is None:
                continue
            stats = result['statistics']
            model_data.append({
                'model': result['model_type'].title(),
                'accuracy_mean': stats['accuracy']['mean'],
                'accuracy_std': stats['accuracy']['std'],
                'miou_mean': stats['miou_foreground']['mean'],
                'miou_std': stats['miou_foreground']['std'],
                'samples': len(result['all_results'])
            })

        # Simplified comparison plots (avoid deprecated params)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        models = [d['model'] for d in model_data]
        accuracies = [d['accuracy_mean'] for d in model_data]
        acc_stds = [d['accuracy_std'] for d in model_data]
        mious = [d['miou_mean'] for d in model_data]
        miou_stds = [d['miou_std'] for d in model_data]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(models)]

        # Accuracy
        x_pos = np.arange(len(models))
        ax1.bar(x_pos, accuracies, color=colors, alpha=0.8)
        ax1.errorbar(x_pos, accuracies, yerr=acc_stds, fmt='none', color='black', capsize=3)

        ax1.set_title('Accuracy Across Models', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, (acc, std) in enumerate(zip(accuracies, acc_stds)):
            ax1.text(i, min(1.0, acc + std + 0.02), f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Foreground mIoU
        ax2.bar(x_pos, mious, color=colors, alpha=0.8)
        ax2.errorbar(x_pos, mious, yerr=miou_stds, fmt='none', color='black', capsize=3)

        ax2.set_title('Foreground mIoU Across Models', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Foreground mIoU', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, (miou, std) in enumerate(zip(mious, miou_stds)):
            ax2.text(i, min(1.0, miou + std + 0.02), f'{miou:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('fixed_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: fixed_model_comparison.png")

        # Print winners
        best_acc_idx = np.argmax(accuracies)
        best_miou_idx = np.argmax(mious)

        print(f"\nWinners:")
        print(f"   Highest accuracy: {models[best_acc_idx]} ({accuracies[best_acc_idx]:.3f})")
        print(f"   Highest mIoU: {models[best_miou_idx]} ({mious[best_miou_idx]:.3f})")

    def run_fixed_evaluation(self):
        """Run the fixed evaluation pipeline"""
        print("Launching fixed model evaluation system")

        all_results = []

        # 1. Multi-class bone segmentation
        try:
            print("\n" + "=" * 60)
            print("Week 1: Multi-class bone segmentation evaluation")
            print("=" * 60)

            dataset = FemurSegmentationDataset(transform=self.transform)
            model = get_model(config.NUM_CLASSES).to(self.device)
            checkpoint = torch.load("best_unet_smp.pth", map_location=self.device)
            model.load_state_dict(checkpoint)

            class_names = ['Background', 'Apophysis', 'Epiphysis', 'Metaphysis',
                           'Diaphysis', 'Surface Tumour', 'In-Bone Tumour', 'Joint']

            all_samples = self.evaluate_all_samples(model, dataset, config.NUM_CLASSES, "Multi-class segmentation")
            best_samples = self.get_best_samples(all_samples)
            stats = self.calculate_overall_statistics(all_samples)

            self.visualize_best_results(best_samples, class_names, stats, 'multiclass')

            all_results.append({
                'model_type': 'multiclass',
                'all_results': all_samples,
                'best_samples': best_samples,
                'statistics': stats
            })

            print(f"\nMulti-class segmentation stats:")
            print(f"   FG mIoU: {stats['miou_foreground']['mean']:.3f} ± {stats['miou_foreground']['std']:.3f}")
            print(f"   Accuracy: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")

        except Exception as e:
            print(f"Failed: multi-class model evaluation: {e}")

        # 2. Tumor segmentation
        try:
            print("\n" + "=" * 60)
            print("Week 2: Tumor segmentation evaluation")
            print("=" * 60)

            dataset = TumorSegmentationDataset(transform=self.transform, only_positive=False)
            model = get_model(cfg_tumor.NUM_CLASSES).to(self.device)
            checkpoint = torch.load(cfg_tumor.MODEL_NAME, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            class_names = ['Background', 'Surface Tumour', 'In-Bone Tumour']

            all_samples = self.evaluate_all_samples(model, dataset, cfg_tumor.NUM_CLASSES, "Tumor segmentation")
            best_samples = self.get_best_samples(all_samples)
            stats = self.calculate_overall_statistics(all_samples)

            self.visualize_best_results(best_samples, class_names, stats, 'tumor')

            all_results.append({
                'model_type': 'tumor',
                'all_results': all_samples,
                'best_samples': best_samples,
                'statistics': stats
            })

            print(f"\nTumor segmentation stats:")
            print(f"   FG mIoU: {stats['miou_foreground']['mean']:.3f} ± {stats['miou_foreground']['std']:.3f}")
            print(f"   Accuracy: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")

        except Exception as e:
            print(f"Failed: tumor model evaluation: {e}")

        # 3. Joint segmentation
        try:
            print("\n" + "=" * 60)
            print("Week 3: Joint segmentation evaluation")
            print("=" * 60)

            dataset = JointSegmentationDataset(transform=self.transform, only_positive=False)
            model = get_model(cfg_joint.NUM_CLASSES).to(self.device)
            checkpoint = torch.load(cfg_joint.MODEL_NAME, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            class_names = ['Background', 'Joint']

            all_samples = self.evaluate_all_samples(model, dataset, cfg_joint.NUM_CLASSES, "Joint segmentation")
            best_samples = self.get_best_samples(all_samples)
            stats = self.calculate_overall_statistics(all_samples)

            self.visualize_best_results(best_samples, class_names, stats, 'joint')

            all_results.append({
                'model_type': 'joint',
                'all_results': all_samples,
                'best_samples': best_samples,
                'statistics': stats
            })

            print(f"\nJoint segmentation stats:")
            print(f"   FG mIoU: {stats['miou_foreground']['mean']:.3f} ± {stats['miou_foreground']['std']:.3f}")
            print(f"   Accuracy: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")

        except Exception as e:
            print(f"Failed: joint model evaluation: {e}")

        # Aggregate comparison
        if len(all_results) > 1:
            self.create_fixed_performance_summary(all_results)

        # Save summary
        summary_data = {}
        for result in all_results:
            model_type = result['model_type']
            summary_data[model_type] = {
                'statistics': result['statistics'],
                'best_samples_indices': [s['idx'] for s in result['best_samples']],
                'total_samples': len(result['all_results'])
            }

        summary_data['evaluation_timestamp'] = datetime.now().isoformat()

        with open('fixed_evaluation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation finished. Generated files:")
        print(f"   fixed_multiclass_results.png - best results for multi-class segmentation")
        print(f"   fixed_tumor_results.png - best results for tumor segmentation")
        print(f"   fixed_joint_results.png - best results for joint segmentation")
        print(f"   fixed_model_comparison.png - cross-model performance comparison")
        print(f"   fixed_evaluation_summary.json - detailed statistics")

        return all_results


def main():
    """Main entry"""
    print("Fixed model evaluation tool")
    print("Resolve matplotlib compatibility issues and export best predictions")
    print()

    evaluator = FixedModelEvaluator()

    try:
        results = evaluator.run_fixed_evaluation()

        if results:
            print(f"\nEvaluation completed!")

            # Final ranking
            model_scores = []
            for result in results:
                stats = result['statistics']
                model_scores.append({
                    'name': result['model_type'].title(),
                    'miou': stats['miou_foreground']['mean'],
                    'accuracy': stats['accuracy']['mean']
                })

            # Sort by FG mIoU
            model_scores.sort(key=lambda x: x['miou'], reverse=True)

            print(f"\nFinal model ranking (by foreground mIoU):")
            for i, model in enumerate(model_scores):
                print(f"   {i + 1}. {model['name']:12} - mIoU: {model['miou']:.3f}, Acc: {model['accuracy']:.3f}")
        else:
            print(f"\nNo models were successfully evaluated")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
