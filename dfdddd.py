# final_complete_evaluation.py
# Complete Academic Model Evaluation Tool - Final Working Version

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from tqdm import tqdm
import warnings
import json
import os
from datetime import datetime
from math import pi

warnings.filterwarnings('ignore')

# Import necessary modules
from dataset import FemurSegmentationDataset
from dataset_joint import JointSegmentationDataset
from dataset_tumor import TumorSegmentationDataset
from unet_smp import get_model

import config
import config_joint as cfg_joint
import config_tumor as cfg_tumor

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


class FinalAcademicEvaluator:
    """Final Complete Academic Evaluation Tool"""

    def __init__(self, device="cuda", output_dir="academic_results"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print("FINAL ACADEMIC MODEL EVALUATION SYSTEM")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Output Directory: {output_dir}")
        print()

        # Set matplotlib parameters - English only to avoid encoding issues
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'

        self.transform = Compose([
            Resize(512, 512),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        # Color schemes
        self.colors = {
            'multiclass': ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292', '#AED581'],
            'tumor': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            'joint': ['black', 'white']
        }

    def denormalize_image(self, img_tensor):
        """Denormalize image for display"""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1.0) / 2.0
        return np.clip(img_np, 0, 1)

    def calculate_metrics(self, pred_mask, gt_mask, num_classes):
        """Calculate evaluation metrics"""
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()

        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        # Overall pixel accuracy
        pixel_accuracy = (pred_flat == gt_flat).mean()

        # Per-class metrics
        class_metrics = []
        ious = []
        dices = []

        for class_id in range(num_classes):
            # Calculate TP, FP, FN, TN
            tp = ((pred_flat == class_id) & (gt_flat == class_id)).sum()
            fp = ((pred_flat == class_id) & (gt_flat != class_id)).sum()
            fn = ((pred_flat != class_id) & (gt_flat == class_id)).sum()
            tn = ((pred_flat != class_id) & (gt_flat != class_id)).sum()

            # Basic metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Segmentation metrics
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            ious.append(float(iou))
            dices.append(float(dice))

            class_metrics.append({
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'iou': float(iou),
                'dice': float(dice),
                'support': int((gt_flat == class_id).sum())
            })

        # Summary metrics
        macro_iou = np.mean(ious)
        macro_dice = np.mean(dices)
        macro_f1 = np.mean([m['f1_score'] for m in class_metrics])

        # Foreground metrics (excluding background)
        if num_classes > 1:
            fg_ious = ious[1:]
            fg_dices = dices[1:]
            foreground_miou = np.mean(fg_ious) if fg_ious else 0.0
            foreground_mdice = np.mean(fg_dices) if fg_dices else 0.0
        else:
            foreground_miou = ious[0]
            foreground_mdice = dices[0]

        return {
            'pixel_accuracy': float(pixel_accuracy),
            'macro_iou': float(macro_iou),
            'macro_dice': float(macro_dice),
            'macro_f1': float(macro_f1),
            'foreground_miou': float(foreground_miou),
            'foreground_mdice': float(foreground_mdice),
            'per_class_metrics': class_metrics,
            'per_class_ious': ious,
            'per_class_dices': dices
        }

    def evaluate_model(self, model, dataset, num_classes, class_names, model_name, n_samples=20):
        """Evaluate model comprehensively"""
        print(f"\nEVALUATING {model_name.upper()} MODEL")
        print(f"Dataset Size: {len(dataset)}")
        print(f"Target Samples: {n_samples}")
        print("-" * 60)

        model.eval()
        all_results = []

        # Evaluate all samples
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
                try:
                    img, gt_mask = dataset[idx]

                    # Model inference
                    img_batch = img.unsqueeze(0).to(self.device)
                    output = model(img_batch)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_mask = output.argmax(dim=1).squeeze().cpu()

                    # Calculate metrics
                    metrics = self.calculate_metrics(pred_mask, gt_mask, num_classes)

                    # Quality score for ranking
                    quality_score = (
                            metrics['foreground_miou'] * 0.4 +
                            metrics['pixel_accuracy'] * 0.2 +
                            metrics['macro_f1'] * 0.2 +
                            metrics['foreground_mdice'] * 0.2
                    )

                    all_results.append({
                        'idx': idx,
                        'image': self.denormalize_image(img),
                        'gt_mask': gt_mask.cpu().numpy(),
                        'pred_mask': pred_mask.numpy(),
                        'pred_probs': pred_probs.squeeze().cpu().numpy(),
                        'metrics': metrics,
                        'quality_score': quality_score
                    })

                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue

        if not all_results:
            print("No samples successfully processed")
            return None

        print(f"Successfully evaluated {len(all_results)} samples")

        # Select diverse samples for display
        display_samples = self.select_diverse_samples(all_results, n_samples)

        # Calculate statistics
        statistics = self.calculate_statistics(all_results, class_names)

        return {
            'model_name': model_name,
            'class_names': class_names,
            'num_classes': num_classes,
            'all_results': all_results,
            'display_samples': display_samples,
            'statistics': statistics,
            'evaluation_timestamp': datetime.now().isoformat()
        }

    def select_diverse_samples(self, all_results, n_samples):
        """Select diverse samples for display"""
        # Sort by quality score
        sorted_results = sorted(all_results, key=lambda x: x['quality_score'], reverse=True)

        n_total = len(sorted_results)
        n_high = min(n_samples // 2, n_total // 3)  # 50% high quality
        n_medium = min(n_samples // 3, n_total // 3)  # 33% medium quality
        n_low = n_samples - n_high - n_medium  # 17% low quality

        selected = []
        # High quality samples
        selected.extend(sorted_results[:n_high])
        # Medium quality samples
        start_medium = n_total // 3
        end_medium = start_medium + n_medium
        selected.extend(sorted_results[start_medium:end_medium])
        # Low quality samples
        if n_low > 0:
            selected.extend(sorted_results[-n_low:])

        print(f"Selected {len(selected)} diverse samples:")
        print(f"  High quality: {n_high}, Medium quality: {n_medium}, Low quality: {n_low}")

        return selected

    def calculate_statistics(self, all_results, class_names):
        """Calculate detailed statistics"""
        stats = {'overall': {}, 'per_class': {}}

        # Overall metrics
        metrics_keys = ['pixel_accuracy', 'macro_iou', 'macro_dice', 'macro_f1',
                        'foreground_miou', 'foreground_mdice', 'quality_score']

        for key in metrics_keys:
            if key == 'quality_score':
                values = [r['quality_score'] for r in all_results]
            else:
                values = [r['metrics'][key] for r in all_results]

            stats['overall'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }

        # Per-class metrics
        for i, class_name in enumerate(class_names):
            stats['per_class'][class_name] = {}
            for metric in ['precision', 'recall', 'f1_score', 'iou', 'dice']:
                values = [r['metrics']['per_class_metrics'][i][metric] for r in all_results]
                stats['per_class'][class_name][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return stats

    def create_sample_visualization(self, eval_result):
        """Create sample visualization"""
        model_name = eval_result['model_name']
        samples = eval_result['display_samples']
        class_names = eval_result['class_names']
        num_classes = eval_result['num_classes']

        print(f"Creating sample visualization for {model_name}")

        n_samples = len(samples)
        n_cols = 5 if num_classes > 2 else 4

        fig, axes = plt.subplots(n_samples, n_cols, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        # Choose colormap
        if model_name == 'multiclass':
            cmap = ListedColormap(self.colors['multiclass'][:num_classes])
        elif model_name == 'tumor':
            cmap = ListedColormap(self.colors['tumor'])
        else:
            cmap = 'gray'

        for i, sample in enumerate(samples):
            metrics = sample['metrics']

            # Original image
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'Sample {sample["idx"]}\nOriginal', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground Truth
            if model_name == 'joint':
                axes[i, 1].imshow(sample['gt_mask'], cmap='gray', vmin=0, vmax=1)
            else:
                axes[i, 1].imshow(sample['gt_mask'], cmap=cmap, vmin=0, vmax=num_classes - 1)
            axes[i, 1].set_title('Ground Truth', fontweight='bold')
            axes[i, 1].axis('off')

            # Prediction
            if model_name == 'joint':
                axes[i, 2].imshow(sample['pred_mask'], cmap='gray', vmin=0, vmax=1)
            else:
                axes[i, 2].imshow(sample['pred_mask'], cmap=cmap, vmin=0, vmax=num_classes - 1)

            fg_miou = metrics['foreground_miou']
            axes[i, 2].set_title(f'Prediction\nFg-mIoU: {fg_miou:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # Probability or error map
            if len(sample['pred_probs'].shape) == 3:
                if num_classes == 2:
                    # Joint probability
                    prob_map = sample['pred_probs'][1]
                    axes[i, 3].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
                    axes[i, 3].set_title('Joint Probability', fontweight='bold')
                elif num_classes == 3:
                    # Surface tumor probability
                    prob_map = sample['pred_probs'][1]
                    axes[i, 3].imshow(prob_map, cmap='Reds', vmin=0, vmax=1)
                    axes[i, 3].set_title('Surface Tumor Prob.', fontweight='bold')
                else:
                    # Max probability
                    prob_map = np.max(sample['pred_probs'], axis=0)
                    axes[i, 3].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
                    axes[i, 3].set_title('Max Probability', fontweight='bold')
            else:
                # Error map
                error_map = (sample['gt_mask'] != sample['pred_mask']).astype(float)
                axes[i, 3].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
                axes[i, 3].set_title('Error Map', fontweight='bold')

            axes[i, 3].axis('off')

            # Metrics (if 5 columns)
            if n_cols == 5:
                axes[i, 4].axis('off')

                metrics_text = f"Quality Score: {sample['quality_score']:.3f}\n\n"
                metrics_text += f"Pixel Acc: {metrics['pixel_accuracy']:.3f}\n"
                metrics_text += f"Fg-mIoU: {metrics['foreground_miou']:.3f}\n"
                metrics_text += f"Macro F1: {metrics['macro_f1']:.3f}\n\n"
                metrics_text += "Per-class IoU:\n"

                for j, class_name in enumerate(class_names):
                    if j < len(metrics['per_class_ious']):
                        iou = metrics['per_class_ious'][j]
                        short_name = class_name[:8]
                        metrics_text += f"{short_name}: {iou:.3f}\n"

                axes[i, 4].text(0.05, 0.95, metrics_text, transform=axes[i, 4].transAxes,
                                fontsize=8, verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'{model_name.title()} Segmentation - Sample Results',
                     fontsize=16, fontweight='bold', y=0.98)

        # Save
        filename = f'final_{model_name}_samples.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")

    def create_performance_analysis(self, eval_result):
        """Create performance analysis charts"""
        model_name = eval_result['model_name']
        stats = eval_result['statistics']
        class_names = eval_result['class_names']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall metrics
        metric_names = ['pixel_accuracy', 'foreground_miou', 'macro_f1', 'macro_dice']
        metric_labels = ['Pixel Accuracy', 'Foreground mIoU', 'Macro F1', 'Macro Dice']

        means = [stats['overall'][m]['mean'] for m in metric_names]
        stds = [stats['overall'][m]['std'] for m in metric_names]

        x_pos = np.arange(len(metric_names))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])

        ax1.set_title('Overall Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_labels, rotation=15, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Per-class IoU
        class_ious = [stats['per_class'][cls]['iou']['mean'] for cls in class_names]
        class_ious_std = [stats['per_class'][cls]['iou']['std'] for cls in class_names]

        x_pos_class = np.arange(len(class_names))
        ax2.bar(x_pos_class, class_ious, yerr=class_ious_std, capsize=3, alpha=0.8,
                color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))

        ax2.set_title('Per-Class IoU Performance', fontweight='bold')
        ax2.set_ylabel('IoU Score')
        ax2.set_xticks(x_pos_class)
        ax2.set_xticklabels([cls[:8] for cls in class_names], rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        for i, (mean, std) in enumerate(zip(class_ious, class_ious_std)):
            ax2.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 3. Quality score distribution
        quality_scores = [r['quality_score'] for r in eval_result['all_results']]
        ax3.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(quality_scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(quality_scores):.3f}')
        ax3.axvline(np.median(quality_scores), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(quality_scores):.3f}')

        ax3.set_title('Quality Score Distribution', fontweight='bold')
        ax3.set_xlabel('Quality Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Metrics correlation
        corr_metrics = ['pixel_accuracy', 'foreground_miou', 'macro_f1', 'macro_dice']
        corr_data = []
        for metric in corr_metrics:
            values = [r['metrics'][metric] for r in eval_result['all_results']]
            corr_data.append(values)

        corr_matrix = np.corrcoef(corr_data)

        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Metrics Correlation', fontweight='bold')

        labels = ['Pixel Acc.', 'Fg-mIoU', 'Macro F1', 'Macro Dice']
        ax4.set_xticks(np.arange(len(labels)))
        ax4.set_yticks(np.arange(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center",
                         color="black", fontweight='bold')

        plt.colorbar(im, ax=ax4, shrink=0.8)

        plt.tight_layout()

        # Save
        filename = f'final_{model_name}_analysis.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")

    def create_model_comparison(self, all_evaluations):
        """Create model comparison charts"""
        if len(all_evaluations) < 2:
            return

        print("Creating model comparison charts")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Prepare data
        model_names = []
        model_display = {'multiclass': 'Multi-class', 'tumor': 'Tumor', 'joint': 'Joint'}

        pixel_acc = []
        fg_miou = []
        macro_f1 = []
        quality = []

        for eval_result in all_evaluations:
            model_name = eval_result['model_name']
            stats = eval_result['statistics']['overall']

            model_names.append(model_display.get(model_name, model_name.title()))
            pixel_acc.append(stats['pixel_accuracy']['mean'])
            fg_miou.append(stats['foreground_miou']['mean'])
            macro_f1.append(stats['macro_f1']['mean'])
            quality.append(stats['quality_score']['mean'])

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(model_names)]
        x_pos = np.arange(len(model_names))

        # 1. Pixel accuracy
        bars1 = axes[0, 0].bar(x_pos, pixel_acc, color=colors, alpha=0.8)
        axes[0, 0].set_title('Pixel Accuracy Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Pixel Accuracy')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)

        for i, val in enumerate(pixel_acc):
            axes[0, 0].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Foreground mIoU
        bars2 = axes[0, 1].bar(x_pos, fg_miou, color=colors, alpha=0.8)
        axes[0, 1].set_title('Foreground mIoU Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('Foreground mIoU')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        for i, val in enumerate(fg_miou):
            axes[0, 1].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Macro F1
        bars3 = axes[0, 2].bar(x_pos, macro_f1, color=colors, alpha=0.8)
        axes[0, 2].set_title('Macro F1 Comparison', fontweight='bold')
        axes[0, 2].set_ylabel('Macro F1')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(model_names)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3)

        for i, val in enumerate(macro_f1):
            axes[0, 2].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. Quality score
        bars4 = axes[1, 0].bar(x_pos, quality, color=colors, alpha=0.8)
        axes[1, 0].set_title('Quality Score Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        for i, val in enumerate(quality):
            axes[1, 0].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 5. Radar chart
        categories = ['Pixel Acc.', 'Fg-mIoU', 'Macro F1', 'Quality']
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax_radar = plt.subplot(2, 3, 5, projection='polar')

        for i, (eval_result, color) in enumerate(zip(all_evaluations, colors)):
            stats = eval_result['statistics']['overall']
            values = [
                stats['pixel_accuracy']['mean'],
                stats['foreground_miou']['mean'],
                stats['macro_f1']['mean'],
                stats['quality_score']['mean']
            ]
            values += values[:1]

            model_label = model_display.get(eval_result['model_name'], eval_result['model_name'].title())
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_label, color=color)
            ax_radar.fill(angles, values, alpha=0.25, color=color)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Radar', fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax_radar.grid(True)

        # 6. Summary table
        axes[1, 2].axis('off')

        table_data = []
        for eval_result in all_evaluations:
            model_name = eval_result['model_name']
            stats = eval_result['statistics']['overall']

            table_data.append([
                model_display.get(model_name, model_name.title()),
                f"{len(eval_result['all_results'])}",
                f"{eval_result['num_classes']}",
                f"{stats['foreground_miou']['mean']:.3f}"
            ])

        table = axes[1, 2].table(cellText=table_data,
                                 colLabels=['Model', 'Samples', 'Classes', 'Fg-mIoU'],
                                 cellLoc='center',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # Style table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        axes[1, 2].set_title('Model Summary', fontweight='bold', pad=20)

        plt.tight_layout()

        # Save
        filename = 'final_model_comparison.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")

    def generate_excel_report(self, all_evaluations):
        """Generate comprehensive Excel report"""
        filename = os.path.join(self.output_dir, 'final_evaluation_report.xlsx')

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Overview sheet
            overview_data = []
            for eval_result in all_evaluations:
                model_name = eval_result['model_name']
                stats = eval_result['statistics']['overall']

                overview_data.append({
                    'Model': model_name.title(),
                    'Total_Samples': len(eval_result['all_results']),
                    'Classes': eval_result['num_classes'],
                    'Pixel_Accuracy_Mean': f"{stats['pixel_accuracy']['mean']:.4f}",
                    'Pixel_Accuracy_Std': f"{stats['pixel_accuracy']['std']:.4f}",
                    'Foreground_mIoU_Mean': f"{stats['foreground_miou']['mean']:.4f}",
                    'Foreground_mIoU_Std': f"{stats['foreground_miou']['std']:.4f}",
                    'Macro_F1_Mean': f"{stats['macro_f1']['mean']:.4f}",
                    'Macro_F1_Std': f"{stats['macro_f1']['std']:.4f}",
                    'Quality_Score_Mean': f"{stats['quality_score']['mean']:.4f}",
                    'Quality_Score_Std': f"{stats['quality_score']['std']:.4f}"
                })

            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Overview', index=False)

            # Detailed sheets for each model
            for eval_result in all_evaluations:
                model_name = eval_result['model_name']

                # Per-class metrics
                class_data = []
                for class_name in eval_result['class_names']:
                    class_stats = eval_result['statistics']['per_class'][class_name]
                    class_data.append({
                        'Class': class_name,
                        'Precision_Mean': f"{class_stats['precision']['mean']:.4f}",
                        'Precision_Std': f"{class_stats['precision']['std']:.4f}",
                        'Recall_Mean': f"{class_stats['recall']['mean']:.4f}",
                        'Recall_Std': f"{class_stats['recall']['std']:.4f}",
                        'F1_Mean': f"{class_stats['f1_score']['mean']:.4f}",
                        'F1_Std': f"{class_stats['f1_score']['std']:.4f}",
                        'IoU_Mean': f"{class_stats['iou']['mean']:.4f}",
                        'IoU_Std': f"{class_stats['iou']['std']:.4f}",
                        'Dice_Mean': f"{class_stats['dice']['mean']:.4f}",
                        'Dice_Std': f"{class_stats['dice']['std']:.4f}",
                        'IoU_Min': f"{class_stats['iou']['min']:.4f}",
                        'IoU_Max': f"{class_stats['iou']['max']:.4f}"
                    })

                class_df = pd.DataFrame(class_data)
                sheet_name = f'{model_name.title()}_Details'
                class_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved Excel report: {filename}")

    def save_json_results(self, all_evaluations):
        """Save detailed JSON results"""
        export_data = {}

        for eval_result in all_evaluations:
            model_name = eval_result['model_name']

            export_data[model_name] = {
                'model_info': {
                    'name': model_name,
                    'num_classes': eval_result['num_classes'],
                    'class_names': eval_result['class_names'],
                    'total_samples': len(eval_result['all_results']),
                    'display_samples': len(eval_result['display_samples'])
                },
                'statistics': eval_result['statistics'],
                'display_sample_indices': [s['idx'] for s in eval_result['display_samples']],
                'display_sample_scores': [s['quality_score'] for s in eval_result['display_samples']],
                'evaluation_timestamp': eval_result['evaluation_timestamp']
            }

        export_data['evaluation_summary'] = {
            'total_models': len(all_evaluations),
            'evaluation_date': datetime.now().isoformat(),
            'sampling_strategy': 'diverse_quality_based',
            'display_samples_per_model': 20
        }

        filename = os.path.join(self.output_dir, 'final_evaluation_data.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Saved JSON data: {filename}")

    def print_summary_report(self, all_evaluations):
        """Print comprehensive summary report"""
        print("\n" + "=" * 100)
        print("FINAL ACADEMIC EVALUATION SUMMARY")
        print("=" * 100)

        # Model ranking
        rankings = []
        for eval_result in all_evaluations:
            stats = eval_result['statistics']['overall']
            rankings.append({
                'name': eval_result['model_name'],
                'fg_miou': stats['foreground_miou']['mean'],
                'pixel_acc': stats['pixel_accuracy']['mean'],
                'quality': stats['quality_score']['mean'],
                'samples': len(eval_result['all_results'])
            })

        # Sort by foreground mIoU
        rankings.sort(key=lambda x: x['fg_miou'], reverse=True)

        print(f"\nMODEL PERFORMANCE RANKING (by Foreground mIoU):")
        print("-" * 80)
        for i, model in enumerate(rankings):
            print(f"  {i + 1}. {model['name'].upper():12} | "
                  f"Fg-mIoU: {model['fg_miou']:.3f} | "
                  f"Pixel-Acc: {model['pixel_acc']:.3f} | "
                  f"Quality: {model['quality']:.3f} | "
                  f"Samples: {model['samples']}")

        print(f"\nDETAILED MODEL ANALYSIS:")
        print("-" * 80)

        for eval_result in all_evaluations:
            model_name = eval_result['model_name']
            stats = eval_result['statistics']['overall']
            n_samples = len(eval_result['all_results'])
            n_display = len(eval_result['display_samples'])

            print(f"\n🔹 {model_name.upper()} MODEL:")
            print(f"   Dataset: {n_samples} total samples, {n_display} displayed")
            print(
                f"   Pixel Accuracy:     {stats['pixel_accuracy']['mean']:.4f} ± {stats['pixel_accuracy']['std']:.4f}")
            print(
                f"   Foreground mIoU:    {stats['foreground_miou']['mean']:.4f} ± {stats['foreground_miou']['std']:.4f}")
            print(f"   Macro F1 Score:     {stats['macro_f1']['mean']:.4f} ± {stats['macro_f1']['std']:.4f}")
            print(f"   Quality Score:      {stats['quality_score']['mean']:.4f} ± {stats['quality_score']['std']:.4f}")

            # Performance assessment
            fg_miou = stats['foreground_miou']['mean']
            if fg_miou >= 0.8:
                level = "EXCELLENT ⭐⭐⭐"
            elif fg_miou >= 0.6:
                level = "GOOD ⭐⭐"
            elif fg_miou >= 0.4:
                level = "FAIR ⭐"
            else:
                level = "NEEDS IMPROVEMENT ❌"

            print(f"   Performance Level:  {level}")

            # Sample quality distribution
            display_scores = [s['quality_score'] for s in eval_result['display_samples']]
            print(f"   Display Samples:    Quality range [{min(display_scores):.3f}, {max(display_scores):.3f}]")

        print(f"\nKEY INSIGHTS:")
        print("-" * 80)

        best_model = max(rankings, key=lambda x: x['fg_miou'])
        worst_model = min(rankings, key=lambda x: x['fg_miou'])

        print(f"🏆 Best Model: {best_model['name'].upper()} (Fg-mIoU: {best_model['fg_miou']:.3f})")
        print(f"🔧 Needs Improvement: {worst_model['name'].upper()} (Fg-mIoU: {worst_model['fg_miou']:.3f})")

        if len(rankings) >= 3:
            avg_performance = np.mean([r['fg_miou'] for r in rankings])
            print(f"📊 Average Performance: {avg_performance:.3f}")

        # Recommendations
        print(f"\nRECOMMENDations:")
        print("-" * 80)
        for model in rankings:
            if model['fg_miou'] < 0.4:
                print(
                    f"• {model['name'].title()}: Consider data augmentation, loss function tuning, or architecture changes")
            elif model['fg_miou'] < 0.6:
                print(f"• {model['name'].title()}: Good baseline, try ensemble methods or hyperparameter optimization")
            else:
                print(f"• {model['name'].title()}: Excellent performance, suitable for deployment")

        print(f"\nGENERATED FILES:")
        print("-" * 80)
        print(f"📊 Visualizations:")
        print(f"   - final_model_comparison.png")
        for eval_result in all_evaluations:
            model_name = eval_result['model_name']
            print(f"   - final_{model_name}_samples.png")
            print(f"   - final_{model_name}_analysis.png")

        print(f"📋 Reports:")
        print(f"   - final_evaluation_report.xlsx")
        print(f"   - final_evaluation_data.json")

        print("\n" + "=" * 100)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 100)

    def run_complete_evaluation(self, n_samples=20):
        """Run the complete evaluation pipeline"""
        print("🚀 STARTING FINAL ACADEMIC EVALUATION")
        print(f"📊 Target samples per model: {n_samples}")
        print(f"📈 Using diverse sampling strategy")

        all_evaluations = []

        # 1. Multiclass bone segmentation
        try:
            print(f"\n{'=' * 60}")
            print("MULTICLASS BONE SEGMENTATION EVALUATION")
            print(f"{'=' * 60}")

            if os.path.exists("best_unet_smp.pth"):
                dataset = FemurSegmentationDataset(transform=self.transform)
                model = get_model(config.NUM_CLASSES).to(self.device)
                checkpoint = torch.load("best_unet_smp.pth", map_location=self.device)
                model.load_state_dict(checkpoint)

                class_names = ['Background', 'Apophysis', 'Epiphysis', 'Metaphysis',
                               'Diaphysis', 'Surface Tumour', 'In-Bone Tumour', 'Joint']

                result = self.evaluate_model(model, dataset, config.NUM_CLASSES,
                                             class_names, 'multiclass', n_samples)

                if result:
                    all_evaluations.append(result)
                    self.create_sample_visualization(result)
                    self.create_performance_analysis(result)
            else:
                print("❌ Model file not found: best_unet_smp.pth")

        except Exception as e:
            print(f"❌ Multiclass evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        # 2. Tumor segmentation
        try:
            print(f"\n{'=' * 60}")
            print("TUMOR SEGMENTATION EVALUATION")
            print(f"{'=' * 60}")

            if os.path.exists(cfg_tumor.MODEL_NAME):
                dataset = TumorSegmentationDataset(transform=self.transform, only_positive=False)
                model = get_model(cfg_tumor.NUM_CLASSES).to(self.device)
                checkpoint = torch.load(cfg_tumor.MODEL_NAME, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                class_names = ['Background', 'Surface Tumour', 'In-Bone Tumour']

                result = self.evaluate_model(model, dataset, cfg_tumor.NUM_CLASSES,
                                             class_names, 'tumor', n_samples)

                if result:
                    all_evaluations.append(result)
                    self.create_sample_visualization(result)
                    self.create_performance_analysis(result)
            else:
                print(f"❌ Model file not found: {cfg_tumor.MODEL_NAME}")

        except Exception as e:
            print(f"❌ Tumor evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        # 3. Joint segmentation
        try:
            print(f"\n{'=' * 60}")
            print("JOINT SEGMENTATION EVALUATION")
            print(f"{'=' * 60}")

            if os.path.exists(cfg_joint.MODEL_NAME):
                dataset = JointSegmentationDataset(transform=self.transform, only_positive=False)
                model = get_model(cfg_joint.NUM_CLASSES).to(self.device)
                checkpoint = torch.load(cfg_joint.MODEL_NAME, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                class_names = ['Background', 'Joint']

                result = self.evaluate_model(model, dataset, cfg_joint.NUM_CLASSES,
                                             class_names, 'joint', n_samples)

                if result:
                    all_evaluations.append(result)
                    self.create_sample_visualization(result)
                    self.create_performance_analysis(result)
            else:
                print(f"❌ Model file not found: {cfg_joint.MODEL_NAME}")

        except Exception as e:
            print(f"❌ Joint evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        # Generate comprehensive reports
        if all_evaluations:
            print(f"\n{'=' * 60}")
            print("GENERATING COMPREHENSIVE REPORTS")
            print(f"{'=' * 60}")

            # Create comparison charts
            self.create_model_comparison(all_evaluations)

            # Generate Excel report
            self.generate_excel_report(all_evaluations)

            # Save JSON data
            self.save_json_results(all_evaluations)

            # Print summary
            self.print_summary_report(all_evaluations)

            print(f"\n🎉 EVALUATION SUCCESSFULLY COMPLETED!")
            print(f"📂 All results saved to: {self.output_dir}")

            return all_evaluations
        else:
            print(f"\n⚠️ No models were successfully evaluated")
            print(f"Please check that model files exist and are accessible")
            return None


def main():
    """Main function to run the evaluation"""
    print("🎓 FINAL ACADEMIC MODEL EVALUATION SYSTEM")
    print("=" * 80)
    print("✨ Features:")
    print("   ✅ Comprehensive evaluation of all 3 models")
    print("   ✅ 20+ diverse samples per model (avoiding accuracy=1.0 issue)")
    print("   ✅ English-only output (no encoding issues)")
    print("   ✅ Academic-quality visualizations")
    print("   ✅ Detailed statistical analysis")
    print("   ✅ Excel and JSON export")
    print("   ✅ Performance ranking and recommendations")
    print("=" * 80)

    # Initialize evaluator
    evaluator = FinalAcademicEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="academic_results"
    )

    try:
        # Run complete evaluation
        results = evaluator.run_complete_evaluation(n_samples=20)

        if results:
            print(f"\n🎊 SUCCESS! Academic evaluation completed")
            print(f"Generated {len(results)} model evaluation reports")

            # Quick summary for immediate feedback
            print(f"\n📈 QUICK SUMMARY:")
            for result in results:
                model_name = result['model_name']
                fg_miou = result['statistics']['overall']['foreground_miou']['mean']
                accuracy = result['statistics']['overall']['pixel_accuracy']['mean']
                print(f"   {model_name.title():12}: Fg-mIoU={fg_miou:.3f}, Accuracy={accuracy:.3f}")

    except KeyboardInterrupt:
        print(f"\n⏹️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()