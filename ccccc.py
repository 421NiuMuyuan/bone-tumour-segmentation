# fixed_evaluation.py
# 修复matplotlib兼容性问题的评估工具

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
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

# 导入必要模块
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
    """修复版模型评估器 - 解决matplotlib兼容性问题"""

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🔧 使用设备: {self.device}")

        # 设置matplotlib样式 - 兼容性设置
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
        """反归一化图像用于显示"""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1)
        return img_np

    def calculate_sample_metrics(self, pred_mask, gt_mask, num_classes):
        """计算单个样本的指标"""
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()

        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        # 整体准确率
        accuracy = (pred_flat == gt_flat).mean()

        # 每类IoU和Dice
        ious = []
        dices = []

        for class_id in range(num_classes):
            # 计算TP, FP, FN
            tp = ((pred_flat == class_id) & (gt_flat == class_id)).sum()
            fp = ((pred_flat == class_id) & (gt_flat != class_id)).sum()
            fn = ((pred_flat != class_id) & (gt_flat == class_id)).sum()

            # IoU和Dice
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

            ious.append(iou)
            dices.append(dice)

        # 排除背景类别的mIoU
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
        """评估所有样本并找到最佳结果"""
        print(f"🔍 正在评估 {model_name} 的所有 {len(dataset)} 个样本...")

        model.eval()
        all_results = []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"评估{model_name}"):
                try:
                    img, gt_mask = dataset[idx]

                    # 模型推理
                    img_batch = img.unsqueeze(0).to(self.device)
                    output = model(img_batch)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_mask = output.argmax(dim=1).squeeze().cpu()

                    # 计算指标
                    metrics = self.calculate_sample_metrics(pred_mask, gt_mask, num_classes)

                    # 保存结果
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
                    print(f"⚠️ 样本 {idx} 处理失败: {e}")
                    continue

        print(f"✅ 成功评估 {len(all_results)} 个样本")
        return all_results

    def get_best_samples(self, all_results, top_k=6):
        """获取效果最好的K个样本"""
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        best_samples = sorted_results[:top_k]

        print(f"🏆 选择了表现最佳的 {len(best_samples)} 个样本进行展示")
        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']
            print(
                f"  #{i + 1}: 样本{sample['idx']}, 前景mIoU={metrics['miou_foreground']:.3f}, 准确率={metrics['accuracy']:.3f}")

        return best_samples

    def calculate_overall_statistics(self, all_results):
        """计算整体统计信息"""
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
        """可视化最佳结果 - 修复版本"""
        n_samples = len(best_samples)

        if model_type == 'multiclass':
            self._visualize_multiclass_fixed(best_samples, class_names, stats)
        elif model_type == 'tumor':
            self._visualize_tumor_fixed(best_samples, class_names, stats)
        elif model_type == 'joint':
            self._visualize_joint_fixed(best_samples, class_names, stats)

    def _visualize_multiclass_fixed(self, best_samples, class_names, stats):
        """多类分割可视化 - 修复版"""
        n_samples = len(best_samples)

        # 创建颜色映射
        colors = ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292', '#AED581']
        cmap = ListedColormap(colors[:len(class_names)])

        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']

            # 原始图像
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'样本 {sample["idx"]}\n原始X光片', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground Truth
            axes[i, 1].imshow(sample['gt_mask'], cmap=cmap, vmin=0, vmax=len(class_names) - 1)
            axes[i, 1].set_title('标准答案', fontweight='bold')
            axes[i, 1].axis('off')

            # 预测结果
            axes[i, 2].imshow(sample['pred_mask'], cmap=cmap, vmin=0, vmax=len(class_names) - 1)
            axes[i, 2].set_title(f'模型预测\nmIoU: {metrics["miou_foreground"]:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # 预测置信度
            max_prob = np.max(sample['pred_probs'], axis=0)
            im = axes[i, 3].imshow(max_prob, cmap='viridis', vmin=0, vmax=1)
            axes[i, 3].set_title(f'预测置信度\n准确率: {metrics["accuracy"]:.3f}', fontweight='bold')
            axes[i, 3].axis('off')

            # 指标展示
            axes[i, 4].axis('off')
            metrics_text = "各类别IoU:\n"
            for j, class_name in enumerate(class_names):
                iou = metrics['per_class_iou'][j]
                metrics_text += f"{class_name[:8]}: {iou:.3f}\n"

            axes[i, 4].text(0.05, 0.95, metrics_text, transform=axes[i, 4].transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'多类骨骼分割最佳结果\n平均前景mIoU: {stats["miou_foreground"]["mean"]:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig('fixed_multiclass_results.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存
        print("💾 保存: fixed_multiclass_results.png")

    def _visualize_tumor_fixed(self, best_samples, class_names, stats):
        """肿瘤分割可视化 - 修复版"""
        n_samples = len(best_samples)

        tumor_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        tumor_cmap = ListedColormap(tumor_colors)

        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']

            # 原始图像
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'样本 {sample["idx"]}\n原始X光片', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground Truth
            axes[i, 1].imshow(sample['gt_mask'], cmap=tumor_cmap, vmin=0, vmax=2)
            axes[i, 1].set_title('标准答案\n(红=表面, 蓝=骨内)', fontweight='bold')
            axes[i, 1].axis('off')

            # 预测结果
            axes[i, 2].imshow(sample['pred_mask'], cmap=tumor_cmap, vmin=0, vmax=2)
            axes[i, 2].set_title(f'模型预测\nmIoU: {metrics["miou_foreground"]:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # 表面肿瘤概率
            if len(sample['pred_probs'].shape) == 3 and sample['pred_probs'].shape[0] > 1:
                surf_prob = sample['pred_probs'][1]
                axes[i, 3].imshow(surf_prob, cmap='Reds', vmin=0, vmax=1)
                axes[i, 3].set_title('表面肿瘤概率', fontweight='bold')
            else:
                axes[i, 3].text(0.5, 0.5, '无概率数据', ha='center', va='center', transform=axes[i, 3].transAxes)
                axes[i, 3].set_title('概率图', fontweight='bold')
            axes[i, 3].axis('off')

            # 指标展示
            axes[i, 4].axis('off')
            metrics_text = f"""肿瘤分割指标:

前景mIoU: {metrics['miou_foreground']:.3f}
整体准确率: {metrics['accuracy']:.3f}
整体mIoU: {metrics['miou_all']:.3f}

各类别IoU:
背景: {metrics['per_class_iou'][0]:.3f}
表面: {metrics['per_class_iou'][1]:.3f}
骨内: {metrics['per_class_iou'][2]:.3f}
            """

            axes[i, 4].text(0.05, 0.95, metrics_text, transform=axes[i, 4].transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'肿瘤分割最佳结果\n平均前景mIoU: {stats["miou_foreground"]["mean"]:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig('fixed_tumor_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("💾 保存: fixed_tumor_results.png")

    def _visualize_joint_fixed(self, best_samples, class_names, stats):
        """关节分割可视化 - 修复版"""
        n_samples = len(best_samples)

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(best_samples):
            metrics = sample['metrics']

            # 原始图像
            axes[i, 0].imshow(sample['image'])
            axes[i, 0].set_title(f'样本 {sample["idx"]}\n原始X光片', fontweight='bold')
            axes[i, 0].axis('off')

            # Ground Truth
            axes[i, 1].imshow(sample['gt_mask'], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('标准答案\n(白=关节)', fontweight='bold')
            axes[i, 1].axis('off')

            # 预测结果
            axes[i, 2].imshow(sample['pred_mask'], cmap='gray', vmin=0, vmax=1)
            joint_iou = metrics['per_class_iou'][1] if len(metrics['per_class_iou']) > 1 else 0
            axes[i, 2].set_title(f'模型预测\n关节IoU: {joint_iou:.3f}', fontweight='bold')
            axes[i, 2].axis('off')

            # 指标展示
            axes[i, 3].axis('off')
            metrics_text = f"""关节分割指标:

关节IoU: {joint_iou:.3f}
关节Dice: {metrics['per_class_dice'][1] if len(metrics['per_class_dice']) > 1 else 0:.3f}
整体准确率: {metrics['accuracy']:.3f}
前景mIoU: {metrics['miou_foreground']:.3f}

评分: {sample['score']:.3f}
            """

            axes[i, 3].text(0.05, 0.95, metrics_text, transform=axes[i, 3].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'关节分割最佳结果\n平均关节IoU: {stats["miou_foreground"]["mean"]:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig('fixed_joint_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("💾 保存: fixed_joint_results.png")

    def create_fixed_performance_summary(self, all_model_results):
        """创建修复版性能汇总对比"""
        if len(all_model_results) < 2:
            return

        print("\n" + "=" * 80)
        print("🏁 所有模型性能汇总对比")
        print("=" * 80)

        # 准备对比数据
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

        # 创建简化的对比图表 - 避免capthick参数
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        models = [d['model'] for d in model_data]
        accuracies = [d['accuracy_mean'] for d in model_data]
        acc_stds = [d['accuracy_std'] for d in model_data]
        mious = [d['miou_mean'] for d in model_data]
        miou_stds = [d['miou_std'] for d in model_data]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(models)]

        # 准确率对比 - 使用简化的errorbar
        x_pos = np.arange(len(models))
        bars1 = ax1.bar(x_pos, accuracies, color=colors, alpha=0.8)
        ax1.errorbar(x_pos, accuracies, yerr=acc_stds, fmt='none', color='black', capsize=3)

        ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (acc, std) in enumerate(zip(accuracies, acc_stds)):
            ax1.text(i, acc + std + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # mIoU对比
        bars2 = ax2.bar(x_pos, mious, color=colors, alpha=0.8)
        ax2.errorbar(x_pos, mious, yerr=miou_stds, fmt='none', color='black', capsize=3)

        ax2.set_title('模型前景mIoU对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('前景mIoU', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (miou, std) in enumerate(zip(mious, miou_stds)):
            ax2.text(i, miou + std + 0.02, f'{miou:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('模型性能全面对比分析', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('fixed_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("💾 保存: fixed_model_comparison.png")

        # 打印排名
        best_acc_idx = np.argmax(accuracies)
        best_miou_idx = np.argmax(mious)

        print(f"\n🏆 性能冠军:")
        print(f"   准确率最高: {models[best_acc_idx]} ({accuracies[best_acc_idx]:.3f})")
        print(f"   mIoU最高: {models[best_miou_idx]} ({mious[best_miou_idx]:.3f})")

    def run_fixed_evaluation(self):
        """运行修复版评估"""
        print("🚀 启动修复版模型评估系统")

        all_results = []

        # 1. 多类骨骼分割
        try:
            print("\n" + "=" * 60)
            print("🦴 Week 1: 多类骨骼分割模型评估")
            print("=" * 60)

            dataset = FemurSegmentationDataset(transform=self.transform)
            model = get_model(config.NUM_CLASSES).to(self.device)
            checkpoint = torch.load("best_unet_smp.pth", map_location=self.device)
            model.load_state_dict(checkpoint)

            class_names = ['Background', 'Apophysis', 'Epiphysis', 'Metaphysis',
                           'Diaphysis', 'Surface Tumour', 'In-Bone Tumour', 'Joint']

            all_samples = self.evaluate_all_samples(model, dataset, config.NUM_CLASSES, "多类骨骼分割")
            best_samples = self.get_best_samples(all_samples)
            stats = self.calculate_overall_statistics(all_samples)

            self.visualize_best_results(best_samples, class_names, stats, 'multiclass')

            all_results.append({
                'model_type': 'multiclass',
                'all_results': all_samples,
                'best_samples': best_samples,
                'statistics': stats
            })

            print(f"\n📊 多类骨骼分割统计:")
            print(f"   前景mIoU: {stats['miou_foreground']['mean']:.3f} ± {stats['miou_foreground']['std']:.3f}")
            print(f"   准确率: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")

        except Exception as e:
            print(f"❌ 多类模型评估失败: {e}")

        # 2. 肿瘤分割
        try:
            print("\n" + "=" * 60)
            print("🎯 Week 2: 肿瘤分割模型评估")
            print("=" * 60)

            dataset = TumorSegmentationDataset(transform=self.transform, only_positive=False)
            model = get_model(cfg_tumor.NUM_CLASSES).to(self.device)
            checkpoint = torch.load(cfg_tumor.MODEL_NAME, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            class_names = ['Background', 'Surface Tumour', 'In-Bone Tumour']

            all_samples = self.evaluate_all_samples(model, dataset, cfg_tumor.NUM_CLASSES, "肿瘤分割")
            best_samples = self.get_best_samples(all_samples)
            stats = self.calculate_overall_statistics(all_samples)

            self.visualize_best_results(best_samples, class_names, stats, 'tumor')

            all_results.append({
                'model_type': 'tumor',
                'all_results': all_samples,
                'best_samples': best_samples,
                'statistics': stats
            })

            print(f"\n📊 肿瘤分割统计:")
            print(f"   前景mIoU: {stats['miou_foreground']['mean']:.3f} ± {stats['miou_foreground']['std']:.3f}")
            print(f"   准确率: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")

        except Exception as e:
            print(f"❌ 肿瘤模型评估失败: {e}")

        # 3. 关节分割
        try:
            print("\n" + "=" * 60)
            print("🔗 Week 3: 关节分割模型评估")
            print("=" * 60)

            dataset = JointSegmentationDataset(transform=self.transform, only_positive=False)
            model = get_model(cfg_joint.NUM_CLASSES).to(self.device)
            checkpoint = torch.load(cfg_joint.MODEL_NAME, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            class_names = ['Background', 'Joint']

            all_samples = self.evaluate_all_samples(model, dataset, cfg_joint.NUM_CLASSES, "关节分割")
            best_samples = self.get_best_samples(all_samples)
            stats = self.calculate_overall_statistics(all_samples)

            self.visualize_best_results(best_samples, class_names, stats, 'joint')

            all_results.append({
                'model_type': 'joint',
                'all_results': all_samples,
                'best_samples': best_samples,
                'statistics': stats
            })

            print(f"\n📊 关节分割统计:")
            print(f"   前景mIoU: {stats['miou_foreground']['mean']:.3f} ± {stats['miou_foreground']['std']:.3f}")
            print(f"   准确率: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")

        except Exception as e:
            print(f"❌ 关节模型评估失败: {e}")

        # 创建综合对比
        if len(all_results) > 1:
            self.create_fixed_performance_summary(all_results)

        # 保存详细结果
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

        print(f"\n✅ 修复版评估完成！生成的文件:")
        print(f"   📊 fixed_multiclass_results.png - 多类分割最佳结果")
        print(f"   🎯 fixed_tumor_results.png - 肿瘤分割最佳结果")
        print(f"   🔗 fixed_joint_results.png - 关节分割最佳结果")
        print(f"   📈 fixed_model_comparison.png - 模型性能对比")
        print(f"   📄 fixed_evaluation_summary.json - 详细统计数据")

        return all_results


def main():
    """主函数"""
    print("🔧 修复版模型评估工具")
    print("解决matplotlib兼容性问题，生成最佳预测结果")
    print()

    evaluator = FixedModelEvaluator()

    try:
        results = evaluator.run_fixed_evaluation()

        if results:
            print(f"\n🎉 评估成功完成！")

            # 打印最终排名
            model_scores = []
            for result in results:
                stats = result['statistics']
                model_scores.append({
                    'name': result['model_type'].title(),
                    'miou': stats['miou_foreground']['mean'],
                    'accuracy': stats['accuracy']['mean']
                })

            # 按mIoU排序
            model_scores.sort(key=lambda x: x['miou'], reverse=True)

            print(f"\n🏆 最终模型排名 (按前景mIoU):")
            for i, model in enumerate(model_scores):
                print(f"   {i + 1}. {model['name']:12} - mIoU: {model['miou']:.3f}, 准确率: {model['accuracy']:.3f}")
        else:
            print(f"\n⚠️ 未能成功评估任何模型")

    except KeyboardInterrupt:
        print("\n⏹️ 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()