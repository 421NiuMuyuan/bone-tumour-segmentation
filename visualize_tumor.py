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
    """创建肉眼易识别的颜色映射"""
    # 定义清晰区分的颜色 (RGB)
    colors = [
        [0.0, 0.0, 0.0],  # 0: 黑色 - 背景
        [1.0, 0.0, 0.0],  # 1: 红色 - 表面肿瘤
        [0.0, 0.0, 1.0],  # 2: 蓝色 - 骨内肿瘤
    ]
    return ListedColormap(colors)


def create_overlay_visualization(image, mask, alpha=0.6):
    """创建原图与mask的叠加可视化"""
    # 确保image是[0,1]范围的RGB图像
    if image.max() > 1.0:
        image = image / 255.0

    # 创建彩色mask
    colored_mask = np.zeros((*mask.shape, 3))

    # 表面肿瘤 - 红色
    surface_tumor = (mask == 1)
    colored_mask[surface_tumor] = [1.0, 0.0, 0.0]

    # 骨内肿瘤 - 蓝色
    inbone_tumor = (mask == 2)
    colored_mask[inbone_tumor] = [0.0, 0.0, 1.0]

    # 叠加显示
    overlay = image.copy()
    tumor_regions = (mask > 0)
    overlay[tumor_regions] = (1 - alpha) * image[tumor_regions] + alpha * colored_mask[tumor_regions]

    return overlay


def analyze_tumor_statistics(mask, image_name=""):
    """分析肿瘤统计信息"""
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
    """可视化Week 2肿瘤Ground Truth，生成肉眼可识别的图像"""
    print("=== Week 2: 肿瘤Ground Truth可视化验证 ===")

    # 数据预处理
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # 加载数据集
    dataset = TumorSegmentationDataset(transform=transform, only_positive=False)
    print(f"数据集大小: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ 数据集为空，请检查数据路径")
        return

    # 创建颜色映射
    tumor_cmap = create_readable_colormap()

    # 随机选择样本，优先选择有肿瘤的样本
    positive_indices = []
    negative_indices = []

    for i in range(len(dataset)):
        _, mask = dataset[i]
        if (mask > 0).any():
            positive_indices.append(i)
        else:
            negative_indices.append(i)

    print(f"阳性样本(有肿瘤): {len(positive_indices)}")
    print(f"阴性样本(无肿瘤): {len(negative_indices)}")

    # 选择展示样本：优先展示阳性样本
    selected_indices = []
    if positive_indices:
        random.shuffle(positive_indices)
        selected_indices.extend(positive_indices[:min(n_samples - 2, len(positive_indices))])

    if negative_indices and len(selected_indices) < n_samples:
        random.shuffle(negative_indices)
        needed = n_samples - len(selected_indices)
        selected_indices.extend(negative_indices[:min(needed, len(negative_indices))])

    # 创建大图展示
    n_cols = 4  # 原图、GT、叠加图、统计图
    n_rows = len(selected_indices)

    fig = plt.figure(figsize=(16, 4 * n_rows))

    all_stats = []

    for row_idx, idx in enumerate(selected_indices):
        img, gt_mask = dataset[idx]

        # 反归一化图像用于显示
        img_display = img.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display + 1.0) / 2.0  # 从[-1,1]转到[0,1]
        img_display = np.clip(img_display, 0, 1)

        gt_mask_np = gt_mask.cpu().numpy()

        # 统计信息
        stats = analyze_tumor_statistics(gt_mask_np, f"Sample_{idx}")
        all_stats.append(stats)

        # 创建叠加图
        overlay = create_overlay_visualization(img_display, gt_mask_np)

        # 绘制原图
        ax1 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 1)
        ax1.imshow(img_display)
        ax1.set_title(f'Sample {idx}\nOriginal X-ray', fontsize=10, fontweight='bold')
        ax1.axis('off')

        # 绘制Ground Truth
        ax2 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 2)
        im2 = ax2.imshow(gt_mask_np, cmap=tumor_cmap, vmin=0, vmax=2)
        ax2.set_title('Ground Truth\n(Black=BG, Red=Surface, Blue=InBone)',
                      fontsize=10, fontweight='bold')
        ax2.axis('off')

        # 绘制叠加图
        ax3 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 3)
        ax3.imshow(overlay)
        ax3.set_title('Overlay Visualization\n(Tumor regions highlighted)',
                      fontsize=10, fontweight='bold')
        ax3.axis('off')

        # 绘制统计信息
        ax4 = plt.subplot(n_rows, n_cols, row_idx * n_cols + 4)
        ax4.axis('off')

        # 格式化统计文本
        stats_text = f"""样本统计信息:

总像素: {stats['total_pixels']:,}

背景: {stats['background']['pixels']:,} 
({stats['background']['percentage']:.1f}%)

表面肿瘤: {stats['surface_tumor']['pixels']:,}
({stats['surface_tumor']['percentage']:.1f}%)

骨内肿瘤: {stats['inbone_tumor']['pixels']:,}
({stats['inbone_tumor']['percentage']:.1f}%)

肿瘤状态:
✅ 表面: {'是' if stats['has_surface'] else '否'}
✅ 骨内: {'是' if stats['has_inbone'] else '否'}
✅ 任意: {'是' if stats['has_tumor'] else '否'}
        """

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax4.set_title('Statistics', fontsize=10, fontweight='bold')

        # 打印控制台信息
        print(f"\n--- Sample {idx} ---")
        print(f"肿瘤类型: ", end="")
        if stats['has_surface'] and stats['has_inbone']:
            print("表面+骨内肿瘤")
        elif stats['has_surface']:
            print("仅表面肿瘤")
        elif stats['has_inbone']:
            print("仅骨内肿瘤")
        else:
            print("无肿瘤")

        print(f"表面肿瘤像素: {stats['surface_tumor']['pixels']:,} ({stats['surface_tumor']['percentage']:.2f}%)")
        print(f"骨内肿瘤像素: {stats['inbone_tumor']['pixels']:,} ({stats['inbone_tumor']['percentage']:.2f}%)")

    # 添加全局色条
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
        print(f"\n✅ 可视化结果已保存: tumor_groundtruth_validation.png")

    plt.show()

    # 数据集整体统计
    print("\n" + "=" * 60)
    print("📊 数据集整体统计")
    print("=" * 60)

    total_samples = len(all_stats)
    samples_with_surface = sum(1 for s in all_stats if s['has_surface'])
    samples_with_inbone = sum(1 for s in all_stats if s['has_inbone'])
    samples_with_any_tumor = sum(1 for s in all_stats if s['has_tumor'])

    print(f"总样本数: {total_samples}")
    print(f"有表面肿瘤的样本: {samples_with_surface} ({samples_with_surface / total_samples * 100:.1f}%)")
    print(f"有骨内肿瘤的样本: {samples_with_inbone} ({samples_with_inbone / total_samples * 100:.1f}%)")
    print(f"有任意肿瘤的样本: {samples_with_any_tumor} ({samples_with_any_tumor / total_samples * 100:.1f}%)")

    return all_stats


def validate_tumor_model_predictions(n_samples=4):
    """验证训练好的肿瘤模型预测结果"""
    print("\n" + "=" * 60)
    print("🤖 肿瘤模型预测验证")
    print("=" * 60)

    # 数据预处理
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # 加载数据集
    dataset = TumorSegmentationDataset(transform=transform, only_positive=False)

    # 加载模型
    try:
        model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ 模型已加载: {cfg.MODEL_NAME}")
    except FileNotFoundError:
        print(f"❌ 模型文件未找到: {cfg.MODEL_NAME}")
        print("请先运行 train_tumor.py 训练模型")
        return

    # 选择样本
    indices = list(range(min(len(dataset), 20)))
    random.shuffle(indices)
    selected_indices = indices[:n_samples]

    # 创建颜色映射
    tumor_cmap = create_readable_colormap()

    fig = plt.figure(figsize=(20, 5 * n_samples))

    with torch.no_grad():
        for row_idx, idx in enumerate(selected_indices):
            img, gt_mask = dataset[idx]

            # 模型预测
            img_batch = img.unsqueeze(0).to(cfg.DEVICE)
            output = model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            # 反归一化图像
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display + 1.0) / 2.0
            img_display = np.clip(img_display, 0, 1)

            gt_mask_np = gt_mask.cpu().numpy()

            # 计算IoU
            intersection = ((gt_mask_np > 0) & (pred_mask > 0)).sum()
            union = ((gt_mask_np > 0) | (pred_mask > 0)).sum()
            iou = intersection / union if union > 0 else 1.0

            # 创建叠加图
            gt_overlay = create_overlay_visualization(img_display, gt_mask_np)
            pred_overlay = create_overlay_visualization(img_display, pred_mask)

            # 5列展示：原图、GT、GT叠加、预测、预测叠加
            n_cols = 5

            # 原图
            ax1 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 1)
            ax1.imshow(img_display)
            ax1.set_title(f'Sample {idx}\nOriginal', fontsize=10, fontweight='bold')
            ax1.axis('off')

            # GT mask
            ax2 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 2)
            ax2.imshow(gt_mask_np, cmap=tumor_cmap, vmin=0, vmax=2)
            ax2.set_title('Ground Truth', fontsize=10, fontweight='bold')
            ax2.axis('off')

            # GT叠加
            ax3 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 3)
            ax3.imshow(gt_overlay)
            ax3.set_title('GT Overlay', fontsize=10, fontweight='bold')
            ax3.axis('off')

            # 预测mask
            ax4 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 4)
            ax4.imshow(pred_mask, cmap=tumor_cmap, vmin=0, vmax=2)
            ax4.set_title(f'Prediction\nIoU: {iou:.3f}', fontsize=10, fontweight='bold')
            ax4.axis('off')

            # 预测叠加
            ax5 = plt.subplot(n_samples, n_cols, row_idx * n_cols + 5)
            ax5.imshow(pred_overlay)
            ax5.set_title('Pred Overlay', fontsize=10, fontweight='bold')
            ax5.axis('off')

            # 打印比较结果
            gt_stats = analyze_tumor_statistics(gt_mask_np)
            pred_stats = analyze_tumor_statistics(pred_mask)

            print(f"\nSample {idx} 对比:")
            print(
                f"  GT  - 表面: {gt_stats['surface_tumor']['pixels']:4d}, 骨内: {gt_stats['inbone_tumor']['pixels']:4d}")
            print(
                f"  预测 - 表面: {pred_stats['surface_tumor']['pixels']:4d}, 骨内: {pred_stats['inbone_tumor']['pixels']:4d}")
            print(f"  IoU: {iou:.3f}")

    plt.tight_layout()
    plt.suptitle('Week 2: Model Prediction vs Ground Truth Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('tumor_prediction_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 预测对比结果已保存: tumor_prediction_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 验证Ground Truth
    print("🔍 开始验证Week 2肿瘤Ground Truth...")
    stats = visualize_tumor_groundtruth(n_samples=6)

    # 验证模型预测（如果模型存在）
    print("\n🤖 验证模型预测效果...")
    validate_tumor_model_predictions(n_samples=4)