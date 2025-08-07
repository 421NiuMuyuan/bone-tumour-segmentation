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

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


def get_eval_transform():
    """验证用的数据预处理"""
    return Compose([
        Resize(512, 512),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def analyze_joint_distribution():
    """详细分析关节数据分布"""
    print("=" * 60)
    print("🔍 详细分析关节数据分布")
    print("=" * 60)

    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    positive_samples = []
    negative_samples = []

    print("分析所有样本...")
    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask_np = mask.cpu().numpy()
        joint_pixels = (mask_np == 1).sum()

        if joint_pixels > 0:
            positive_samples.append((i, joint_pixels))
        else:
            negative_samples.append(i)

    print(f"\n📊 样本分布:")
    print(f"阳性样本 (有关节): {len(positive_samples)}")
    print(f"阴性样本 (无关节): {len(negative_samples)}")

    if positive_samples:
        print(f"\n🔍 阳性样本详情:")
        positive_samples.sort(key=lambda x: x[1], reverse=True)  # 按像素数排序
        for i, (idx, pixels) in enumerate(positive_samples[:10]):
            print(f"  Sample {idx}: {pixels:,} 关节像素")

    return positive_samples, negative_samples


def test_model_predictions():
    """测试模型预测行为"""
    print("\n" + "=" * 60)
    print("🤖 测试模型预测行为")
    print("=" * 60)

    # 加载数据集和模型
    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    try:
        model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ 模型已加载: {cfg.MODEL_NAME}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 分析所有样本的预测结果
    all_predictions = []
    all_gt_labels = []

    print("\n分析模型预测...")
    with torch.no_grad():
        for i in range(min(50, len(dataset))):  # 分析前50个样本
            img, gt_mask = dataset[i]

            # 模型预测
            img_batch = img.unsqueeze(0).to(cfg.DEVICE)
            output = model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            # 统计预测结果
            gt_has_joint = (gt_mask == 1).any().item()
            pred_has_joint = (pred_mask == 1).any()

            max_joint_prob = pred_probs[0, 1].max().item()  # 关节类别的最大概率

            all_predictions.append({
                'sample_idx': i,
                'gt_has_joint': gt_has_joint,
                'pred_has_joint': pred_has_joint,
                'max_joint_prob': max_joint_prob,
                'gt_joint_pixels': (gt_mask == 1).sum().item(),
                'pred_joint_pixels': (pred_mask == 1).sum()
            })

            if gt_has_joint:
                all_gt_labels.append(1)
            else:
                all_gt_labels.append(0)

    # 分析结果
    print(f"\n📊 预测统计 (前{len(all_predictions)}个样本):")

    # 预测类别分布
    pred_joint_count = sum(1 for p in all_predictions if p['pred_has_joint'])
    pred_bg_count = len(all_predictions) - pred_joint_count

    print(f"模型预测分布:")
    print(f"  预测为关节: {pred_joint_count}")
    print(f"  预测为背景: {pred_bg_count}")

    # 概率分布
    joint_probs = [p['max_joint_prob'] for p in all_predictions]
    print(f"\n关节概率统计:")
    print(f"  最大概率: {max(joint_probs):.4f}")
    print(f"  平均概率: {np.mean(joint_probs):.4f}")
    print(f"  概率>0.5的样本: {sum(1 for p in joint_probs if p > 0.5)}")
    print(f"  概率>0.1的样本: {sum(1 for p in joint_probs if p > 0.1)}")

    # 按GT分组分析
    positive_preds = [p for p in all_predictions if p['gt_has_joint']]
    negative_preds = [p for p in all_predictions if not p['gt_has_joint']]

    print(f"\n🔴 阳性样本 (GT有关节, {len(positive_preds)}个):")
    if positive_preds:
        pos_probs = [p['max_joint_prob'] for p in positive_preds]
        print(f"  平均关节概率: {np.mean(pos_probs):.4f}")
        print(f"  最大关节概率: {max(pos_probs):.4f}")
        print(f"  预测正确的: {sum(1 for p in positive_preds if p['pred_has_joint'])}")

    print(f"\n⚪ 阴性样本 (GT无关节, {len(negative_preds)}个):")
    if negative_preds:
        neg_probs = [p['max_joint_prob'] for p in negative_preds]
        print(f"  平均关节概率: {np.mean(neg_probs):.4f}")
        print(f"  最大关节概率: {max(neg_probs):.4f}")
        print(f"  预测正确的: {sum(1 for p in negative_preds if not p['pred_has_joint'])}")

    return all_predictions


def visualize_positive_samples():
    """专门可视化有关节的阳性样本"""
    print("\n" + "=" * 60)
    print("🎨 可视化阳性样本")
    print("=" * 60)

    dataset = JointSegmentationDataset(transform=get_eval_transform(), only_positive=False)

    # 找到所有阳性样本
    positive_indices = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        if (mask == 1).any():
            positive_indices.append(i)

    print(f"找到 {len(positive_indices)} 个阳性样本")

    if len(positive_indices) == 0:
        print("❌ 没有找到阳性样本")
        return

    # 加载模型
    try:
        model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
        checkpoint = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 可视化前4个阳性样本
    n_samples = min(4, len(positive_indices))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(positive_indices[:n_samples]):
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
            joint_prob = pred_probs[0, 1].cpu().numpy()

            # 计算指标
            gt_joint_pixels = (gt_mask_np == 1).sum()
            pred_joint_pixels = (pred_mask == 1).sum()
            max_prob = joint_prob.max()

            # 显示图像
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
    print(f"✅ 阳性样本分析图已保存: joint_positive_samples_analysis.png")


def suggest_solutions():
    """建议解决方案"""
    print("\n" + "=" * 60)
    print("💡 问题诊断与解决建议")
    print("=" * 60)

    print("可能的问题:")
    print("1. 数据极度不平衡 (782.8:1) 导致模型退化")
    print("2. 学习率过高，模型过快收敛到懒惰策略")
    print("3. 损失函数权重不当")
    print("4. 训练epochs不足")

    print("\n建议解决方案:")
    print("1. 重新训练时使用更强的类别权重:")
    print("   - 将关节类别权重提高到 50-100")
    print("2. 降低学习率:")
    print("   - LR = 5e-4 或 1e-4")
    print("3. 增加Focal Loss:")
    print("   - 更好处理不平衡数据")
    print("4. 数据增强:")
    print("   - 对阳性样本进行重复采样")
    print("5. 调整训练策略:")
    print("   - 先在阳性样本上预训练")


if __name__ == "__main__":
    # 分析数据分布
    positive_samples, negative_samples = analyze_joint_distribution()

    # 测试模型预测
    predictions = test_model_predictions()

    # 可视化阳性样本
    visualize_positive_samples()

    # 给出建议
    suggest_solutions()