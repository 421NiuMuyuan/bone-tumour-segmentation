# diagnose_data.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import config_tumor as cfg

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


def check_directory_structure():
    """检查数据目录结构"""
    print("=" * 60)
    print("📁 检查数据目录结构")
    print("=" * 60)

    print(f"数据集根目录: {cfg.DATASET_ROOT}")
    print(f"存在: {os.path.exists(cfg.DATASET_ROOT)}")

    if os.path.exists(cfg.DATASET_ROOT):
        print(f"\n根目录下的内容:")
        for item in os.listdir(cfg.DATASET_ROOT):
            item_path = os.path.join(cfg.DATASET_ROOT, item)
            if os.path.isdir(item_path):
                count = len(os.listdir(item_path)) if os.path.exists(item_path) else 0
                print(f"  📁 {item}/ ({count} files)")
            else:
                print(f"  📄 {item}")

    print(f"\n表面肿瘤目录: {cfg.SURF_DIR}")
    print(f"存在: {os.path.exists(cfg.SURF_DIR)}")

    print(f"\n骨内肿瘤目录: {cfg.INBONE_DIR}")
    print(f"存在: {os.path.exists(cfg.INBONE_DIR)}")

    if os.path.exists(cfg.SURF_DIR):
        surf_files = [f for f in os.listdir(cfg.SURF_DIR) if f.endswith('.png')]
        print(f"\n表面肿瘤PNG文件数: {len(surf_files)}")
        if len(surf_files) > 0:
            print(f"前5个文件: {surf_files[:5]}")

    if os.path.exists(cfg.INBONE_DIR):
        inbone_files = [f for f in os.listdir(cfg.INBONE_DIR) if f.endswith('.png')]
        print(f"骨内肿瘤PNG文件数: {len(inbone_files)}")
        if len(inbone_files) > 0:
            print(f"前5个文件: {inbone_files[:5]}")


def analyze_mask_files():
    """分析mask文件内容"""
    print("\n" + "=" * 60)
    print("🔍 分析Mask文件内容")
    print("=" * 60)

    # 检查表面肿瘤mask
    if os.path.exists(cfg.SURF_DIR):
        surf_files = [f for f in os.listdir(cfg.SURF_DIR) if f.endswith('.png')]
        print(f"\n📊 表面肿瘤Mask分析 ({len(surf_files)} files):")

        non_empty_surf = 0
        total_surf_pixels = 0

        for i, filename in enumerate(surf_files[:10]):  # 检查前10个
            filepath = os.path.join(cfg.SURF_DIR, filename)
            mask = cv2.imread(filepath, 0)

            if mask is not None:
                unique_values = np.unique(mask)
                non_zero_pixels = (mask > 0).sum()
                total_surf_pixels += non_zero_pixels

                if non_zero_pixels > 0:
                    non_empty_surf += 1

                print(
                    f"  {filename}: shape={mask.shape}, unique_values={unique_values}, non_zero_pixels={non_zero_pixels}")
            else:
                print(f"  {filename}: 无法读取!")

        print(f"\n表面肿瘤统计:")
        print(f"  非空mask数量: {non_empty_surf}/{min(10, len(surf_files))}")
        print(f"  总非零像素数: {total_surf_pixels}")

    # 检查骨内肿瘤mask
    if os.path.exists(cfg.INBONE_DIR):
        inbone_files = [f for f in os.listdir(cfg.INBONE_DIR) if f.endswith('.png')]
        print(f"\n📊 骨内肿瘤Mask分析 ({len(inbone_files)} files):")

        non_empty_inbone = 0
        total_inbone_pixels = 0

        for i, filename in enumerate(inbone_files[:10]):  # 检查前10个
            filepath = os.path.join(cfg.INBONE_DIR, filename)
            mask = cv2.imread(filepath, 0)

            if mask is not None:
                unique_values = np.unique(mask)
                non_zero_pixels = (mask > 0).sum()
                total_inbone_pixels += non_zero_pixels

                if non_zero_pixels > 0:
                    non_empty_inbone += 1

                print(
                    f"  {filename}: shape={mask.shape}, unique_values={unique_values}, non_zero_pixels={non_zero_pixels}")
            else:
                print(f"  {filename}: 无法读取!")

        print(f"\n骨内肿瘤统计:")
        print(f"  非空mask数量: {non_empty_inbone}/{min(10, len(inbone_files))}")
        print(f"  总非零像素数: {total_inbone_pixels}")


def visualize_sample_masks():
    """可视化几个样本mask"""
    print("\n" + "=" * 60)
    print("🎨 可视化样本Mask")
    print("=" * 60)

    if not os.path.exists(cfg.SURF_DIR) or not os.path.exists(cfg.INBONE_DIR):
        print("❌ 肿瘤mask目录不存在")
        return

    surf_files = [f for f in os.listdir(cfg.SURF_DIR) if f.endswith('.png')]
    inbone_files = [f for f in os.listdir(cfg.INBONE_DIR) if f.endswith('.png')]

    # 找到共同的文件名
    common_files = list(set(surf_files) & set(inbone_files))

    if len(common_files) == 0:
        print("❌ 没有找到共同的mask文件")
        return

    print(f"找到 {len(common_files)} 个共同mask文件")

    # 可视化前4个
    n_samples = min(4, len(common_files))
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, filename in enumerate(common_files[:n_samples]):
        # 读取表面肿瘤mask
        surf_path = os.path.join(cfg.SURF_DIR, filename)
        surf_mask = cv2.imread(surf_path, 0)

        # 读取骨内肿瘤mask
        inbone_path = os.path.join(cfg.INBONE_DIR, filename)
        inbone_mask = cv2.imread(inbone_path, 0)

        # 显示表面肿瘤
        axes[0, i].imshow(surf_mask, cmap='hot', vmin=0, vmax=255)
        axes[0, i].set_title(f'Surface Tumor\n{filename}', fontsize=8)
        axes[0, i].axis('off')

        # 显示骨内肿瘤
        axes[1, i].imshow(inbone_mask, cmap='hot', vmin=0, vmax=255)
        axes[1, i].set_title(f'In-Bone Tumor\n{filename}', fontsize=8)
        axes[1, i].axis('off')

        # 打印统计
        surf_pixels = (surf_mask > 0).sum() if surf_mask is not None else 0
        inbone_pixels = (inbone_mask > 0).sum() if inbone_mask is not None else 0

        print(f"{filename}:")
        print(f"  表面肿瘤像素: {surf_pixels}")
        print(f"  骨内肿瘤像素: {inbone_pixels}")

    plt.tight_layout()
    plt.suptitle('Raw Tumor Mask Files Analysis', fontsize=12, y=0.98)
    plt.savefig('mask_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def check_original_json_files():
    """检查原始JSON标注文件"""
    print("\n" + "=" * 60)
    print("📋 检查原始JSON标注文件")
    print("=" * 60)

    json_files = [f for f in os.listdir(cfg.DATASET_ROOT) if f.endswith('.json')]
    print(f"找到JSON文件: {json_files}")

    if len(json_files) == 0:
        print("❌ 没有找到JSON标注文件")
        return

    import json

    for json_file in json_files:
        json_path = os.path.join(cfg.DATASET_ROOT, json_file)
        print(f"\n分析 {json_file}:")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"  总标注项目: {len(data)}")

            # 统计标签类型
            label_counts = {}
            tumor_annotations = 0

            for item in data:
                for ann in item.get('annotations', []):
                    for result in ann.get('result', []):
                        if result.get('type') == 'polygonlabels':
                            labels = result['value'].get('polygonlabels', [])
                            for label in labels:
                                label_counts[label] = label_counts.get(label, 0) + 1
                                if 'Tumour' in label or 'Tumor' in label:
                                    tumor_annotations += 1

            print(f"  标签统计:")
            for label, count in label_counts.items():
                print(f"    {label}: {count}")

            print(f"  肿瘤相关标注: {tumor_annotations}")

        except Exception as e:
            print(f"  ❌ 读取失败: {e}")


def suggest_solutions():
    """建议解决方案"""
    print("\n" + "=" * 60)
    print("💡 问题诊断与解决建议")
    print("=" * 60)

    # 检查convert_labels.py是否正确执行
    print("可能的问题原因:")
    print("1. convert_labels.py 没有正确执行")
    print("2. JSON文件中没有肿瘤标注")
    print("3. 标签名称不匹配 (Surface Tumour vs Surface Tumor)")
    print("4. mask生成过程中的bug")

    print("\n建议解决步骤:")
    print("1. 重新运行 convert_labels.py")
    print("2. 检查JSON文件中的标签名称")
    print("3. 手动验证生成的mask文件")
    print("4. 如果必要，创建一些人工肿瘤样本进行测试")


if __name__ == "__main__":
    # 执行诊断
    check_directory_structure()
    analyze_mask_files()
    visualize_sample_masks()
    check_original_json_files()
    suggest_solutions()