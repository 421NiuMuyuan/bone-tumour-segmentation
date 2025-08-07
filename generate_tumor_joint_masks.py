# generate_tumor_joint_masks.py
# 专门为Week 2 (肿瘤分割) 和 Week 3 (关节分割) 生成正确的mask文件

import os, json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# 自动定位本脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "bones-annotated")
OUT_MULTI = os.path.join(DATA_ROOT, "masks_multi")

# JSON中的实际标签名称（基于诊断结果）
TUMOR_JOINT_LABELS = {
    "Tumour on bone surface": "Surface Tumour",  # 表面肿瘤
    "Tumour": "In-Bone Tumour",  # 骨内肿瘤
    "Joint space": "Joint",  # 关节区域
}


def find_image(name):
    """查找对应的图像文件"""
    for sub in ("2", "3"):
        p = os.path.join(DATA_ROOT, sub, name)
        if os.path.isfile(p):
            return p
    return None


def analyze_tumor_joint_labels():
    """分析JSON文件中的肿瘤和关节标签"""
    print("=" * 60)
    print("🔍 分析肿瘤和关节标签")
    print("=" * 60)

    jsons = [f for f in os.listdir(DATA_ROOT) if f.endswith(".json")]

    total_tumor_surface = 0
    total_tumor_inbone = 0
    total_joint = 0
    images_with_tumor = set()
    images_with_joint = set()

    for jf in jsons:
        data = json.load(open(os.path.join(DATA_ROOT, jf), encoding="utf-8"))
        print(f"\n📄 {jf}:")

        file_stats = {"Tumour on bone surface": 0, "Tumour": 0, "Joint space": 0}

        for item in data:
            image_name = os.path.basename(item["data"].get("image") or item["data"].get("image_url"))
            item_has_tumor = False
            item_has_joint = False

            for ann in item.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") == "polygonlabels":
                        lbl_list = r["value"].get("polygonlabels", [])
                        for lbl in lbl_list:
                            if lbl in TUMOR_JOINT_LABELS:
                                file_stats[lbl] += 1

                                if lbl in ["Tumour on bone surface", "Tumour"]:
                                    item_has_tumor = True
                                    if lbl == "Tumour on bone surface":
                                        total_tumor_surface += 1
                                    else:
                                        total_tumor_inbone += 1
                                elif lbl == "Joint space":
                                    item_has_joint = True
                                    total_joint += 1

            if item_has_tumor:
                images_with_tumor.add(image_name)
            if item_has_joint:
                images_with_joint.add(image_name)

        for label, count in file_stats.items():
            if count > 0:
                print(f"   {label}: {count}")

    print(f"\n📊 总体统计:")
    print(f"表面肿瘤标注: {total_tumor_surface}")
    print(f"骨内肿瘤标注: {total_tumor_inbone}")
    print(f"关节标注: {total_joint}")
    print(f"有肿瘤的图像: {len(images_with_tumor)}")
    print(f"有关节的图像: {len(images_with_joint)}")

    return images_with_tumor, images_with_joint


def generate_masks():
    """生成肿瘤和关节的mask文件"""
    print("\n" + "=" * 60)
    print("🎨 生成肿瘤和关节mask文件")
    print("=" * 60)

    # 确保输出目录存在
    for dir_name in TUMOR_JOINT_LABELS.values():
        os.makedirs(os.path.join(OUT_MULTI, dir_name), exist_ok=True)

    jsons = [f for f in os.listdir(DATA_ROOT) if f.endswith(".json")]

    total_images = 0
    images_with_surface_tumor = 0
    images_with_inbone_tumor = 0
    images_with_joint = 0

    surface_tumor_pixels = 0
    inbone_tumor_pixels = 0
    joint_pixels = 0

    for jf in jsons:
        data = json.load(open(os.path.join(DATA_ROOT, jf), encoding="utf-8"))
        print(f"\n-- 处理 {jf}")

        for item in data:
            ref = item["data"].get("image") or item["data"].get("image_url")
            name = os.path.basename(ref)
            img_path = find_image(name)

            if not img_path:
                continue

            img = Image.open(img_path)
            W, H = img.size
            total_images += 1

            # 为每种类型创建mask
            masks = {}
            draws = {}
            for original_label, dir_name in TUMOR_JOINT_LABELS.items():
                masks[dir_name] = Image.new("L", (W, H), 0)  # 使用灰度图
                draws[dir_name] = ImageDraw.Draw(masks[dir_name])

            # 统计这张图像的标注
            image_surface_count = 0
            image_inbone_count = 0
            image_joint_count = 0

            for ann in item.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "polygonlabels":
                        continue

                    lbl_list = r["value"].get("polygonlabels", [])
                    if not lbl_list:
                        continue

                    lbl = lbl_list[0]
                    if lbl not in TUMOR_JOINT_LABELS:
                        continue

                    dir_name = TUMOR_JOINT_LABELS[lbl]
                    pts = r["value"]["points"]
                    poly = [(x * W / 100.0, y * H / 100.0) for x, y in pts]

                    # 绘制mask (使用255作为前景值)
                    draws[dir_name].polygon(poly, fill=255)

                    # 统计
                    if lbl == "Tumour on bone surface":
                        image_surface_count += 1
                    elif lbl == "Tumour":
                        image_inbone_count += 1
                    elif lbl == "Joint space":
                        image_joint_count += 1

            # 保存mask并统计像素
            base = os.path.splitext(name)[0]

            for dir_name, mask in masks.items():
                output_path = os.path.join(OUT_MULTI, dir_name, f"{base}.png")
                mask.save(output_path)

                # 统计非零像素
                mask_array = np.array(mask)
                non_zero_pixels = (mask_array > 0).sum()

                if dir_name == "Surface Tumour" and non_zero_pixels > 0:
                    surface_tumor_pixels += non_zero_pixels
                elif dir_name == "In-Bone Tumour" and non_zero_pixels > 0:
                    inbone_tumor_pixels += non_zero_pixels
                elif dir_name == "Joint" and non_zero_pixels > 0:
                    joint_pixels += non_zero_pixels

            # 统计有标注的图像
            if image_surface_count > 0:
                images_with_surface_tumor += 1
            if image_inbone_count > 0:
                images_with_inbone_tumor += 1
            if image_joint_count > 0:
                images_with_joint += 1

            # 显示处理状态
            status_parts = []
            if image_surface_count > 0:
                status_parts.append(f"🔴表面×{image_surface_count}")
            if image_inbone_count > 0:
                status_parts.append(f"🔵骨内×{image_inbone_count}")
            if image_joint_count > 0:
                status_parts.append(f"🟡关节×{image_joint_count}")

            if status_parts:
                status = " ".join(status_parts)
                print(f"   ✅ {name}: {status}")

    print(f"\n📊 生成完成统计:")
    print(f"总处理图像: {total_images}")
    print(f"有表面肿瘤的图像: {images_with_surface_tumor}")
    print(f"有骨内肿瘤的图像: {images_with_inbone_tumor}")
    print(f"有关节的图像: {images_with_joint}")
    print(f"表面肿瘤总像素: {surface_tumor_pixels:,}")
    print(f"骨内肿瘤总像素: {inbone_tumor_pixels:,}")
    print(f"关节总像素: {joint_pixels:,}")


def verify_generated_masks():
    """验证生成的mask文件"""
    print(f"\n🔍 验证生成的mask文件:")

    for original_label, dir_name in TUMOR_JOINT_LABELS.items():
        mask_dir = os.path.join(OUT_MULTI, dir_name)

        if not os.path.exists(mask_dir):
            print(f"❌ {dir_name} 目录不存在")
            continue

        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        print(f"\n📁 {dir_name}: {len(mask_files)} 文件")

        non_empty_count = 0
        total_pixels = 0

        # 检查前10个文件
        for filename in mask_files[:10]:
            filepath = os.path.join(mask_dir, filename)
            mask = cv2.imread(filepath, 0)

            if mask is not None:
                non_zero_pixels = (mask > 0).sum()
                total_pixels += non_zero_pixels

                if non_zero_pixels > 0:
                    non_empty_count += 1
                    print(f"  ✅ {filename}: {non_zero_pixels} 像素")
                else:
                    print(f"  ⚪ {filename}: 空mask")

        print(f"  📊 前10个文件: {non_empty_count}/10 非空, 总像素: {total_pixels:,}")


def main():
    print("=" * 60)
    print("🦴 Week 2&3: 生成肿瘤和关节Mask文件")
    print("=" * 60)

    # 分析标签
    images_with_tumor, images_with_joint = analyze_tumor_joint_labels()

    # 生成mask
    generate_masks()

    # 验证结果
    verify_generated_masks()

    print(f"\n✅ 完成! 现在可以运行:")
    print(f"   python train_tumor.py    # Week 2 肿瘤分割训练")
    print(f"   python train_joint.py    # Week 3 关节分割训练")
    print(f"   python visualize_readable.py  # 验证数据质量")


if __name__ == "__main__":
    main()