# generate_tumor_joint_masks.py
# Generate correct mask files specifically for Week 2 (Tumor Segmentation) and Week 3 (Joint Segmentation)

import os, json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Auto-detect this script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "bones-annotated")
OUT_MULTI = os.path.join(DATA_ROOT, "masks_multi")

# Actual label names used in JSON (based on diagnostics)
TUMOR_JOINT_LABELS = {
    "Tumour on bone surface": "Surface Tumour",  # surface tumor
    "Tumour": "In-Bone Tumour",                  # in-bone tumor
    "Joint space": "Joint",                      # joint region
}


def find_image(name):
    """Find the corresponding image file"""
    for sub in ("2", "3"):
        p = os.path.join(DATA_ROOT, sub, name)
        if os.path.isfile(p):
            return p
    return None


def analyze_tumor_joint_labels():
    """Analyze tumor and joint labels in JSON files"""
    print("=" * 60)
    print("🔍 Analyzing tumor & joint labels")
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

    print(f"\n📊 Overall stats:")
    print(f"Surface tumor annotations: {total_tumor_surface}")
    print(f"In-bone tumor annotations: {total_tumor_inbone}")
    print(f"Joint annotations:         {total_joint}")
    print(f"Images with tumor:         {len(images_with_tumor)}")
    print(f"Images with joint:         {len(images_with_joint)}")

    return images_with_tumor, images_with_joint


def generate_masks():
    """Generate tumor and joint mask files"""
    print("\n" + "=" * 60)
    print("🎨 Generating tumor & joint masks")
    print("=" * 60)

    # Ensure output directories exist
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
        print(f"\n-- Processing {jf}")

        for item in data:
            ref = item["data"].get("image") or item["data"].get("image_url")
            name = os.path.basename(ref)
            img_path = find_image(name)

            if not img_path:
                continue

            img = Image.open(img_path)
            W, H = img.size
            total_images += 1

            # Create masks for each category
            masks = {}
            draws = {}
            for original_label, dir_name in TUMOR_JOINT_LABELS.items():
                masks[dir_name] = Image.new("L", (W, H), 0)  # grayscale
                draws[dir_name] = ImageDraw.Draw(masks[dir_name])

            # Per-image annotation counters
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

                    # Draw mask (use 255 for foreground)
                    draws[dir_name].polygon(poly, fill=255)

                    # Update counters
                    if lbl == "Tumour on bone surface":
                        image_surface_count += 1
                    elif lbl == "Tumour":
                        image_inbone_count += 1
                    elif lbl == "Joint space":
                        image_joint_count += 1

            # Save masks and accumulate pixels
            base = os.path.splitext(name)[0]

            for dir_name, mask in masks.items():
                output_path = os.path.join(OUT_MULTI, dir_name, f"{base}.png")
                mask.save(output_path)

                # Count non-zero pixels
                mask_array = np.array(mask)
                non_zero_pixels = (mask_array > 0).sum()

                if dir_name == "Surface Tumour" and non_zero_pixels > 0:
                    surface_tumor_pixels += non_zero_pixels
                elif dir_name == "In-Bone Tumour" and non_zero_pixels > 0:
                    inbone_tumor_pixels += non_zero_pixels
                elif dir_name == "Joint" and non_zero_pixels > 0:
                    joint_pixels += non_zero_pixels

            # Track images containing annotations
            if image_surface_count > 0:
                images_with_surface_tumor += 1
            if image_inbone_count > 0:
                images_with_inbone_tumor += 1
            if image_joint_count > 0:
                images_with_joint += 1

            # Print status line
            status_parts = []
            if image_surface_count > 0:
                status_parts.append(f"🔴Surface×{image_surface_count}")
            if image_inbone_count > 0:
                status_parts.append(f"🔵In-bone×{image_inbone_count}")
            if image_joint_count > 0:
                status_parts.append(f"🟡Joint×{image_joint_count}")

            if status_parts:
                status = " ".join(status_parts)
                print(f"   ✅ {name}: {status}")

    print(f"\n📊 Generation summary:")
    print(f"Total processed images:  {total_images}")
    print(f"Images with surface tumor: {images_with_surface_tumor}")
    print(f"Images with in-bone tumor: {images_with_inbone_tumor}")
    print(f"Images with joint:         {images_with_joint}")
    print(f"Total surface tumor pixels: {surface_tumor_pixels:,}")
    print(f"Total in-bone tumor pixels: {inbone_tumor_pixels:,}")
    print(f"Total joint pixels:         {joint_pixels:,}")


def verify_generated_masks():
    """Verify generated mask files"""
    print(f"\n🔍 Verifying generated mask files:")

    for original_label, dir_name in TUMOR_JOINT_LABELS.items():
        mask_dir = os.path.join(OUT_MULTI, dir_name)

        if not os.path.exists(mask_dir):
            print(f"❌ Directory does not exist: {dir_name}")
            continue

        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        print(f"\n📁 {dir_name}: {len(mask_files)} files")

        non_empty_count = 0
        total_pixels = 0

        # Inspect first 10 files
        for filename in mask_files[:10]:
            filepath = os.path.join(mask_dir, filename)
            mask = cv2.imread(filepath, 0)

            if mask is not None:
                non_zero_pixels = (mask > 0).sum()
                total_pixels += non_zero_pixels

                if non_zero_pixels > 0:
                    non_empty_count += 1
                    print(f"  ✅ {filename}: {non_zero_pixels} pixels")
                else:
                    print(f"  ⚪ {filename}: empty mask")

        print(f"  📊 First 10 files: {non_empty_count}/10 non-empty, total pixels: {total_pixels:,}")


def main():
    print("=" * 60)
    print("🦴 Week 2 & 3: Generate tumor & joint masks")
    print("=" * 60)

    # Analyze labels
    images_with_tumor, images_with_joint = analyze_tumor_joint_labels()

    # Generate masks
    generate_masks()

    # Verify results
    verify_generated_masks()

    print(f"\n✅ Done! You can now run:")
    print(f"   python train_tumor.py         # Week 2 tumor segmentation training")
    print(f"   python train_joint.py         # Week 3 joint segmentation training")
    print(f"   python visualize_readable.py  # Validate data quality")


if __name__ == "__main__":
    main()
