# diagnose_data.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import config_tumor as cfg

# Configure matplotlib to support Unicode (so emojis/symbols display correctly)
rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


def check_directory_structure():
    """Check dataset directory structure"""
    print("=" * 60)
    print("📁 Checking dataset directory structure")
    print("=" * 60)

    print(f"Dataset root: {cfg.DATASET_ROOT}")
    print(f"Exists: {os.path.exists(cfg.DATASET_ROOT)}")

    if os.path.exists(cfg.DATASET_ROOT):
        print(f"\nContents under root:")
        for item in os.listdir(cfg.DATASET_ROOT):
            item_path = os.path.join(cfg.DATASET_ROOT, item)
            if os.path.isdir(item_path):
                count = len(os.listdir(item_path)) if os.path.exists(item_path) else 0
                print(f"  📁 {item}/ ({count} files)")
            else:
                print(f"  📄 {item}")

    print(f"\nSurface tumour mask dir: {cfg.SURF_DIR}")
    print(f"Exists: {os.path.exists(cfg.SURF_DIR)}")

    print(f"\nIn-bone tumour mask dir: {cfg.INBONE_DIR}")
    print(f"Exists: {os.path.exists(cfg.INBONE_DIR)}")

    if os.path.exists(cfg.SURF_DIR):
        surf_files = [f for f in os.listdir(cfg.SURF_DIR) if f.endswith('.png')]
        print(f"\n# of surface tumour PNG files: {len(surf_files)}")
        if len(surf_files) > 0:
            print(f"First 5 files: {surf_files[:5]}")

    if os.path.exists(cfg.INBONE_DIR):
        inbone_files = [f for f in os.listdir(cfg.INBONE_DIR) if f.endswith('.png')]
        print(f"# of in-bone tumour PNG files: {len(inbone_files)}")
        if len(inbone_files) > 0:
            print(f"First 5 files: {inbone_files[:5]}")


def analyze_mask_files():
    """Analyze mask file contents"""
    print("\n" + "=" * 60)
    print("🔍 Analyzing mask file contents")
    print("=" * 60)

    # Surface tumour masks
    if os.path.exists(cfg.SURF_DIR):
        surf_files = [f for f in os.listdir(cfg.SURF_DIR) if f.endswith('.png')]
        print(f"\n📊 Surface tumour mask analysis ({len(surf_files)} files):")

        non_empty_surf = 0
        total_surf_pixels = 0

        for i, filename in enumerate(surf_files[:10]):  # check first 10
            filepath = os.path.join(cfg.SURF_DIR, filename)
            mask = cv2.imread(filepath, 0)

            if mask is not None:
                unique_values = np.unique(mask)
                non_zero_pixels = (mask > 0).sum()
                total_surf_pixels += non_zero_pixels

                if non_zero_pixels > 0:
                    non_empty_surf += 1

                print(f"  {filename}: shape={mask.shape}, unique_values={unique_values}, non_zero_pixels={non_zero_pixels}")
            else:
                print(f"  {filename}: cannot read!")

        print(f"\nSurface tumour stats:")
        print(f"  Non-empty masks: {non_empty_surf}/{min(10, len(surf_files))}")
        print(f"  Total non-zero pixels: {total_surf_pixels}")

    # In-bone tumour masks
    if os.path.exists(cfg.INBONE_DIR):
        inbone_files = [f for f in os.listdir(cfg.INBONE_DIR) if f.endswith('.png')]
        print(f"\n📊 In-bone tumour mask analysis ({len(inbone_files)} files):")

        non_empty_inbone = 0
        total_inbone_pixels = 0

        for i, filename in enumerate(inbone_files[:10]):  # check first 10
            filepath = os.path.join(cfg.INBONE_DIR, filename)
            mask = cv2.imread(filepath, 0)

            if mask is not None:
                unique_values = np.unique(mask)
                non_zero_pixels = (mask > 0).sum()
                total_inbone_pixels += non_zero_pixels

                if non_zero_pixels > 0:
                    non_empty_inbone += 1

                print(f"  {filename}: shape={mask.shape}, unique_values={unique_values}, non_zero_pixels={non_zero_pixels}")
            else:
                print(f"  {filename}: cannot read!")

        print(f"\nIn-bone tumour stats:")
        print(f"  Non-empty masks: {non_empty_inbone}/{min(10, len(inbone_files))}")
        print(f"  Total non-zero pixels: {total_inbone_pixels}")


def visualize_sample_masks():
    """Visualize several sample masks"""
    print("\n" + "=" * 60)
    print("🎨 Visualizing sample masks")
    print("=" * 60)

    if not os.path.exists(cfg.SURF_DIR) or not os.path.exists(cfg.INBONE_DIR):
        print("❌ Tumour mask directories do not exist")
        return

    surf_files = [f for f in os.listdir(cfg.SURF_DIR) if f.endswith('.png')]
    inbone_files = [f for f in os.listdir(cfg.INBONE_DIR) if f.endswith('.png')]

    # Intersection of filenames
    common_files = list(set(surf_files) & set(inbone_files))

    if len(common_files) == 0:
        print("❌ No common mask files found")
        return

    print(f"Found {len(common_files)} common mask files")

    # Visualize up to 4
    n_samples = min(4, len(common_files))
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, filename in enumerate(common_files[:n_samples]):
        # Surface tumour
        surf_path = os.path.join(cfg.SURF_DIR, filename)
        surf_mask = cv2.imread(surf_path, 0)

        # In-bone tumour
        inbone_path = os.path.join(cfg.INBONE_DIR, filename)
        inbone_mask = cv2.imread(inbone_path, 0)

        # Show surface tumour
        axes[0, i].imshow(surf_mask, cmap='hot', vmin=0, vmax=255)
        axes[0, i].set_title(f'Surface Tumor\n{filename}', fontsize=8)
        axes[0, i].axis('off')

        # Show in-bone tumour
        axes[1, i].imshow(inbone_mask, cmap='hot', vmin=0, vmax=255)
        axes[1, i].set_title(f'In-Bone Tumor\n{filename}', fontsize=8)
        axes[1, i].axis('off')

        # Print stats
        surf_pixels = (surf_mask > 0).sum() if surf_mask is not None else 0
        inbone_pixels = (inbone_mask > 0).sum() if inbone_mask is not None else 0

        print(f"{filename}:")
        print(f"  Surface tumour pixels: {surf_pixels}")
        print(f"  In-bone tumour pixels: {inbone_pixels}")

    plt.tight_layout()
    plt.suptitle('Raw Tumor Mask Files Analysis', fontsize=12, y=0.98)
    plt.savefig('mask_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def check_original_json_files():
    """Check original JSON annotation files"""
    print("\n" + "=" * 60)
    print("📋 Checking original JSON annotation files")
    print("=" * 60)

    json_files = [f for f in os.listdir(cfg.DATASET_ROOT) if f.endswith('.json')]
    print(f"Found JSON files: {json_files}")

    if len(json_files) == 0:
        print("❌ No JSON annotation files found")
        return

    import json

    for json_file in json_files:
        json_path = os.path.join(cfg.DATASET_ROOT, json_file)
        print(f"\nAnalyzing {json_file}:")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"  Total annotated items: {len(data)}")

            # Count label types
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

            print(f"  Label counts:")
            for label, count in label_counts.items():
                print(f"    {label}: {count}")

            print(f"  Tumour-related annotations: {tumor_annotations}")

        except Exception as e:
            print(f"  ❌ Failed to read: {e}")


def suggest_solutions():
    """Suggest possible solutions"""
    print("\n" + "=" * 60)
    print("💡 Diagnosis and suggested fixes")
    print("=" * 60)

    # Verify convert_labels.py pipeline
    print("Possible causes:")
    print("1. convert_labels.py did not run correctly")
    print("2. No tumour annotations in JSON files")
    print("3. Label name mismatch (Surface Tumour vs Surface Tumor)")
    print("4. Bug in mask generation pipeline")

    print("\nSuggested steps:")
    print("1. Re-run convert_labels.py")
    print("2. Inspect label names in JSON files")
    print("3. Manually verify several generated mask files")
    print("4. If necessary, craft a few synthetic tumour samples for testing")


if __name__ == "__main__":
    # Run diagnostics
    check_directory_structure()
    analyze_mask_files()
    visualize_sample_masks()
    check_original_json_files()
    suggest_solutions()
