import os
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_class_imbalance(class_to_percent: Dict[str, float],
                         out_path: str = "/mnt/data/fig_class_imbalance.png") -> str:
    """
    Bar chart for class imbalance (%). Keys are class names, values are percentages (0-100).
    """
    _ensure_dir(out_path)
    classes = list(class_to_percent.keys())
    values = [class_to_percent[k] for k in classes]

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(classes))
    plt.bar(x, values)
    plt.xticks(x, classes, rotation=15, ha="right")
    plt.ylabel("Pixel Ratio (%)")
    plt.title("Class Imbalance")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_training_curves_from_csv(csv_path: str,
                                  out_path: str = "/mnt/data/fig_training_curves.png",
                                  epoch_col: str = "epoch",
                                  train_metric_col: str = "train_dice",
                                  val_metric_col: str = "val_dice") -> str:
    """
    Plot training vs validation curves from a CSV with columns:
    epoch, train_dice, val_dice (customizable via args).
    """
    import csv

    _ensure_dir(out_path)
    epochs, train_vals, val_vals = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row[epoch_col]))
                train_vals.append(float(row[train_metric_col]))
                val_vals.append(float(row[val_metric_col]))
            except Exception:
                continue

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_vals, marker="o", label="Training")
    plt.plot(epochs, val_vals, marker="o", linestyle="--", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Dice (or chosen metric)")
    plt.title("Training / Validation Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_bar_comparison(labels: List[str],
                        values: List[float],
                        ylabel: str,
                        title: str,
                        out_path: str) -> str:
    """
    Simple bar chart for a single series of values.
    """
    _ensure_dir(out_path)
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_grouped_bar_comparison(group_labels: List[str],
                                series: Dict[str, List[float]],
                                ylabel: str,
                                title: str,
                                out_path: str) -> str:
    """
    Grouped bar plot. `series` is a dict of {series_name: values_per_group}.
    All lists must have the same length as group_labels.
    """
    _ensure_dir(out_path)
    keys = list(series.keys())
    n_groups = len(group_labels)
    n_series = len(keys)

    x = np.arange(n_groups)
    width = 0.8 / max(n_series, 1)

    plt.figure(figsize=(8, 4.5))
    for i, k in enumerate(keys):
        vals = series[k]
        plt.bar(x + i * width, vals, width=width, label=k)

    plt.xticks(x + (n_series - 1) * width / 2.0, group_labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def _to_numpy_gray(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("L"))  # 0..255


def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def overlay_mask(image_path: str,
                 mask_path: str,
                 alpha: float = 0.4) -> Image.Image:
    """
    Overlay a single-channel mask onto the grayscale image with simple colormap-like encoding.
    We avoid setting specific colors; we use normalized mask as intensity overlay.
    """
    img = Image.open(image_path)
    base = _to_numpy_gray(img) / 255.0
    h, w = base.shape

    mask = Image.open(mask_path)
    mask_np = np.array(mask.resize((w, h)))

    # Normalize mask to [0,1] for overlay intensity.
    mask_norm = _normalize_to_01(mask_np)

    # Create a 3-channel overlay by stacking base and adding mask to the first channel.
    # No explicit colors; this makes masked areas appear brighter.
    overlay = np.stack([base, base, base], axis=-1)
    overlay[..., 0] = np.clip(overlay[..., 0] * (1 - alpha) + mask_norm * alpha, 0.0, 1.0)

    out = (overlay * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def visualize_triptych(image_path: str,
                       gt_mask_path: str,
                       pred_mask_path: str,
                       out_path: str = "/mnt/data/fig_triptych.png",
                       titles: Tuple[str, str, str] = ("Input", "Ground Truth", "Prediction")) -> str:
    """
    Create a 3-panel figure: Input / GT-overlay / Pred-overlay.
    """
    _ensure_dir(out_path)
    img = Image.open(image_path).convert("L")
    gt_overlay = overlay_mask(image_path, gt_mask_path)
    pred_overlay = overlay_mask(image_path, pred_mask_path)

    plt.figure(figsize=(12, 4))
    for i, (panel, t) in enumerate([(img, titles[0]), (gt_overlay, titles[1]), (pred_overlay, titles[2])]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(panel, cmap="gray" if i == 0 else None)
        plt.axis("off")
        plt.title(t)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_failure_grid(cases: List[Tuple[str, str, str]],
                      out_path: str = "/mnt/data/fig_failures.png",
                      n_cols: int = 3) -> str:
    """
    Plot a grid of failure cases. Each case is (image_path, gt_mask_path, pred_mask_path).
    """
    _ensure_dir(out_path)
    n = len(cases)
    n_cols = max(1, n_cols)
    n_rows = int(math.ceil(n / n_cols))

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    idx = 1
    for (img_path, gt_path, pred_path) in cases:
        trip = [Image.open(img_path).convert("L"),
                overlay_mask(img_path, gt_path),
                overlay_mask(img_path, pred_path)]
        titles = ["Input", "GT", "Pred"]
        for j in range(3):
            plt.subplot(n_rows, 3 * n_cols, (idx - 1) * 3 + j + 1)
            plt.imshow(trip[j], cmap="gray" if j == 0 else None)
            plt.axis("off")
            if idx == 1:
                plt.title(titles[j])
        idx += 1
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_metrics_from_json(json_path: str,
                           out_path: str = "/mnt/data/fig_metrics_from_json.png",
                           metrics_keys: Optional[List[str]] = None) -> str:
    """
    Read a JSON summary (e.g., fixed_evaluation_summary.json) and plot selected metrics as a bar chart.
    Expects a flat dict like {"Bone Dice": 0.91, "Joint Dice": 0.87, ...}.
    """
    _ensure_dir(out_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    if metrics_keys is None:
        metrics_keys = list(data.keys())
    vals = [float(data[k]) for k in metrics_keys if k in data]
    labels = [k for k in metrics_keys if k in data]

    return plot_bar_comparison(labels, vals, ylabel="Score", title="Evaluation Summary", out_path=out_path)


def plot_architecture_comparison(out_path: str,
                                 bone: List[float],
                                 joint: List[float],
                                 surface: List[float],
                                 intra: List[float],
                                 arch_labels: List[str]) -> str:
    """
    Grouped bars for architecture comparison (Bone/Joint/Surface/Intra across methods).
    """
    series = {
        "Bone Dice (%)": bone,
        "Joint Dice (%)": joint,
        "Surface Tumor Dice (%)": surface,
        "In-bone Tumor Dice (%)": intra,
    }
    return plot_grouped_bar_comparison(arch_labels, series, ylabel="Dice (%)",
                                       title="Architecture Comparison", out_path=out_path)


def plot_loss_ablation(out_path: str,
                       surface: List[float],
                       intra: List[float],
                       loss_labels: List[str]) -> str:
    """
    Grouped bars for loss ablation (Surface/In-bone across losses).
    """
    series = {
        "Surface Tumor Dice (%)": surface,
        "In-bone Tumor Dice (%)": intra,
    }
    return plot_grouped_bar_comparison(loss_labels, series, ylabel="Dice (%)",
                                       title="Loss Function Ablation", out_path=out_path)


def plot_sampling_ablation(out_path: str,
                           surface: List[float],
                           intra: List[float],
                           labels: List[str]) -> str:
    """
    Grouped bars for sampling/cropping ablation.
    """
    series = {
        "Surface Tumor Dice (%)": surface,
        "In-bone Tumor Dice (%)": intra,
    }
    return plot_grouped_bar_comparison(labels, series, ylabel="Dice (%)",
                                       title="Sampling/Cropping Ablation", out_path=out_path)


if __name__ == "__main__":
    # Minimal demo calls with placeholders; replace with real values/paths when running.
    # 1) Class imbalance
    plot_class_imbalance(
        {"Background": 92.0, "Bone": 6.0, "Joint": 1.5, "Surface Tumor": 0.3, "In-bone Tumor": 0.2},
        out_path="/mnt/data/fig_class_imbalance.png"
    )

    # 2) Architecture comparison (placeholders; replace with measured means)
    plot_architecture_comparison(
        out_path="/mnt/data/fig_architecture_comparison.png",
        bone=[88.0, 91.2, 89.0, 90.0],
        joint=[84.0, 87.3, 85.0, 86.0],
        surface=[60.0, 72.4, 65.0, 66.0],
        intra=[45.0, 60.7, 50.0, 52.0],
        arch_labels=["U-Net (scratch)", "U-Net (ResNet-34)", "DeepLabV3+", "nnU-Net (2D)"]
    )

    # 3) Loss ablation (placeholders)
    plot_loss_ablation(
        out_path="/mnt/data/fig_loss_ablation.png",
        surface=[54.1, 65.3, 69.0, 71.0, 72.4],
        intra=[38.7, 49.5, 52.0, 58.0, 60.7],
        loss_labels=["CE", "WCE", "Dice", "Tversky", "WCE+Focal Tversky"]
    )

    # 4) Sampling ablation (placeholders)
    plot_sampling_ablation(
        out_path="/mnt/data/fig_sampling_ablation.png",
        surface=[62.8, 72.4],
        intra=[51.3, 60.7],
        labels=["Uniform crops", "Tumor-aware crops"]
    )

    # 5) Triptych (requires real file paths)
    # visualize_triptych("/path/to/image.png", "/path/to/gt_mask.png", "/path/to/pred_mask.png",
    #                    out_path="/mnt/data/fig_triptych.png")

    # 6) Failures grid (requires real file paths)
    # cases = [
    #   ("/path/img1.png", "/path/gt1.png", "/path/pred1.png"),
    #   ("/path/img2.png", "/path/gt2.png", "/path/pred2.png"),
    # ]
    # plot_failure_grid(cases, out_path="/mnt/data/fig_failures.png", n_cols=2)

    # 7) Metrics from JSON (if available)
    # plot_metrics_from_json("/mnt/data/fixed_evaluation_summary.json",
    #                        out_path="/mnt/data/fig_metrics_from_json.png")
