import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 配置：根据实际路径修改
MASK_DIR = r"C:\Users\yilin\Desktop\week1-unet-segmentation\bones-annotated\masks_single"
OUT_DIR = r"C:\Users\yilin\Desktop\week1-unet-segmentation\masks_single_viz"

os.makedirs(OUT_DIR, exist_ok=True)

# 遍历所有 single-mask 文件
for fname in sorted(os.listdir(MASK_DIR)):
    if not fname.lower().endswith(".png"):
        continue
    mask_path = os.path.join(MASK_DIR, fname)
    mask = np.array(Image.open(mask_path))

    # 可视化
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=7)
    ax.set_title(fname, fontsize=8)
    ax.axis('off')
    fig.colorbar(cax, ticks=range(0, 8), fraction=0.046, pad=0.04)

    # 保存图像
    out_path = os.path.join(OUT_DIR, fname.replace(".png", "_viz.png"))
    fig.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

print(f"Finished visualizing all masks. Check the folder:\n{OUT_DIR}")
