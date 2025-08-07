# masks_tumor_viz.py

import os, cv2, matplotlib.pyplot as plt
import config_tumor as cfg

# 收集所有表面/骨内 mask 文件
surf = sorted([
    os.path.join(cfg.SURF_DIR, f)
    for f in os.listdir(cfg.SURF_DIR) if f.endswith(".png")
])
inb  = sorted([
    os.path.join(cfg.INBONE_DIR, f)
    for f in os.listdir(cfg.INBONE_DIR) if f.endswith(".png")
])
mask_paths = surf + inb
print(f">>> Found {len(mask_paths)} tumour-mask files")

# 可视化前 9 张
N = min(9, len(mask_paths))
rows = (N + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(12, 4*rows))
axes = axes.flatten()

for i, p in enumerate(mask_paths[:N]):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None: continue
    # 如果多通道取最大
    if m.ndim == 3: m = m.max(axis=2)

    axes[i].imshow(m, cmap="nipy_spectral", vmin=0, vmax=2)
    axes[i].set_title(os.path.basename(p))
    axes[i].axis("off")

for ax in axes[N:]:
    ax.axis("off")

plt.tight_layout()
plt.show()
