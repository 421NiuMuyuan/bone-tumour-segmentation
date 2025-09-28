# visualize_tumor_gt.py

import os, cv2, matplotlib.pyplot as plt
from glob import glob
import config_tumor as cfg
import random

# Find all positive samples (masks with non-zero pixels)
positive = []
for f in os.listdir(cfg.SURF_DIR):
    if not f.lower().endswith(".png"):
        continue
    m = cv2.imread(os.path.join(cfg.SURF_DIR, f), 0)
    if m is None or m.sum() == 0:
        continue
    positive.append(f)
# You can similarly check INBONE_DIR if needed

print(f">>> Positive tumor masks: {len(positive)}")

# Randomly display N samples
random.shuffle(positive)
N = min(4, len(positive))
fig, axes = plt.subplots(N, 3, figsize=(12, 4 * N))

for i, fname in enumerate(positive[:N]):
    # Original image
    base = os.path.splitext(fname)[0]
    # Look for image under subfolders 2/3
    img_path = None
    for sub in ("2", "3"):
        p = os.path.join(cfg.DATASET_ROOT, sub, base + ".jpg")
        if os.path.isfile(p):
            img_path = p
            break
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Merge Surface & In-Bone masks
    surf = cv2.imread(os.path.join(cfg.SURF_DIR, fname), 0)
    inb  = cv2.imread(os.path.join(cfg.INBONE_DIR, fname), 0)
    mask = surf.copy()
    mask[inb > 0] = 2

    axes[i, 0].imshow(img);       axes[i, 0].set_title("Image");      axes[i, 0].axis("off")
    axes[i, 1].imshow(surf, cmap="nipy_spectral", vmin=0, vmax=2)
    axes[i, 1].set_title("Surface GT"); axes[i, 1].axis("off")
    axes[i, 2].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=2)
    axes[i, 2].set_title("Merged GT");  axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
