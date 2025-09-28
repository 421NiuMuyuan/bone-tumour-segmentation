# dataset.py  ——  Week-1 femur segmentation dataset (7 classes for bone regions)
import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
import config  # depends on config.py

class FemurSegmentationDataset(Dataset):
    """
    Output:
      - image : (3,H,W) float32 0-1
      - mask  : (H,W)   long   0-NUM_CLASSES-1
    """
    def __init__(self, transform=None):
        self.transform = transform
        root = config.DATASET_ROOT

        # 1. Prefer masks_single; if not found, fall back to masks_multi
        cand_dirs = [
            os.path.join(root, "masks_single"),
            os.path.join(root, "masks_multi"),
        ]
        self.mask_dir = None
        for d in cand_dirs:
            if os.path.isdir(d) and any(f.endswith(".png") for f in os.listdir(d)):
                self.mask_dir = d
                break
        if self.mask_dir is None:
            raise FileNotFoundError("Cannot find masks_single/ or masks_multi/ directory under DATASET_ROOT")

        # 2. Collect base names (without extension)
        mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith(".png")]
        img_dirs   = [os.path.join(root, sub) for sub in ("2", "3")]

        self.names = []
        for m in mask_files:
            base = os.path.splitext(m)[0]
            # Check if corresponding image exists
            if any(os.path.isfile(os.path.join(d, base + ".jpg")) for d in img_dirs):
                self.names.append(base)

        print(f">>> Found {len(self.names)} usable samples "
              f"(mask dir: {os.path.basename(self.mask_dir)})")

        if len(self.names) == 0:
            raise RuntimeError("No usable samples found in the dataset. Please check MASK_DIR path and files.")

        # Save image dirs for __getitem__
        self.img_dirs = img_dirs

    # ----------------------------
    def __len__(self):  
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        # --- load mask ---
        mask = cv2.imread(os.path.join(self.mask_dir, base + ".png"), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {base}.png")
        mask = mask.astype(np.int64)  # ensure long

        # --- load original image ---
        img_path = None
        for d in self.img_dirs:
            cand = os.path.join(d, base + ".jpg")
            if os.path.isfile(cand):
                img_path = cand
                break
        if img_path is None:
            raise FileNotFoundError(f"Missing image: {base}.jpg")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # --- apply transform ---
        if self.transform:
            out = self.transform(image=img, mask=mask)
            return out["image"], out["mask"]

        # No transform: manual tensor conversion
        img_t  = torch.from_numpy(img).permute(2,0,1).float()/255.0
        mask_t = torch.from_numpy(mask)
        return img_t, mask_t
