# dataset.py  ——  Week-1 骨区七分类数据集
import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
import config                      # 依赖 config.py

class FemurSegmentationDataset(Dataset):
    """
    输出:
      - image : (3,H,W) float32 0-1
      - mask  : (H,W)   long   0-NUM_CLASSES-1
    """
    def __init__(self, transform=None):
        self.transform = transform
        root = config.DATASET_ROOT

        # 1. 先找单通道 masks_single；若没找到则用 masks_multi
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
            raise FileNotFoundError("在 DATASET_ROOT 下找不到 masks_single/ 或 masks_multi/ 目录")

        # 2. 收集 base 名（无后缀）
        mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith(".png")]
        img_dirs   = [os.path.join(root, sub) for sub in ("2", "3")]

        self.names = []
        for m in mask_files:
            base = os.path.splitext(m)[0]
            # 检查原图是否存在
            if any(os.path.isfile(os.path.join(d, base + ".jpg")) for d in img_dirs):
                self.names.append(base)

        print(f">>> Found {len(self.names)} usable samples "
              f"(mask dir: {os.path.basename(self.mask_dir)})")

        if len(self.names) == 0:
            raise RuntimeError("数据集中没有可用样本，请确认 MASK_DIR 路径和文件")

        # 保存 image dirs 以便 __getitem__
        self.img_dirs = img_dirs

    # ----------------------------
    def __len__(self):  return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        # --- 读 mask ---
        mask = cv2.imread(os.path.join(self.mask_dir, base + ".png"), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"mask 缺失: {base}.png")
        mask = mask.astype(np.int64)  # 确保 long

        # --- 读原图 ---
        img_path = None
        for d in self.img_dirs:
            cand = os.path.join(d, base + ".jpg")
            if os.path.isfile(cand):
                img_path = cand
                break
        if img_path is None:
            raise FileNotFoundError(f"原图缺失: {base}.jpg")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # --- transform ---
        if self.transform:
            out = self.transform(image=img, mask=mask)
            return out["image"], out["mask"]

        # 无增强：手动转 tensor
        img_t  = torch.from_numpy(img).permute(2,0,1).float()/255.0
        mask_t = torch.from_numpy(mask)
        return img_t, mask_t
