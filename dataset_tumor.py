# dataset_tumor.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import config_tumor as cfg


class TumorSegmentationDataset(Dataset):
    """
    肿瘤三分类数据集：
    - 0: 背景
    - 1: 表面肿瘤 (Surface Tumour)
    - 2: 骨内肿瘤 (In-Bone Tumour)
    """

    def __init__(self, transform=None, only_positive=True):
        self.transform = transform

        # 获取所有可能的mask文件名
        surf_masks = set(f for f in os.listdir(cfg.SURF_DIR) if f.endswith(".png"))
        inb_masks = set(f for f in os.listdir(cfg.INBONE_DIR) if f.endswith(".png"))
        all_names = sorted(surf_masks | inb_masks)

        self.names = []
        for name in all_names:
            # 检查是否存在对应的原始图像
            base = os.path.splitext(name)[0]
            img_path = self._find_image(base)
            if img_path is None:
                continue

            # 构造mask以检查是否有阳性样本
            mask = self._construct_mask(name)

            if only_positive:
                if mask.sum() > 0:  # 有肿瘤标注
                    self.names.append(name)
            else:
                self.names.append(name)

        print(f">>> 肿瘤数据集: 总候选文件 {len(all_names)}, 有效样本 {len(self.names)}")

    def _find_image(self, base_name):
        """查找对应的原始图像文件"""
        for sub in ("2", "3"):
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(cfg.DATASET_ROOT, sub, base_name + ext)
                if os.path.isfile(p):
                    return p
        return None

    def _construct_mask(self, mask_name):
        """构造三分类mask: 0=背景, 1=表面肿瘤, 2=骨内肿瘤"""
        # 先读取表面肿瘤作为基础
        surf_path = os.path.join(cfg.SURF_DIR, mask_name)
        if os.path.isfile(surf_path):
            mask = cv2.imread(surf_path, 0)
            mask = (mask > 0).astype(np.uint8)  # 二值化为0/1
        else:
            # 如果没有表面肿瘤，需要从骨内肿瘤推断尺寸
            inb_path = os.path.join(cfg.INBONE_DIR, mask_name)
            if os.path.isfile(inb_path):
                temp = cv2.imread(inb_path, 0)
                mask = np.zeros_like(temp, dtype=np.uint8)
            else:
                # 这种情况不应该发生，因为我们已经筛选过了
                raise ValueError(f"Neither surface nor in-bone mask found for {mask_name}")

        # 叠加骨内肿瘤 (标签为2)
        inb_path = os.path.join(cfg.INBONE_DIR, mask_name)
        if os.path.isfile(inb_path):
            inb_mask = cv2.imread(inb_path, 0)
            mask[inb_mask > 0] = 2

        return mask

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        mask_name = self.names[idx]
        base = os.path.splitext(mask_name)[0]

        # 读取原始图像
        img_path = self._find_image(base)
        if img_path is None:
            raise ValueError(f"Image not found for {mask_name}")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 构造mask
        mask = self._construct_mask(mask_name)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            return augmented["image"], augmented["mask"].long()

        # 如果没有transform，手动转换
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor