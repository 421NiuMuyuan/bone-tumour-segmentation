# dataset_joint.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import config_joint as cfg


class JointSegmentationDataset(Dataset):
    """
    关节二分类数据集：
    - 0: 背景
    - 1: 关节区域 (Joint)
    """

    def __init__(self, transform=None, only_positive=True):
        self.transform = transform

        # 获取所有关节mask文件
        if not os.path.exists(cfg.JOINT_DIR):
            raise ValueError(f"关节mask目录不存在: {cfg.JOINT_DIR}")

        joint_masks = [f for f in os.listdir(cfg.JOINT_DIR) if f.endswith(".png")]

        self.names = []
        for mask_name in joint_masks:
            # 检查是否存在对应的原始图像
            base = os.path.splitext(mask_name)[0]
            img_path = self._find_image(base)
            if img_path is None:
                continue

            # 检查mask是否有内容
            if only_positive:
                mask_path = os.path.join(cfg.JOINT_DIR, mask_name)
                mask = cv2.imread(mask_path, 0)
                if mask is not None and mask.sum() > 0:  # 有关节标注
                    self.names.append(mask_name)
            else:
                self.names.append(mask_name)

        print(f">>> 关节数据集: 总候选文件 {len(joint_masks)}, 有效样本 {len(self.names)}")

    def _find_image(self, base_name):
        """查找对应的原始图像文件"""
        for sub in ("2", "3"):
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(cfg.DATASET_ROOT, sub, base_name + ext)
                if os.path.isfile(p):
                    return p
        return None

    def _construct_mask(self, mask_name):
        """构造二分类mask: 0=背景, 1=关节"""
        mask_path = os.path.join(cfg.JOINT_DIR, mask_name)

        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, 0)
            # 二值化：任何非零值都设为1
            mask = (mask > 0).astype(np.uint8)
        else:
            raise ValueError(f"Joint mask not found: {mask_path}")

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