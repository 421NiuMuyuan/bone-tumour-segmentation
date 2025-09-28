# dataset_joint.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import config_joint as cfg


class JointSegmentationDataset(Dataset):
    """
    Joint binary segmentation dataset:
    - 0: background
    - 1: joint region (Joint)
    """

    def __init__(self, transform=None, only_positive=True):
        self.transform = transform

        # Collect all joint mask files
        if not os.path.exists(cfg.JOINT_DIR):
            raise ValueError(f"Joint mask directory does not exist: {cfg.JOINT_DIR}")

        joint_masks = [f for f in os.listdir(cfg.JOINT_DIR) if f.endswith(".png")]

        self.names = []
        for mask_name in joint_masks:
            # Check if the corresponding original image exists
            base = os.path.splitext(mask_name)[0]
            img_path = self._find_image(base)
            if img_path is None:
                continue

            # Keep only masks with content when only_positive=True
            if only_positive:
                mask_path = os.path.join(cfg.JOINT_DIR, mask_name)
                mask = cv2.imread(mask_path, 0)
                if mask is not None and mask.sum() > 0:  # has joint annotation
                    self.names.append(mask_name)
            else:
                self.names.append(mask_name)

        print(f">>> Joint dataset: total candidates {len(joint_masks)}, valid samples {len(self.names)}")

    def _find_image(self, base_name):
        """Find the corresponding original image file."""
        for sub in ("2", "3"):
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(cfg.DATASET_ROOT, sub, base_name + ext)
                if os.path.isfile(p):
                    return p
        return None

    def _construct_mask(self, mask_name):
        """Build a binary mask: 0=background, 1=joint."""
        mask_path = os.path.join(cfg.JOINT_DIR, mask_name)

        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, 0)
            # Binarize: any non-zero becomes 1
            mask = (mask > 0).astype(np.uint8)
        else:
            raise ValueError(f"Joint mask not found: {mask_path}")

        return mask

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        mask_name = self.names[idx]
        base = os.path.splitext(mask_name)[0]

        # Read original image
        img_path = self._find_image(base)
        if img_path is None:
            raise ValueError(f"Image not found for {mask_name}")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Build mask
        mask = self._construct_mask(mask_name)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            return augmented["image"], augmented["mask"].long()

        # Fallback if no transform: manual tensor conversion
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor
