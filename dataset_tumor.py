# dataset_tumor.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import config_tumor as cfg


class TumorSegmentationDataset(Dataset):
    """
    Tumor segmentation dataset (3 classes):
    - 0: Background
    - 1: Surface Tumour
    - 2: In-Bone Tumour
    """

    def __init__(self, transform=None, only_positive=True):
        self.transform = transform

        # Collect all possible mask file names
        surf_masks = set(f for f in os.listdir(cfg.SURF_DIR) if f.endswith(".png"))
        inb_masks = set(f for f in os.listdir(cfg.INBONE_DIR) if f.endswith(".png"))
        all_names = sorted(surf_masks | inb_masks)

        self.names = []
        for name in all_names:
            # Check if the corresponding original image exists
            base = os.path.splitext(name)[0]
            img_path = self._find_image(base)
            if img_path is None:
                continue

            # Build mask to check whether positive samples exist
            mask = self._construct_mask(name)

            if only_positive:
                if mask.sum() > 0:  # has tumor annotation
                    self.names.append(name)
            else:
                self.names.append(name)

        print(f">>> Tumor dataset: total candidates {len(all_names)}, valid samples {len(self.names)}")

    def _find_image(self, base_name):
        """Find the corresponding original image file."""
        for sub in ("2", "3"):
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(cfg.DATASET_ROOT, sub, base_name + ext)
                if os.path.isfile(p):
                    return p
        return None

    def _construct_mask(self, mask_name):
        """Build a 3-class mask: 0=background, 1=surface tumour, 2=in-bone tumour"""
        # Start with surface tumour as base
        surf_path = os.path.join(cfg.SURF_DIR, mask_name)
        if os.path.isfile(surf_path):
            mask = cv2.imread(surf_path, 0)
            mask = (mask > 0).astype(np.uint8)  # binarize to 0/1
        else:
            # If no surface mask, infer size from in-bone mask
            inb_path = os.path.join(cfg.INBONE_DIR, mask_name)
            if os.path.isfile(inb_path):
                temp = cv2.imread(inb_path, 0)
                mask = np.zeros_like(temp, dtype=np.uint8)
            else:
                # Should not happen, since we already filtered
                raise ValueError(f"Neither surface nor in-bone mask found for {mask_name}")

        # Overlay in-bone tumour (label = 2)
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
