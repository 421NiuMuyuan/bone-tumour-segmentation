# inference_system.py

import os
import sys
import torch
import numpy as np
import cv2
from typing import Dict, Any
from unet_smp import get_model
import config, config_tumor, config_joint
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# Device
device = config.DEVICE

# Function to load checkpoints
def load_checkpoint(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)

# Initialize models
bone_m  = get_model(config.NUM_CLASSES).to(device)
tum_m   = get_model(config_tumor.NUM_CLASSES).to(device)
joint_m = get_model(config_joint.NUM_CLASSES).to(device)

# Load weights
load_checkpoint(bone_m,  "best_unet_smp.pth")
load_checkpoint(tum_m,   "best_tumor.pth")
load_checkpoint(joint_m, "best_joint_unet_smp.pth")

for m in (bone_m, tum_m, joint_m):
    m.eval()

# Preprocessing
tform = Compose([
    Resize(512,512),
    Normalize((0.5,)*3, (0.5,)*3),
    ToTensorV2()
])

# Bone region mapping
BONE_REGIONS = {
    1: "apophysis",
    2: "epiphysis",
    3: "metaphysis",
    4: "diaphysis",
    5: "greater_trochanter",
    6: "lesser_trochanter",
    7: "neck"
}

# Analysis function
def analyze_image(img_path: str) -> Dict[str, Any]:
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aug = tform(image=img_rgb)
    x   = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        bone_pred  = bone_m(x).argmax(1).squeeze().cpu().numpy()
        tumor_pred = tum_m(x).argmax(1).squeeze().cpu().numpy()
        joint_pred = joint_m(x).argmax(1).squeeze().cpu().numpy()

    has_tumor = bool((tumor_pred > 0).any())
    surf = bool((tumor_pred == 1).any())
    inb  = bool((tumor_pred == 2).any())
    if surf and inb:
        depth = "mixed"
    elif surf:
        depth = "surface"
    elif inb:
        depth = "in-bone"
    else:
        depth = None

    region = None
    if has_tumor:
        labels, counts = np.unique(bone_pred[tumor_pred > 0], return_counts=True)
        d = {int(l):int(c) for l,c in zip(labels,counts) if l != 0}
        if d:
            best = max(d, key=d.get)
            region = BONE_REGIONS.get(best, f"label_{best}")

    in_joint = bool(((tumor_pred > 0) & (joint_pred > 0)).any())

    return {
        "has_tumor": has_tumor,
        "tumor_depth": depth,
        "bone_region": region,
        "in_joint_region": in_joint
    }

# Main entry
if __name__ == "__main__":
    # 可以通过命令行参数指定 '2' 或 '3'，默认 '3' (含肿瘤)
    folder = os.path.join(config.DATASET_ROOT, sys.argv[1] if len(sys.argv)>1 else "3")
    print(f">>> Inferencing on: {folder}")
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".jpg",".jpeg",".png")): continue
        info = analyze_image(os.path.join(folder, fn))
        print(f"{fn}: {info}")
