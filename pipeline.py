# pipeline.py

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from unet_smp       import get_model
import config, config_tumor, config_joint
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

device = config.DEVICE

# --- 辅助函数：加载 checkpoint 中的 state_dict ---
def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=device)
    # 如果是完整 checkpoint dict，就取 'model_state_dict'
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)

# 1) 初始化三个模型
bone_model  = get_model(config.NUM_CLASSES).to(device)
tum_model   = get_model(config_tumor.NUM_CLASSES).to(device)
joint_model = get_model(config_joint.NUM_CLASSES).to(device)

# 2) 加载权重（根据你的文件名调整）
load_checkpoint(bone_model,  "best_unet_smp.pth")
load_checkpoint(tum_model,   "best_tumor.pth")
load_checkpoint(joint_model, "best_joint_unet_smp.pth")

for m in (bone_model, tum_model, joint_model):
    m.eval()

# 3) 预处理管线
transform = Compose([
    Resize(512, 512),
    Normalize((0.5,)*3, (0.5,)*3),
    ToTensorV2()
])

# 4) 批量推理并可视化
def segment_all(input_folder, max_images=10):
    jpgs = sorted(glob(os.path.join(input_folder, "*.jpg")))[:max_images]
    for img_path in jpgs:
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug     = transform(image=img_rgb)
        x       = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            bone_pred  = bone_model(x).argmax(1).squeeze().cpu().numpy()
            tumor_pred = tum_model(x).argmax(1).squeeze().cpu().numpy()
            joint_pred = joint_model(x).argmax(1).squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16,4))
        axes[0].imshow(img_rgb); axes[0].set_title("Input");   axes[0].axis("off")
        axes[1].imshow(bone_pred,  cmap="nipy_spectral", vmin=0, vmax=config.NUM_CLASSES-1)
        axes[1].set_title("Bone");  axes[1].axis("off")
        axes[2].imshow(tumor_pred, cmap="nipy_spectral", vmin=0, vmax=config_tumor.NUM_CLASSES-1)
        axes[2].set_title("Tumor"); axes[2].axis("off")
        axes[3].imshow(joint_pred, cmap="gray",          vmin=0, vmax=1)
        axes[3].set_title("Joint"); axes[3].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    demo_folder = os.path.join(config.DATASET_ROOT, "2")
    segment_all(demo_folder, max_images=10)
