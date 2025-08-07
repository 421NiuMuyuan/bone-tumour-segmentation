# config_joint.py

import os

# 绝对路径，指向包含 2/, 3/, masks_single/, masks_multi/ 的 bones-annotated 目录
DATASET_ROOT = r"C:\Users\yilin\Desktop\week1-unet-segmentation\bones-annotated"

# 关节相关路径
JOINT_DIR    = os.path.join(DATASET_ROOT, "masks_multi", "Joint")

# 二分类：0=背景, 1=关节(Joint)
NUM_CLASSES  = 2
BATCH_SIZE   = 4
NUM_EPOCHS   = 50
LR           = 1e-3
DEVICE       = "cuda" # 或 "cpu"
ES_PATIENCE  = 5      # EarlyStopping 容忍次数

# 输出模型文件名
MODEL_NAME   = "best_joint_unet_smp.pth"