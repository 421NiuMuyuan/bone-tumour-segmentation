# config_tumor.py

import os

# 绝对路径，指向包含 2/, 3/, masks_single/, masks_multi/ 的 bones-annotated 目录
DATASET_ROOT = r"C:\Users\yilin\Desktop\week1-unet-segmentation\bones-annotated"

# 肿瘤相关路径
SURF_DIR     = os.path.join(DATASET_ROOT, "masks_multi", "Surface Tumour")
INBONE_DIR   = os.path.join(DATASET_ROOT, "masks_multi", "In-Bone Tumour")

# 三分类：0=背景, 1=表面肿瘤(Surface Tumour), 2=骨内肿瘤(In-Bone Tumour)
NUM_CLASSES  = 3
BATCH_SIZE   = 4
NUM_EPOCHS   = 50
LR           = 1e-3
DEVICE       = "cuda" # 或 "cpu"
ES_PATIENCE  = 5      # EarlyStopping 容忍次数

# 输出模型文件名
MODEL_NAME   = "best_tumor_unet_smp.pth"