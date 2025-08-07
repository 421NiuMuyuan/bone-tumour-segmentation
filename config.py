# config.py

# 绝对路径，指向包含 2/, 3/, masks_single/, masks_multi/ 的 bones-annotated 目录
DATASET_ROOT = r"C:\Users\yilin\Desktop\week1-unet-segmentation\bones-annotated"
MASK_DIR     = DATASET_ROOT + r"\masks_single"

NUM_CLASSES  = 8      # 背景(0) + 7 类
BATCH_SIZE   = 4
NUM_EPOCHS   = 50     # 提高到 50 轮
LR           = 1e-3
DEVICE       = "cuda" # 或 "cpu"
ES_PATIENCE  = 5      # EarlyStopping 容忍次数
