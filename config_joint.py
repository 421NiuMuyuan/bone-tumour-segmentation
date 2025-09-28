# config_joint.py

import os

DATASET_ROOT = r"Desktop\week1-unet-segmentation\bones-annotated"

JOINT_DIR    = os.path.join(DATASET_ROOT, "masks_multi", "Joint")

NUM_CLASSES  = 2
BATCH_SIZE   = 4
NUM_EPOCHS   = 50
LR           = 1e-3
DEVICE       = "cuda" # "cpu"
ES_PATIENCE  = 5      # 


MODEL_NAME   = "best_joint_unet_smp.pth"