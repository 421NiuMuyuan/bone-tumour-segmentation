# config_tumor.py

import os


DATASET_ROOT = r"Desktop\week1-unet-segmentation\bones-annotated"


SURF_DIR     = os.path.join(DATASET_ROOT, "masks_multi", "Surface Tumour")
INBONE_DIR   = os.path.join(DATASET_ROOT, "masks_multi", "In-Bone Tumour")

NUM_CLASSES  = 3
BATCH_SIZE   = 4
NUM_EPOCHS   = 50
LR           = 1e-3
DEVICE       = "cuda" #  "cpu"
ES_PATIENCE  = 5      


MODEL_NAME   = "best_tumor_unet_smp.pth"