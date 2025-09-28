# config.py


DATASET_ROOT = r"Desktop\week1-unet-segmentation\bones-annotated"
MASK_DIR     = DATASET_ROOT + r"\masks_single"

NUM_CLASSES  = 8      
BATCH_SIZE   = 4
NUM_EPOCHS   = 50    
LR           = 1e-3
DEVICE       = "cuda" # "cpu"
ES_PATIENCE  = 5      #
