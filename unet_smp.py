# unet_smp.py

import segmentation_models_pytorch as smp

def get_model(num_classes):
    """
    U-Net with ResNet34 encoder pretrained on ImageNet
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None  # raw logits
    )
