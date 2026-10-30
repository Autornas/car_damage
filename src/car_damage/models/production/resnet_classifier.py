"""
resnet_classifier.py

Thin wrapper around torchvision's ResNet18 for phase 1 car damage
classification (damaged vs not_damaged). This code rarely changes -- it
adapts a well-established architecture to this project's input/output
shape, nothing more.

Allowed imports: torch, torchvision. Must not import from
car_damage.models.research or car_damage.layers.
"""

import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=2, freeze_backbone=True):
    """
    ResNet18 pretrained on ImageNet with a new final FC layer.
    freeze_backbone=True freezes everything except the new head -- good for
    a first training run with limited data. Unfreeze later for fine-tuning.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final layer -- this one is always trainable
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_last_block(model):
    """
    Call this after a few epochs of head-only training to fine-tune the
    last conv block + fc layer together, usually with a lower learning rate.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    return model
