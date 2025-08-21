# model.py
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Build a ResNet50 with an adjustable classification head.
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
