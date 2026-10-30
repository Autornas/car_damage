"""
scratch_cnn.py

A CNN built entirely from car_damage.layers.scratch_layers -- no pretrained
weights, no nn.Conv2d/nn.Linear/nn.BatchNorm2d. Every parameter starts from
random (He) initialization and has to learn everything from your car
photos.

Architecture: 5 conv blocks that double (then hold) channel depth while
halving spatial size each time, followed by global average pooling and
a linear classifier head.

    224x224x3
      -> block1 ->  112x112x32
      -> block2 ->   56x56x64
      -> block3 ->   28x28x128
      -> block4 ->   14x14x256
      -> block5 ->    7x7x256
      -> global avg pool -> 256
      -> dropout -> linear -> 2 classes

Compare to models.production.resnet_classifier's ResNet18: that one starts
with 11M+ pretrained parameters already tuned on 1.4M ImageNet images. This
one starts at random and only ever sees whatever you put in data/.

Allowed imports: torch, car_damage.layers. Must not be imported by anything
in car_damage.models.production.
"""

import torch.nn as nn
from car_damage.layers.scratch_layers import (
    Conv2dScratch, BatchNorm2dScratch, ReLUScratch,
    MaxPool2dScratch, GlobalAvgPoolScratch, LinearScratch, DropoutScratch
)


class ConvBlock(nn.Module):
    """conv -> batchnorm -> relu -> maxpool, the repeating unit of the network."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Conv2dScratch(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = BatchNorm2dScratch(out_ch)
        self.relu = ReLUScratch()
        self.pool = MaxPool2dScratch(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ScratchCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.block5 = ConvBlock(256, 256)
        self.gap = GlobalAvgPoolScratch()
        self.dropout = DropoutScratch(dropout)
        self.fc = LinearScratch(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: does a forward pass run and produce the right shape?
    import torch
    model = ScratchCNN(num_classes=2)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect (4, 2)
    print(f"Trainable parameters: {count_parameters(model):,}")
