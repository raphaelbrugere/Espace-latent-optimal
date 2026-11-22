"""Tete de segmentation semantique (config `conv_seg`, configs/head/conv_seg.yaml), issue du
decodeur FPN (Lin et al., 2017).

Elle lit les cartes fusionnees par un neck (`FPNNeck`, config `fpn`) et en fait des logits
par pixel. La sortie est au stride le plus fin de `feature_info` ; le passage a la
resolution cible (bilineaire) est laisse a la loss et a la metrique.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvSegHead(nn.Module):
    """Carte (B, C, H, W) -> logits (B, K, H, W) : classifieur a deux convolutions."""

    in_kind = "map"

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.logits = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(F.relu(x))
        return self.logits(F.relu(x))
