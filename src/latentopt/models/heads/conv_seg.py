"""Tete de segmentation semantique : carte (B, C, H, W) -> logits par pixel (B, K, H, W).

La sortie reste a la resolution de la carte d'entree ; la mise a la resolution cible
(bilineaire) est laissee a la loss et a la metrique.
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
