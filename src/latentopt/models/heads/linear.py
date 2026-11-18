"""Tête linéaire : la sonde de classification, protocole d'évaluation standard en SSL.

Recette `linear` (configs/head/linear.yaml) : neck `EmbedNorm` (LayerNorm sur `embed`) puis
cette tête. La partie affine de la LayerNorm se replie dans la projection : la frontière de
décision reste linéaire à une normalisation près.
"""

import torch.nn as nn
from torch import Tensor


class LinearHead(nn.Module):
    """Vecteur (B, D) -> logits (B, K) : projection affine."""

    in_kind = "vector"

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
