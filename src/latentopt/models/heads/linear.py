"""Tete lineaire : sonde de classification, vecteur (B, D) -> logits (B, K)."""

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
