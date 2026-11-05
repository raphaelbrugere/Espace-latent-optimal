"""Localisation mono-instance par soft-argmax spatial (régression intégrale).

Chaque cellule de la carte au stride le plus proche de 8 émet un logit d'attention,
un offset sub-cellule du centre et une taille de boîte normalisée (sigmoïdes). La boîte
est l'espérance de ces prédictions sous la softmax spatiale, et la classe est prédite
depuis les features poolées par la même attention. Contrairement au décodage argmax
de CenterNet, tout est dérivable : la tête s'entraîne directement sous L1 + GIoU, sans
heatmap cible (cf. Sun et al., 2018, Integral Human Pose Regression).

Limites : une seule instance par image, et sous une attention multimodale l'espérance
tombe entre les modes, de même que la taille moyennée.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SoftArgmaxHead(nn.Module):
    """(B, C, H, W) -> {"box": (B, 4) xyxy normalisé dans [0, 1], "logits": (B, K)}."""

    def __init__(
        self,
        feature_info: list[dict],
        num_classes: int,
        channels: int = 128,
        target_stride: int = 8,
    ):
        super().__init__()
        # stride le plus proche de target_stride ; à égalité (ViT simple), le plus profond
        level = min(reversed(feature_info), key=lambda f: abs(f["stride"] - target_stride))
        self.name = level["name"]
        self.in_keys = [self.name]  # entrées lues dans le dict du backbone

        self.conv1 = nn.Conv2d(level["channels"], channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.geometry = nn.Conv2d(channels, 4, kernel_size=1)  # dx, dy, w, h
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, feats: dict[str, Tensor]) -> dict[str, Tensor]:
        x = F.relu(self.bn1(self.conv1(feats[self.name])))
        x = F.relu(self.bn2(self.conv2(x)))  # (B, C, H, W)
        _, _, H, W = x.shape

        weights = self.attention(x).flatten(1).softmax(dim=1)  # (B, HW)

        dx, dy, width, height = self.geometry(x).flatten(2).sigmoid().unbind(dim=1)  # (B, HW)
        rows, cols = torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"
        )
        center_x = (cols.flatten() + dx) / W
        center_y = (rows.flatten() + dy) / H

        # espérance sous l'attention
        cx, cy, w, h = ((v * weights).sum(dim=1) for v in (center_x, center_y, width, height))
        box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1).clamp(0, 1)

        pooled = (x.flatten(2) * weights.unsqueeze(1)).sum(dim=2)  # (B, C)
        return {"box": box, "logits": self.classifier(pooled)}
