"""Localisation mono-instance par soft-argmax spatial (regression integrale).

Chaque cellule de la carte emet un logit d'attention, un offset sub-cellule du centre et
une taille de boite normalisee (sigmoides). La boite est l'esperance de ces predictions
sous la softmax spatiale, et la classe est predite depuis les features poolees par la meme
attention. Contrairement au decodage argmax de CenterNet, tout est derivable : la tete
s'entraine directement sous L1 + GIoU, sans heatmap cible (cf. Sun et al., 2018, Integral
Human Pose Regression).

Recette `soft_argmax` (configs/head/soft_argmax.yaml) : neck `ConvTrunk`, qui lit la carte au
stride le plus proche de 8, puis cette tete.

Limites : une seule instance par image, et sous une attention multimodale l'esperance
tombe entre les modes, de meme que la taille moyennee.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SoftArgmaxDecoder(nn.Module):
    """(B, C, H, W) -> {"box": (B, 4) xyxy normalise dans [0, 1], "logits": (B, K)}."""

    in_kind = "map"

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.attention = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.geometry = nn.Conv2d(in_dim, 4, kernel_size=1)  # dx, dy, w, h
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        _, _, H, W = x.shape

        weights = self.attention(x).flatten(1).softmax(dim=1)  # (B, HW)

        dx, dy, width, height = self.geometry(x).flatten(2).sigmoid().unbind(dim=1)  # (B, HW)
        rows, cols = torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"
        )
        center_x = (cols.flatten() + dx) / W
        center_y = (rows.flatten() + dy) / H

        # esperance sous l'attention
        cx, cy, w, h = ((v * weights).sum(dim=1) for v in (center_x, center_y, width, height))
        box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1).clamp(0, 1)

        pooled = (x.flatten(2) * weights.unsqueeze(1)).sum(dim=2)  # (B, C)
        return {"box": box, "logits": self.classifier(pooled)}
