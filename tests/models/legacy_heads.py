"""Têtes monolithiques d'avant la coupe neck / tête : référence des tests d'équivalence.

Copie fidèle des anciennes `LinearProbe`, `FPNHead` et `SoftArgmaxHead`. Ne pas les modifier :
elles servent à vérifier que les têtes recomposées (neck + tête) font exactement le même calcul,
avec les mêmes poids initiaux pour une même graine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearProbe(nn.Module):
    """LayerNorm puis projection affine : `embed` (B, D) -> logits (B, K).

    La LayerNorm absorbe l'échelle propre à chaque backbone gelé, ce qui rend les
    hyperparamètres d'optimisation de la sonde transférables d'un backbone à l'autre.
    Sa partie affine se replie dans `fc` : la frontière de décision reste linéaire
    à une normalisation près. `norm=False` donne la sonde linéaire stricte.
    """

    def __init__(self, embed_dim: int, num_classes: int, norm: bool = True) -> None:
        super().__init__()
        self.in_keys = ["embed"]  # entrées lues dans le dict du backbone
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        x = self.norm(feats["embed"])
        return self.fc(x)


class FPNHead(nn.Module):
    """{name_i: (B, C_i, H_i, W_i)} -> logits (B, K, H_0, W_0), niveau 0 = le plus fin."""

    def __init__(
        self,
        feature_info: list[dict],
        num_classes: int,
        channels: int = 64,
        levels: list[str] | None = None,
        upsample: str = "nearest",
    ) -> None:
        """`levels` : noms des cartes à fusionner, dans l'ordre de `feature_info` (défaut :
        toutes). `upsample` : mode de F.interpolate du chemin top-down."""
        super().__init__()
        if levels is not None:
            unknown = set(levels) - {level["name"] for level in feature_info}
            if unknown:
                raise ValueError(f"niveaux inconnus du backbone : {sorted(unknown)}")
            feature_info = [level for level in feature_info if level["name"] in levels]
        self.names = [level["name"] for level in feature_info]
        self.in_keys = self.names  # entrées lues dans le dict du backbone
        self.upsample = upsample
        self.lateral = nn.ModuleList(
            [nn.Conv2d(level["channels"], channels, kernel_size=1) for level in feature_info]
        )
        self.smooth = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.logits = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        levels = [conv(feats[name]) for conv, name in zip(self.lateral, self.names, strict=True)]

        fused = levels[-1]
        for finer in reversed(levels[:-1]):
            fused = finer + F.interpolate(fused, size=finer.shape[-2:], mode=self.upsample)

        x = self.smooth(fused)
        x = self.conv(F.relu(x))
        return self.logits(F.relu(x))


class SoftArgmaxHead(nn.Module):
    """(B, C, H, W) -> {"box": (B, 4) xyxy normalisé dans [0, 1], "logits": (B, K)}."""

    def __init__(
        self,
        feature_info: list[dict],
        num_classes: int,
        channels: int = 128,
        target_stride: int = 8,
    ) -> None:
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
