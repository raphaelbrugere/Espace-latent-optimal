"""Décodeur FPN top-down (Lin et al., 2017) pour la segmentation sémantique.

Seul le chemin top-down est appris : la voie bottom-up est le backbone gelé. Projections
latérales 1x1 vers une largeur commune, upsampling nearest et somme, lissage 3x3, puis
classifieur à deux convolutions. La sortie est au stride le plus fin de `feature_info` ;
le passage à la résolution cible (bilinéaire) est laissé à la loss et à la métrique.

Sur des cartes iso-stride (ViT simple), la pyramide dégénère en une somme de features
de profondeurs différentes, sans gain de résolution : cf. le simple feature pyramid de
ViTDet (Li et al., 2022) pour en reconstruire une.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FPNHead(nn.Module):
    """{name_i: (B, C_i, H_i, W_i)} -> logits (B, K, H_0, W_0), niveau 0 = le plus fin."""

    def __init__(
        self,
        feature_info: list[dict],
        num_classes: int,
        channels: int = 64,
        levels: list[str] | None = None,
        upsample: str = "nearest",
    ):
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
