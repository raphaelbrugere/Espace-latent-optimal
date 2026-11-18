"""Neck FPN top-down (Lin et al., 2017) : fusion des cartes du backbone en une seule.

Seul le chemin top-down est appris : la voie bottom-up est le backbone gele. Projections
laterales 1x1 vers une largeur commune, upsampling nearest et somme, lissage 3x3. La sortie
est au stride le plus fin de `feature_info`.

Sur des cartes iso-stride (ViT simple), la pyramide degenere en une somme de features
de profondeurs differentes, sans gain de resolution : cf. le simple feature pyramid de
ViTDet (Li et al., 2022) pour en reconstruire une.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FPNNeck(nn.Module):
    """{name_i: (B, C_i, H_i, W_i)} -> (B, channels, H_0, W_0), niveau 0 = le plus fin."""

    out_kind = "map"

    def __init__(
        self,
        feature_info: list[dict],
        channels: int = 64,
        levels: list[str] | None = None,
        upsample: str = "nearest",
    ) -> None:
        """`levels` : noms des cartes a fusionner, dans l'ordre de `feature_info` (defaut :
        toutes). `upsample` : mode de F.interpolate du chemin top-down."""
        super().__init__()
        if levels is not None:
            unknown = set(levels) - {level["name"] for level in feature_info}
            if unknown:
                raise ValueError(f"niveaux inconnus du backbone : {sorted(unknown)}")
            feature_info = [level for level in feature_info if level["name"] in levels]
        self.names = [level["name"] for level in feature_info]
        self.in_keys = self.names  # entrees lues dans le dict du backbone
        self.out_dim = channels
        self.upsample = upsample
        self.lateral = nn.ModuleList(
            [nn.Conv2d(level["channels"], channels, kernel_size=1) for level in feature_info]
        )
        self.smooth = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        levels = [conv(feats[name]) for conv, name in zip(self.lateral, self.names, strict=True)]

        fused = levels[-1]
        for finer in reversed(levels[:-1]):
            fused = finer + F.interpolate(fused, size=finer.shape[-2:], mode=self.upsample)

        return self.smooth(fused)
