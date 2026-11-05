"""Backbones au contrat de `latentopt.models.contract`.

archs/     architectures (poids aléatoires)
external/  poids externes chargés dans une architecture d'archs/ (aucun réseau défini)
"""

from latentopt.models.backbones.archs.vit import TimmViT
from latentopt.models.backbones.external.dino import dino_vits8

__all__ = ["TimmViT", "dino_vits8"]
