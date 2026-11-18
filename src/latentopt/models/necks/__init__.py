"""Necks : du dict de features du backbone a ce que lit la tete, entraines avec elle.

Un neck par recette d'origine (configs/head/) : `EmbedNorm` (linear), `FPNNeck` (fpn),
`ConvTrunk` (soft_argmax)."""

from latentopt.models.necks.conv_trunk import ConvTrunk
from latentopt.models.necks.embed_norm import EmbedNorm
from latentopt.models.necks.fpn import FPNNeck

__all__ = ["ConvTrunk", "EmbedNorm", "FPNNeck"]
