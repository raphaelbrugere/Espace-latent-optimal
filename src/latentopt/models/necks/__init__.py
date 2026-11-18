"""Necks : du dict de features du backbone à ce que lit la tête, entraînés avec elle.

Un neck par recette d'origine (configs/head/) : `EmbedNorm` (linear), `FPNNeck` (fpn),
`ConvTrunk` (soft_argmax)."""

from latentopt.models.necks.conv_trunk import ConvTrunk
from latentopt.models.necks.embed_norm import EmbedNorm
from latentopt.models.necks.fpn import FPNNeck

__all__ = ["ConvTrunk", "EmbedNorm", "FPNNeck"]
