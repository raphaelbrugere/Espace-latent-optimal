"""Necks : du dict de features du backbone a ce que lit la tete, entraines avec elle.

Un neck par config de configs/neck/ : `EmbedNorm` (embed_norm), `FPNNeck` (fpn),
`ConvTrunk` (conv_trunk)."""

from latentopt.models.necks.conv_trunk import ConvTrunk
from latentopt.models.necks.embed_norm import EmbedNorm
from latentopt.models.necks.fpn import FPNNeck

__all__ = ["ConvTrunk", "EmbedNorm", "FPNNeck"]
