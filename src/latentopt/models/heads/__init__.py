"""Tetes downstream : chacune lit la sortie d'un neck (voir models/necks/). Configs dans
configs/head/, assemblage neck + tete dans models/recipe.py."""

from latentopt.models.heads.conv_seg import ConvSegHead
from latentopt.models.heads.linear import LinearHead
from latentopt.models.heads.soft_argmax import SoftArgmaxDecoder

__all__ = ["ConvSegHead", "LinearHead", "SoftArgmaxDecoder"]
