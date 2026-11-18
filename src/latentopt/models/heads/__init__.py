"""Têtes downstream : chacune lit la sortie d'un neck (voir models/necks/). Un neck et une tête
forment une recette (configs/head/, models/recipe.py)."""

from latentopt.models.heads.fpn import ConvSegHead
from latentopt.models.heads.linear import LinearHead
from latentopt.models.heads.soft_argmax import SoftArgmaxDecoder

__all__ = ["ConvSegHead", "LinearHead", "SoftArgmaxDecoder"]
