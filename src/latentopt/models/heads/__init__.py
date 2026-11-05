"""Têtes downstream. Chacune prend le dict de features du backbone en entrée."""

from latentopt.models.heads.fpn import FPNHead
from latentopt.models.heads.linear import LinearProbe
from latentopt.models.heads.soft_argmax import SoftArgmaxHead

__all__ = ["FPNHead", "LinearProbe", "SoftArgmaxHead"]
