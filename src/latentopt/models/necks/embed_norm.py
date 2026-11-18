"""Neck vectoriel : normalisation de l'embedding global du backbone."""

import torch.nn as nn
from torch import Tensor


class EmbedNorm(nn.Module):
    """`embed` (B, D) -> (B, D), normalise par une LayerNorm.

    La LayerNorm absorbe l'echelle propre a chaque backbone gele, ce qui rend les
    hyperparametres d'optimisation de la tete transferables d'un backbone a l'autre.
    `norm=False` : pas de normalisation, `embed` est rendu tel quel (`nn.Identity`, comme
    dans la sonde lineaire stricte d'origine) ; c'est le cas " pas de neck ".
    """

    out_kind = "vector"

    def __init__(self, embed_dim: int, norm: bool = True) -> None:
        super().__init__()
        self.in_keys = ["embed"]  # entrees lues dans le dict du backbone
        self.out_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        return self.norm(feats["embed"])
