"""Neck vectoriel : normalisation de l'embedding global du backbone."""

import torch.nn as nn
from torch import Tensor


class EmbedNorm(nn.Module):
    """`embed` (B, D) -> (B, D), normalisé par une LayerNorm.

    La LayerNorm absorbe l'échelle propre à chaque backbone gelé, ce qui rend les
    hyperparamètres d'optimisation de la tête transférables d'un backbone à l'autre.
    `norm=False` : pas de normalisation, `embed` est rendu tel quel (`nn.Identity`, comme
    dans la sonde linéaire stricte d'origine) ; c'est le cas « pas de neck ».
    """

    out_kind = "vector"

    def __init__(self, embed_dim: int, norm: bool = True) -> None:
        super().__init__()
        self.in_keys = ["embed"]  # entrées lues dans le dict du backbone
        self.out_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        return self.norm(feats["embed"])
