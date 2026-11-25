"""Neck vectoriel : normalisation, au choix, de l'embedding global du backbone."""

import torch.nn as nn
from torch import Tensor

# nom de la normalisation -> constructeur (embed_dim)
NORMS = {"layernorm": nn.LayerNorm, "none": lambda dim: nn.Identity()}


class EmbedNorm(nn.Module):
    """`embed` (B, D) -> (B, D), passe par la normalisation choisie par `norm`.

    `norm` : un nom de `NORMS`. "layernorm" (defaut) absorbe l'echelle propre a chaque
    backbone gele, ce qui rend les hyperparametres d'optimisation de la tete transferables
    d'un backbone a l'autre. "none" rend `embed` tel quel (`nn.Identity`, comme dans la
    sonde lineaire stricte d'origine : le cas "pas de neck").
    """

    out_kind = "vector"

    def __init__(self, embed_dim: int, norm: str = "layernorm") -> None:
        super().__init__()
        self.in_keys = ["embed"]  # entrees lues dans le dict du backbone
        self.out_dim = embed_dim
        if norm not in NORMS:
            raise ValueError(f"norm inconnue : {norm!r} (choix : {sorted(NORMS)})")
        self.norm = NORMS[norm](embed_dim)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        return self.norm(feats["embed"])
