"""Linear probe sur l'embedding global, protocole d'évaluation standard en SSL."""

import torch.nn as nn
from torch import Tensor


class LinearProbe(nn.Module):
    """LayerNorm puis projection affine : `embed` (B, D) -> logits (B, K).

    La LayerNorm absorbe l'échelle propre à chaque backbone gelé, ce qui rend les
    hyperparamètres d'optimisation de la sonde transférables d'un backbone à l'autre.
    Sa partie affine se replie dans `fc` : la frontière de décision reste linéaire
    à une normalisation près. `norm=False` donne la sonde linéaire stricte.
    """

    def __init__(self, embed_dim: int, num_classes: int, norm: bool = True):
        super().__init__()
        self.in_keys = ["embed"]  # entrées lues dans le dict du backbone
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        x = self.norm(feats["embed"])
        return self.fc(x)
