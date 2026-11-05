"""ViT timm au contrat Backbone : `embed` + une carte de features par bloc lu.

Un ViT simple n'a pas de pyramide : toutes les cartes sont au stride du patch. C'est la
principale faiblesse de ce backbone pour les tâches denses (cf. simple feature pyramid
de ViTDet, ou un ConvNet).

`embed` est la sortie de `forward_head` de timm : pooling (token CLS par défaut), puis,
si `embed_dim` est donné, une projection linéaire apprise avec le backbone pendant le
pré-entraînement. `embed_dim=None` : pas de projection, `embed` = token CLS (largeur
du ViT), comme dans le protocole d'évaluation de DINO.

Poids aléatoires uniquement : les poids externes sont chargés par `backbones/external/`.
"""

import timm
import torch.nn as nn
from torch import Tensor


class TimmViT(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small_patch8_224",
        img_size: int = 128,
        embed_dim: int | None = 512,
        blocks: tuple[int, ...] = (2, 5, 8, 11),
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.vit = timm.create_model(
            arch,
            img_size=img_size,
            num_classes=embed_dim or 0,
            drop_path_rate=drop_path_rate,
            dynamic_img_size=True,  # pré-entraînement et tâches à des résolutions différentes
        )
        self.vit.set_grad_checkpointing(grad_checkpointing)

        self.blocks = list(blocks)
        self.embed_dim = embed_dim or self.vit.num_features
        stride = self.vit.patch_embed.patch_size[0]
        self.feature_info = [
            {"name": f"block{b + 1}", "channels": self.vit.embed_dim, "stride": stride}
            for b in self.blocks
        ]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        final, maps = self.vit.forward_intermediates(
            x, indices=self.blocks, intermediates_only=False
        )
        out = {f["name"]: m for f, m in zip(self.feature_info, maps, strict=True)}
        out["embed"] = self.vit.forward_head(final)
        return out
