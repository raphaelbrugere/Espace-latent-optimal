"""Poids DINO (Caron et al., 2021) du ViT-S/8, auto-supervises sur ImageNet-1k.

Architecture : `archs.vit.TimmViT`. Ce module ne definit aucun reseau : il telecharge
les poids (`download`, appele par `lo data prepare`) et les charge en lecture seule
dans un TimmViT dont il fixe les reglages imposes par ces poids (ViT-S/8, pas de
projection : `embed` = token CLS, 384).

Poids pre-entraines en 224px : timm reechantillonne les position embeddings (bicubique)
vers la grille de `img_size`.
"""

from pathlib import Path

import torch.nn as nn
from safetensors.torch import load_file
from timm.models.vision_transformer import checkpoint_filter_fn

from latentopt.data.paths import models_root
from latentopt.models.backbones.archs.vit import TimmViT

NAME = "dino_vits8"
REPO, FILE = "timm/vit_small_patch8_224.dino", "model.safetensors"


def local_path() -> Path:
    return models_root() / NAME / FILE


def download() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, FILE, local_dir=models_root() / NAME))


def load_into(model: TimmViT, path: Path) -> None:
    """Charge strictement le safetensors timm (cles remappees par `checkpoint_filter_fn`)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lance `uv run lo data prepare {NAME}`")
    state_dict = checkpoint_filter_fn(load_file(str(path)), model.vit)
    model.vit.load_state_dict(state_dict, strict=True)


def dino_vits8(
    img_size: int = 128,
    blocks: tuple[int, ...] = (2, 5, 8, 11),
    drop_path_rate: float = 0.0,
    grad_checkpointing: bool = False,
) -> nn.Module:
    model = TimmViT(
        "vit_small_patch8_224",
        img_size=img_size,
        embed_dim=None,
        blocks=blocks,
        drop_path_rate=drop_path_rate,
        grad_checkpointing=grad_checkpointing,
    )
    load_into(model, local_path())
    return model
