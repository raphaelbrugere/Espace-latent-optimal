"""Fixtures communes. Aucun test ne touche au reseau, aux vrais datasets ni a data/ :
DATA_ROOT et FEATURES_ROOT pointent vers un dossier temporaire pour chaque test, et
MLFLOW_TRACKING_URI est retire (le suivi MLflow va dans le dossier de resultats du test).

Deux backbones de test, tous deux au contrat de `models/contract.py` :
- `tiny_vit` : un vrai `TimmViT` (vit_tiny, patch 16), cartes iso-stride ;
- `pyramid` : un ConvNet jouet a deux strides (4, 8), pour les tetes multi-echelles.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from latentopt.models.contract import normalize_input, register_input_norm

TINY_VIT = {
    "_target_": "latentopt.models.backbones.archs.vit.TimmViT",
    "arch": "vit_tiny_patch16_224",
    "img_size": 32,
    "embed_dim": 24,
    "blocks": [0, 1],
    "drop_path_rate": 0.0,
}


class PyramidBackbone(nn.Module):
    """ConvNet jouet au contrat : cartes aux strides 4 et 8, `embed` poole puis projete."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.down = nn.Conv2d(8, 12, kernel_size=2, stride=2)
        self.proj = nn.Linear(12, embed_dim)
        register_input_norm(self)
        self.embed_dim = embed_dim
        self.feature_info = [
            {"name": "c4", "channels": 8, "stride": 4},
            {"name": "c8", "channels": 12, "stride": 8},
        ]

    def forward(self, x):
        c4 = F.relu(self.stem(normalize_input(self, x)))
        c8 = F.relu(self.down(c4))
        return {"c4": c4, "c8": c8, "embed": self.proj(c8.mean((2, 3)))}


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch) -> Path:
    """Chaque test a son propre DATA_ROOT : le cache de features n'ecrit jamais dans data/."""
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    monkeypatch.delenv("FEATURES_ROOT", raising=False)
    monkeypatch.delenv("MODELS_ROOT", raising=False)
    # jamais le serveur de l'utilisateur ; setenv puis delenv : la valeur est aussi retiree a la
    # fin du test si quelque chose l'a definie entre-temps (elle fuirait vers les tests suivants)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    # comme run.quiet_libraries, que les tests qui appellent mlflow_logger directement sautent
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return root


@pytest.fixture
def tiny_vit():
    from latentopt.models.backbones.archs.vit import TimmViT

    torch.manual_seed(0)
    kwargs = {k: v for k, v in TINY_VIT.items() if k != "_target_"}
    return TimmViT(**kwargs).eval()


@pytest.fixture
def pyramid():
    torch.manual_seed(0)
    return PyramidBackbone().eval()


@pytest.fixture
def tiny_vit_cfg() -> dict:
    """Config Hydra de `tiny_vit` (format de backbone de config.yaml d'un run)."""
    return dict(TINY_VIT, blocks=list(TINY_VIT["blocks"]))


# couples neck + tete de la baseline, par famille de tache (evaluation/tasks.py)
PAIRS = {
    "linear": ("embed_norm", "linear"),
    "fpn": ("fpn", "conv_seg"),
    "soft_argmax": ("conv_trunk", "soft_argmax"),
}


@pytest.fixture
def recipe_head():
    """Neck + tete d'un couple de PAIRS (configs/neck/, configs/head/), hyperparametres du
    neck surchargeables (comme `neck.<cle>=...` en ligne de commande)."""
    from omegaconf import OmegaConf

    from latentopt.models import recipe

    def build(name, backbone, num_classes, **neck):
        neck_name, head_name = PAIRS[name]
        cfg = OmegaConf.merge(recipe.load("neck", neck_name), neck)
        return recipe.build(cfg, recipe.load("head", head_name), backbone, num_classes)

    return build
