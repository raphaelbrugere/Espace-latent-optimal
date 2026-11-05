"""Backbone issu d'un de nos runs de pré-entraînement (format de latentopt/run.py).

L'architecture est relue dans le config.yaml du run, jamais redemandée : impossible de
charger les poids dans une architecture différente. Les poids (backbone.pt) sont vérifiés
contre l'empreinte enregistrée dans results.json.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from latentopt.data.paths import PROJECT_ROOT
from latentopt.models.contract import weights_hash


def resolve_run(run: str | Path) -> Path:
    """Chemin absolu, ou relatif à la racine du projet."""
    path = Path(run)
    return path if path.is_absolute() else PROJECT_ROOT / path


def from_run(run: str) -> nn.Module:
    run_dir = resolve_run(run)
    for name in ("config.yaml", "backbone.pt", "results.json"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir / name} absent : ce n'est pas un run terminé")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    backbone = instantiate(cfg.backbone)  # poids externes éventuels écrasés juste après
    backbone.load_state_dict(torch.load(run_dir / "backbone.pt", map_location="cpu"))

    expected = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["backbone_sha256"]
    if weights_hash(backbone) != expected:
        raise ValueError(f"{run_dir}/backbone.pt ne correspond pas à l'empreinte de results.json")
    return backbone
