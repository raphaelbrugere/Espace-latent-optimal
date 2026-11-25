"""Backbone issu d'un de nos runs de pre-entrainement : results/<run>/backbone/ (format de
latentopt/run.py, emplacement de latentopt/hub.py).

L'architecture est relue dans le config.yaml du backbone, jamais redemandee : impossible de
charger les poids dans une architecture differente. Les poids (backbone.pt) sont verifies
contre l'empreinte enregistree dans results.json.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from latentopt import hub
from latentopt.models.contract import weights_hash


def resolve_run(run: str | Path) -> Path:
    """Dossier backbone/ d'un run designe par son chemin (results/<run>, ou un dossier qu'il
    contient) ou son nom : absent en local, le run est telecharge depuis le Hub."""
    return hub.run_root(hub.locate(run)) / hub.BACKBONE


def from_run(run: str) -> nn.Module:
    """Reconstruit le backbone d'un run termine (config.yaml, backbone.pt, results.json requis).

    Leve FileNotFoundError si un fichier manque, ValueError si les poids ne correspondent
    pas a l'empreinte de results.json.
    """
    run_dir = resolve_run(run)
    for name in ("config.yaml", "backbone.pt", "results.json"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir / name} absent : ce n'est pas un run termine")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    backbone = instantiate(cfg.backbone)  # poids externes eventuels ecrases juste apres
    backbone.load_state_dict(torch.load(run_dir / "backbone.pt", map_location="cpu"))

    expected = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["backbone_sha256"]
    if weights_hash(backbone) != expected:
        raise ValueError(f"{run_dir}/backbone.pt ne correspond pas a l'empreinte de results.json")
    return backbone
