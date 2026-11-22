"""Configs nommees des necks et des tetes, et leur construction sur un backbone.

    configs/neck/<nom>.yaml   config partielle d'un neck (models/necks/)
    configs/head/<nom>.yaml   config partielle d'une tete (models/heads/)

Leurs noms sont les parties <neck> et <tete> du dossier d'entrainement
(downstream/<tache>-<neck>-<tete>-<date>/, voir docs/HUB.md). Pour essayer
une variante : `neck.channels=128` en ligne de commande (le dossier garde le nom du neck) ;
une variante retenue merite sa propre config, donc son propre nom.
"""

from pathlib import Path

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from latentopt.data.paths import PROJECT_ROOT
from latentopt.models.contract import Downstream, check_pair, make_head, make_neck

CONFIGS = PROJECT_ROOT / "configs"
GROUPS = ("neck", "head")


def path_of(group: str, name: str) -> Path:
    return CONFIGS / group / f"{name}.yaml"


def load(group: str, name: str) -> DictConfig:
    """Config `name` du groupe `neck` ou `head` ; FileNotFoundError si elle n'existe pas."""
    if group not in GROUPS:
        raise ValueError(f"groupe {group!r} : attendu l'un de {GROUPS}")
    path = path_of(group, name)
    if not path.exists():
        known = sorted(p.stem for p in (CONFIGS / group).glob("*.yaml"))
        raise FileNotFoundError(f"{group} inconnu '{name}', disponibles : {known}")
    return OmegaConf.load(path)


def build_neck(cfg: DictConfig, backbone: nn.Module) -> nn.Module:
    """Neck de cette config, sur ce backbone."""
    return make_neck(instantiate(cfg), backbone)


def build_head(cfg: DictConfig, neck: nn.Module, num_classes: int) -> nn.Module:
    """Tete de cette config, sur la sortie de ce neck ; ValueError si les formes different."""
    head = make_head(instantiate(cfg), neck, num_classes)
    check_pair(neck, head)
    return head


def build(neck: DictConfig, head: DictConfig, backbone: nn.Module, num_classes: int) -> Downstream:
    """Neck puis tete, construits dans cet ordre (il fixe l'init sous une graine donnee).
    Utilise par lo train et lo eval ; un neck repris se charge ensuite (load_neck)."""
    built = build_neck(neck, backbone)
    return Downstream(built, build_head(head, built, num_classes))
