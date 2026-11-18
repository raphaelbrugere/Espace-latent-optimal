"""Contrat Backbone / Neck / Head.

Backbone
    attributs  embed_dim: int
               feature_info: list[{"name": str, "channels": int, "stride": int}],
                             du stride le plus fin au plus grossier
    forward    x (B, 3, H, W), normalisé ImageNet, H et W multiples du plus grand stride
               -> {"embed": (B, embed_dim),
                   f["name"]: (B, f["channels"], H / f["stride"], W / f["stride"]) pour f}
    règles     <= BACKBONE_MAX_PARAMS paramètres, déterministe en eval(), reconstructible
               depuis sa config puis son state_dict

Neck
    construit avec (feature_info, embed_dim) + ses hyperparamètres
    attributs  in_keys: list[str], les entrées du dict du backbone qu'il lit (cache de
               features : seules celles-ci sont calculées et stockées)
               out_kind: "vector" (B, out_dim) ou "map" (B, out_dim, H, W) ; out_dim: int
    forward    le dict du backbone -> sa sortie

Head
    construite avec (in_dim, num_classes) + ses hyperparamètres
    attribut   in_kind: "vector" | "map", la forme de sortie de neck qu'elle accepte
    forward    la sortie du neck -> sortie de la tâche (voir evaluation/tasks.py)

Downstream(neck, head) est ce qu'on entraîne : mêmes `in_keys`
que son neck, forward = head(neck(feats)). Neck et tête sont décrits par des configs nommées
(configs/neck/, configs/head/, voir models/recipe.py) et sauvegardés séparément :

    neck.pt   poids du neck, empreinte du backbone, empreinte du neck, nombre de paramètres
    head.pt   poids de la tête seule, empreintes du backbone et du neck, nombre de paramètres

Le backbone est gelé pendant l'entraînement des têtes, et le neck aussi quand il est repris
d'un entraînement précédent : chaque fichier enregistre les empreintes (`weights_hash`) des
poids sur lesquels il a été appris.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

BACKBONE_MAX_PARAMS = 30_000_000


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def weights_hash(module: nn.Module) -> str:
    """SHA-256 des paramètres et buffers, dans l'ordre des clés : empreinte des poids."""
    h = hashlib.sha256()
    for k, t in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def freeze(backbone: nn.Module) -> nn.Module:
    """Gèle tous les paramètres (en place) et passe en eval() ; renvoie le même module."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.eval()


def check_pair(neck: nn.Module, head: nn.Module) -> None:
    """Échoue tôt, avec un message clair, si la sortie du neck n'est pas ce que la tête lit."""
    for module, attr in ((neck, "out_kind"), (neck, "out_dim"), (head, "in_kind")):
        if not hasattr(module, attr):
            raise AttributeError(f"{type(module).__name__} doit définir `{attr}`")
    if neck.out_kind != head.in_kind:
        raise ValueError(
            f"{type(neck).__name__} produit un '{neck.out_kind}', "
            f"{type(head).__name__} attend un '{head.in_kind}'"
        )


class Downstream(nn.Module):
    """Neck puis tête, entraînés ensemble sur un backbone gelé : l'unité que le pipeline
    budgète, optimise et sauvegarde (un seul `state_dict`, clés `neck.*` et `head.*`).

    Le neck est construit avant la tête par l'appelant : l'ordre des couches fixe l'init
    sous une graine donnée."""

    def __init__(self, neck: nn.Module, head: nn.Module) -> None:
        check_pair(neck, head)
        super().__init__()
        self.neck, self.head = neck, head
        self.in_keys = neck.in_keys  # entrées lues dans le dict du backbone

    def forward(self, feats: dict[str, torch.Tensor]) -> Any:
        return self.head(self.neck(feats))


@torch.no_grad()
def check_backbone(backbone: nn.Module, size: int = 64) -> None:
    """Échoue tôt, avec un message clair, si le backbone ne respecte pas le contrat."""
    for attr in ("feature_info", "embed_dim"):
        if not hasattr(backbone, attr):
            raise AttributeError(f"le backbone doit définir `{attr}` (voir models/contract.py)")
    strides = [f["stride"] for f in backbone.feature_info]
    if strides != sorted(strides):
        raise ValueError(
            f"feature_info doit aller du stride le plus fin au plus grossier : {strides}"
        )
    if count_params(backbone) > BACKBONE_MAX_PARAMS:
        raise ValueError(f"{count_params(backbone):,} paramètres > {BACKBONE_MAX_PARAMS:,}")

    was_training = backbone.training
    backbone.eval()
    device = next(backbone.parameters()).device
    out = backbone(torch.zeros(1, 3, size, size, device=device))
    backbone.train(was_training)

    if out.get("embed") is None or tuple(out["embed"].shape) != (1, backbone.embed_dim):
        raise ValueError(f"'embed' doit avoir la forme (B, {backbone.embed_dim})")
    for f in backbone.feature_info:
        t = out.get(f["name"])
        expected = (1, f["channels"], size // f["stride"], size // f["stride"])
        if t is None or tuple(t.shape) != expected:
            got = None if t is None else tuple(t.shape)
            raise ValueError(f"carte '{f['name']}' : forme attendue {expected}, obtenue {got}")


def _save(module: nn.Module, path: str | Path, seed: int, **fields: Any) -> None:
    import sys

    payload = {"state_dict": module.state_dict(), "params": count_params(module), "seed": seed}
    torch.save(payload | {"argv": sys.argv} | fields, path)


def save_neck(
    neck: nn.Module, path: str | Path, backbone: nn.Module, seed: int, **extra: Any
) -> str:
    """neck.pt : poids du neck, preuve du backbone gelé sur lequel il a été appris, et sa
    propre empreinte (vérifiée par les têtes qui le reprennent). Renvoie l'empreinte du neck."""
    sha = weights_hash(neck)
    _save(neck, path, seed, backbone_sha256=weights_hash(backbone), neck_sha256=sha, **extra)
    return sha


def save_head(
    head: nn.Module,
    path: str | Path,
    backbone: nn.Module,
    neck: nn.Module,
    seed: int,
    **extra: Any,
) -> str:
    """head.pt : poids de la tête seule, preuves du backbone et du neck sur lesquels elle a
    été entraînée. Renvoie l'empreinte du backbone."""
    sha = weights_hash(backbone)
    _save(head, path, seed, backbone_sha256=sha, neck_sha256=weights_hash(neck), **extra)
    return sha


def load_neck(neck: nn.Module, path: str | Path, backbone: nn.Module) -> dict[str, Any]:
    """Charge les poids d'un neck ; ValueError (poids chargés quand même) s'ils ne
    correspondent pas à leur empreinte ou ont été appris sur un autre backbone."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    neck.load_state_dict(ckpt["state_dict"])
    if ckpt.get("neck_sha256") != weights_hash(neck):
        raise ValueError(f"{path} : poids différents de l'empreinte enregistrée")
    if ckpt.get("backbone_sha256") != weights_hash(backbone):
        raise ValueError(f"{path} : neck entraîné sur un autre backbone que celui chargé")
    return ckpt


def load_head(
    head: nn.Module, path: str | Path, backbone: nn.Module, neck: nn.Module
) -> dict[str, Any]:
    """Charge les poids d'une tête ; ValueError (poids chargés quand même) si elle a été
    entraînée sur un autre backbone ou un autre neck que ceux donnés."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    if ckpt.get("backbone_sha256") != weights_hash(backbone):
        raise ValueError(f"{path} : tête entraînée sur un autre backbone que celui chargé")
    if ckpt.get("neck_sha256") != weights_hash(neck):
        raise ValueError(f"{path} : tête entraînée sur un autre neck que celui chargé")
    return ckpt


def _call_with(make: Callable[..., nn.Module], available: dict[str, Any]) -> nn.Module:
    """Appelle `make` (classe ou config partielle) avec les seuls arguments qu'il demande."""
    import inspect

    params = inspect.signature(make).parameters
    return make(**{k: v for k, v in available.items() if k in params})


def make_neck(make: Callable[..., nn.Module], backbone: nn.Module) -> nn.Module:
    """Construit un neck depuis sa config partielle (`_partial_: true`) : ne lui passe, parmi
    feature_info / embed_dim, que ce que son constructeur demande."""
    return _call_with(
        make, {"feature_info": backbone.feature_info, "embed_dim": backbone.embed_dim}
    )


def make_head(make: Callable[..., nn.Module], neck: nn.Module, num_classes: int) -> nn.Module:
    """Construit une tête depuis sa config partielle : ne lui passe, parmi in_dim (sortie du
    neck) / num_classes, que ce que son constructeur demande."""
    return _call_with(make, {"in_dim": neck.out_dim, "num_classes": num_classes})


def make_downstream(
    neck: Callable[..., nn.Module],
    head: Callable[..., nn.Module],
    backbone: nn.Module,
    num_classes: int,
) -> Downstream:
    """Neck puis tête, construits dans cet ordre (il fixe l'init sous une graine donnée)."""
    built = make_neck(neck, backbone)
    return Downstream(built, make_head(head, built, num_classes))
