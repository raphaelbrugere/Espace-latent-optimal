"""Contrat Backbone / Head, repris de `projects/lejepa_arena/arena/contract.py`.

Backbone
    attributs  embed_dim: int
               feature_info: list[{"name": str, "channels": int, "stride": int}],
                             du stride le plus fin au plus grossier
    forward    x (B, 3, H, W), normalisé ImageNet, H et W multiples du plus grand stride
               -> {"embed": (B, embed_dim),
                   f["name"]: (B, f["channels"], H / f["stride"], W / f["stride"]) pour f}
    règles     <= BACKBONE_MAX_PARAMS paramètres, déterministe en eval(), reconstructible
               depuis sa config puis son state_dict

Head
    construite avec (feature_info, embed_dim, num_classes) + ses hyperparamètres
    attribut   in_keys: list[str], les entrées du dict du backbone qu'elle lit (cache de
               features : seules celles-ci sont calculées et stockées)
    forward    le dict du backbone -> sortie de la tâche (voir evaluation/tasks.py)

Le backbone est gelé pendant l'entraînement des têtes ; chaque tête enregistre le
`weights_hash` du backbone sur lequel elle a été entraînée.
"""

import hashlib

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
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.eval()


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


def save_head(head: nn.Module, path, backbone: nn.Module, seed: int, **extra) -> str:
    """Enregistre la tête avec la preuve du backbone gelé sur lequel elle a été entraînée.
    Renvoie l'empreinte du backbone."""
    import sys

    sha = weights_hash(backbone)
    torch.save(
        {"state_dict": head.state_dict(), "backbone_sha256": sha, "seed": seed, "argv": sys.argv}
        | extra,
        path,
    )
    return sha


def load_head(head: nn.Module, path, backbone: nn.Module) -> dict:
    """Charge les poids d'une tête et vérifie qu'ils ont été appris sur ce backbone.
    Renvoie le checkpoint (seed, métriques…) ; lève une erreur si le backbone diffère."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    if ckpt.get("backbone_sha256") != weights_hash(backbone):
        raise ValueError(f"{path} : tête entraînée sur un autre backbone que celui chargé")
    return ckpt


def make_head(make, backbone: nn.Module, num_classes: int) -> nn.Module:
    """Construit une tête depuis sa config partielle (`_partial_: true`) : ne lui passe,
    parmi feature_info / embed_dim / num_classes, que ce que son constructeur demande."""
    import inspect

    available = {
        "feature_info": backbone.feature_info,
        "embed_dim": backbone.embed_dim,
        "num_classes": num_classes,
    }
    params = inspect.signature(make).parameters
    return make(**{k: v for k, v in available.items() if k in params})
