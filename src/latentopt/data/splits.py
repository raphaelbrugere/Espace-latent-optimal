"""Splits figés : calculés une fois, commités dans `splits/<dataset>.json`.

Les splits dérivés (holdout, stratifié) sont stockés par identifiant stable (chemin
relatif, nom de fichier, ligne d'une révision HF épinglée), jamais par indice : un
changement d'ordre de listage côté torchvision ou HF ne peut pas les déplacer. Les
algorithmes et graines sont ceux de `projects/lejepa_arena`, donc les ensembles
obtenus sont identiques aux siens.

Les splits officiels (Flowers-102, tests d'Imagenette et de Pets) ne sont pas figés ici.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SPLITS_DIR = Path(__file__).parent / "splits"


def holdout(n: int, frac: float = 0.05, seed: int = 0) -> dict[str, list[int]]:
    """Permutation fixe ; les `frac` premiers en val, le reste en train."""
    perm = np.random.RandomState(seed).permutation(n)
    k = max(1, int(n * frac))
    return {"train": perm[k:].tolist(), "val": perm[:k].tolist()}


def stratified(
    labels: Sequence[int], fracs: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> dict[str, list[int]]:
    """Partition train / val / test par classe, permutation fixe dans chaque classe."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        a, b = int(len(idx) * fracs[0]), int(len(idx) * (fracs[0] + fracs[1]))
        out["train"] += idx[:a].tolist()
        out["val"] += idx[a:b].tolist()
        out["test"] += idx[b:].tolist()
    return {k: sorted(v) for k, v in out.items()}


def save(name: str, splits: dict[str, list[int]], ids: Sequence[str], **meta) -> Path:
    """Convertit des indices en identifiants et écrit le JSON (un id par ligne)."""
    named = {split: sorted(str(ids[i]) for i in idx) for split, idx in splits.items()}
    path = SPLITS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": name, **meta, "splits": named}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def load(name: str) -> dict:
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent : lance `python tools/prepare_data.py --freeze-splits`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def indices(name: str, split: str, ids: Sequence[str]) -> list[int]:
    """Indices, dans la source chargée, des identifiants du split figé."""
    position = {str(i): k for k, i in enumerate(ids)}
    wanted = load(name)["splits"][split]
    missing = [i for i in wanted if i not in position]
    if missing:
        raise KeyError(
            f"{name}/{split} : {len(missing)} identifiants introuvables, ex. {missing[:3]}"
        )
    return [position[i] for i in wanted]
