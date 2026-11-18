"""Splits figés : calculés une fois, commités dans `splits/<dataset>.json`.

Les splits dérivés (holdout, stratifié) sont stockés par identifiant stable (chemin
relatif, nom de fichier, ligne d'une révision HF épinglée), jamais par indice : un
changement d'ordre de listage côté torchvision ou HF ne peut pas les déplacer.

Les splits officiels (Flowers-102, tests d'Imagenette et de Pets) passent par le même
mécanisme : le JSON liste les identifiants que la définition officielle du dataset donne.

Chaque dataset définit son calcul dans son propre module (`freeze_splits`), appelé par
`build` : un JSON existant est lu tel quel, sinon il est créé. `freeze` ne le réécrit
qu'avec `force`.
"""

import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SPLITS_DIR = Path(__file__).parent / "splits"
log = logging.getLogger(__name__)

Computed = tuple[dict[str, list[int]], Sequence[str], dict[str, Any]]


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


def save(name: str, splits: dict[str, list[int]], ids: Sequence[str], **meta: Any) -> Path:
    """Convertit des indices en identifiants et écrit le JSON (un id par ligne)."""
    named = {split: sorted(str(ids[i]) for i in idx) for split, idx in splits.items()}
    path = SPLITS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": name, **meta, "splits": named}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomique : un job concurrent ne lit jamais un JSON à moitié écrit
    return path


def freeze(name: str, compute: Callable[[], Computed], force: bool = False) -> Path:
    """Écrit `splits/<name>.json` avec `compute()` = (indices par split, ids, méta).

    Un JSON existant est conservé tel quel, sans appeler `compute` (donc sans toucher au
    dataset) : le réécrire change les ensembles, et avec eux la comparabilité des scores.
    `force` ne sert qu'à le faire exprès."""
    path = SPLITS_DIR / f"{name}.json"
    existed = path.exists()
    if existed and not force:
        log.info("splits %s : déjà figés, conservés (%s)", name, path)
        return path
    parts, ids, meta = compute()
    path = save(name, parts, ids, **meta)
    sizes = ", ".join(f"{k} {len(v)}" for k, v in parts.items())
    log.info("splits %s : %s (%s) -> %s", name, "réécrits" if existed else "créés", sizes, path)
    return path


def load(name: str) -> dict[str, Any]:
    """JSON figé du dataset ; `FileNotFoundError` si les splits n'ont pas été gelés."""
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lance `uv run lo data prepare`")
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
