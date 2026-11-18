"""Brique commune des datasets : `Subset` (source + indices + transform) et `require`.

Une source est tout objet indexable renvoyant (image PIL, cible brute) ; le transform
reçoit les deux ensemble pour garder image et cible alignées (flip, crop).
"""

import importlib
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class Subset(Dataset):
    """Vue `indices` d'une source (image PIL, cible brute), avec transform conjoint."""

    def __init__(self, source: Sequence, indices: Sequence[int], transform: Callable) -> None:
        self.source, self.indices, self.transform = source, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Any:
        img, target = self.source[self.indices[i]]
        return self.transform(img, target)


def raw(img: Any, target: Any) -> tuple[Any, Any]:
    """Transform identité de `Subset` : (image, cible) inchangées. Fonction de module, donc
    picklable par les workers d'un DataLoader, contrairement à une lambda."""
    return img, target


def reset(*paths: Path) -> None:
    """Supprime des téléchargements existants (dossier ou fichier) : `download(force=True)`."""
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def abbrev(target: str) -> str:
    """Nom court (`ABBREV`) du module d'un dataset, d'après une fonction de ce module, ex.
    `latentopt.data.datasets.imagenette.pretraining_images` -> `imnt160`. Le nom court vit
    dans le code du dataset, jamais dans les configs."""
    module = importlib.import_module(target.rpartition(".")[0])
    if not hasattr(module, "ABBREV"):
        raise AttributeError(f"{module.__name__} : ABBREV manquant (nom court du dataset)")
    return module.ABBREV


def require(path: Path, name: str) -> None:
    """Échoue tout de suite, avec la commande à lancer, si les données manquent."""
    if not path.exists():
        raise FileNotFoundError(f"{name} absent de {path} : lance `uv run lo data prepare {name}`")
