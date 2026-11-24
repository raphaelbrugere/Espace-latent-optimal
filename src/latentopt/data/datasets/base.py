"""Brique commune des datasets : sources (`Source`), classes meres (`DatasetSpec`,
`DownloadableDataset`), `Subset` (source + indices + transform) et `require`.

Trois niveaux : un `DatasetSpec` (sans etat, une instance `DATASET` par module, listee dans
`registry.py`) fabrique des `Source` (donnees brutes), que `Subset` filtre par split et
transforme pour le DataLoader. Le transform recoit image et cible ensemble pour les garder
alignees (flip, crop).
"""

import importlib
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar

from PIL import Image
from torch.utils.data import Dataset

T = TypeVar("T", covariant=True)


class Source(Protocol[T]):
    """Donnees brutes : `source[i]` -> (image PIL RGB, cible brute), et l'identifiant stable
    de chaque element, cle des splits figes (splits.py). Les sources en heritent
    explicitement (`ClsSource`...) pour annoncer le format de leur cible et recevoir
    `__repr__`."""

    def __len__(self) -> int: ...

    def __getitem__(self, i: int) -> tuple[Image.Image, T]: ...

    def ids(self) -> list[str]: ...

    def __repr__(self) -> str:
        split = getattr(self, "split", None)
        where = f"split={split!r}, " if split else ""
        return f"{type(self).__name__}({where}{len(self)} images)"


ClsSource = Source[int]  # label de classe
SegSource = Source[Image.Image]  # masque "L" de la taille de l'image, valeurs = classes
DetSource = Source[dict[str, Any]]  # {"box": tenseur xyxy normalise dans [0, 1], "label": int}


class DatasetSpec(ABC):
    """Un dataset : son nom, son nom court, son pool de pre-entrainement et ses taches.

    Une sous-classe incomplete (methode abstraite ou attribut manquant) ne peut pas etre
    instanciee : l'oubli se voit des l'import de son module."""

    NAME: ClassVar[str]  # nom dans `lo data prepare` et dans les configs
    # CRITIQUE : nom court dans les noms de runs et de depots Hub. NE JAMAIS LE MODIFIER : il
    # renommerait tous les runs existants. Unique (tests/data/test_abbreviations.py).
    ABBREV: ClassVar[str]

    def __init__(self) -> None:
        missing = [a for a in ("NAME", "ABBREV") if not hasattr(self, a)]
        if missing:
            raise TypeError(f"{type(self).__name__} : {', '.join(missing)} manquant(s)")

    def __repr__(self) -> str:
        kinds = ", ".join(self.tasks())
        return f"{type(self).__name__}(name={self.NAME!r}, abbrev={self.ABBREV!r}, tasks=[{kinds}])"

    @abstractmethod
    def pretraining_images(self, *args: Any, **kwargs: Any) -> ClsSource:
        """Pool de pre-entrainement (configs/data/<NAME>.yaml). Le label ne sert qu'a la sonde
        en ligne."""

    @abstractmethod
    def tasks(self) -> dict[str, Callable]:
        """Constructeurs de splits par type de tache ("cls", "seg", "det") :
        (split, transform) -> Subset. Le protocole (transforms, metriques) est dans
        evaluation/tasks.py."""


class DownloadableDataset(DatasetSpec):
    """Dataset telecharge sur disque : ce que `lo data prepare` appelle, dans cet ordre."""

    @abstractmethod
    def present(self, root: Path | None = None) -> bool:
        """Le dataset est-il deja telecharge dans `root` ?"""

    @abstractmethod
    def download(self, root: Path | None = None, force: bool = False) -> None:
        """Telecharge ; `force` supprime d'abord l'ancien telechargement."""

    @abstractmethod
    def freeze_splits(self, root: Path | None = None, force: bool = False) -> list[Path]:
        """Fige les splits (splits.freeze), sauf si deja fait ; renvoie les JSON."""


class Subset(Dataset):
    """Vue `indices` d'une source (image PIL, cible brute), avec transform conjoint."""

    def __init__(self, source: Source, indices: Sequence[int], transform: Callable) -> None:
        self.source, self.indices, self.transform = source, list(indices), transform

    def __repr__(self) -> str:
        t = getattr(self.transform, "__name__", type(self.transform).__name__)
        return f"Subset({self.source!r}, {len(self)} images, transform={t})"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Any:
        img, target = self.source[self.indices[i]]
        return self.transform(img, target)


def reset(*paths: Path) -> None:
    """Supprime des telechargements existants (dossier ou fichier) : `download(force=True)`."""
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def abbrev(target: str) -> str:
    """Nom court (`ABBREV`) d'un dataset d'apres le `_target_` d'une de ses methodes, ex.
    `latentopt.data.datasets.imagenette.DATASET.pretraining_images` -> `imnt160`. Le nom
    court vit dans le code du dataset, jamais dans les configs."""
    module, _, attr = target.rpartition(".")[0].rpartition(".")
    dataset = getattr(importlib.import_module(module), attr, None)
    if not isinstance(dataset, DatasetSpec):
        raise TypeError(
            f"{target} : pas une methode d'un DatasetSpec (`<module>.DATASET.<methode>`)"
        )
    return dataset.ABBREV


def require(path: Path, name: str) -> None:
    """Echoue tout de suite, avec la commande a lancer, si les donnees manquent."""
    if not path.exists():
        raise FileNotFoundError(f"{name} absent de {path} : lance `uv run lo data prepare {name}`")
