"""Imagenette 160px (fast.ai) : 10 classes faciles d'ImageNet, ~9,5k images de train.

Splits : train / val = holdout 95/5 du train officiel (fige) ; test = val officiel.
Pool de pre-entrainement du protocole : tout le train officiel, sans labels.
"""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torchvision.datasets import Imagenette

from latentopt.data import splits
from latentopt.data.datasets.base import Subset, require, reset
from latentopt.data.paths import datasets_root

NAME = "imagenette"
NUM_CLASSES = 10
SIZE = "160px"
FOLDER = "imagenette2-160"
# CRITIQUE : nom court du dataset dans les noms de runs et de depots Hub. NE JAMAIS LE
# MODIFIER : il renommerait tous les runs existants. Il doit aussi etre DIFFERENT de celui
# de tous les autres datasets (verifie par tests/data/test_abbreviations.py).
# Il suit la variante chargee (SIZE) : Imagenette 320px ne doit pas porter ce nom.
ABBREV = f"imnt{SIZE.removesuffix('px')}"


class Source:
    """Split officiel ("train" | "val") -> (image PIL RGB, label)."""

    def __init__(self, official_split: str, root: Path | None = None) -> None:
        root = root or datasets_root()
        require(root / FOLDER, NAME)
        self.ds = Imagenette(root, split=official_split, size=SIZE, download=False)

    def ids(self) -> list[str]:
        # Attributs prives de torchvision : seul moyen d'obtenir les chemins sans relire le disque.
        base = Path(self.ds._image_root)
        return [Path(path).relative_to(base).as_posix() for path, _ in self.ds._samples]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        return self.ds[i]


def present(root: Path | None = None) -> bool:
    """Le dataset est-il deja telecharge dans `root` ?"""
    return ((root or datasets_root()) / FOLDER).exists()


def download(root: Path | None = None, force: bool = False) -> None:
    root = root or datasets_root()
    if force:
        reset(root / FOLDER, root / f"{FOLDER}.tgz")  # l'archive aussi, sinon elle est reutilisee
    if not (root / FOLDER).exists():
        Imagenette(root, split="train", size=SIZE, download=True)  # une archive, les deux splits


def freeze_splits(root: Path | None = None, force: bool = False) -> Path:
    """Fige train / val (holdout 95/5 du train) et test (val officiel), sauf si deja fait."""

    def compute() -> splits.Computed:
        train, test = Source("train", root), Source("val", root)
        n = len(train)
        parts = splits.holdout(n)
        parts["test"] = list(range(n, n + len(test)))  # ids du test a la suite de ceux du train
        meta = {"pool": "train", "test": "val officiel", "seed": 0, "frac": 0.05}
        return parts, train.ids() + test.ids(), meta

    return splits.freeze(NAME, compute, force)


def build(split: str, transform: Callable, root: Path | None = None) -> Subset:
    freeze_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
    source = Source("val" if split == "test" else "train", root)
    return Subset(source, splits.indices(NAME, split, source.ids()), transform)


def pretraining_images(root: Path | None = None) -> Source:
    return Source("train", root)
