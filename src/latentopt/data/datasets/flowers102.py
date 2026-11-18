"""Oxford Flowers-102 : classification fine-grained, 102 classes, splits officiels."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torchvision.datasets import Flowers102

from latentopt.data import splits
from latentopt.data.datasets.base import Subset, require, reset
from latentopt.data.paths import datasets_root

NAME = "flowers102"
NUM_CLASSES = 102
FOLDER = "flowers-102"
# CRITIQUE : nom court du dataset dans les noms de runs et de depots Hub. NE JAMAIS LE
# MODIFIER : il renommerait tous les runs existants. Il doit aussi etre DIFFERENT de celui
# de tous les autres datasets (verifie par tests/data/test_abbreviations.py).
ABBREV = "fl102"


class Source:
    """Split officiel ("train" | "val" | "test") -> (image PIL RGB, label)."""

    def __init__(self, split: str, root: Path | None = None) -> None:
        root = root or datasets_root()
        require(root / FOLDER, NAME)
        self.ds = Flowers102(root, split=split, download=False)

    def ids(self) -> list[str]:
        return [path.name for path in self.ds._image_files]  # attribut prive de torchvision

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        img, label = self.ds[i]
        return img.convert("RGB"), label


def present(root: Path | None = None) -> bool:
    """Le dataset est-il deja telecharge et intact dans `root` (verification de torchvision) ?"""
    try:
        Flowers102(root or datasets_root(), split="train", download=False)
    except RuntimeError:  # " Dataset not found or corrupted "
        return False
    return True


def download(root: Path | None = None, force: bool = False) -> None:
    root = root or datasets_root()
    if force:
        reset(root / FOLDER)
    Flowers102(root, split="train", download=True)  # tous les fichiers


def freeze_splits(root: Path | None = None, force: bool = False) -> Path:
    """Fige les trois splits officiels, sauf si deja fait : aucun calcul, seulement leurs ids."""

    def compute() -> splits.Computed:
        parts, ids = {}, []
        for split in ("train", "val", "test"):
            split_ids = Source(split, root).ids()
            parts[split] = list(range(len(ids), len(ids) + len(split_ids)))
            ids += split_ids
        return parts, ids, {"official": True}

    return splits.freeze(NAME, compute, force)


def build(split: str, transform: Callable, root: Path | None = None) -> Subset:
    freeze_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
    source = Source(split, root)
    return Subset(source, splits.indices(NAME, split, source.ids()), transform)


def pretraining_images(root: Path | None = None) -> Source:
    """Pool de pre-entrainement (configs/data/flowers102.yaml) : train officiel seul (1 020
    images). Le val sert a la sonde en ligne, le test n'y entre jamais. Hors protocole,
    dont le pool est Imagenette."""
    return Source("train", root)
