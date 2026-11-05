"""Imagenette 160px (fast.ai) : 10 classes faciles d'ImageNet, ~9,5k images de train.

Splits : train / val = holdout 95/5 du train officiel (figé) ; test = val officiel.
Pool de pré-entraînement de la small league : tout le train officiel, sans labels.
"""

from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import Imagenette

from latentopt.data import splits
from latentopt.data.datasets.base import Subset, require
from latentopt.data.paths import datasets_root

NAME = "imagenette"
NUM_CLASSES = 10
SIZE = "160px"
FOLDER = "imagenette2-160"


class Source:
    """Split officiel ("train" | "val") -> (image PIL RGB, label)."""

    def __init__(self, official_split: str, root: Path | None = None):
        root = root or datasets_root()
        require(root / FOLDER, NAME)
        self.ds = Imagenette(root, split=official_split, size=SIZE, download=False)

    def ids(self) -> list[str]:
        base = Path(self.ds._image_root)
        return [Path(path).relative_to(base).as_posix() for path, _ in self.ds._samples]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds[i]


def download(root: Path | None = None) -> None:
    root = root or datasets_root()
    if not (root / FOLDER).exists():
        Imagenette(root, split="train", size=SIZE, download=True)  # une archive, les deux splits


def freeze_splits(root: Path | None = None) -> Path:
    source = Source("train", root)
    return splits.save(
        NAME, splits.holdout(len(source)), source.ids(), pool="train", seed=0, frac=0.05
    )


def build(split: str, transform: Callable, root: Path | None = None) -> Subset:
    if split == "test":
        source = Source("val", root)
        return Subset(source, range(len(source)), transform)
    source = Source("train", root)
    return Subset(source, splits.indices(NAME, split, source.ids()), transform)


def pretraining_images(root: Path | None = None) -> Source:
    return Source("train", root)
