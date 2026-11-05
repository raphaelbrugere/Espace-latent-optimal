"""Oxford Flowers-102 : classification fine-grained, 102 classes, splits officiels."""

from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import Flowers102

from latentopt.data.datasets.base import Subset, require
from latentopt.data.paths import datasets_root

NAME = "flowers102"
NUM_CLASSES = 102
FOLDER = "flowers-102"


class Source:
    """Split officiel ("train" | "val" | "test") -> (image PIL RGB, label)."""

    def __init__(self, split: str, root: Path | None = None):
        root = root or datasets_root()
        require(root / FOLDER, NAME)
        self.ds = Flowers102(root, split=split, download=False)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        img, label = self.ds[i]
        return img.convert("RGB"), label


def download(root: Path | None = None) -> None:
    Flowers102(root or datasets_root(), split="train", download=True)  # tous les fichiers


def build(split: str, transform: Callable, root: Path | None = None) -> Subset:
    source = Source(split, root)
    return Subset(source, range(len(source)), transform)
