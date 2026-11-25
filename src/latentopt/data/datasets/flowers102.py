"""Oxford Flowers-102 : classification fine-grained, 102 classes, splits officiels."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torchvision.datasets import Flowers102 as TvFlowers102

from latentopt.data import splits
from latentopt.data.datasets.base import ClsSource, DownloadableDataset, Subset, require, reset
from latentopt.data.paths import datasets_root

FOLDER = "flowers-102"


class Flowers102Source(ClsSource):
    """Split officiel ("train" | "val" | "test") -> (image PIL RGB, label)."""

    def __init__(self, split: str, root: Path | None = None) -> None:
        root = root or datasets_root()
        require(root / FOLDER, Flowers102.NAME)
        self.split = split
        self.ds = TvFlowers102(root, split=split, download=False)

    def ids(self) -> list[str]:
        return [path.name for path in self.ds._image_files]  # attribut prive de torchvision

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        img, label = self.ds[i]
        return img.convert("RGB"), label


class Flowers102(DownloadableDataset):
    """Classification 102 classes (`cls`), splits officiels ; pool : train officiel."""

    NAME = "flowers102"
    ABBREV = "fl102"  # CRITIQUE : fige, voir DatasetSpec.ABBREV

    def present(self, root: Path | None = None) -> bool:
        """Telecharge et intact (verification de torchvision) ?"""
        try:
            TvFlowers102(root or datasets_root(), split="train", download=False)
        except RuntimeError:  # " Dataset not found or corrupted "
            return False
        return True

    def download(self, root: Path | None = None, force: bool = False) -> None:
        root = root or datasets_root()
        if force:
            reset(root / FOLDER)
        TvFlowers102(root, split="train", download=True)  # tous les fichiers

    def freeze_splits(self, root: Path | None = None, force: bool = False) -> list[Path]:
        """Fige les trois splits officiels, sauf si deja fait : aucun calcul, juste leurs ids."""

        def compute() -> splits.Computed:
            parts, ids = {}, []
            for split in ("train", "val", "test"):
                split_ids = Flowers102Source(split, root).ids()
                parts[split] = list(range(len(ids), len(ids) + len(split_ids)))
                ids += split_ids
            return splits.Computed(indices=parts, ids=ids, meta={"official": True})

        return [splits.freeze(self.NAME, compute, force)]

    def tasks(self) -> dict[str, Callable]:
        return {"cls": self.build}

    def build(self, split: str, transform: Callable, root: Path | None = None) -> Subset:
        self.freeze_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
        source = Flowers102Source(split, root)
        return Subset(source, splits.indices(self.NAME, split, source.ids()), transform)

    def pretraining_images(self, root: Path | None = None) -> Flowers102Source:
        """Pool de pre-entrainement (configs/data/flowers102.yaml) : train officiel seul (1 020
        images). Le val sert a la sonde en ligne, le test n'y entre jamais. Hors protocole,
        dont le pool est Imagenette."""
        return Flowers102Source("train", root)


DATASET = Flowers102()
