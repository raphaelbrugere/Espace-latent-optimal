"""Imagenette 160px (fast.ai) : 10 classes faciles d'ImageNet, ~9,5k images de train.

Splits : train / val = holdout 95/5 du train officiel (fige) ; test = val officiel.
Pool de pre-entrainement du protocole : tout le train officiel, sans labels.
"""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torchvision.datasets import Imagenette as TvImagenette

from latentopt.data import splits
from latentopt.data.datasets.base import ClsSource, DownloadableDataset, Subset, require, reset
from latentopt.data.paths import datasets_root

SIZE = "160px"
FOLDER = "imagenette2-160"


class ImagenetteSource(ClsSource):
    """Split officiel ("train" | "val") -> (image PIL RGB, label)."""

    def __init__(self, official_split: str, root: Path | None = None) -> None:
        root = root or datasets_root()
        require(root / FOLDER, Imagenette.NAME)
        self.split = official_split
        self.ds = TvImagenette(root, split=official_split, size=SIZE, download=False)

    def ids(self) -> list[str]:
        # Attributs prives de torchvision : seul moyen d'obtenir les chemins sans relire le disque.
        base = Path(self.ds._image_root)
        return [Path(path).relative_to(base).as_posix() for path, _ in self.ds._samples]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        return self.ds[i]


class Imagenette(DownloadableDataset):
    """Classification 10 classes (`cls`) ; pool du protocole : train officiel."""

    NAME = "imagenette"
    # CRITIQUE : fige, voir DatasetSpec.ABBREV. Il suit la variante chargee (SIZE) :
    # Imagenette 320px ne doit pas porter ce nom.
    ABBREV = f"imnt{SIZE.removesuffix('px')}"

    def present(self, root: Path | None = None) -> bool:
        return ((root or datasets_root()) / FOLDER).exists()

    def download(self, root: Path | None = None, force: bool = False) -> None:
        root = root or datasets_root()
        if force:
            reset(root / FOLDER, root / f"{FOLDER}.tgz")  # l'archive aussi, sinon reutilisee
        if not (root / FOLDER).exists():
            TvImagenette(root, split="train", size=SIZE, download=True)  # une archive, 2 splits

    def freeze_splits(self, root: Path | None = None, force: bool = False) -> list[Path]:
        """Fige train / val (holdout 95/5 du train) et test (val officiel), sauf si deja fait."""

        def compute() -> splits.Computed:
            train, test = ImagenetteSource("train", root), ImagenetteSource("val", root)
            n = len(train)
            parts = splits.holdout(n)
            parts["test"] = list(range(n, n + len(test)))  # ids du test a la suite du train
            meta = {"pool": "train", "test": "val officiel", "seed": 0, "frac": 0.05}
            return splits.Computed(indices=parts, ids=train.ids() + test.ids(), meta=meta)

        return [splits.freeze(self.NAME, compute, force)]

    def tasks(self) -> dict[str, Callable]:
        return {"cls": self.build}

    def build(self, split: str, transform: Callable, root: Path | None = None) -> Subset:
        self.freeze_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
        source = ImagenetteSource("val" if split == "test" else "train", root)
        return Subset(source, splits.indices(self.NAME, split, source.ids()), transform)

    def pretraining_images(self, root: Path | None = None) -> ImagenetteSource:
        return ImagenetteSource("train", root)


DATASET = Imagenette()
