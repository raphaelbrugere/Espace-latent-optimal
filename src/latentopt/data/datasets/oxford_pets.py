"""Oxford-IIIT Pet : une source, deux taches.

Segmentation : trimap {1, 2, 3} ramene a {0, 1, 2} (animal, fond, contour).
    train / val = holdout 95/5 de trainval (fige) ; test = test officiel.
Detection : boite de la tete et espece (0 chat, 1 chien), une instance par image.
    Les boites n'existent que pour une partie de trainval (annotations/xmls) :
    stratifie 80/10/10 par espece sur ce sous-ensemble (fige).
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import OxfordIIITPet as TvOxfordIIITPet

from latentopt.data import splits
from latentopt.data.datasets.base import (
    ClsSource,
    DetSource,
    DownloadableDataset,
    SegSource,
    Subset,
    require,
    reset,
)
from latentopt.data.paths import datasets_root

SEG_NAME, DET_NAME = "oxford_pets_seg", "oxford_pets_det"
FOLDER = "oxford-iiit-pet"


class PetsSegSource(SegSource):
    """Split officiel ("trainval" | "test") -> (image PIL RGB, masque PIL dans {0, 1, 2})."""

    def __init__(self, official_split: str, root: Path | None = None) -> None:
        root = root or datasets_root()
        require(root / FOLDER, OxfordPets.NAME)
        self.split = official_split
        self.ds = TvOxfordIIITPet(
            root, split=official_split, target_types="segmentation", download=False
        )

    def ids(self) -> list[str]:
        return [path.stem for path in self.ds._images]  # attribut prive de torchvision

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, Image.Image]:
        img, trimap = self.ds[i]
        # trimap {1, 2, 3} -> {0, 1, 2}
        mask = Image.fromarray((np.asarray(trimap) - 1).astype(np.uint8))
        return img.convert("RGB"), mask


def trainval_rows(base: Path) -> list[tuple[str, int]]:
    """(nom, espece : 0 chat, 1 chien) de annotations/trainval.txt, dans l'ordre du fichier."""
    rows = []
    for line in (base / "annotations" / "trainval.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, _, species, _ = line.split()
            rows.append((name, int(species) - 1))
    return rows


class PetsDetSource(DetSource):
    """Images de trainval annotees -> (image PIL RGB, {"box": xyxy dans [0, 1], "label"})."""

    def __init__(self, root: Path | None = None) -> None:
        base = (root or datasets_root()) / FOLDER
        require(base, OxfordPets.NAME)
        self.images, xmls = base / "images", base / "annotations" / "xmls"
        self.items = [
            (n, s, read_box(xmls / f"{n}.xml"))
            for n, s in trainval_rows(base)
            if (xmls / f"{n}.xml").exists()
        ]

    def ids(self) -> list[str]:
        return [name for name, _, _ in self.items]

    def labels(self) -> list[int]:
        return [label for _, label, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Image.Image, dict[str, torch.Tensor | int]]:
        name, label, box = self.items[i]
        img = Image.open(self.images / f"{name}.jpg").convert("RGB")
        return img, {"box": torch.tensor(box).clamp(0, 1), "label": label}


def read_box(path: Path) -> list[float]:
    """Boite VOC en pixels -> xyxy normalise par la taille de l'image."""
    root = ET.parse(path).getroot()
    w, h = float(root.find("size/width").text), float(root.find("size/height").text)
    b = root.find("object/bndbox")
    x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
    return [x1 / w, y1 / h, x2 / w, y2 / h]


class PetsPool(ClsSource):
    """Images de trainval hors test de la detection -> (image PIL RGB, espece : 0 chat, 1 chien).

    Le test de `pets_det` est tire de trainval (stratifie) : sans cette exclusion, le
    pre-entrainement verrait des images de test. Le test de la segmentation est le test
    officiel, hors de trainval."""

    def __init__(self, root: Path | None = None) -> None:
        base = (root or datasets_root()) / FOLDER
        require(base, OxfordPets.NAME)
        DATASET.freeze_det_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
        excluded = set(splits.load(DET_NAME)["splits"]["test"])
        self.images = base / "images"
        self.items = [(n, s) for n, s in trainval_rows(base) if n not in excluded]

    def ids(self) -> list[str]:
        return [name for name, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        name, label = self.items[i]
        return Image.open(self.images / f"{name}.jpg").convert("RGB"), label


class OxfordPets(DownloadableDataset):
    """Segmentation (`seg`) et detection (`det`) ; pool : trainval sans le test de `det`."""

    NAME = "oxford_pets"
    ABBREV = "pets"  # CRITIQUE : fige, voir DatasetSpec.ABBREV

    def present(self, root: Path | None = None) -> bool:
        """Images et annotations des deux splits sont-elles deja telechargees dans `root` ?"""
        try:
            for split in ("trainval", "test"):
                TvOxfordIIITPet(root or datasets_root(), split=split, download=False)
        except RuntimeError:  # " Dataset not found "
            return False
        return True

    def download(self, root: Path | None = None, force: bool = False) -> None:
        root = root or datasets_root()
        if force:
            reset(root / FOLDER)
        for split in ("trainval", "test"):  # images + annotations, xmls compris
            TvOxfordIIITPet(root, split=split, download=True)

    def freeze_seg_splits(self, root: Path | None = None, force: bool = False) -> Path:
        """Fige la segmentation (holdout de trainval + test officiel), sauf si deja fait."""

        def compute() -> splits.Computed:
            trainval, test = PetsSegSource("trainval", root), PetsSegSource("test", root)
            n = len(trainval)
            parts = splits.holdout(n)
            parts["test"] = list(
                range(n, n + len(test))
            )  # ids du test a la suite de ceux de trainval
            meta = {"pool": "trainval", "test": "test officiel", "seed": 0, "frac": 0.05}
            return parts, trainval.ids() + test.ids(), meta

        return splits.freeze(SEG_NAME, compute, force)

    def freeze_det_splits(self, root: Path | None = None, force: bool = False) -> Path:
        """Fige la detection (stratifie 80/10/10 par espece), sauf si deja fait."""

        def compute() -> splits.Computed:
            source = PetsDetSource(root)
            meta = {"pool": "trainval+xmls", "seed": 0}
            return splits.stratified(source.labels()), source.ids(), meta

        return splits.freeze(DET_NAME, compute, force)

    def freeze_splits(self, root: Path | None = None, force: bool = False) -> list[Path]:
        """Fige les splits de segmentation et de detection, sauf si deja fait."""
        return [self.freeze_seg_splits(root, force), self.freeze_det_splits(root, force)]

    def build_seg(self, split: str, transform: Callable, root: Path | None = None) -> Subset:
        self.freeze_seg_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
        source = PetsSegSource("test" if split == "test" else "trainval", root)
        return Subset(source, splits.indices(SEG_NAME, split, source.ids()), transform)

    def build_det(self, split: str, transform: Callable, root: Path | None = None) -> Subset:
        self.freeze_det_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
        source = PetsDetSource(root)
        return Subset(source, splits.indices(DET_NAME, split, source.ids()), transform)

    def tasks(self) -> dict[str, Callable]:
        return {"seg": self.build_seg, "det": self.build_det}

    def pretraining_images(self, root: Path | None = None) -> PetsPool:
        """Pool de pre-entrainement (configs/data/oxford_pets.yaml) : trainval sans le test de
        la detection. Pas de sonde en ligne : Pets n'a pas de tache de classification. Hors
        protocole, dont le pool est Imagenette."""
        return PetsPool(root)


DATASET = OxfordPets()
