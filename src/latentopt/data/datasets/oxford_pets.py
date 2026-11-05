"""Oxford-IIIT Pet : une source, deux tâches.

Segmentation : trimap {1, 2, 3} ramené à {0, 1, 2} (animal, fond, contour).
    train / val = holdout 95/5 de trainval (figé) ; test = test officiel.
Détection : boîte de la tête et espèce (0 chat, 1 chien), une instance par image.
    Les boîtes n'existent que pour une partie de trainval (annotations/xmls) :
    stratifié 80/10/10 par espèce sur ce sous-ensemble (figé).
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from latentopt.data import splits
from latentopt.data.datasets.base import Subset, require
from latentopt.data.paths import datasets_root

SEG_NAME, DET_NAME = "oxford_pets_seg", "oxford_pets_det"
SEG_CLASSES, DET_CLASSES = 3, 2
FOLDER = "oxford-iiit-pet"


class SegSource:
    """Split officiel ("trainval" | "test") -> (image PIL RGB, masque PIL dans {0, 1, 2})."""

    def __init__(self, official_split: str, root: Path | None = None):
        root = root or datasets_root()
        require(root / FOLDER, "oxford_pets")
        self.ds = OxfordIIITPet(
            root, split=official_split, target_types="segmentation", download=False
        )

    def ids(self) -> list[str]:
        return [path.stem for path in self.ds._images]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        img, trimap = self.ds[i]
        mask = Image.fromarray((np.asarray(trimap) - 1).astype(np.uint8))
        return img.convert("RGB"), mask


class DetSource:
    """Images de trainval annotées -> (image PIL RGB, {"box": xyxy dans [0, 1], "label"})."""

    def __init__(self, root: Path | None = None):
        base = (root or datasets_root()) / FOLDER
        require(base, "oxford_pets")
        self.images, xmls = base / "images", base / "annotations" / "xmls"
        self.items = []
        for line in (base / "annotations" / "trainval.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, species, _ = line.split()
            if (xmls / f"{name}.xml").exists():
                self.items.append((name, int(species) - 1, read_box(xmls / f"{name}.xml")))

    def ids(self) -> list[str]:
        return [name for name, _, _ in self.items]

    def labels(self) -> list[int]:
        return [label for _, label, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        name, label, box = self.items[i]
        img = Image.open(self.images / f"{name}.jpg").convert("RGB")
        return img, {"box": torch.tensor(box).clamp(0, 1), "label": label}


def read_box(path: Path) -> list[float]:
    """Boîte VOC en pixels -> xyxy normalisé par la taille de l'image."""
    root = ET.parse(path).getroot()
    w, h = float(root.find("size/width").text), float(root.find("size/height").text)
    b = root.find("object/bndbox")
    x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def download(root: Path | None = None) -> None:
    root = root or datasets_root()
    for split in ("trainval", "test"):  # images + annotations, xmls compris
        OxfordIIITPet(root, split=split, download=True)


def freeze_splits(root: Path | None = None) -> list[Path]:
    seg, det = SegSource("trainval", root), DetSource(root)
    return [
        splits.save(
            SEG_NAME, splits.holdout(len(seg)), seg.ids(), pool="trainval", seed=0, frac=0.05
        ),
        splits.save(
            DET_NAME, splits.stratified(det.labels()), det.ids(), pool="trainval+xmls", seed=0
        ),
    ]


def build_seg(split: str, transform: Callable, root: Path | None = None) -> Subset:
    if split == "test":
        source = SegSource("test", root)
        return Subset(source, range(len(source)), transform)
    source = SegSource("trainval", root)
    return Subset(source, splits.indices(SEG_NAME, split, source.ids()), transform)


def build_det(split: str, transform: Callable, root: Path | None = None) -> Subset:
    source = DetSource(root)
    return Subset(source, splits.indices(DET_NAME, split, source.ids()), transform)
