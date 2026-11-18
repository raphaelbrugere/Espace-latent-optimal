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
from latentopt.data.datasets.base import Subset, require, reset
from latentopt.data.paths import datasets_root

SEG_NAME, DET_NAME = "oxford_pets_seg", "oxford_pets_det"
SEG_CLASSES, DET_CLASSES = 3, 2
FOLDER = "oxford-iiit-pet"
# CRITIQUE : nom court du dataset dans les noms de runs et de dépôts Hub. NE JAMAIS LE
# MODIFIER : il renommerait tous les runs existants. Il doit aussi être DIFFÉRENT de celui
# de tous les autres datasets (vérifié par tests/data/test_abbreviations.py).
ABBREV = "pets"


class SegSource:
    """Split officiel ("trainval" | "test") -> (image PIL RGB, masque PIL dans {0, 1, 2})."""

    def __init__(self, official_split: str, root: Path | None = None) -> None:
        root = root or datasets_root()
        require(root / FOLDER, "oxford_pets")
        self.ds = OxfordIIITPet(
            root, split=official_split, target_types="segmentation", download=False
        )

    def ids(self) -> list[str]:
        return [path.stem for path in self.ds._images]  # attribut privé de torchvision

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, Image.Image]:
        img, trimap = self.ds[i]
        # trimap {1, 2, 3} -> {0, 1, 2}
        mask = Image.fromarray((np.asarray(trimap) - 1).astype(np.uint8))
        return img.convert("RGB"), mask


class DetSource:
    """Images de trainval annotées -> (image PIL RGB, {"box": xyxy dans [0, 1], "label"})."""

    def __init__(self, root: Path | None = None) -> None:
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

    def __getitem__(self, i: int) -> tuple[Image.Image, dict[str, torch.Tensor | int]]:
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


def present(root: Path | None = None) -> bool:
    """Images et annotations des deux splits sont-elles déjà téléchargées dans `root` ?"""
    try:
        for split in ("trainval", "test"):
            OxfordIIITPet(root or datasets_root(), split=split, download=False)
    except RuntimeError:  # « Dataset not found »
        return False
    return True


def download(root: Path | None = None, force: bool = False) -> None:
    root = root or datasets_root()
    if force:
        reset(root / FOLDER)
    for split in ("trainval", "test"):  # images + annotations, xmls compris
        OxfordIIITPet(root, split=split, download=True)


def freeze_seg_splits(root: Path | None = None, force: bool = False) -> Path:
    """Fige la segmentation (holdout de trainval + test officiel), sauf si déjà fait."""

    def compute() -> splits.Computed:
        trainval, test = SegSource("trainval", root), SegSource("test", root)
        n = len(trainval)
        parts = splits.holdout(n)
        parts["test"] = list(range(n, n + len(test)))  # ids du test à la suite de ceux de trainval
        meta = {"pool": "trainval", "test": "test officiel", "seed": 0, "frac": 0.05}
        return parts, trainval.ids() + test.ids(), meta

    return splits.freeze(SEG_NAME, compute, force)


def freeze_det_splits(root: Path | None = None, force: bool = False) -> Path:
    """Fige la détection (stratifié 80/10/10 par espèce), sauf si déjà fait."""

    def compute() -> splits.Computed:
        source = DetSource(root)
        meta = {"pool": "trainval+xmls", "seed": 0}
        return splits.stratified(source.labels()), source.ids(), meta

    return splits.freeze(DET_NAME, compute, force)


def freeze_splits(root: Path | None = None, force: bool = False) -> list[Path]:
    """Fige les splits de segmentation et de détection."""
    return [freeze_seg_splits(root, force), freeze_det_splits(root, force)]


def build_seg(split: str, transform: Callable, root: Path | None = None) -> Subset:
    freeze_seg_splits(root)  # crée le JSON s'il manque, sinon ne fait rien
    source = SegSource("test" if split == "test" else "trainval", root)
    return Subset(source, splits.indices(SEG_NAME, split, source.ids()), transform)


def build_det(split: str, transform: Callable, root: Path | None = None) -> Subset:
    freeze_det_splits(root)  # crée le JSON s'il manque, sinon ne fait rien
    source = DetSource(root)
    return Subset(source, splits.indices(DET_NAME, split, source.ids()), transform)


class PoolSource:
    """Images de trainval hors test de la détection -> (image PIL RGB, espèce : 0 chat, 1 chien).

    Le test de `pets_det` est tiré de trainval (stratifié) : sans cette exclusion, le
    pré-entraînement verrait des images de test. Le test de la segmentation est le test
    officiel, hors de trainval."""

    def __init__(self, root: Path | None = None) -> None:
        base = (root or datasets_root()) / FOLDER
        require(base, "oxford_pets")
        freeze_det_splits(root)  # crée le JSON s'il manque, sinon ne fait rien
        excluded = set(splits.load(DET_NAME)["splits"]["test"])
        self.images, self.items = base / "images", []
        for line in (base / "annotations" / "trainval.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, species, _ = line.split()
            if name not in excluded:
                self.items.append((name, int(species) - 1))

    def ids(self) -> list[str]:
        return [name for name, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        name, label = self.items[i]
        return Image.open(self.images / f"{name}.jpg").convert("RGB"), label


def pretraining_images(root: Path | None = None) -> PoolSource:
    """Pool de pré-entraînement (configs/data/oxford_pets.yaml) : trainval sans le test de la
    détection. Pas de sonde en ligne : Pets n'a pas de tâche de classification. Hors protocole,
    dont le pool est Imagenette."""
    return PoolSource(root)
