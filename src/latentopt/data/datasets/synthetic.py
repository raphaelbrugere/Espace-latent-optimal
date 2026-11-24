"""Datasets generes, pour le smoke run, la CI et les tests : aucun fichier, CPU, quelques
secondes. Memes formats de cible que les vrais datasets ; les scores n'ont aucun sens.

Images de segmentation et de detection non carrees : elles exercent la metrique a
resolution native et la normalisation des boites.
"""

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from latentopt.data.datasets.base import ClsSource, DatasetSpec, DetSource, SegSource, Subset

SIZES = {"train": 1000, "val": 150, "test": 300}
SEEDS = {"train": 0, "val": 2, "test": 1}


def noise(rng: np.random.RandomState, w: int, h: int) -> Image.Image:
    return Image.fromarray(np.clip(rng.randn(h, w, 3) * 20 + 128, 0, 255).astype(np.uint8))


class ShapesSource(ClsSource):
    """Classification : la classe fixe la couleur de fond et le nombre d'ellipses."""

    def __init__(self, n: int, num_classes: int, seed: int, size: int = 80) -> None:
        self.n, self.k, self.seed, self.size = n, num_classes, seed, size
        self.palette = np.random.RandomState(1234).randint(30, 225, size=(num_classes, 3))

    def __len__(self) -> int:
        return self.n

    def ids(self) -> list[str]:
        return [str(i) for i in range(self.n)]  # pas de split fige : la graine fait le split

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        # Graine unique par (split, index) : `seed` differe par split, i < 1_000_003.
        rng = np.random.RandomState(self.seed * 1_000_003 + i)
        label = i % self.k
        bg = np.clip(self.palette[label] + rng.randn(3) * 15, 0, 255)
        img = Image.new("RGB", (self.size, self.size), tuple(int(c) for c in bg))
        draw = ImageDraw.Draw(img)
        for _ in range(1 + label % 4):
            x, y = rng.randint(0, self.size - 12, size=2)
            w, h = rng.randint(8, self.size // 2, size=2)
            draw.ellipse([x, y, x + w, y + h], fill=tuple(int(c) for c in rng.randint(0, 255, 3)))
        return img, label


class BlobsSource(SegSource):
    """Segmentation a 3 classes : ellipses colorees sur bruit, masque aligne."""

    palette = [(40, 40, 40), (220, 60, 60), (60, 200, 90)]

    def __init__(self, n: int, seed: int, size: tuple[int, int] = (96, 72)) -> None:
        self.n, self.seed, self.size = n, seed, size

    def __len__(self) -> int:
        return self.n

    def ids(self) -> list[str]:
        return [str(i) for i in range(self.n)]  # pas de split fige : la graine fait le split

    def __getitem__(self, i: int) -> tuple[Image.Image, Image.Image]:
        rng = np.random.RandomState(self.seed * 1_000_003 + i)  # cf. ShapesSource
        w0, h0 = self.size
        img, mask = noise(rng, w0, h0), Image.new("L", (w0, h0), 0)
        for _ in range(3):
            c = int(rng.randint(1, 3))
            x, y = rng.randint(0, w0 - 16), rng.randint(0, h0 - 16)
            w, h = rng.randint(12, w0 // 2), rng.randint(12, h0 // 2)
            ImageDraw.Draw(img).ellipse([x, y, x + w, y + h], fill=self.palette[c])
            ImageDraw.Draw(mask).ellipse([x, y, x + w, y + h], fill=c)
        return img, mask


class HeadsSource(DetSource):
    """Detection : une ellipse par image, classe = famille de couleur."""

    palette = [(220, 80, 60), (60, 120, 220)]

    def __init__(self, n: int, seed: int, size: tuple[int, int] = (96, 72)) -> None:
        self.n, self.seed, self.size = n, seed, size

    def __len__(self) -> int:
        return self.n

    def ids(self) -> list[str]:
        return [str(i) for i in range(self.n)]  # pas de split fige : la graine fait le split

    def __getitem__(self, i: int) -> tuple[Image.Image, dict[str, torch.Tensor | int]]:
        rng = np.random.RandomState(self.seed * 1_000_003 + i)  # cf. ShapesSource
        w0, h0 = self.size
        img, label = noise(rng, w0, h0), i % 2
        w, h = rng.randint(14, w0 // 2), rng.randint(14, h0 // 2)
        x, y = rng.randint(0, w0 - w), rng.randint(0, h0 - h)
        color = np.clip(np.array(self.palette[label]) + rng.randn(3) * 20, 0, 255)
        ImageDraw.Draw(img).ellipse([x, y, x + w, y + h], fill=tuple(int(c) for c in color))
        box = torch.tensor([x / w0, y / h0, (x + w) / w0, (y + h) / h0])
        # box en coordonnees normalisees [0, 1] (x0, y0, x1, y1).
        return img, {"box": box, "label": label}


class Synthetic(DatasetSpec):
    """Genere a la volee : rien a telecharger, donc absent de `lo data prepare`."""

    NAME = "synthetic"
    ABBREV = "synth"  # CRITIQUE : fige, voir DatasetSpec.ABBREV

    def pretraining_images(self, num_classes: int = 10) -> ShapesSource:
        """Pendant de `Imagenette.pretraining_images` : (image PIL, label)."""
        return ShapesSource(SIZES["train"], num_classes, SEEDS["train"])

    def tasks(self) -> dict[str, Callable]:
        return {"cls": self.build_cls, "seg": self.build_seg, "det": self.build_det}

    def build_cls(self, split: str, transform: Callable, num_classes: int = 10) -> Subset:
        source = ShapesSource(SIZES[split], num_classes, SEEDS[split])
        return Subset(source, range(len(source)), transform)

    def build_seg(self, split: str, transform: Callable) -> Subset:
        source = BlobsSource(SIZES[split], SEEDS[split])
        return Subset(source, range(len(source)), transform)

    def build_det(self, split: str, transform: Callable) -> Subset:
        source = HeadsSource(SIZES[split], SEEDS[split])
        return Subset(source, range(len(source)), transform)


DATASET = Synthetic()
