"""Datasets générés, pour le smoke run, la CI et les tests : aucun fichier, CPU, quelques
secondes. Mêmes formats de cible que les vrais datasets ; les scores n'ont aucun sens.

Images de segmentation et de détection non carrées : elles exercent la métrique à
résolution native et la normalisation des boîtes.
"""

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from latentopt.data.datasets.base import Subset

SIZES = {"train": 1000, "val": 150, "test": 300}
SEEDS = {"train": 0, "val": 2, "test": 1}


def noise(rng: np.random.RandomState, w: int, h: int) -> Image.Image:
    return Image.fromarray(np.clip(rng.randn(h, w, 3) * 20 + 128, 0, 255).astype(np.uint8))


class Shapes:
    """Classification : la classe fixe la couleur de fond et le nombre d'ellipses."""

    def __init__(self, n: int, num_classes: int, seed: int, size: int = 80):
        self.n, self.k, self.seed, self.size = n, num_classes, seed, size
        self.palette = np.random.RandomState(1234).randint(30, 225, size=(num_classes, 3))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
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


class Blobs:
    """Segmentation à 3 classes : ellipses colorées sur bruit, masque aligné."""

    palette = [(40, 40, 40), (220, 60, 60), (60, 200, 90)]

    def __init__(self, n: int, seed: int, size: tuple[int, int] = (96, 72)):
        self.n, self.seed, self.size = n, seed, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.RandomState(self.seed * 1_000_003 + i)
        w0, h0 = self.size
        img, mask = noise(rng, w0, h0), Image.new("L", (w0, h0), 0)
        for _ in range(3):
            c = int(rng.randint(1, 3))
            x, y = rng.randint(0, w0 - 16), rng.randint(0, h0 - 16)
            w, h = rng.randint(12, w0 // 2), rng.randint(12, h0 // 2)
            ImageDraw.Draw(img).ellipse([x, y, x + w, y + h], fill=self.palette[c])
            ImageDraw.Draw(mask).ellipse([x, y, x + w, y + h], fill=c)
        return img, mask


class Heads:
    """Détection : une ellipse par image, classe = famille de couleur."""

    palette = [(220, 80, 60), (60, 120, 220)]

    def __init__(self, n: int, seed: int, size: tuple[int, int] = (96, 72)):
        self.n, self.seed, self.size = n, seed, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.RandomState(self.seed * 1_000_003 + i)
        w0, h0 = self.size
        img, label = noise(rng, w0, h0), i % 2
        w, h = rng.randint(14, w0 // 2), rng.randint(14, h0 // 2)
        x, y = rng.randint(0, w0 - w), rng.randint(0, h0 - h)
        color = np.clip(np.array(self.palette[label]) + rng.randn(3) * 20, 0, 255)
        ImageDraw.Draw(img).ellipse([x, y, x + w, y + h], fill=tuple(int(c) for c in color))
        box = torch.tensor([x / w0, y / h0, (x + w) / w0, (y + h) / h0])
        return img, {"box": box, "label": label}


def pretraining_images(num_classes: int = 10) -> Shapes:
    """Pendant de `imagenette.pretraining_images` : source (image PIL, label)."""
    return Shapes(SIZES["train"], num_classes, SEEDS["train"])


def build_cls(split: str, transform: Callable, num_classes: int = 10) -> Subset:
    source = Shapes(SIZES[split], num_classes, SEEDS[split])
    return Subset(source, range(len(source)), transform)


def build_seg(split: str, transform: Callable) -> Subset:
    source = Blobs(SIZES[split], SEEDS[split])
    return Subset(source, range(len(source)), transform)


def build_det(split: str, transform: Callable) -> Subset:
    source = Heads(SIZES[split], SEEDS[split])
    return Subset(source, range(len(source)), transform)
