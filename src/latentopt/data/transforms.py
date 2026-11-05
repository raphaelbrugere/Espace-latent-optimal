"""Transforms du protocole d'évaluation, appliqués conjointement à (image, cible).

Les transforms d'éval font partie du protocole : les modifier change les scores et
casse la comparabilité avec `projects/lejepa_arena`. Les transforms d'entraînement
`hflip` (probabilité de retournement horizontal : 0.5 pour la baseline ; 0 et 1 pour
les deux variantes du cache de features) sont l'augmentation minimale de la baseline.
"""

import random

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(img: Image.Image) -> Tensor:
    return TF.normalize(
        TF.to_dtype(TF.to_image(img), torch.float32, scale=True), IMAGENET_MEAN, IMAGENET_STD
    )


def check_hflip(hflip: float) -> float:
    """`hflip` est une probabilité : True vaudrait 1 (toujours retourner), pas 0.5."""
    if isinstance(hflip, bool) or not 0.0 <= hflip <= 1.0:
        raise ValueError(
            f"hflip est une probabilité dans [0, 1] (0.5 pour la baseline), reçu {hflip!r}"
        )
    return float(hflip)


def mask_to_tensor(mask: Image.Image) -> Tensor:
    return torch.from_numpy(np.array(mask, dtype=np.int64))


class CenterCrop:
    """Classification : petit côté à `size`, puis crop central (exemple minimal LeJEPA)."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image, label: int) -> tuple[Tensor, int]:
        return normalize(TF.center_crop(TF.resize(img, self.size), self.size)), label


class SquareResize:
    """Classification d'objets centrés (Galaxy10) : resize anisotrope, sans crop."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image, label: int) -> tuple[Tensor, int]:
        return normalize(TF.resize(img, [self.size, self.size])), label


class SegmentationTransform:
    """Image en carré `size` (scaling anisotrope réversible).

    Éval (`native_mask=True`) : le masque garde sa résolution native ; la métrique y
    remonte les logits, donc la perte de résolution de la tête est pénalisée.
    Train : le masque suit l'image (nearest), ce qui permet le collate par défaut.
    """

    def __init__(self, size: int, native_mask: bool = True, hflip: float = 0.0):
        self.size, self.native_mask, self.hflip = size, native_mask, check_hflip(hflip)

    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        img = TF.resize(img, [self.size, self.size])
        if not self.native_mask:
            mask = TF.resize(
                mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST
            )
        if self.hflip and random.random() < self.hflip:
            img, mask = TF.hflip(img), TF.hflip(mask)
        return normalize(img), mask_to_tensor(mask)


class DetectionTransform:
    """Image en carré `size`. Les boîtes sont normalisées dans [0, 1] et l'IoU entre
    boîtes alignées est invariante par scaling des axes : aucune correction à faire."""

    def __init__(self, size: int, hflip: float = 0.0):
        self.size, self.hflip = size, check_hflip(hflip)

    def __call__(self, img: Image.Image, target: dict) -> tuple[Tensor, dict[str, Tensor]]:
        img = normalize(TF.resize(img, [self.size, self.size]))
        box = target["box"].clone()
        if self.hflip and random.random() < self.hflip:
            img, box = TF.hflip(img), torch.stack([1 - box[2], box[1], 1 - box[0], box[3]])
        return img, {"box": box, "label": torch.tensor(target["label"])}


def collate_native(batch: list) -> tuple[Tensor, list[Tensor]]:
    """Images de taille commune, masques à leur résolution native : liste, pas de stack."""
    return torch.stack([x for x, _ in batch]), [y for _, y in batch]
