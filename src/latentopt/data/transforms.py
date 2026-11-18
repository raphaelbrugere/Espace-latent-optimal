"""Transforms du protocole d'evaluation, appliques conjointement a (image, cible).

Les transforms d'eval font partie du protocole : les modifier change les scores et
casse la comparabilite avec les scores existants. Les transforms d'entrainement
`hflip` (probabilite de retournement horizontal : 0.5 pour la baseline ; 0 et 1 pour
les deux variantes du cache de features) sont l'augmentation minimale de la baseline.
"""

import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(img: Image.Image) -> Tensor:
    """PIL -> (3, H, W) float32 dans [0, 1], puis centre-reduit aux statistiques ImageNet."""
    return TF.normalize(
        TF.to_dtype(TF.to_image(img), torch.float32, scale=True), IMAGENET_MEAN, IMAGENET_STD
    )


def check_hflip(hflip: float) -> float:
    """`hflip` est une probabilite : True vaudrait 1 (toujours retourner), pas 0.5."""
    if isinstance(hflip, bool) or not 0.0 <= hflip <= 1.0:
        raise ValueError(
            f"hflip est une probabilite dans [0, 1] (0.5 pour la baseline), recu {hflip!r}"
        )
    return float(hflip)


def mask_to_tensor(mask: Image.Image) -> Tensor:
    """Masque PIL (mode L, un indice de classe par pixel) -> (H, W) int64."""
    return torch.from_numpy(np.array(mask, dtype=np.int64))


class CenterCrop:
    """Classification : petit cote a `size`, puis crop central (exemple minimal LeJEPA)."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image, label: int) -> tuple[Tensor, int]:
        return normalize(TF.center_crop(TF.resize(img, self.size), self.size)), label


class SquareResize:
    """Classification d'objets centres (Galaxy10) : resize anisotrope, sans crop."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image, label: int) -> tuple[Tensor, int]:
        return normalize(TF.resize(img, [self.size, self.size])), label


class SegmentationTransform:
    """Image en carre `size` (scaling anisotrope reversible).

    Eval (`native_mask=True`) : le masque garde sa resolution native ; la metrique y
    remonte les logits, donc la perte de resolution de la tete est penalisee.
    Train : le masque suit l'image (nearest), ce qui permet le collate par defaut.
    """

    def __init__(self, size: int, native_mask: bool = True, hflip: float = 0.0) -> None:
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
    """Image en carre `size`. Les boites sont normalisees dans [0, 1] et l'IoU entre
    boites alignees est invariante par scaling des axes : aucune correction a faire."""

    def __init__(self, size: int, hflip: float = 0.0) -> None:
        self.size, self.hflip = size, check_hflip(hflip)

    def __call__(
        self, img: Image.Image, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        img = normalize(TF.resize(img, [self.size, self.size]))
        box = target["box"].clone()
        if self.hflip and random.random() < self.hflip:
            img, box = TF.hflip(img), torch.stack([1 - box[2], box[1], 1 - box[0], box[3]])
        return img, {"box": box, "label": torch.tensor(target["label"])}


def collate_native(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, list[Tensor]]:
    """Images de taille commune, masques a leur resolution native : liste, pas de stack."""
    return torch.stack([x for x, _ in batch]), [y for _, y in batch]
