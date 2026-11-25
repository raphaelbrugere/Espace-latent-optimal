"""Transforms des taches aval (evaluation et entrainement des tetes), appliques
conjointement a (image, cible) pour garder les deux alignees : un flip retourne aussi le
masque ou la boite. Les vues du pre-entrainement sont ailleurs (methods/lejepa.py).

Nommes par tache puis variante (`Cls...`, `Seg...`, `Det...`), comme les cles de
`DatasetSpec.tasks()`. Les transforms d'eval font partie du protocole : les modifier change
les scores et casse la comparabilite avec les scores existants. A l'entrainement, la seule
augmentation est `hflip`, la probabilite de retournement horizontal (0.5 pour la baseline).

Les images sortent en float32 dans [0, 1], sans normalisation : c'est le backbone qui
normalise (voir models/contract.py).
"""

import random
from typing import Any, Protocol, TypeVar

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import functional as TF

Target = TypeVar("Target", contravariant=True)


class JointTransform(Protocol[Target]):
    """(image PIL, cible brute de la source) -> (image (3, size, size) float32 [0, 1], cible)."""

    def __call__(self, img: Image.Image, target: Target) -> tuple[Tensor, Any]: ...


def to_tensor(img: Image.Image) -> Tensor:
    """PIL -> (3, H, W) float32 dans [0, 1]. Pas de normalisation : c'est le backbone qui s'en
    charge (models/contract.py)."""
    return TF.to_dtype(TF.to_image(img), torch.float32, scale=True)


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


class ClsCenterCrop:
    """Classification : petit cote a `size`, puis crop central carre."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image, label: int) -> tuple[Tensor, int]:
        return to_tensor(TF.center_crop(TF.resize(img, self.size), self.size)), label


class ClsSquareResize:
    """Classification d'objets centres (Galaxy10) : resize anisotrope, sans crop."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image, label: int) -> tuple[Tensor, int]:
        return to_tensor(TF.resize(img, [self.size, self.size])), label


class SegTransform:
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
        return to_tensor(img), mask_to_tensor(mask)


class DetTransform:
    """Image en carre `size`. Les boites sont normalisees dans [0, 1] et l'IoU entre
    boites alignees est invariante par scaling des axes : aucune correction a faire."""

    def __init__(self, size: int, hflip: float = 0.0) -> None:
        self.size, self.hflip = size, check_hflip(hflip)

    def __call__(
        self, img: Image.Image, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        img = to_tensor(TF.resize(img, [self.size, self.size]))
        box = target["box"].clone()
        if self.hflip and random.random() < self.hflip:
            img, box = TF.hflip(img), torch.stack([1 - box[2], box[1], 1 - box[0], box[3]])
        return img, {"box": box, "label": torch.tensor(target["label"])}


def collate_native(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, list[Tensor]]:
    """Images de taille commune, masques a leur resolution native : liste, pas de stack."""
    return torch.stack([x for x, _ in batch]), [y for _, y in batch]
