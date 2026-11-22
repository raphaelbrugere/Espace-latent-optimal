"""Registre des taches downstream : dataset, transforms, loss, metrique.

Cinq taches de protocole (entree 128px, splits figes, transforms et metriques fixes, voir
docs/DATASETS.md), plus trois taches synthetiques pour le smoke run (64px).

`default_neck`, `default_head` et `epochs` forment le couple neck + tete de la baseline : choix par
defaut, remplacables (`neck=...`, `head=...`, `epochs=...`), contrairement au protocole.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from latentopt.data import transforms as T
from latentopt.data.datasets import flowers102, galaxy10, imagenette, oxford_pets, synthetic
from latentopt.evaluation import metrics as M
from latentopt.losses import detection_loss, segmentation_loss

SIZE, SMOKE_SIZE = 128, 64


@dataclass(frozen=True)
class Task:
    """Protocole d'une tache aval : donnees, transforms, perte et metrique.

    Les taches sont enregistrees dans `TASKS` ; `get_task(nom)` les retrouve."""

    name: str
    num_classes: int
    primary_metric: str
    build: Callable  # (split, transform) -> Dataset
    eval_transform: Callable
    train_transform: Callable  # augmentation minimale de la baseline
    loss: Callable
    metric: Callable  # () -> metrique vierge
    default_neck: str  # neck de la baseline : configs/neck/<nom>.yaml
    default_head: str  # tete de la baseline : configs/head/<nom>.yaml
    epochs: int  # epochs de la baseline
    collate_fn: Callable | None = None  # split d'eval seulement

    def dataset(self, split: str, train: bool = False) -> Any:
        return self.build(split, self.train_transform if train else self.eval_transform)


def classification(
    name: str,
    build: Callable,
    num_classes: int,
    transform: Callable,
    epochs: int,
    metric: str | None = None,
) -> Task:
    """Pas d'augmentation a l'entrainement (baseline) : train = eval."""
    balanced = metric == "balanced_acc"
    return Task(
        name=name,
        num_classes=num_classes,
        primary_metric=metric or "top1",
        build=build,
        eval_transform=transform,
        train_transform=transform,
        loss=F.cross_entropy,
        metric=partial(M.BalancedAccuracy, num_classes) if balanced else M.Top1,
        default_neck="embed_norm",
        default_head="linear",
        epochs=epochs,
    )


def segmentation(name: str, build: Callable, size: int) -> Task:
    return Task(
        name=name,
        num_classes=3,
        primary_metric="miou",
        build=build,
        eval_transform=T.SegmentationTransform(size, native_mask=True),
        train_transform=T.SegmentationTransform(size, native_mask=False, hflip=0.5),
        loss=segmentation_loss,
        metric=partial(M.MeanIoU, 3),
        default_neck="fpn",
        default_head="conv_seg",
        epochs=30,
        collate_fn=T.collate_native,
    )


def detection(name: str, build: Callable, size: int) -> Task:
    return Task(
        name=name,
        num_classes=2,
        primary_metric="loc_acc",
        build=build,
        eval_transform=T.DetectionTransform(size),
        train_transform=T.DetectionTransform(size, hflip=0.5),
        loss=detection_loss,
        metric=M.LocalizationAccuracy,
        default_neck="conv_trunk",
        default_head="soft_argmax",
        epochs=30,
    )


TASKS = {
    t.name: t
    for t in (
        classification("imagenette_cls", imagenette.build, 10, T.CenterCrop(SIZE), 20),
        classification("flowers102_cls", flowers102.build, 102, T.CenterCrop(SIZE), 30),
        classification(
            "galaxy10_cls", galaxy10.build, 10, T.SquareResize(SIZE), 20, "balanced_acc"
        ),
        segmentation("pets_seg", oxford_pets.build_seg, SIZE),
        detection("pets_det", oxford_pets.build_det, SIZE),
        classification("synthetic_cls", synthetic.build_cls, 10, T.CenterCrop(SMOKE_SIZE), 20),
        segmentation("synthetic_seg", synthetic.build_seg, SMOKE_SIZE),
        detection("synthetic_det", synthetic.build_det, SMOKE_SIZE),
    )
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"tache inconnue '{name}', disponibles : {sorted(TASKS)}")
    return TASKS[name]


def to_device(y: Any, device: torch.device) -> Any:
    """Deplace des cibles sur `device`, quelle que soit leur structure (tenseur, liste, dict)."""
    if torch.is_tensor(y):
        return y.to(device, non_blocking=True)
    if isinstance(y, dict):
        return {k: to_device(v, device) for k, v in y.items()}
    if isinstance(y, (list, tuple)):
        return [to_device(v, device) for v in y]
    return y


def to_float(out: Any) -> Any:
    """Sorties de tete en fp32 (apres autocast), quelle que soit leur structure."""
    if torch.is_tensor(out):
        return out.float()
    if isinstance(out, dict):
        return {k: v.float() for k, v in out.items()}
    return out
