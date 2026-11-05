"""Registre des tâches downstream : dataset, transforms, loss, métrique, budget de tête.

Reprend la small league de `projects/lejepa_arena` (entrée 128px, mêmes splits, mêmes
transforms, métriques et budgets), plus trois tâches synthétiques pour le smoke run (64px).

`default_head` et `epochs` forment la recette de la baseline (train_head.py de l'arène) :
choix par défaut, remplaçables (`head=...`, `epochs=...`), contrairement au protocole.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

from latentopt.data import transforms as T
from latentopt.data.datasets import flowers102, galaxy10, imagenette, oxford_pets, synthetic
from latentopt.evaluation import metrics as M
from latentopt.losses import detection_loss, segmentation_loss

SIZE, SMOKE_SIZE = 128, 64
CLS_HEAD_MAX_PARAMS = 2_000_000
DENSE_HEAD_MAX_PARAMS = 3_000_000


@dataclass(frozen=True)
class Task:
    name: str
    num_classes: int
    primary_metric: str
    head_max_params: int
    build: Callable  # (split, transform) -> Dataset
    eval_transform: Callable
    train_transform: Callable  # augmentation minimale de la baseline
    loss: Callable
    metric: Callable  # () -> métrique vierge
    default_head: str  # configs/head/<nom>.yaml
    epochs: int  # epochs de la baseline
    collate_fn: Callable | None = None  # split d'éval seulement

    def dataset(self, split: str, train: bool = False):
        return self.build(split, self.train_transform if train else self.eval_transform)


def classification(name, build, num_classes, transform, epochs, metric=None) -> Task:
    """Pas d'augmentation à l'entraînement (baseline) : train = éval."""
    balanced = metric == "balanced_acc"
    return Task(
        name=name,
        num_classes=num_classes,
        primary_metric=metric or "top1",
        head_max_params=CLS_HEAD_MAX_PARAMS,
        build=build,
        eval_transform=transform,
        train_transform=transform,
        loss=F.cross_entropy,
        metric=partial(M.BalancedAccuracy, num_classes) if balanced else M.Top1,
        default_head="linear",
        epochs=epochs,
    )


def segmentation(name, build, size) -> Task:
    return Task(
        name=name,
        num_classes=3,
        primary_metric="miou",
        head_max_params=DENSE_HEAD_MAX_PARAMS,
        build=build,
        eval_transform=T.SegmentationTransform(size, native_mask=True),
        train_transform=T.SegmentationTransform(size, native_mask=False, hflip=0.5),
        loss=segmentation_loss,
        metric=partial(M.MeanIoU, 3),
        default_head="fpn",
        epochs=30,
        collate_fn=T.collate_native,
    )


def detection(name, build, size) -> Task:
    return Task(
        name=name,
        num_classes=2,
        primary_metric="loc_acc",
        head_max_params=DENSE_HEAD_MAX_PARAMS,
        build=build,
        eval_transform=T.DetectionTransform(size),
        train_transform=T.DetectionTransform(size, hflip=0.5),
        loss=detection_loss,
        metric=M.LocalizationAccuracy,
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
        raise KeyError(f"tâche inconnue '{name}', disponibles : {sorted(TASKS)}")
    return TASKS[name]


def to_device(y, device):
    """Déplace des cibles sur `device`, quelle que soit leur structure (tenseur, liste, dict)."""
    if torch.is_tensor(y):
        return y.to(device, non_blocking=True)
    if isinstance(y, dict):
        return {k: to_device(v, device) for k, v in y.items()}
    if isinstance(y, (list, tuple)):
        return [to_device(v, device) for v in y]
    return y


def to_float(out):
    """Sorties de tête en fp32 (après autocast), quelle que soit leur structure."""
    if torch.is_tensor(out):
        return out.float()
    if isinstance(out, dict):
        return {k: v.float() for k, v in out.items()}
    return out
