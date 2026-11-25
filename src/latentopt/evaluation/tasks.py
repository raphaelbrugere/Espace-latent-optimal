"""Registre des taches downstream : dataset, traitement des images, loss, metrique.

Cinq taches de protocole (entree 128px, splits figes, traitement et metriques fixes, voir
docs/DATASETS.md), plus trois taches synthetiques pour le smoke run (64px). Chaque tache
declare sa preparation des images (`prepare`, protocole) et son augmentation a
l'entrainement des tetes (`augment`, None = aucune) : voir data/transforms.py.

`default_neck`, `default_head` et `epochs` forment le couple neck + tete de la baseline : choix par
defaut, remplacables (`neck=...`, `head=...`, `epochs=...`), contrairement au protocole.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from latentopt.data import transforms as T
from latentopt.data.datasets.registry import BY_NAME as DATASETS
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
    build: Callable  # (split, pipeline) -> Dataset, fourni par le dataset (DatasetSpec.tasks)
    eval_pipeline: T.Pipeline  # sans augmentation
    train_pipeline: T.Pipeline  # avec l'augmentation de la tache, s'il y en a une
    loss: Callable
    metric: Callable  # () -> metrique vierge
    default_neck: str  # neck de la baseline : configs/neck/<nom>.yaml
    default_head: str  # tete de la baseline : configs/head/<nom>.yaml
    epochs: int  # epochs de la baseline
    collate_fn: Callable | None = None  # split d'eval seulement

    def dataset(self, split: str, train: bool = False) -> Any:
        return self.build(split, self.train_pipeline if train else self.eval_pipeline)


def classification(
    name: str,
    dataset: str,
    num_classes: int,
    prepare: v2.Transform,
    epochs: int,
    metric: str | None = None,
    augment: v2.Transform | None = None,
) -> Task:
    balanced = metric == "balanced_acc"
    return Task(
        name=name,
        num_classes=num_classes,
        primary_metric=metric or "top1",
        build=DATASETS[dataset].tasks()["cls"],
        eval_pipeline=T.Pipeline(prepare),
        train_pipeline=T.Pipeline(prepare, augment),
        loss=F.cross_entropy,
        metric=partial(M.BalancedAccuracy, num_classes) if balanced else M.Top1,
        default_neck="embed_norm",
        default_head="linear",
        epochs=epochs,
    )


def segmentation(
    name: str,
    dataset: str,
    num_classes: int,
    size: int,
    epochs: int,
    augment: v2.Transform | None = None,
) -> Task:
    """Image en carre ; a l'evaluation, le masque garde sa resolution d'origine."""
    return Task(
        name=name,
        num_classes=num_classes,
        primary_metric="miou",
        build=DATASETS[dataset].tasks()["seg"],
        eval_pipeline=T.Pipeline(T.square(size), native_target=True),
        train_pipeline=T.Pipeline(T.square(size), augment),
        loss=segmentation_loss,
        metric=partial(M.MeanIoU, num_classes),
        default_neck="fpn",
        default_head="conv_seg",
        epochs=epochs,
        collate_fn=T.collate_native,
    )


def detection(
    name: str,
    dataset: str,
    num_classes: int,
    size: int,
    epochs: int,
    augment: v2.Transform | None = None,
) -> Task:
    return Task(
        name=name,
        num_classes=num_classes,
        primary_metric="loc_acc",
        build=DATASETS[dataset].tasks()["det"],
        eval_pipeline=T.Pipeline(T.square(size)),
        train_pipeline=T.Pipeline(T.square(size), augment),
        loss=detection_loss,
        metric=M.LocalizationAccuracy,
        default_neck="conv_trunk",
        default_head="soft_argmax",
        epochs=epochs,
    )


TASKS = {
    t.name: t
    for t in (
        classification("imagenette_cls", "imagenette", 10, T.center_crop(SIZE), 20),
        classification("flowers102_cls", "flowers102", 102, T.center_crop(SIZE), 30),
        classification("galaxy10_cls", "galaxy10", 10, T.square(SIZE), 20, "balanced_acc"),
        segmentation("pets_seg", "oxford_pets", 3, SIZE, 30, augment=T.hflip()),
        detection("pets_det", "oxford_pets", 2, SIZE, 30, augment=T.hflip()),
        classification("synthetic_cls", "synthetic", 10, T.center_crop(SMOKE_SIZE), 20),
        segmentation("synthetic_seg", "synthetic", 3, SMOKE_SIZE, 30, augment=T.hflip()),
        detection("synthetic_det", "synthetic", 2, SMOKE_SIZE, 30, augment=T.hflip()),
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
