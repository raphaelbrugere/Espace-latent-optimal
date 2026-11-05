"""Entraînement d'une tête sur un backbone gelé (train_head.py de projects/lejepa_arena).

Le backbone reste en eval() et sans gradient pendant tout l'entraînement : Lightning
appelle train() sur le module entier à chaque epoch, `train` est donc redéfini pour ne
remettre que la tête en mode entraînement (sinon le drop-path du backbone se réactive).

Optimisation de la baseline : AdamW, décroissance cosinus sans warmup, par pas.

Les batches sont soit des images (le backbone est appliqué à la volée), soit des features
déjà calculées (dict, voir evaluation/features.py) : même calcul pour la tête.
"""

import math

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from latentopt.evaluation.tasks import Task, to_float
from latentopt.models.contract import freeze


class FrozenBackboneModule(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        task: Task,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.backbone = freeze(backbone)
        self.head = head
        self.task = task
        self.lr, self.weight_decay = lr, weight_decay
        self.metric = task.metric()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x) -> dict[str, Tensor]:
        if isinstance(x, dict):  # cache de features
            return x
        with torch.no_grad():
            return self.backbone(x)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        feats = self.features(x)
        out = self.head({k: v.float() for k, v in feats.items()})
        loss = self.task.loss(to_float(out), y)
        self.log("train/loss", loss.detach(), batch_size=len(next(iter(feats.values()))))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.metric = self.task.metric()

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        self.metric.update(to_float(self.head(self.features(x))), y)

    def on_validation_epoch_end(self) -> None:
        scores = self.metric.compute()
        self.log_dict({f"val/{k}": v for k, v in scores.items()})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total))
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
