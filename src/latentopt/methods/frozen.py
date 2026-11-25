"""Entrainement d'un neck et d'une tete sur un backbone gele.

Le backbone reste en eval() et sans gradient pendant tout l'entrainement : Lightning
appelle train() sur le module entier a chaque epoch, `train` est donc redefini pour ne
remettre que la tete en mode entrainement (sinon le drop-path du backbone se reactive).
Avec `freeze_neck` (neck repris d'un entrainement precedent), le neck est traite comme le
backbone : eval(), sans gradient, hors de l'optimiseur.

Optimisation de la baseline : AdamW, decroissance cosinus sans warmup, par pas.

Les batches sont soit des images (le backbone est applique a la volee), soit des features
deja calculees (dict, voir evaluation/features.py) : meme calcul pour la tete.
"""

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from latentopt.evaluation.tasks import Task, to_float
from latentopt.models.contract import freeze


class FrozenBackboneModule(L.LightningModule):
    """Entraine `downstream` (neck + tete) sur un backbone gele (sondage lineaire, FPN,
    detection), ou la tete seule sur un neck gele (`freeze_neck`).

    Le backbone reste en mode eval meme sous `train()` ; seuls neck et tete recoivent des
    gradients, ou la tete seule."""

    def __init__(
        self,
        backbone: nn.Module,
        downstream: nn.Module,
        task: Task,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_neck: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = freeze(backbone)
        self.downstream = downstream
        self.freeze_neck = freeze_neck
        if freeze_neck:
            freeze(downstream.neck)
        self.task = task
        self.lr, self.weight_decay = lr, weight_decay
        self.metric = task.metric()

    def train(self, mode: bool = True) -> "FrozenBackboneModule":
        super().train(mode)
        self.backbone.eval()
        if self.freeze_neck:
            self.downstream.neck.eval()
        return self

    def features(self, x: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Features du backbone gele, sans gradient ; un dict (cache) est rendu tel quel."""
        if isinstance(x, dict):  # cache de features
            return x
        with torch.no_grad():
            return self.backbone(x)

    def training_step(self, batch: tuple[Any, Any], batch_idx: int) -> Tensor:
        x, y = batch
        feats = self.features(x)
        out = self.downstream({k: v.float() for k, v in feats.items()})
        loss = self.task.loss(to_float(out), y)
        self.log("train/loss", loss.detach(), batch_size=len(next(iter(feats.values()))))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.metric = self.task.metric()

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> None:
        x, y = batch
        self.metric.update(to_float(self.downstream(self.features(x))), y)

    def on_validation_epoch_end(self) -> None:
        scores = self.metric.compute()
        self.log_dict({f"val/{k}": v for k, v in scores.items()})

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.downstream.parameters() if p.requires_grad]  # sans le neck gele
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total))
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
