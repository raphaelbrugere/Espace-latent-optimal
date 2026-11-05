"""LeJEPA (Balestriero & LeCun, 2025) : pré-entraînement joint-embedding sans heuristique.

    L = λ · SIGReg(z) + (1 - λ) · E_v ||z_v - z̄||²

où z = g(f(x_v)) est la projection des V vues d'une image et z̄ leur centroïde. SIGReg
remplace à lui seul les mécanismes anti-collapse habituels (EMA teacher, stop-gradient,
centrage, négatifs) : la cible N(0, I) minimise le risque downstream dans le pire cas.

Les losses portent sur le projecteur g ; c'est `embed` = f(x) qui sert en aval. La sonde
en ligne lit des embeddings détachés : elle suit l'entraînement sans fuite de labels
vers le backbone.

Valeurs par défaut : celles de l'exemple minimal de LeJEPA, reprises de la baseline de
`projects/lejepa_arena` (ViT-S/8 sur Imagenette, 128px).

`LeJEPA` est un nn.Module pur (testable sans Lightning) ; `LeJEPAModule` l'enveloppe
pour le Trainer de Lightning, sans changer le calcul.
"""

from collections.abc import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import MLP
from torchvision.transforms import v2

from latentopt.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from latentopt.losses import SIGReg
from latentopt.models.heads import LinearProbe


class MultiViewTransform:
    """(image PIL, label) -> ((V, 3, size, size), label) : V vues i.i.d. sous les
    augmentations LeJEPA. Signature conjointe des transforms de `data/` : s'utilise avec
    `data.datasets.base.Subset`. Le label ne sert qu'à la sonde en ligne."""

    def __init__(self, size: int = 128, num_views: int = 4):
        self.num_views = num_views
        self.augment = v2.Compose(
            [
                v2.RandomResizedCrop(size, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, img, label: int) -> tuple[Tensor, int]:
        return torch.stack([self.augment(img) for _ in range(self.num_views)]), label


class LeJEPA(nn.Module):
    """Backbone + projecteur + SIGReg, et sonde en ligne optionnelle.

    `backbone` respecte le contrat des têtes : `forward(x)["embed"]` de forme (B, D)
    et un attribut `embed_dim` = D.
    """

    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 16,
        hidden_dim: int = 2048,
        lamb: float = 0.02,
        num_probe_classes: int | None = None,
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(
            backbone.embed_dim, [hidden_dim, hidden_dim, proj_dim], norm_layer=nn.BatchNorm1d
        )
        self.sigreg = sigreg if sigreg is not None else SIGReg()
        self.lamb = lamb
        self.probe = (
            LinearProbe(backbone.embed_dim, num_probe_classes) if num_probe_classes else None
        )

    def forward(self, views: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        """views (N, V, 3, H, W), labels (N,) pour la sonde -> dict des losses.

        `loss` est la quantité à rétropropager ; elle inclut la loss de la sonde, qui ne
        reçoit que des embeddings détachés.
        """
        N, V = views.shape[:2]
        emb = self.backbone(views.flatten(0, 1))["embed"]  # (N·V, D)
        z = self.projector(emb).reshape(N, V, -1).transpose(0, 1)  # (V, N, d)

        inv_loss = (z.mean(0) - z).square().mean()
        sigreg_loss = self.sigreg(z)
        out = {
            "loss": self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
        }
        if self.probe is not None and labels is not None:
            logits = self.probe({"embed": emb.detach()})
            out["probe_loss"] = F.cross_entropy(logits, labels.repeat_interleave(V))
            out["loss"] = out["loss"] + out["probe_loss"]
        return out

    def ssl_parameters(self) -> list[nn.Parameter]:
        return [*self.backbone.parameters(), *self.projector.parameters()]


def build_optimizer(
    model: LeJEPA,
    total_steps: int,
    warmup_steps: int,
    lr: float = 2e-3,
    weight_decay: float = 5e-2,
    eta_min: float = 1e-3,
    probe_lr: float = 1e-3,
    probe_weight_decay: float = 1e-7,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW, warmup linéaire depuis 1 % du lr puis décroissance cosinus, par step.

    `eta_min` est un lr absolu, commun à tous les groupes (sémantique de
    CosineAnnealingLR), pas une fraction de `lr`.
    """
    groups = [{"params": model.ssl_parameters(), "lr": lr, "weight_decay": weight_decay}]
    if model.probe is not None:
        groups.append(
            {"params": model.probe.parameters(), "lr": probe_lr, "weight_decay": probe_weight_decay}
        )
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
            ),
        ],
        milestones=[warmup_steps],
    )
    return opt, sched


class LeJEPAModule(L.LightningModule):
    """Enveloppe Lightning de `LeJEPA`.

    Entraînement : batches (views (N, V, 3, H, W), labels (N,)) ; les labels ne servent
    qu'à la sonde en ligne. Validation : batches (x (B, 3, H, W), y (B,)) ; précision de
    la sonde, pondérée par la taille des batches (= correct / total sur l'epoch).

    Le scheduler avance à chaque pas d'optimisation (interval="step"), warmup et cosinus
    étant définis en pas : le défaut de Lightning (une fois par epoch) étirerait le
    warmup d'un facteur égal au nombre de pas par epoch.
    """

    def __init__(self, model: LeJEPA, make_optimizer: Callable, warmup_epochs: int = 1):
        super().__init__()
        self.model = model
        self.make_optimizer = make_optimizer
        self.warmup_epochs = warmup_epochs

    def training_step(self, batch, batch_idx: int) -> Tensor:
        views, labels = batch
        out = self.model(views, labels)
        self.log_dict({f"train/{k}": v.detach() for k, v in out.items()}, batch_size=len(views))
        return out["loss"]

    def validation_step(self, batch, batch_idx: int) -> None:
        if self.model.probe is None:
            return
        x, y = batch
        logits = self.model.probe({"embed": self.model.backbone(x)["embed"]})
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val/probe_acc", acc, batch_size=len(y), prog_bar=True)

    def configure_optimizers(self):
        total = self.trainer.estimated_stepping_batches  # tient compte de limit_train_batches
        warmup = total // self.trainer.max_epochs * self.warmup_epochs
        opt, sched = self.make_optimizer(self.model, total_steps=total, warmup_steps=warmup)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
