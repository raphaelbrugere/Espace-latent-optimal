"""Metriques accumulees par batch : `update(out, target)` puis `compute() -> dict`."""

import torch
from torch import Tensor

from latentopt.losses import resize_logits


class Top1:
    """Precision top-1 : logits (B, K), cibles (B,) d'indices de classe."""

    def __init__(self) -> None:
        self.correct = self.total = 0

    def update(self, logits: Tensor, target: Tensor) -> None:
        self.correct += (logits.argmax(1) == target).sum().item()
        self.total += target.numel()

    def compute(self) -> dict[str, float]:
        return {"top1": self.correct / max(1, self.total)}


class BalancedAccuracy:
    """Rappel moyen par classe ; seules les classes presentes comptent.

    Logits (B, K), cibles (B,) d'indices de classe. Une classe frequente ne masque pas
    les erreurs sur les classes rares (Galaxy10 est desequilibre).
    `compute()` renvoie `balanced_acc` et aussi `top1` (precision globale, pour comparaison)."""

    def __init__(self, num_classes: int) -> None:
        self.hit_per_cls, self.count_per_cls = torch.zeros(num_classes), torch.zeros(num_classes)

    def update(self, logits: Tensor, target: Tensor) -> None:
        pred, target = logits.argmax(1).cpu(), target.cpu()
        n_cls = len(self.count_per_cls)
        self.count_per_cls += torch.bincount(target, minlength=n_cls).float()
        self.hit_per_cls += torch.bincount(target[pred == target], minlength=n_cls).float()

    def compute(self) -> dict[str, float]:
        seen = self.count_per_cls > 0
        recall = self.hit_per_cls[seen] / self.count_per_cls[seen]
        return {
            "balanced_acc": recall.mean().item(),
            "top1": (self.hit_per_cls.sum() / self.count_per_cls.sum()).item(),
        }


class MeanIoU:
    """mIoU dataset-level : intersections et unions sommees sur tous les masques avant la
    division, calcule a la resolution native de chaque masque.

    Logits (B, K, H, W), cibles : masques (H, W) d'indices de classe, de taille variable
    (liste) ou commune (tenseur). Renvoie `miou` (classes absentes : IoU 0, comptees)."""

    def __init__(self, num_classes: int) -> None:
        self.n_cls = num_classes
        self.inter, self.union = torch.zeros(num_classes), torch.zeros(num_classes)

    def update(self, logits: Tensor, target: Tensor | list[Tensor]) -> None:
        k = self.n_cls
        for lg, mask in zip(logits, list(target), strict=True):
            mask = mask.to(lg.device)
            pred = resize_logits(lg[None], mask.shape)[0].argmax(0)  # (H, W)
            # matrice de confusion (K, K) : ligne = classe vraie, colonne = classe predite
            conf = torch.bincount((k * mask + pred).flatten(), minlength=k * k).view(k, k)
            inter = conf.diagonal()
            self.inter += inter.cpu()
            self.union += (conf.sum(0) + conf.sum(1) - inter).cpu()

    def compute(self) -> dict[str, float]:
        return {"miou": (self.inter / self.union.clamp(min=1)).mean().item()}


def paired_iou(a: Tensor, b: Tensor) -> Tensor:
    """IoU de la i-eme boite de `a` avec la i-eme de `b` : (N, 4) xyxy -> (N,)."""
    wh = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter)


class LocalizationAccuracy:
    """Boites [x1, y1, x2, y2] normalisees dans [0, 1] (`out["box"]`, `target["box"]`).

    Correct si la classe est juste ET IoU >= seuil. Metrique principale : moyenne sur
    les seuils 0.50:0.05:0.95 (a la COCO), qui recompense la precision des boites.
    Renvoie `loc_acc` (cette moyenne), `acc_iou50`, `acc_iou75`, `mean_iou` (IoU moyen,
    sans condition de classe) et `cls_acc`."""

    iou_thresholds = torch.arange(0.5, 0.96, 0.05)

    def __init__(self) -> None:
        self.ious, self.correct_cls = [], []  # une entree par batch

    def update(self, out: dict[str, Tensor], target: dict[str, Tensor]) -> None:
        self.ious.append(paired_iou(out["box"].float(), target["box"].float()).cpu())
        self.correct_cls.append((out["logits"].argmax(1) == target["label"]).cpu())

    def compute(self) -> dict[str, float]:
        iou, cls_ok = torch.cat(self.ious), torch.cat(self.correct_cls)
        hits = torch.stack([(iou >= thr) & cls_ok for thr in self.iou_thresholds])
        acc_per_thr = hits.float().mean(1)
        # accuracy retrouvee par seuil (arrondi : evite l'indice magique et l'erreur flottante)
        acc = {
            round(t, 2): a for t, a in zip(self.iou_thresholds.tolist(), acc_per_thr, strict=True)
        }
        return {
            "loc_acc": acc_per_thr.mean().item(),
            "acc_iou50": acc[0.5].item(),
            "acc_iou75": acc[0.75].item(),
            "mean_iou": iou.mean().item(),
            "cls_acc": cls_ok.float().mean().item(),
        }
