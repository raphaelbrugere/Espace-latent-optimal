"""Métriques accumulées par batch : `update(out, target)` puis `compute() -> dict`."""

import torch
from torch import Tensor
from torchvision.ops import box_iou

from latentopt.losses import resize_logits


class Top1:
    def __init__(self):
        self.correct = self.total = 0

    def update(self, logits: Tensor, y: Tensor) -> None:
        self.correct += (logits.argmax(1) == y).sum().item()
        self.total += y.numel()

    def compute(self) -> dict[str, float]:
        return {"top1": self.correct / max(1, self.total)}


class BalancedAccuracy:
    """Rappel moyen par classe ; seules les classes présentes comptent."""

    def __init__(self, num_classes: int):
        self.hit, self.count = torch.zeros(num_classes), torch.zeros(num_classes)

    def update(self, logits: Tensor, y: Tensor) -> None:
        pred, y = logits.argmax(1).cpu(), y.cpu()
        self.count += torch.bincount(y, minlength=len(self.count)).float()
        self.hit += torch.bincount(y[pred == y], minlength=len(self.hit)).float()

    def compute(self) -> dict[str, float]:
        seen = self.count > 0
        return {
            "balanced_acc": (self.hit[seen] / self.count[seen]).mean().item(),
            "top1": (self.hit.sum() / self.count.sum()).item(),
        }


class MeanIoU:
    """mIoU dataset-level (intersections et unions sommées avant division), calculé à la
    résolution native de chaque masque."""

    def __init__(self, num_classes: int):
        self.k = num_classes
        self.inter, self.union = torch.zeros(num_classes), torch.zeros(num_classes)

    def update(self, logits: Tensor, targets: Tensor | list[Tensor]) -> None:
        for lg, y in zip(logits, list(targets), strict=True):
            y = y.to(lg.device)
            pred = resize_logits(lg[None], y.shape)[0].argmax(0)
            for c in range(self.k):
                p, t = pred == c, y == c
                self.inter[c] += (p & t).sum().item()
                self.union[c] += (p | t).sum().item()

    def compute(self) -> dict[str, float]:
        return {"miou": (self.inter / self.union.clamp(min=1)).mean().item()}


class LocalizationAccuracy:
    """Correct si la classe est juste ET IoU >= seuil. Métrique principale : moyenne sur
    les seuils 0.50:0.05:0.95 (à la COCO), qui récompense la précision des boîtes."""

    thresholds = torch.arange(0.5, 0.96, 0.05)

    def __init__(self):
        self.ious, self.correct_cls = [], []

    def update(self, out: dict[str, Tensor], y: dict[str, Tensor]) -> None:
        self.ious.append(box_iou(out["box"].float(), y["box"].float()).diagonal().cpu())
        self.correct_cls.append((out["logits"].argmax(1) == y["label"]).cpu())

    def compute(self) -> dict[str, float]:
        iou, cls = torch.cat(self.ious), torch.cat(self.correct_cls)
        hits = torch.stack([(iou >= t) & cls for t in self.thresholds]).float().mean(1)
        return {
            "loc_acc": hits.mean().item(),
            "acc_iou50": hits[0].item(),
            "acc_iou75": hits[5].item(),
            "mean_iou": iou.mean().item(),
            "cls_acc": cls.float().mean().item(),
        }
