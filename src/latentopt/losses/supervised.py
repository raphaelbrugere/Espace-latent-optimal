"""Losses supervisees des tetes downstream (baselines ; libres d'etre remplacees)."""

import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import generalized_box_iou_loss


def resize_logits(logits: Tensor, size: tuple[int, int]) -> Tensor:
    """Remonte des logits denses a `size` (bilineaire, fp32) ; no-op si deja a la taille."""
    if logits.shape[-2:] != tuple(size):
        logits = F.interpolate(
            logits.float(), size=tuple(size), mode="bilinear", align_corners=False
        )
    return logits


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    """CE pixel a la resolution du masque : la tete peut sortir a n'importe quel stride."""
    return F.cross_entropy(resize_logits(logits, target.shape[-2:]), target)


def detection_loss(out: dict[str, Tensor], target: dict[str, Tensor]) -> Tensor:
    """CE (classe) + 5*L1 + 2*GIoU (boite), ponderations DETR."""
    box = out["box"].float()
    return (
        F.cross_entropy(out["logits"].float(), target["label"])
        + 5 * F.l1_loss(box, target["box"])
        + 2 * generalized_box_iou_loss(box, target["box"], reduction="mean")
    )
