"""Métriques : valeurs calculées à la main, accumulation par batch, cas limites."""

import pytest
import torch

from latentopt.evaluation import metrics as M


def one_hot_logits(pred, k):
    return torch.nn.functional.one_hot(torch.tensor(pred), k).float()


# --- Top1 / BalancedAccuracy -------------------------------------------------------------


def test_top1_accumulates_over_batches():
    m = M.Top1()
    m.update(one_hot_logits([0, 1, 2], 3), torch.tensor([0, 1, 0]))
    m.update(one_hot_logits([2], 3), torch.tensor([2]))
    assert m.compute() == {"top1": 3 / 4}


def test_top1_empty_is_zero_not_nan():
    assert M.Top1().compute() == {"top1": 0.0}


def test_balanced_accuracy_averages_recall_per_class():
    m = M.BalancedAccuracy(3)
    # classe 0 : 3 exemples, 3 justes ; classe 1 : 1 exemple, faux ; classe 2 absente
    m.update(one_hot_logits([0, 0, 0, 0], 3), torch.tensor([0, 0, 0, 1]))
    out = m.compute()
    assert out["balanced_acc"] == pytest.approx((1.0 + 0.0) / 2)  # classe absente ignorée
    assert out["top1"] == pytest.approx(3 / 4)


def test_balanced_accuracy_differs_from_top1_under_imbalance():
    m = M.BalancedAccuracy(2)
    m.update(one_hot_logits([0] * 10, 2), torch.tensor([0] * 9 + [1]))
    out = m.compute()
    assert out["top1"] == pytest.approx(0.9) and out["balanced_acc"] == pytest.approx(0.5)


# --- MeanIoU -----------------------------------------------------------------------------


def seg_logits(pred: torch.Tensor, k: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(pred, k).permute(0, 3, 1, 2).float()


def test_miou_perfect_prediction_is_one():
    y = torch.randint(0, 3, (2, 6, 6))
    y[0, 0, :3] = torch.tensor([0, 1, 2])  # toutes les classes présentes
    m = M.MeanIoU(3)
    m.update(seg_logits(y, 3), y)
    assert m.compute()["miou"] == pytest.approx(1.0)


def test_miou_hand_computed():
    y = torch.tensor([[[0, 0], [1, 1]]])
    pred = torch.tensor([[[0, 1], [1, 1]]])
    m = M.MeanIoU(2)
    m.update(seg_logits(pred, 2), y)
    # classe 0 : inter 1, union 2 ; classe 1 : inter 2, union 3
    assert m.compute()["miou"] == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_miou_is_dataset_level_not_image_mean():
    m = M.MeanIoU(2)
    y_big = torch.zeros(1, 10, 10, dtype=torch.long)
    m.update(seg_logits(y_big, 2), y_big)  # classe 0 : 100/100
    y_small = torch.tensor([[[1]]])
    m.update(seg_logits(torch.tensor([[[0]]]), 2), y_small)  # erreur sur 1 pixel
    out = m.compute()["miou"]
    assert out == pytest.approx((100 / 101 + 0 / 1) / 2)


def test_miou_upsamples_logits_to_native_mask_list():
    m = M.MeanIoU(2)
    logits = torch.zeros(2, 2, 4, 4)
    logits[:, 1] = 1.0  # tout prédit en classe 1
    masks = [torch.ones(8, 6, dtype=torch.long), torch.ones(5, 9, dtype=torch.long)]
    m.update(logits, masks)
    assert m.compute()["miou"] == pytest.approx(0.5)  # classe 0 jamais vue : IoU 0


# --- LocalizationAccuracy ------------------------------------------------------------------


def det(boxes, pred_cls, k=2):
    return {"box": torch.tensor(boxes), "logits": one_hot_logits(pred_cls, k)}


def test_localization_perfect_is_one():
    boxes = [[0.1, 0.1, 0.5, 0.5], [0.2, 0.3, 0.9, 0.8]]
    m = M.LocalizationAccuracy()
    m.update(det(boxes, [0, 1]), {"box": torch.tensor(boxes), "label": torch.tensor([0, 1])})
    out = m.compute()
    for key in ("loc_acc", "acc_iou50", "acc_iou75", "mean_iou", "cls_acc"):
        assert out[key] == pytest.approx(1.0), key


def test_localization_requires_correct_class():
    box = [[0.1, 0.1, 0.5, 0.5]]
    m = M.LocalizationAccuracy()
    m.update(det(box, [1]), {"box": torch.tensor(box), "label": torch.tensor([0])})
    out = m.compute()
    assert out["loc_acc"] == 0 and out["mean_iou"] == pytest.approx(1.0)


def test_localization_thresholds():
    target = {"box": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "label": torch.tensor([0])}
    m = M.LocalizationAccuracy()
    m.update(det([[0.0, 0.0, 0.62, 1.0]], [0]), target)  # IoU 0.62
    out = m.compute()
    assert out["acc_iou50"] == 1.0 and out["acc_iou75"] == 0.0
    assert out["loc_acc"] == pytest.approx(0.3)  # seuils 0.50, 0.55, 0.60 franchis sur 10


def test_localization_thresholds_are_coco_style():
    assert len(M.LocalizationAccuracy.thresholds) == 10
    assert M.LocalizationAccuracy.thresholds[0] == pytest.approx(0.5)
    assert M.LocalizationAccuracy.thresholds[5] == pytest.approx(0.75)
    assert M.LocalizationAccuracy.thresholds[-1] == pytest.approx(0.95)
