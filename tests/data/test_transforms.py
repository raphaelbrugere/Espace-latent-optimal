"""Transforms du protocole : formes, normalisation, retournement conjoint image / cible."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from latentopt.data import transforms as T


def rgb(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, size=(h, w, 3)).astype(np.uint8))


def left_half_mask(w: int, h: int) -> Image.Image:
    m = np.zeros((h, w), dtype=np.uint8)
    m[:, : w // 2] = 2
    return Image.fromarray(m, mode="L")


def test_normalize_uses_imagenet_statistics():
    x = T.normalize(Image.new("RGB", (4, 3), (0, 255, 128)))
    assert x.shape == (3, 3, 4) and x.dtype == torch.float32
    expected = [
        (0 - T.IMAGENET_MEAN[0]) / T.IMAGENET_STD[0],
        (1 - T.IMAGENET_MEAN[1]) / T.IMAGENET_STD[1],
        (128 / 255 - T.IMAGENET_MEAN[2]) / T.IMAGENET_STD[2],
    ]
    assert torch.allclose(x[:, 0, 0], torch.tensor(expected), atol=1e-5)


@pytest.mark.parametrize("p", [0, 0.0, 0.5, 1, 1.0])
def test_check_hflip_accepts_probabilities(p):
    assert T.check_hflip(p) == float(p)


@pytest.mark.parametrize("p", [True, False, -0.1, 1.5])
def test_check_hflip_rejects_bools_and_out_of_range(p):
    with pytest.raises(ValueError, match="probabilite"):
        T.check_hflip(p)


@pytest.mark.parametrize("w,h", [(160, 120), (90, 200), (64, 64)])
def test_center_crop_gives_square(w, h):
    x, label = T.CenterCrop(32)(rgb(w, h), 7)
    assert x.shape == (3, 32, 32) and label == 7


def test_square_resize_is_anisotropic_without_crop():
    img = Image.new("RGB", (100, 50), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 10, 50))  # bande blanche a gauche, 10 % de la largeur
    x, _ = T.SquareResize(20)(img, 0)
    assert x.shape == (3, 20, 20)
    # sans crop, la bande reste presente sur toute la hauteur
    assert (x[0, :, 0] > 0).all()


def test_segmentation_eval_keeps_native_mask():
    x, m = T.SegmentationTransform(32, native_mask=True)(rgb(50, 40), left_half_mask(50, 40))
    assert x.shape == (3, 32, 32)
    assert m.shape == (40, 50) and m.dtype == torch.int64
    assert set(m.unique().tolist()) == {0, 2}


def test_segmentation_train_resizes_mask_with_nearest():
    _, m = T.SegmentationTransform(32, native_mask=False)(rgb(50, 40), left_half_mask(50, 40))
    assert m.shape == (32, 32)
    assert set(m.unique().tolist()) <= {0, 2}  # nearest : aucune valeur interpolee


def test_segmentation_hflip_flips_image_and_mask_together():
    img, mask = rgb(40, 40), left_half_mask(40, 40)
    x0, m0 = T.SegmentationTransform(32, native_mask=False)(img, mask)
    x1, m1 = T.SegmentationTransform(32, native_mask=False, hflip=1.0)(img, mask)
    assert torch.equal(x1, x0.flip(-1))
    assert torch.equal(m1, m0.flip(-1))


def test_detection_transform_keeps_normalized_box():
    box = torch.tensor([0.1, 0.2, 0.4, 0.9])
    x, t = T.DetectionTransform(32)(rgb(80, 60), {"box": box, "label": 1})
    assert x.shape == (3, 32, 32)
    assert torch.equal(t["box"], box) and t["label"].item() == 1


def test_detection_hflip_mirrors_box_and_does_not_mutate_input():
    box = torch.tensor([0.1, 0.2, 0.4, 0.9])
    target = {"box": box, "label": 0}
    x0, _ = T.DetectionTransform(32)(rgb(80, 60), target)
    x1, t1 = T.DetectionTransform(32, hflip=1.0)(rgb(80, 60), target)
    assert torch.allclose(t1["box"], torch.tensor([0.6, 0.2, 0.9, 0.9]))
    assert torch.equal(x1, x0.flip(-1))
    assert torch.equal(target["box"], torch.tensor([0.1, 0.2, 0.4, 0.9]))
    # le retournement reste une boite valide (x1 < x2)
    assert t1["box"][0] < t1["box"][2]


def test_hflip_probability_is_respected():
    random.seed(0)
    t = T.DetectionTransform(16, hflip=0.5)
    img, target = rgb(16, 16), {"box": torch.tensor([0.0, 0.0, 0.25, 1.0]), "label": 0}
    flips = sum(t(img, target)[1]["box"][0].item() > 0.5 for _ in range(400))
    assert 140 < flips < 260


def test_collate_native_keeps_masks_as_list():
    batch = [(torch.zeros(3, 8, 8), torch.zeros(5, 6)), (torch.ones(3, 8, 8), torch.zeros(7, 4))]
    x, y = T.collate_native(batch)
    assert x.shape == (2, 3, 8, 8)
    assert isinstance(y, list) and [t.shape for t in y] == [(5, 6), (7, 4)]
