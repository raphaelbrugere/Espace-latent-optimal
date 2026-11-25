"""Traitement des images (Pipeline) : formes, cible qui suit l'image, versions, empreinte."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from latentopt.data import transforms as T


def rgb(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, size=(h, w, 3)).astype(np.uint8))


def left_half_mask(w: int, h: int) -> tv_tensors.Mask:
    m = torch.zeros((h, w), dtype=torch.uint8)
    m[:, : w // 2] = 2
    return tv_tensors.Mask(m)


def det_target(box: list[float], w: int, h: int, label: int = 0, dtype=torch.float32) -> dict:
    """Boite en pixels sur une image (w, h), comme les sources de detection."""
    box = torch.tensor([box], dtype=dtype)
    return {
        "box": tv_tensors.BoundingBoxes(box, format="XYXY", canvas_size=(h, w)),
        "label": label,
    }


FLIP = v2.RandomHorizontalFlip(p=1.0)


def test_images_are_unnormalized_float32_in_unit_range():
    x = T.ToFloatImage()(Image.new("RGB", (4, 3), (0, 255, 128)))
    assert x.shape == (3, 3, 4) and x.dtype == torch.float32
    assert torch.allclose(x[:, 0, 0], torch.tensor([0.0, 1.0, 128 / 255]), atol=1e-6)


def test_every_task_kind_returns_float32_in_unit_range():
    img = rgb(50, 40)
    outs = [
        T.Pipeline(T.center_crop(32))(img, 0)[0],
        T.Pipeline(T.square(32))(img, 0)[0],
        T.Pipeline(T.square(32), native_target=True)(img, left_half_mask(50, 40))[0],
        T.Pipeline(T.square(32))(img, det_target([5, 8, 20, 36], 50, 40))[0],
    ]
    for x in outs:
        assert x.shape == (3, 32, 32) and x.dtype == torch.float32
        assert 0.0 <= x.min() and x.max() <= 1.0


@pytest.mark.parametrize("w,h", [(160, 120), (90, 200), (64, 64)])
def test_center_crop_gives_square_and_keeps_label(w, h):
    x, label = T.Pipeline(T.center_crop(32))(rgb(w, h), 7)
    assert x.shape == (3, 32, 32) and label == 7


def test_square_is_anisotropic_without_crop():
    img = Image.new("RGB", (100, 50), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 10, 50))  # bande blanche a gauche, 10 % de la largeur
    x, _ = T.Pipeline(T.square(20))(img, 0)
    assert x.shape == (3, 20, 20)
    assert (x[0, :, 0] > 0).all()  # sans crop, la bande reste presente sur toute la hauteur


def test_segmentation_eval_keeps_native_mask():
    x, m = T.Pipeline(T.square(32), native_target=True)(rgb(50, 40), left_half_mask(50, 40))
    assert x.shape == (3, 32, 32)
    assert m.shape == (40, 50) and m.dtype == torch.int64 and type(m) is torch.Tensor
    assert set(m.unique().tolist()) == {0, 2}


def test_segmentation_train_resizes_mask_with_nearest():
    _, m = T.Pipeline(T.square(32))(rgb(50, 40), left_half_mask(50, 40))
    assert m.shape == (32, 32) and m.dtype == torch.int64
    assert set(m.unique().tolist()) <= {0, 2}  # plus proche voisin : aucune valeur interpolee


def test_mask_resize_matches_pil_nearest():
    """Le plus proche voisin de `Resize` est celui de PIL (l'ancien chemin), pixel pour pixel."""
    from torchvision.transforms.v2 import functional as TF

    mask = left_half_mask(97, 61)
    mask[10:30, 40:70] = 1
    pil = Image.fromarray(mask.numpy(), mode="L")
    expected = np.array(TF.resize(pil, [32, 32], interpolation=TF.InterpolationMode.NEAREST))
    assert np.array_equal(T.Resize([32, 32])(mask).numpy(), expected)


def test_flip_moves_image_and_mask_together():
    img, mask = rgb(40, 40), left_half_mask(40, 40)
    x0, m0 = T.Pipeline(T.square(32))(img, mask)
    x1, m1 = T.Pipeline(T.square(32), FLIP)(img, mask)
    assert torch.equal(x1, x0.flip(-1)) and torch.equal(m1, m0.flip(-1))


def test_detection_box_is_normalized_after_resize():
    x, t = T.Pipeline(T.square(32))(rgb(80, 60), det_target([8, 12, 32, 54], 80, 60, label=1))
    assert x.shape == (3, 32, 32)
    assert torch.allclose(t["box"], torch.tensor([0.1, 0.2, 0.4, 0.9]), atol=1e-6)
    assert t["box"].shape == (4,) and t["label"].item() == 1


def test_integer_boxes_are_refused_instead_of_silently_rounded():
    with pytest.raises(TypeError, match="float32"):
        T.Pipeline(T.square(32))(
            rgb(80, 60), det_target([8, 12, 32, 54], 80, 60, dtype=torch.int64)
        )


def test_detection_flip_mirrors_box_and_does_not_mutate_input():
    target = det_target([8, 12, 32, 54], 80, 60)
    x0, _ = T.Pipeline(T.square(32))(rgb(80, 60), target)
    x1, t1 = T.Pipeline(T.square(32), FLIP)(rgb(80, 60), target)
    assert torch.allclose(t1["box"], torch.tensor([0.6, 0.2, 0.9, 0.9]), atol=1e-6)
    assert torch.equal(x1, x0.flip(-1))
    assert target["box"].tolist() == [[8, 12, 32, 54]]
    assert t1["box"][0] < t1["box"][2]  # le retournement reste une boite valide


def test_hflip_flips_about_half_the_time():
    torch.manual_seed(0)
    pipeline = T.Pipeline(T.square(16), T.hflip())
    target = det_target([0, 0, 4, 16], 16, 16)
    flips = sum(pipeline(rgb(16, 16), target)[1]["box"][0].item() > 0.5 for _ in range(400))
    assert 140 < flips < 260


def test_variants_enumerate_every_version_deterministically():
    pipeline = T.Pipeline(T.square(16), T.hflip())
    versions = pipeline.variants()
    assert list(versions) == ["", "flip"]
    img, target = rgb(16, 16), det_target([0, 0, 4, 16], 16, 16)
    for _ in range(5):  # chaque version est deterministe
        assert versions[""](img, target)[1]["box"][0].item() == 0.0
        assert versions["flip"](img, target)[1]["box"][0].item() == 0.75


def test_variants_without_augmentation_is_the_pipeline_itself():
    pipeline = T.Pipeline(T.center_crop(16))
    assert pipeline.variants() == {"": pipeline}


def test_non_enumerable_augmentation_is_refused_by_the_cache():
    with pytest.raises(NotImplementedError, match="Enumerate"):
        T.Pipeline(T.square(16), v2.RandomResizedCrop(16)).variants()


def test_fingerprint_changes_with_every_part_of_the_pipeline():
    base = T.Pipeline(T.square(32))
    others = [
        T.Pipeline(T.square(64)),
        T.Pipeline(T.center_crop(32)),
        T.Pipeline(T.square(32), native_target=True),
        T.Pipeline(T.square(32), T.hflip()),
        T.Pipeline(T.square(32), T.Enumerate({"": v2.Identity(), "v": v2.RandomVerticalFlip(1.0)})),
    ]
    prints = [p.fingerprint() for p in (base, *others)]
    assert len(set(prints)) == len(prints)
    assert T.Pipeline(T.square(32)).fingerprint() == base.fingerprint()  # stable


def test_collate_native_keeps_masks_as_list():
    batch = [(torch.zeros(3, 8, 8), torch.zeros(5, 6)), (torch.ones(3, 8, 8), torch.zeros(7, 4))]
    x, y = T.collate_native(batch)
    assert x.shape == (2, 3, 8, 8)
    assert isinstance(y, list) and [t.shape for t in y] == [(5, 6), (7, 4)]


def test_repr_shows_every_step_and_its_parameters():
    text = repr(T.Pipeline(T.square(32), T.hflip(), native_target=True))
    for part in ("Resize(size=[32, 32]", "masks=nearest_exact", "float32", "'flip'", "origine"):
        assert part in text, part
