"""Datasets synthetiques (smoke run, CI) : deterministes et au format des vrais datasets."""

import numpy as np
import pytest
import torch
from PIL import Image

from latentopt.data.datasets import synthetic
from latentopt.data.datasets.base import Subset, require


def pixels(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


@pytest.mark.parametrize(
    "make",
    [
        lambda seed: synthetic.Shapes(20, 10, seed),
        lambda seed: synthetic.Blobs(20, seed),
        lambda seed: synthetic.Heads(20, seed),
    ],
    ids=["shapes", "blobs", "heads"],
)
def test_sources_are_deterministic_per_index_and_seed(make):
    a, b, other = make(0), make(0), make(1)
    assert len(a) == 20
    img_a, img_b, img_o = a[5][0], b[5][0], other[5][0]
    assert np.array_equal(pixels(img_a), pixels(img_b))
    assert not np.array_equal(pixels(img_a), pixels(img_o))
    # l'acces est sans etat : relire un indice redonne la meme image
    assert np.array_equal(pixels(a[5][0]), pixels(img_a))


def test_shapes_labels_and_format():
    src = synthetic.Shapes(30, 10, seed=0, size=40)
    labels = [src[i][1] for i in range(len(src))]
    assert labels == [i % 10 for i in range(30)]
    img = src[0][0]
    assert img.mode == "RGB" and img.size == (40, 40)


def test_blobs_mask_is_aligned_and_has_three_classes():
    src = synthetic.Blobs(10, seed=0)
    for i in range(len(src)):
        img, mask = src[i]
        assert img.size == mask.size == (96, 72)  # non carre, comme les vrais masques
        assert mask.mode == "L"
        assert set(np.unique(np.asarray(mask))) <= {0, 1, 2}


def test_heads_box_is_valid_and_normalized():
    src = synthetic.Heads(50, seed=0)
    for i in range(len(src)):
        _, target = src[i]
        box = target["box"]
        assert box.shape == (4,) and box.dtype == torch.float32
        assert (box >= 0).all() and (box <= 1).all()
        assert box[0] < box[2] and box[1] < box[3]
        assert target["label"] == i % 2


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_builders_lengths(split):
    identity = lambda img, y: (img, y)  # noqa: E731
    n = synthetic.SIZES[split]
    assert len(synthetic.build_cls(split, identity)) == n
    assert len(synthetic.build_seg(split, identity)) == n
    assert len(synthetic.build_det(split, identity)) == n


def test_splits_use_distinct_seeds():
    assert len(set(synthetic.SEEDS.values())) == len(synthetic.SEEDS)


def test_subset_applies_indices_and_joint_transform():
    src = synthetic.Shapes(10, 10, seed=0)
    sub = Subset(src, [3, 7], lambda img, y: (img.size, y * 10))
    assert len(sub) == 2
    assert sub[0] == ((80, 80), 30) and sub[1] == ((80, 80), 70)


def test_require_gives_the_command_to_run(tmp_path):
    require(tmp_path, "x")  # existe : rien
    with pytest.raises(FileNotFoundError, match="lo data prepare imagenette"):
        require(tmp_path / "absent", "imagenette")
