"""Splits figés : partitions exactes, déterministes, stockées par identifiant."""

import json
from collections import Counter

import numpy as np
import pytest

from latentopt.data import splits


@pytest.mark.parametrize("n", [1, 7, 20, 1000, 9469])
def test_holdout_is_a_partition(n):
    s = splits.holdout(n)
    train, val = s["train"], s["val"]
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(n))
    assert len(val) == max(1, int(n * 0.05))


def test_holdout_is_deterministic_and_depends_on_seed():
    assert splits.holdout(500, seed=0) == splits.holdout(500, seed=0)
    assert splits.holdout(500, seed=0) != splits.holdout(500, seed=1)


def test_holdout_frac():
    assert len(splits.holdout(1000, frac=0.2)["val"]) == 200


def labels_fixture():
    rng = np.random.RandomState(3)
    return rng.randint(0, 5, size=503).tolist()


def test_stratified_is_a_sorted_partition():
    labels = labels_fixture()
    s = splits.stratified(labels)
    all_idx = s["train"] + s["val"] + s["test"]
    assert sorted(all_idx) == list(range(len(labels)))
    assert len(set(all_idx)) == len(labels)
    for part in s.values():
        assert part == sorted(part)


def test_stratified_respects_proportions_per_class():
    labels = labels_fixture()
    s = splits.stratified(labels, fracs=(0.8, 0.1, 0.1))
    per_class = Counter(labels)
    for c, n in per_class.items():
        train_c = sum(labels[i] == c for i in s["train"])
        val_c = sum(labels[i] == c for i in s["val"])
        assert train_c == int(n * 0.8)
        assert val_c == int(n * (0.8 + 0.1)) - int(n * 0.8)


def test_stratified_is_deterministic():
    labels = labels_fixture()
    assert splits.stratified(labels) == splits.stratified(labels)
    assert splits.stratified(labels, seed=0) != splits.stratified(labels, seed=1)


@pytest.fixture
def splits_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")
    return tmp_path / "splits"


def test_save_load_indices_roundtrip(splits_dir):
    ids = [f"img_{i:03d}.jpg" for i in range(40)]
    s = splits.holdout(len(ids), frac=0.25)
    path = splits.save("toy", s, ids, seed=0, frac=0.25)
    assert path == splits_dir / "toy.json"

    payload = splits.load("toy")
    assert payload["dataset"] == "toy" and payload["seed"] == 0 and payload["frac"] == 0.25
    # stocké par identifiant, trié
    assert payload["splits"]["val"] == sorted(ids[i] for i in s["val"])

    # robuste à un changement d'ordre de listage de la source
    shuffled = list(reversed(ids))
    for split in ("train", "val"):
        got = [shuffled[i] for i in splits.indices("toy", split, shuffled)]
        assert sorted(got) == sorted(ids[i] for i in s[split])


def test_indices_missing_ids_raise(splits_dir):
    ids = [str(i) for i in range(10)]
    splits.save("toy", splits.holdout(10), ids)
    with pytest.raises(KeyError, match="introuvables"):
        splits.indices("toy", "train", ids[:3])


def test_freeze_keeps_an_existing_file_unless_forced(splits_dir):
    calls = []

    def compute(seed):
        def inner():
            calls.append(seed)
            ids = [str(i) for i in range(20)]
            return splits.holdout(20, seed=seed), ids, {"seed": seed}

        return inner

    path = splits.freeze("toy", compute(0))
    first = path.read_text(encoding="utf-8")
    # existant : rien n'est recalculé ni réécrit, même si le calcul donnerait autre chose
    assert splits.freeze("toy", compute(1)) == path
    assert path.read_text(encoding="utf-8") == first and calls == [0]
    # force : réécrit
    splits.freeze("toy", compute(1), force=True)
    assert path.read_text(encoding="utf-8") != first and calls == [0, 1]


def test_load_missing_split_file_explains_command(splits_dir):
    with pytest.raises(FileNotFoundError, match="lo data prepare"):
        splits.load("absent")


# Splits commités : un changement involontaire casse la comparabilité avec les scores existants.


def committed():
    return sorted(splits.SPLITS_DIR.glob("*.json"))


@pytest.mark.parametrize("path", committed(), ids=lambda p: p.stem)
def test_committed_splits_are_disjoint_sorted_and_unique(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"] == path.stem
    parts = payload["splits"]
    seen = set()
    for name, ids in parts.items():
        assert ids == sorted(ids), f"{name} non trié"
        assert len(ids) == len(set(ids)), f"{name} contient des doublons"
        assert not seen & set(ids), f"{name} recouvre un autre split"
        seen |= set(ids)


def test_committed_imagenette_holdout_matches_its_metadata():
    payload = splits.load("imagenette")
    assert payload["pool"] == "train" and payload["seed"] == 0 and payload["frac"] == 0.05
    n_train, n_val = len(payload["splits"]["train"]), len(payload["splits"]["val"])
    assert n_val == max(1, int((n_train + n_val) * payload["frac"]))
    # Imagenette 160px : 9469 images dans le train officiel
    assert n_train + n_val == 9469
