"""Galaxy10 : des splits figes sur une autre revision HF sont refuses, par la tache comme
par le pool de pre-entrainement (leurs numeros de ligne designeraient d'autres images)."""

import json

import pytest

from latentopt.data import splits
from latentopt.data.datasets import galaxy10


@pytest.fixture
def other_revision(tmp_path, monkeypatch):
    """JSON de splits galaxy10 fige sur une autre revision ; aucune donnee sur disque."""
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path)
    payload = {"dataset": "galaxy10", "revision": "autre", "splits": {"train": ["0"]}}
    (tmp_path / "galaxy10.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path / "data"


def test_task_refuses_splits_of_another_revision(other_revision):
    with pytest.raises(ValueError, match="revision autre"):
        galaxy10.DATASET.build("train", lambda img, y: (img, y), other_revision)


def test_pretraining_pool_refuses_splits_of_another_revision(other_revision):
    with pytest.raises(ValueError, match="revision autre"):
        galaxy10.Galaxy10Pool(other_revision)


def test_matching_revision_passes_the_check(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path)
    payload = {"dataset": "galaxy10", "revision": galaxy10.REVISION, "splits": {"train": ["0"]}}
    (tmp_path / "galaxy10.json").write_text(json.dumps(payload), encoding="utf-8")
    galaxy10.DATASET.check_splits(tmp_path / "data")  # ne leve pas
