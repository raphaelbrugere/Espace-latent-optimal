"""Registre des datasets : complet, coherent, et classes meres qui refusent un dataset incomplet."""

import importlib
from pathlib import Path

import pytest

from latentopt.data import datasets
from latentopt.data.datasets.base import DatasetSpec, DownloadableDataset, Source
from latentopt.data.datasets.registry import ALL, BY_NAME, DOWNLOADABLE

NOT_DATASETS = {"__init__", "base", "registry"}


def test_every_dataset_module_is_registered():
    """Un module de datasets/ oublie dans `ALL` serait invisible de `lo data prepare`."""
    folder = Path(datasets.__file__).parent
    names = sorted(p.stem for p in folder.glob("*.py") if p.stem not in NOT_DATASETS)
    modules = [importlib.import_module(f"latentopt.data.datasets.{n}") for n in names]
    assert {id(m.DATASET) for m in modules} == {id(d) for d in ALL}


def test_names_are_unique():
    assert len(BY_NAME) == len(ALL)


def test_prepare_offers_downloadable_datasets_only():
    assert set(DOWNLOADABLE) == {"imagenette", "flowers102", "galaxy10", "oxford_pets"}


@pytest.mark.parametrize("dataset", ALL, ids=lambda d: d.NAME)
def test_task_kinds_are_known(dataset):
    assert dataset.tasks() and set(dataset.tasks()) <= {"cls", "seg", "det"}


def test_a_dataset_missing_a_method_cannot_be_instantiated():
    class Incomplete(DownloadableDataset):
        NAME, ABBREV = "x", "x"

        def pretraining_images(self):
            return []

        def tasks(self):
            return {}

    with pytest.raises(TypeError, match="present"):
        Incomplete()


def test_a_dataset_missing_its_names_cannot_be_instantiated():
    class Nameless(DatasetSpec):
        def pretraining_images(self):
            return []

        def tasks(self):
            return {}

    with pytest.raises(TypeError, match="NAME, ABBREV"):
        Nameless()


def keep(img, target):
    return img, target


def test_reprs_say_what_the_object_holds():
    synthetic = BY_NAME["synthetic"]
    assert repr(synthetic) == "Synthetic(name='synthetic', abbrev='synth', tasks=[cls, seg, det])"
    subset = synthetic.tasks()["seg"]("val", keep)
    assert repr(subset) == "Subset(BlobsSource(150 images), 150 images, transform=keep)"


@pytest.mark.parametrize("dataset", ALL, ids=lambda d: d.NAME)
def test_sources_declare_their_target_format(dataset):
    """Chaque source herite de ClsSource / SegSource / DetSource : le format de sa cible."""
    module = importlib.import_module(type(dataset).__module__)
    classes = [c for c in vars(module).values() if isinstance(c, type)]
    sources = [c for c in classes if c.__module__ == module.__name__ and c.__name__.endswith(
        ("Source", "Pool"))]  # fmt: skip
    assert sources and all(Source in c.__mro__ for c in sources)


def test_synthetic_sources_have_ids():
    synthetic = BY_NAME["synthetic"]
    for kind, build in synthetic.tasks().items():
        source = build("val", keep).source
        assert source.ids() == [str(i) for i in range(len(source))], kind
