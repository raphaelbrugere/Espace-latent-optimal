"""Noms courts des datasets (ABBREV) : uniques, valides dans un nom de run, et figes."""

import re

import pytest
from omegaconf import OmegaConf

from latentopt.data.datasets import imagenette
from latentopt.data.datasets.base import abbrev
from latentopt.data.datasets.registry import BY_NAME as DATASETS
from latentopt.data.paths import PROJECT_ROOT

CONFIGS = PROJECT_ROOT / "configs" / "data"

# Un nom court est un identifiant dans des depots publies : le changer renomme les runs.
# Si ce test echoue, c'est qu'un nom a change ; l'actualiser ici doit etre un choix assume.
PINNED = {
    "imagenette": "imnt160",
    "flowers102": "fl102",
    "galaxy10": "gal10d",
    "oxford_pets": "pets",
    "synthetic": "synth",
}


def test_abbreviations_are_unique():
    abbrevs = [d.ABBREV for d in DATASETS.values()]
    assert len(set(abbrevs)) == len(abbrevs), f"noms courts en double : {abbrevs}"


@pytest.mark.parametrize("name", DATASETS)
def test_abbreviation_is_a_valid_name_part(name):
    abbrev = DATASETS[name].ABBREV
    assert re.fullmatch(r"[a-z0-9]+", abbrev), f"{name} : {abbrev!r}"  # ni " - " ni " _ "
    assert len(abbrev) <= 8, f"{name} : {abbrev!r} trop long pour etre un nom court"


def test_abbreviations_are_pinned():
    assert {name: d.ABBREV for name, d in DATASETS.items()} == PINNED


def test_imagenette_abbreviation_follows_its_size():
    assert imagenette.DATASET.ABBREV == f"imnt{imagenette.SIZE.removesuffix('px')}"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_pool_names_its_runs_with_its_abbreviation(path):
    """Le <donnees> d'un nom de run vient du dataset du pool (data.source._target_)."""
    target = OmegaConf.load(path).source._target_
    assert abbrev(target) == DATASETS[path.stem].ABBREV


def test_abbrev_rejects_a_target_that_is_not_a_dataset_method():
    with pytest.raises(TypeError, match="DatasetSpec"):
        abbrev("latentopt.data.datasets.imagenette.ImagenetteSource")
