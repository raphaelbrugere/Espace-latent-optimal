"""Noms courts des datasets (ABBREV) : uniques, valides dans un nom de run, et figés."""

import re

import pytest
from omegaconf import OmegaConf

from latentopt.data.datasets import flowers102, galaxy10, imagenette, oxford_pets, synthetic
from latentopt.data.datasets.base import abbrev
from latentopt.data.paths import PROJECT_ROOT

CONFIGS = PROJECT_ROOT / "configs" / "data"

MODULES = {
    "imagenette": imagenette,
    "flowers102": flowers102,
    "galaxy10": galaxy10,
    "oxford_pets": oxford_pets,
    "synthetic": synthetic,
}

# Un nom court est un identifiant dans des dépôts publiés : le changer renomme les runs.
# Si ce test échoue, c'est qu'un nom a changé ; l'actualiser ici doit être un choix assumé.
PINNED = {
    "imagenette": "imnt160",
    "flowers102": "fl102",
    "galaxy10": "gal10d",
    "oxford_pets": "pets",
    "synthetic": "synth",
}


def test_abbreviations_are_unique():
    abbrevs = [m.ABBREV for m in MODULES.values()]
    assert len(set(abbrevs)) == len(abbrevs), f"noms courts en double : {abbrevs}"


@pytest.mark.parametrize("name", MODULES)
def test_abbreviation_is_a_valid_name_part(name):
    abbrev = MODULES[name].ABBREV
    assert re.fullmatch(r"[a-z0-9]+", abbrev), f"{name} : {abbrev!r}"  # ni « - » ni « _ »
    assert len(abbrev) <= 8, f"{name} : {abbrev!r} trop long pour être un nom court"


def test_abbreviations_are_pinned():
    assert {name: m.ABBREV for name, m in MODULES.items()} == PINNED


def test_imagenette_abbreviation_follows_its_size():
    assert imagenette.ABBREV == f"imnt{imagenette.SIZE.removesuffix('px')}"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_pool_names_its_runs_with_its_abbreviation(path):
    """Le <données> d'un nom de run vient du module du pool (data.source._target_)."""
    target = OmegaConf.load(path).source._target_
    module = target.rpartition(".")[0].rpartition(".")[2]
    assert abbrev(target) == MODULES[module].ABBREV
