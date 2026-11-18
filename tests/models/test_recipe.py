"""Configs nommées (configs/neck/, configs/head/) : chargées et construites sur un backbone."""

import pytest
import torch
from omegaconf import OmegaConf

from latentopt.evaluation.tasks import TASKS
from latentopt.models import recipe
from latentopt.models.contract import Downstream

NECKS = sorted(p.stem for p in (recipe.CONFIGS / "neck").glob("*.yaml"))
HEADS = sorted(p.stem for p in (recipe.CONFIGS / "head").glob("*.yaml"))


def test_every_task_default_exists():
    assert {t.default_neck for t in TASKS.values()} <= set(NECKS)
    assert {t.default_head for t in TASKS.values()} <= set(HEADS)


@pytest.mark.parametrize(
    ("group", "name"), [("neck", n) for n in NECKS] + [("head", h) for h in HEADS]
)
def test_each_config_targets_its_module(group, name):
    assert f".{group}s." in recipe.load(group, name)._target_


@pytest.mark.parametrize("task", sorted(TASKS))
def test_each_default_pair_builds_on_the_backbone_and_runs(pyramid, task):
    t = TASKS[task]
    head = recipe.build(recipe.load("neck", t.default_neck), recipe.load("head", t.default_head),
                        pyramid, t.num_classes)  # fmt: skip
    assert isinstance(head, Downstream)
    with torch.no_grad():
        out = head(pyramid(torch.randn(2, 3, 32, 32)))
    assert out is not None


def test_unknown_config_lists_the_available_ones():
    with pytest.raises(FileNotFoundError, match="neck inconnu 'absent'.*fpn"):
        recipe.load("neck", "absent")
    with pytest.raises(ValueError, match="groupe"):
        recipe.load("recette", "fpn")


def test_incompatible_neck_and_head_fail_early(pyramid):
    """Un neck qui sort une carte et une tête qui attend un vecteur : refusé à la construction."""
    with pytest.raises(ValueError, match="attend un 'vector'"):
        recipe.build(recipe.load("neck", "fpn"), recipe.load("head", "linear"), pyramid, 3)


def test_command_line_override_reaches_the_neck(pyramid):
    """`neck.channels=8` : même neck, plus étroit."""
    cfg = OmegaConf.merge(recipe.load("neck", "fpn"), {"channels": 8})
    assert recipe.build_neck(cfg, pyramid).out_dim == 8
