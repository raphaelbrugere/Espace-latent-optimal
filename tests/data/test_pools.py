"""Pools de pre-entrainement (configs/data/) : chaque YAML est coherent avec le code (fonction
source, tache et classes de la sonde, resolution) et se compose dans `lo pretrain`."""

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_method
from omegaconf import OmegaConf

from latentopt.config import register_resolvers
from latentopt.data.paths import PROJECT_ROOT
from latentopt.evaluation import tasks

CONFIGS = PROJECT_ROOT / "configs"
POOLS = sorted(p.stem for p in (CONFIGS / "data").glob("*.yaml"))
register_resolvers()


def pool(name):
    return OmegaConf.load(CONFIGS / "data" / f"{name}.yaml")


def compose_pretrain(overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="pretrain", overrides=overrides)


def test_every_dataset_has_a_pool():
    assert set(POOLS) == {"imagenette", "flowers102", "galaxy10", "oxford_pets", "synthetic"}


@pytest.mark.parametrize("name", POOLS)
def test_pool_declares_exactly_its_keys(name):
    assert set(pool(name)) == {"source", "img_size", "num_probe_classes", "probe_task"}


@pytest.mark.parametrize("name", POOLS)
def test_pool_source_is_the_pretraining_function_of_its_dataset(name):
    target = pool(name).source._target_
    assert target == f"latentopt.data.datasets.{name}.DATASET.pretraining_images"
    assert callable(get_method(target))


@pytest.mark.parametrize("name", POOLS)
def test_probe_matches_a_classification_task(name):
    cfg = pool(name)
    if cfg.probe_task is None:  # pas de sonde : pas de classes non plus
        assert cfg.num_probe_classes is None
        return
    task = tasks.get_task(cfg.probe_task)
    assert task.default_head == "linear", "la sonde en ligne est une classification"
    assert cfg.num_probe_classes == task.num_classes


@pytest.mark.parametrize("name", POOLS)
def test_pool_resolution_is_the_protocol_one(name):
    expected = tasks.SMOKE_SIZE if name == "synthetic" else tasks.SIZE
    assert pool(name).img_size == expected


@pytest.mark.parametrize("name", POOLS)
def test_pool_composes_for_pretraining(name):
    cfg = compose_pretrain(["backbone=vit_s8", f"data={name}"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["method"]["model"]["num_probe_classes"] == pool(name).num_probe_classes
    assert resolved["backbone"]["img_size"] == pool(name).img_size


def test_pretraining_without_probe_has_no_validation():
    """Pool sans tache de classification (Pets) : ni sonde, ni DataLoader de validation."""
    from latentopt.cli.run.pretrain import build

    cfg = compose_pretrain([
        "backbone=vit_s8", "backbone.arch=vit_tiny_patch16_224", "backbone.embed_dim=24",
        "backbone.blocks=[0,1]", "data=synthetic", "data.img_size=32", "data.probe_task=null",
        "data.num_probe_classes=null", "env=ci",
    ])  # fmt: skip
    module, train, val = build(cfg)
    assert val is None and module.model.probe is None
    assert len(train.dataset) > 0
