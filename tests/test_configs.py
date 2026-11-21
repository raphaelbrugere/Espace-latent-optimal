"""Configs Hydra : chaque point d'entree et chaque experience se compose et se resout ;
la separation " cluster = environnement, methode = resultat " est respectee."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from latentopt.config import register_resolvers
from latentopt.data.paths import PROJECT_ROOT

CONFIGS = PROJECT_ROOT / "configs"
register_resolvers()


def load(name: str, overrides: list[str]):
    """Config composee et entierement resolue (hors noeud `hydra`, defini a l'execution)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose(config_name=name, overrides=overrides)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def experiments() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("cluster", ["local", "ci", "dce"])
def test_pretrain_composes_on_every_cluster(cluster):
    cfg = load("pretrain", ["backbone=vit_s8", f"cluster={cluster}"])
    assert cfg.method.name == "lejepa" and cfg.cluster.name == cluster


@pytest.mark.parametrize("exp", [e for e in experiments() if e != "smoke_train"])
def test_pretrain_experiments_compose(exp):
    cfg = load("pretrain", [f"experiment={exp}"])
    assert cfg.experiment_name == exp  # nom du dossier de resultats


def test_smoke_pretrain_is_tiny_and_synthetic():
    cfg = load("pretrain", ["experiment=smoke", "cluster=ci"])
    assert cfg.data.source._target_.endswith("synthetic.pretraining_images")
    assert cfg.method.epochs <= 2 and cfg.method.max_steps_per_epoch <= 5
    assert cfg.method.precision == "fp32" and cfg.cluster.device == "cpu"


def test_smoke_train_composes():
    cfg = load("train", ["experiment=smoke_train", "cluster=ci"])
    assert cfg.task.startswith("synthetic_") and cfg.precision == "fp32"


@pytest.mark.parametrize("task", ["imagenette_cls", "pets_seg", "pets_det"])
def test_train_composes_for_protocol_tasks(task):
    cfg = load("train", ["backbone=vit_s8", f"task={task}"])
    assert cfg.get("head") is None and cfg.epochs is None  # recette de la baseline, remplie au run


def test_eval_composes():
    cfg = load("eval", ["downstream=[a,b]"])
    assert cfg.split == "val" and list(cfg.downstream) == ["a", "b"]


ENV_KEYS = {
    "name", "data_root", "results_root", "mlflow", "num_workers", "device", "progress_bar",
    "grad_checkpointing", "compile",  # resultat identique au bruit numerique pres (docs/CONFIGS.md)
}  # fmt: skip


@pytest.mark.parametrize("path", sorted((CONFIGS / "cluster").glob("*.yaml")), ids=lambda p: p.stem)
def test_cluster_configs_only_hold_environment(path: Path):
    """Rien dans cluster/ ne doit changer le resultat d'un run (hors reglages Slurm)."""
    cfg = OmegaConf.load(path)
    assert set(cfg.cluster) <= ENV_KEYS
    assert set(cfg) <= {"cluster", "defaults", "hydra"}


def test_backbone_config_instantiates_contract_backbone():
    from hydra.utils import instantiate

    from latentopt.models.contract import check_backbone

    cfg = load("train", ["backbone=vit_s8", "task=synthetic_cls", "backbone.img_size=32"])
    check_backbone(instantiate(cfg.backbone), size=32)
