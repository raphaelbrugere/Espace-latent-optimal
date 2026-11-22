"""Configs Hydra : chaque point d'entree et chaque experience se compose et se resout ;
la separation " env = machine, slurm/ = ressources Slurm, le reste = resultat " est respectee."""

import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from latentopt.config import register_resolvers
from latentopt.data.paths import PROJECT_ROOT

CONFIGS = PROJECT_ROOT / "configs"
LAUNCHERS = PROJECT_ROOT / "slurm" / "launcher"
register_resolvers()


def load(name: str, overrides: list[str]):
    """Config composee et entierement resolue (hors noeud `hydra`, defini a l'execution)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose(config_name=name, overrides=overrides)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def launcher_of(name: str, overrides: list[str]):
    """Noeud hydra.launcher de la config composee (choisi par launcher=, avec -m)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose(config_name=name, overrides=overrides, return_hydra_config=True)
    return cfg.env.name, cfg.hydra.launcher


def experiments() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("env", ["local", "ci", "dce"])
def test_pretrain_composes_on_every_env(env):
    cfg = load("pretrain", ["backbone=vit_s8", f"env={env}"])
    assert cfg.method.name == "lejepa" and cfg.env.name == env


ENTRY_POINTS = {
    "pretrain": ["backbone=vit_s8"],
    "train": ["backbone=vit_s8", "task=pets_seg"],
    "eval": ["downstream=[a]"],
}


@pytest.mark.parametrize("name", ENTRY_POINTS)
@pytest.mark.parametrize("profile", ["long", "short"])
def test_slurm_profiles_compose_and_select_dce(name, profile):
    env, launcher = launcher_of(name, [*ENTRY_POINTS[name], f"launcher={profile}"])
    assert env == "dce" and launcher._target_.endswith("SlurmLauncher")
    assert launcher.gpus_per_node is None  # pas de gres sur le DCE : sbatch refuserait le job
    assert launcher.exclude == "sh[10-16]"  # GTX 1080 Ti


def test_long_profile_only_serves_bf16_gpus():
    _, launcher = launcher_of("pretrain", ["experiment=vit_lejepa_imagenette", "launcher=long"])
    assert launcher.partition == "gpu_prod_long" and launcher.timeout_min == 2880


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_no_profile_runs_locally(name):
    env, launcher = launcher_of(name, ENTRY_POINTS[name])
    assert env == "local" and launcher._target_.endswith("BasicLauncher")


@pytest.mark.parametrize("exp", [e for e in experiments() if e != "smoke_train"])
def test_pretrain_experiments_compose(exp):
    cfg = load("pretrain", [f"experiment={exp}"])
    assert cfg.experiment_name == exp  # nom du dossier de resultats


def test_smoke_pretrain_is_tiny_and_synthetic():
    cfg = load("pretrain", ["experiment=smoke", "env=ci"])
    assert cfg.data.source._target_.endswith("synthetic.pretraining_images")
    assert cfg.method.epochs <= 2 and cfg.method.max_steps_per_epoch <= 5
    assert cfg.method.precision == "fp32" and cfg.env.device == "cpu"


def test_smoke_train_composes():
    cfg = load("train", ["experiment=smoke_train", "env=ci"])
    assert cfg.task.startswith("synthetic_") and cfg.precision == "fp32"


@pytest.mark.parametrize(("name", "exp"), [("pretrain", "smoke"), ("train", "smoke_train")])
def test_smoke_configs_skip_the_dirty_tree_guard(name, exp):
    # seule exemption du garde-fou git sur le DCE (run.check_launch)
    assert load(name, [f"experiment={exp}", "env=dce"]).allow_dirty is True


@pytest.mark.parametrize("task", ["imagenette_cls", "pets_seg", "pets_det"])
def test_train_composes_for_protocol_tasks(task):
    cfg = load("train", ["backbone=vit_s8", f"task={task}"])
    assert cfg.get("head") is None and cfg.epochs is None  # remplis au run


def test_eval_composes():
    cfg = load("eval", ["downstream=[a,b]"])
    assert cfg.split == "val" and list(cfg.downstream) == ["a", "b"]


ENV_KEYS = {
    "name", "data_root", "results_root", "mlflow", "num_workers", "device", "progress_bar",
    "grad_checkpointing", "compile",  # resultat identique au bruit numerique pres (docs/CONFIGS.md)
}  # fmt: skip


@pytest.mark.parametrize("path", sorted((CONFIGS / "env").glob("*.yaml")), ids=lambda p: p.stem)
def test_env_configs_only_hold_environment(path: Path):
    """Rien dans env/ ne doit changer le resultat d'un run, ni demander des ressources Slurm."""
    cfg = OmegaConf.load(path)
    assert set(cfg.env) <= ENV_KEYS
    assert set(cfg) <= {"env", "defaults"}


@pytest.mark.parametrize("path", sorted(LAUNCHERS.glob("*.yaml")), ids=lambda p: p.stem)
def test_slurm_profiles_only_hold_resources(path: Path):
    """Un profil de slurm/launcher/ ne choisit que l'environnement et les ressources du job."""
    assert set(OmegaConf.load(path)) <= {"defaults", "hydra"}


SLURM_TERMS = ("submitit", "hydra/launcher", "gpu_prod", "gpu_inter", "sbatch", "srun", "SBATCH")


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.stem)
def test_configs_say_nothing_about_slurm(path: Path):
    """Partitions, noeuds et launcher : dans slurm/, jamais dans configs/ (slurm/README.md)."""
    text = path.read_text(encoding="utf-8")
    assert not [t for t in SLURM_TERMS if t in text]


RECIPE_GROUPS = ("backbone", "method", "data", "neck", "head", "experiment")


@pytest.mark.parametrize(
    "path", sorted(p for g in RECIPE_GROUPS for p in (CONFIGS / g).glob("*.yaml")),
    ids=lambda p: f"{p.parent.name}/{p.stem}",
)  # fmt: skip
def test_recipe_does_not_read_env(path: Path):
    """Une config ne depend pas de la machine : aucune interpolation vers env.*"""
    assert not re.search(r"\$\{[^}]*\benv\.", path.read_text(encoding="utf-8"))


def test_backbone_config_instantiates_contract_backbone():
    from hydra.utils import instantiate

    from latentopt.models.contract import check_backbone

    cfg = load("train", ["backbone=vit_s8", "task=synthetic_cls", "backbone.img_size=32"])
    check_backbone(instantiate(cfg.backbone), size=32)


# --- noms des dossiers de run (hydra.sweep.subdir), resolus depuis les yaml ---------------

DATE = "20261116_183012"
BB = f"imnt160-vit_s8-lejepa-{DATE}"


def sweep_subdir(name: str, overrides: list[str], num: int | None = None) -> str:
    """hydra.sweep.subdir tel que run_job le resout (config complete, noeud hydra compris)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose(config_name=name, overrides=overrides, return_hydra_config=True)
    if num is not None:
        cfg.hydra.job.num = num  # job n d'un balayage -m
    return cfg.hydra.sweep.subdir


@pytest.fixture
def fixed_stamp(monkeypatch):
    from latentopt import hub

    monkeypatch.setattr(hub, "_process_stamp", lambda: DATE)


@pytest.fixture
def pretrained(tmp_path):
    (tmp_path / BB / "backbone").mkdir(parents=True)
    return (tmp_path / BB).as_posix()


NONE_BB = f"none-vit_s8-none-{DATE}/downstream"


@pytest.mark.parametrize(
    "overrides, num, expected",
    [
        # neck et tete de la baseline de la tache
        (["task=pets_seg"], None, f"{NONE_BB}/pets_seg-fpn-conv_seg-{DATE}"),
        # neck et tete choisis : une inversion des arguments 7 et 8 changerait le nom
        (["task=pets_seg", "neck=embed_norm", "head=linear"], None,
         f"{NONE_BB}/pets_seg-embed_norm-linear-{DATE}"),
        # neck repris : son nom vient du dossier neck_from
        (["task=pets_det", "head=soft_argmax", f"neck_from=pets_seg-fpn-conv_seg-{DATE}"], None,
         f"{NONE_BB}/pets_det-fpn-soft_argmax-{DATE}"),
        # backbone externe : donnees et methode dans le nom, ou backbone_run impose
        (["backbone=vit_s8_dino", "backbone_data=imagenet1k", "backbone_method=dino",
          "task=pets_seg"], None,
         f"imagenet1k-vit_s8_dino-dino-{DATE}/downstream/pets_seg-fpn-conv_seg-{DATE}"),
        (["backbone=vit_s8_dino", f"backbone_run={BB}", "task=pets_seg"], None,
         f"{BB}/downstream/pets_seg-fpn-conv_seg-{DATE}"),
        # job 2 d'un balayage : suffixe sur la date du dossier d'entrainement
        (["task=pets_det"], 2, f"{NONE_BB}/pets_det-conv_trunk-soft_argmax-{DATE}_2"),
    ],
)  # fmt: skip
def test_train_dir_from_config(fixed_stamp, overrides, num, expected):
    defaults = [] if any(o.startswith("backbone=") for o in overrides) else ["backbone=vit_s8"]
    assert sweep_subdir("train", [*defaults, *overrides], num) == expected


def test_train_dir_from_config_of_a_pretrained_backbone(fixed_stamp, pretrained):
    overrides = ["backbone=from_run", f"backbone.run={pretrained}", "task=pets_seg"]
    assert sweep_subdir("train", overrides) == f"{BB}/downstream/pets_seg-fpn-conv_seg-{DATE}"


@pytest.mark.parametrize("num, suffix", [(None, ""), (1, "_1")])
def test_pretrain_dir_from_config(fixed_stamp, num, suffix):
    assert sweep_subdir("pretrain", ["backbone=vit_s8"], num) == f"{BB}{suffix}/backbone"


def test_eval_log_dir_has_the_date_of_the_eval_folders(fixed_stamp):
    """Le journal results/.eval/<date>/ et les dossiers eval/<split>-<date>/ partagent une date."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose(config_name="eval", overrides=["downstream=x"], return_hydra_config=True)
    assert cfg.hydra.run.dir.endswith(f"/.eval/{DATE}")
