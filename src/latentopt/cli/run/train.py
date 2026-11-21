"""Entrainement d'un neck et d'une tete sur un backbone gele (configs/train.yaml).

    lo train backbone=from_run backbone.run=<run> task=pets_seg
    lo train backbone=from_run backbone.run=<run> task=imagenette_cls,pets_seg -m
    lo train backbone=from_run backbone.run=<run> task=pets_det \
        neck_from=pets_seg-fpn-conv_seg-20261117_090000

Le run ecrit dans results/<run>/downstream/<tache>-<neck>-<tete>-<AAAAMMJJ_HHMMSS>/
(latentopt/hub.py) : neck.pt (poids, empreintes du backbone et du neck), head.pt (poids de
la tete seule, empreintes du backbone et du neck, graine), README.md, results.json (scores
val, parametres du neck et de la tete separes), plus le dossier de run commun ; suivi dans
MLflow (train/<experiment_name>). Avec `neck_from`, pas de neck.pt : le neck repris reste
gele, la tete seule apprend, et head.pt et results.json designent le dossier du neck.

Par defaut (cache_features=true), les features du backbone sont extraites une fois dans
data/features/ (evaluation/features.py), puis neck et tete s'entrainent dessus.
"""

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from latentopt import hub, run
from latentopt.config import CONFIGS_DIR, register_resolvers
from latentopt.evaluation.features import CachedFeatures, collate, load_or_extract
from latentopt.evaluation.tasks import Task, get_task
from latentopt.methods import FrozenBackboneModule
from latentopt.models import recipe
from latentopt.models.contract import (
    Downstream,
    count_params,
    load_neck,
    save_head,
    save_neck,
    weights_hash,
)

register_resolvers()
log = logging.getLogger("train")


def neck_dir(neck_from: str, run_dir: Path) -> Path:
    """Dossier du neck repris : `<tache>-<neck>-<tete>-<date>` dans downstream/ du backbone
    de ce run, sinon un chemin ou un nom complet (<run>/downstream/<...>, telecharge s'il
    manque). Il doit contenir neck.pt : un entrainement sur neck repris n'en a pas."""
    if "/" not in neck_from and "\\" not in neck_from:
        hub.parse_downstream(neck_from)
        folder = hub.run_root(run_dir) / hub.DOWNSTREAM / neck_from
    else:
        folder = hub.locate(neck_from)
    if not (folder / "neck.pt").exists():
        hint = ""
        if (folder / "results.json").exists():
            origin = json.loads((folder / "results.json").read_text(encoding="utf-8"))["neck"]
            hint = f" : neck repris de {origin}, le designer directement"
        raise FileNotFoundError(f"{folder / 'neck.pt'} absent{hint}")
    return folder


def fill_defaults(cfg: DictConfig) -> None:
    """neck, head et epochs absents -> ceux de la baseline de la tache (ecrits dans la config).
    Avec `neck_from` (chemin resolu), la config du neck est celle enregistree dans son neck.pt."""
    task = get_task(cfg.task)
    with open_dict(cfg):
        if cfg.get("neck_from"):
            ckpt = torch.load(Path(cfg.neck_from) / "neck.pt", map_location="cpu",
                              weights_only=False)  # fmt: skip
            cfg.neck = ckpt["config"]
        elif cfg.get("neck") is None:
            cfg.neck = recipe.load("neck", task.default_neck)
        if cfg.get("head") is None:
            cfg.head = recipe.load("head", task.default_head)
        if cfg.epochs is None:
            cfg.epochs = task.epochs


def cached_datasets(
    cfg: DictConfig, task: Task, backbone: torch.nn.Module, keys: Sequence[str]
) -> tuple[CachedFeatures, CachedFeatures]:
    """Features train (deux variantes si retournement) et val, extraites ou relues."""
    use_cuda = run.resolve_device(cfg.env.device) == "cuda"
    common = dict(
        backbone=backbone,
        sha=weights_hash(backbone),
        task=task,
        keys=keys,
        batch_size=256,
        num_workers=cfg.env.num_workers,
        device=torch.device("cuda" if use_cuda else "cpu"),
        precision=cfg.precision,
        info={"backbone": OmegaConf.to_container(cfg.backbone, resolve=True)},
    )
    train = load_or_extract(split="train", transform=task.train_transform, **common)
    val = load_or_extract(split="val", transform=task.eval_transform, **common)
    backbone.cpu()
    return (
        CachedFeatures(train, getattr(task.train_transform, "hflip", 0.0)),
        CachedFeatures(val),
    )


def build(cfg: DictConfig) -> tuple[FrozenBackboneModule, DataLoader, DataLoader]:
    """Config -> module Lightning et DataLoaders (testable).

    Ordre fixe : backbone construit, puis graine, puis neck et tete (init) ; la graine
    ne depend donc pas du backbone. L'extraction des features se fait avant la graine, pour
    ne decaler ni l'init ni l'ordre des donnees. Avec `neck_from`, le neck recoit ses poids
    enregistres (ValueError s'ils ont ete appris sur un autre backbone) et reste gele.
    """
    task = get_task(cfg.task)
    L.seed_everything(cfg.seed, workers=False)  # backbone a poids aleatoires : reproductible
    backbone = instantiate(cfg.backbone)
    if cfg.cache_features:
        # avant la graine, donc jete : seulement pour connaitre les cles lues
        keys = recipe.build_neck(cfg.neck, backbone).in_keys
        train_ds, val_ds = cached_datasets(cfg, task, backbone, keys)
        workers, val_collate = 0, collate(native_targets=task.collate_fn is not None)
        train_collate = collate(native_targets=False)
    else:
        train_ds, val_ds = task.dataset("train", train=True), task.dataset("val")
        workers, val_collate, train_collate = cfg.env.num_workers, task.collate_fn, None

    L.seed_everything(cfg.seed, workers=False)
    neck = recipe.build_neck(cfg.neck, backbone)
    frozen = bool(cfg.get("neck_from"))
    if frozen:
        load_neck(neck, Path(cfg.neck_from) / "neck.pt", backbone)
    head = Downstream(neck, recipe.build_head(cfg.head, neck, task.num_classes))
    module = FrozenBackboneModule(backbone, head, task, **cfg.optimizer, freeze_neck=frozen)
    train = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=train_collate,
    )
    val = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=workers, collate_fn=val_collate)
    return module, train, val


def names(task: Task, neck_from: str | None) -> tuple[str, str]:
    """Noms du neck et de la tete du run : choix `neck=` / `head=`, sinon ceux de la tache ;
    avec `neck_from`, le nom du neck repris."""
    choices = HydraConfig.get().runtime.choices

    def chosen(group: str, default: str) -> str:
        choice = choices.get(group)
        return choice if choice not in (None, "null", "None") else default

    neck = hub.parse_downstream(Path(neck_from).name).neck if neck_from else None
    return neck or chosen("neck", task.default_neck), chosen("head", task.default_head)


@hydra.main(version_base="1.3", config_path=CONFIGS_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    os.environ["DATA_ROOT"] = cfg.env.data_root
    os.environ["OUTPUT_ROOT"] = cfg.env.results_root  # runs designes par leur nom (hub.locate)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if cfg.get("neck_from"):
        with open_dict(cfg):
            cfg.neck_from = neck_dir(str(cfg.neck_from), run_dir).as_posix()
    fill_defaults(cfg)
    git = run.prepare(cfg, run_dir)
    settings = run.lightning_settings(cfg.env.device, cfg.precision)
    mlflow = run.mlflow_logger(cfg, "train", run_dir, git)

    module, train, val = build(cfg)
    task, head = module.task, module.head
    neck_name, head_name = names(task, cfg.get("neck_from"))
    log.info("%s : neck %s (%s, %s parametres%s), tete %s (%s, %s parametres)",
             task.name, neck_name, type(head.neck).__name__, f"{count_params(head.neck):,}",
             ", gele" if module.freeze_neck else "", head_name, type(head.head).__name__,
             f"{count_params(head.head):,}")  # fmt: skip
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.max_steps_per_epoch,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=run_dir,
        logger=[CSVLogger(save_dir=run_dir, name="", version=""), *filter(None, [mlflow])],
        callbacks=[
            ModelCheckpoint(dirpath=run_dir / "checkpoints", save_last=True, save_top_k=0),
            run.LogProgress(),
        ],
        enable_progress_bar=cfg.env.progress_bar,
        devices=1,
        **settings,
    )
    trainer.fit(module, train, val, ckpt_path=cfg.resume)

    scores = {k.removeprefix("val/"): float(v) for k, v in trainer.callback_metrics.items()
              if k.startswith("val/")}  # fmt: skip
    backbone, neck = module.backbone.cpu(), head.neck.cpu()
    if module.freeze_neck:
        folder = Path(cfg.neck_from)
        neck_sha = weights_hash(neck)
    else:  # neck appris avec la tete : dans le meme dossier
        folder = run_dir
        neck_cfg = OmegaConf.to_container(cfg.neck, resolve=True)
        neck_sha = save_neck(neck, run_dir / "neck.pt", backbone, seed=cfg.seed, config=neck_cfg,
                             name=neck_name, task=task.name)  # fmt: skip
    try:  # dossier du neck relatif au dossier du backbone : downstream/<...>
        root = hub.run_root(run_dir)
        neck_ref = folder.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # hors de l'arborescence (hydra.run.dir surcharge, neck d'ailleurs)
        neck_ref = folder.resolve().as_posix()
    sha = save_head(head.head.cpu(), run_dir / "head.pt", backbone, neck, seed=cfg.seed,
                    val=scores, epochs=cfg.epochs, task=task.name, neck_dir=neck_ref)  # fmt: skip
    results = {
        "task": task.name,
        "metric": task.primary_metric,
        "val": scores,
        "neck": neck_ref,
        "neck_frozen": module.freeze_neck,
        "names": {"neck": neck_name, "head": head_name},
        "classes": {"neck": type(head.neck).__name__, "head": type(head.head).__name__},
        "params": {"neck": count_params(head.neck), "head": count_params(head.head)},
        "backbone_sha256": sha,
        "neck_sha256": neck_sha,
        "backbone_params": count_params(backbone),
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "precision": settings["precision"],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
    hub.write_card(run_dir, results)
    # en dernier : marque le run termine (hub.parts)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    run.mlflow_finish(mlflow, results)
    log.info("%s : val %s = %.4f", task.name, task.primary_metric, scores[task.primary_metric])


if __name__ == "__main__":
    main()
