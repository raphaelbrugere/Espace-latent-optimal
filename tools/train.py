"""Entraînement d'une tête sur un backbone gelé (configs/train.yaml).

    python tools/train.py backbone=from_run backbone.run=results/pretrain/<exp>/<date> task=pets_seg
    python tools/train.py backbone=from_run backbone.run=<run> task=imagenette_cls,pets_seg -m

Le run écrit dans results/train/<experiment_name>/<task>/<date>/ : head.pt (poids, empreinte
du backbone, seed), results.json (scores val, budget), plus le dossier de run commun.

Par défaut (cache_features=true), les features du backbone sont extraites une fois dans
data/features/ (evaluation/features.py), puis la tête s'entraîne dessus.
"""

import json
import logging
import os
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

from latentopt import run
from latentopt.config import register_resolvers
from latentopt.data.paths import PROJECT_ROOT
from latentopt.evaluation.features import CachedFeatures, collate, load_or_extract
from latentopt.evaluation.tasks import get_task
from latentopt.methods import FrozenBackboneModule
from latentopt.models.contract import count_params, make_head, save_head, weights_hash

register_resolvers()
log = logging.getLogger("train")


def fill_defaults(cfg: DictConfig) -> None:
    """head et epochs absents -> recette de la baseline de la tâche (écrite dans la config)."""
    task = get_task(cfg.task)
    with open_dict(cfg):
        if cfg.get("head") is None:
            cfg.head = OmegaConf.load(
                PROJECT_ROOT / "configs" / "head" / f"{task.default_head}.yaml"
            )
        if cfg.epochs is None:
            cfg.epochs = task.epochs


def cached_datasets(cfg, task, backbone, keys) -> tuple[CachedFeatures, CachedFeatures]:
    """Features train (deux variantes si retournement) et val, extraites ou relues."""
    use_cuda = cfg.cluster.device != "cpu" and torch.cuda.is_available()
    common = dict(
        backbone=backbone,
        sha=weights_hash(backbone),
        task=task,
        keys=keys,
        batch_size=256,
        num_workers=cfg.cluster.num_workers,
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

    Ordre de l'arène : backbone construit, puis graine, puis tête (init) ; la graine ne
    dépend donc pas du backbone. L'extraction des features se fait avant la graine, pour
    ne décaler ni l'init de la tête ni l'ordre des données.
    """
    task = get_task(cfg.task)
    L.seed_everything(cfg.seed, workers=False)  # backbone à poids aléatoires : reproductible
    backbone = instantiate(cfg.backbone)
    make = instantiate(cfg.head)
    if cfg.cache_features:
        keys = make_head(make, backbone, task.num_classes).in_keys  # avant la graine : jetée
        train_ds, val_ds = cached_datasets(cfg, task, backbone, keys)
        workers, val_collate = 0, collate(native_targets=task.collate_fn is not None)
        train_collate = collate(native_targets=False)
    else:
        train_ds, val_ds = task.dataset("train", train=True), task.dataset("val")
        workers, val_collate, train_collate = cfg.cluster.num_workers, task.collate_fn, None

    L.seed_everything(cfg.seed, workers=False)
    head = make_head(make, backbone, task.num_classes)
    module = FrozenBackboneModule(backbone, head, task, **cfg.optimizer)
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    os.environ["DATA_ROOT"] = cfg.cluster.data_root
    fill_defaults(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    git = run.prepare(cfg, run_dir)
    settings = run.lightning_settings(cfg.cluster.device, cfg.precision)

    module, train, val = build(cfg)
    task, head = module.task, module.head
    log.info("%s : tête %s, %s paramètres (budget %s)", task.name, type(head).__name__,
             f"{count_params(head):,}", f"{task.head_max_params:,}")  # fmt: skip
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.max_steps_per_epoch,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        callbacks=[
            ModelCheckpoint(dirpath=run_dir / "checkpoints", save_last=True, save_top_k=0),
            run.LogProgress(),
        ],
        enable_progress_bar=cfg.cluster.progress_bar,
        devices=1,
        **settings,
    )
    trainer.fit(module, train, val, ckpt_path=cfg.resume)

    scores = {k.removeprefix("val/"): float(v) for k, v in trainer.callback_metrics.items()
              if k.startswith("val/")}  # fmt: skip
    backbone = module.backbone.cpu()
    sha = save_head(head.cpu(), run_dir / "head.pt", backbone, seed=cfg.seed, val=scores,
                    epochs=cfg.epochs, task=task.name)  # fmt: skip
    n = count_params(head)
    results = {
        "task": task.name,
        "metric": task.primary_metric,
        "val": scores,
        "head": type(head).__name__,
        "head_params": n,
        "head_max_params": task.head_max_params,
        "within_budget": n <= task.head_max_params,
        "backbone_sha256": sha,
        "backbone_params": count_params(backbone),
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "precision": settings["precision"],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log.info("%s : val %s = %.4f", task.name, task.primary_metric, scores[task.primary_metric])


if __name__ == "__main__":
    main()
