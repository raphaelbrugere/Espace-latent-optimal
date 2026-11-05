"""Pré-entraînement d'un backbone (configs/pretrain.yaml).

    python tools/pretrain.py experiment=smoke                     # smoke run, CPU, ~1 min
    python tools/pretrain.py backbone=vit_s8                      # local
    python tools/pretrain.py experiment=<nom> cluster=dce -m      # job Slurm

Le run écrit dans results/pretrain/<experiment_name>/<date>/ (voir latentopt/run.py).
"""

import logging
import os
import time
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from latentopt import run
from latentopt.config import register_resolvers
from latentopt.data.datasets.base import Subset
from latentopt.evaluation.tasks import get_task
from latentopt.methods import LeJEPAModule, MultiViewTransform

register_resolvers()
log = logging.getLogger("pretrain")


def build(cfg: DictConfig) -> tuple[LeJEPAModule, DataLoader, DataLoader]:
    """Config -> module Lightning et DataLoaders, sans effet de bord (testable)."""
    m = cfg.method
    source = instantiate(cfg.data.source)
    views = MultiViewTransform(size=cfg.data.img_size, num_views=m.num_views)
    workers = cfg.cluster.num_workers
    train = DataLoader(
        Subset(source, range(len(source)), views),
        batch_size=m.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    # val non persistant, comme l'arène : recréer ses workers à chaque epoch tire un nombre
    # aléatoire, et le rendre persistant décalerait l'aléa de l'entraînement
    val = DataLoader(
        get_task(cfg.data.probe_task).dataset("val"), batch_size=256, num_workers=workers
    )

    model = instantiate(m.model)(backbone=instantiate(cfg.backbone))
    module = LeJEPAModule(model, instantiate(m.optimizer), warmup_epochs=m.warmup_epochs)
    return module, train, val


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    # la config fait foi pour l'emplacement des données (lu par latentopt.data.paths)
    os.environ["DATA_ROOT"] = cfg.cluster.data_root
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    git = run.prepare(cfg, run_dir)
    settings = run.lightning_settings(cfg.cluster.device, cfg.method.precision)
    # torch.manual_seed comme l'arène, plus random et numpy ; workers seedés par torch
    L.seed_everything(cfg.seed, workers=False)

    module, train, val = build(cfg)
    trainer = L.Trainer(
        max_epochs=cfg.method.epochs,
        limit_train_batches=cfg.method.max_steps_per_epoch,
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
    log.info("Trainer %s | %d images de pré-entraînement", settings, len(train.dataset))

    t0 = time.time()
    trainer.fit(module, train, val, ckpt_path=cfg.resume)
    hours = (time.time() - t0) / 3600

    final = {k: float(v) for k, v in trainer.callback_metrics.items()}
    run.save_backbone(
        run_dir,
        module.model.backbone,
        method=cfg.method.name,
        epochs=trainer.current_epoch,
        steps=trainer.global_step,
        precision=settings["precision"],
        gpu_hours=round(hours, 3)
        if trainer.accelerator.__class__.__name__ == "CUDAAccelerator"
        else 0.0,
        final_metrics=final,
        git_commit=git["commit"],
        git_dirty=git["dirty"],
    )


if __name__ == "__main__":
    main()
