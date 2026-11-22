"""Pre-entrainement d'un backbone (configs/pretrain.yaml).

    lo pretrain experiment=smoke                     # smoke run, CPU
    lo pretrain backbone=vit_s8                      # local
    lo pretrain experiment=<nom> launcher=long -m    # job Slurm (slurm/README.md)

Le run ecrit dans results/<donnees>-<archi>-<methode>-<date>/backbone/ (voir latentopt/run.py
et latentopt/hub.py), et est suivi dans MLflow (experience pretrain/<experiment_name>).
Les entrainements downstream de ce backbone iront dans le meme dossier results/<run>/.
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
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from latentopt import run
from latentopt.config import CONFIGS_DIR, register_resolvers
from latentopt.data.datasets.base import Subset
from latentopt.evaluation.tasks import get_task
from latentopt.methods import LeJEPAModule, MultiViewTransform

register_resolvers()
log = logging.getLogger("pretrain")


def build(cfg: DictConfig) -> tuple[LeJEPAModule, DataLoader, DataLoader | None]:
    """Config -> module Lightning et DataLoaders ; aucun entrainement (le sondage du GPU de
    run.resolve_device est mis en cache)."""
    m = cfg.method
    source = instantiate(cfg.data.source)
    views = MultiViewTransform(size=cfg.data.img_size, num_views=m.num_views)
    workers = cfg.env.num_workers
    train = DataLoader(
        Subset(source, range(len(source)), views),
        batch_size=m.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        # copie vers le GPU asynchrone ; ~100 Mo par pas, seul DataLoader ou cela compte
        pin_memory=run.resolve_device(cfg.env.device) == "cuda",
    )
    # val de la sonde en ligne ; probe_task null (pool sans tache de classification) : pas de
    # sonde, donc pas de validation. Non persistant, a garder : recreer ses workers a
    # chaque epoch tire un nombre aleatoire, et le rendre persistant decalerait l'alea
    probe_task = cfg.data.get("probe_task")
    val = (
        DataLoader(get_task(probe_task).dataset("val"), batch_size=256, num_workers=workers)
        if probe_task is not None
        else None
    )

    model = instantiate(m.model)(backbone=instantiate(cfg.backbone))
    module = LeJEPAModule(model, instantiate(m.optimizer), warmup_epochs=m.warmup_epochs)
    return module, train, val


@hydra.main(version_base="1.3", config_path=CONFIGS_DIR, config_name="pretrain")
def main(cfg: DictConfig) -> None:
    # la config fait foi pour l'emplacement des donnees (lu par latentopt.data.paths)
    os.environ["DATA_ROOT"] = cfg.env.data_root
    os.environ["OUTPUT_ROOT"] = cfg.env.results_root  # runs designes par leur nom (hub.locate)
    ckpt = run.resume_checkpoint(cfg.resume)  # avant tout calcul : chemin faux = arret
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    git = run.prepare(cfg, run_dir)
    settings = run.lightning_settings(cfg.env.device, cfg.method.precision)
    if "grad_checkpointing" in cfg.backbone:  # auto : resolu ici, sur le noeud qui calcule
        gc = run.resolve_grad_checkpointing(cfg.env.grad_checkpointing, settings["accelerator"])
        with open_dict(cfg):
            cfg.backbone.grad_checkpointing = gc
        run.write_config(cfg, run_dir)  # config.yaml (et from_run) : la valeur utilisee
    mlflow = run.mlflow_logger(cfg, "pretrain", run_dir, git)
    # torch.manual_seed, plus random et numpy ; workers seedes par torch
    L.seed_everything(cfg.seed, workers=False)

    module, train, val = build(cfg)
    compiled = run.resolve_compile(cfg.env.compile, settings["accelerator"])
    if compiled:
        os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # defaut actuel, sans avertissement
        module.model.backbone.compile()  # en place : state_dict inchange (backbone.pt, last.ckpt)
    trainer = L.Trainer(
        max_epochs=cfg.method.epochs,
        limit_train_batches=cfg.method.max_steps_per_epoch,
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
    log.info(
        "Trainer %s | compile %s | grad_checkpointing %s | %d images de pre-entrainement",
        settings, compiled, cfg.backbone.get("grad_checkpointing"), len(train.dataset),
    )  # fmt: skip

    t0 = time.time()
    trainer.fit(module, train, val, ckpt_path=ckpt)
    hours = (time.time() - t0) / 3600

    final = {k: float(v) for k, v in trainer.callback_metrics.items()}
    payload = run.save_backbone(
        run_dir,
        module.model.backbone,
        method=cfg.method.name,
        epochs=trainer.current_epoch,
        steps=trainer.global_step,
        precision=settings["precision"],
        compiled=compiled,
        gpu_hours=round(hours, 3)
        if trainer.accelerator.__class__.__name__ == "CUDAAccelerator"
        else 0.0,
        final_metrics=final,
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        resumed_from=run.resumed_from(cfg, ckpt),
    )
    run.mlflow_finish(mlflow, payload)


if __name__ == "__main__":
    main()
