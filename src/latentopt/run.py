"""Dossier de run, commun à tous les points d'entrée de tools/.

    <run_dir>/
    ├── config.yaml     config complète résolue (se suffit à elle-même)
    ├── git.txt         commit, branche, fichiers non commités
    ├── run.log         avancement (handler fichier de Hydra + LogProgress)
    ├── metrics.csv     métriques par pas et par epoch (CSVLogger de Lightning)
    ├── checkpoints/    last.ckpt : reprise d'un run interrompu
    ├── backbone.pt     state_dict du backbone (save_backbone)
    └── results.json    hash du backbone, résultats finaux, temps GPU

`backbone.pt` et `results.json` forment le format relu par `from_run` : le modifier
casse le rechargement des runs existants.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import lightning as L
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from latentopt.data.paths import PROJECT_ROOT
from latentopt.models.contract import count_params, weights_hash

log = logging.getLogger(__name__)

ACCELERATORS = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}
PRECISIONS = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def git_info() -> dict:
    def git(*args):
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    changed = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "changed": changed.splitlines() if changed else [],
    }


def check_launch(cfg: DictConfig, git: dict) -> None:
    """Garde-fous avant tout calcul (CONTRIBUTING.md, docs/CLUSTER.md)."""
    if cfg.cluster.name == "dce" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            "cluster=dce hors d'un job Slurm : le run tournerait sur la passerelle. "
            "Ajouter -m pour soumettre via submitit, ou lancer depuis une session srun."
        )
    long_run = cfg.cluster.name == "dce" and cfg.experiment_name != "smoke"
    if long_run and git["dirty"] and not cfg.allow_dirty:
        raise RuntimeError(
            "job long avec un arbre git non commité (CONTRIBUTING.md) : commiter, ou "
            "allow_dirty=true en connaissance de cause. Fichiers : " + ", ".join(git["changed"][:5])
        )


def lightning_settings(device: str, precision: str) -> dict:
    """device (cluster) et precision (méthode) de la config -> arguments du Trainer."""
    accelerator = ACCELERATORS[device]
    on_cpu = accelerator == "cpu" or (accelerator == "auto" and not _cuda())
    if on_cpu and precision == "fp16":
        raise ValueError("fp16 n'est pas disponible sur CPU : precision=fp32 (ou bf16)")
    return {"accelerator": accelerator, "precision": PRECISIONS[precision]}


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def prepare(cfg: DictConfig, run_dir: Path) -> dict:
    """Vérifie le lancement, écrit config.yaml (résolue) et git.txt. Renvoie l'état git."""
    git = git_info()
    check_launch(cfg, git)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    lines = [f"commit: {git['commit']}", f"branch: {git['branch']}", f"dirty: {git['dirty']}"]
    (run_dir / "git.txt").write_text("\n".join(lines + git["changed"]) + "\n", encoding="utf-8")
    log.info("run dir %s | commit %s%s", run_dir, git["commit"], " (dirty)" if git["dirty"] else "")
    return git


def save_backbone(run_dir: Path, backbone: nn.Module, **results) -> dict:
    """backbone.pt + results.json, avec l'empreinte des poids enregistrés."""
    import torch

    backbone = backbone.cpu().eval()
    torch.save(backbone.state_dict(), run_dir / "backbone.pt")
    payload = {
        "backbone_sha256": weights_hash(backbone),
        "backbone_params": count_params(backbone),
        **results,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("backbone.pt sauvegardé, sha256 %s", payload["backbone_sha256"][:12])
    return payload


class LogProgress(L.Callback):
    """Une ligne par epoch dans run.log : métriques d'epoch, lr, temps écoulé."""

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()}
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        elapsed = (time.time() - self.t0) / 60
        log.info(
            "epoch %d/%d | %s | lr %.2e | %.1f min",
            trainer.current_epoch + 1, trainer.max_epochs, metrics, lr, elapsed,
        )  # fmt: skip
