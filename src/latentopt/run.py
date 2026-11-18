"""Dossier de run, commun à `lo pretrain` (results/<run>/backbone/), `lo train`
(results/<run>/downstream/<tâche>-<neck>-<tête>-<date>/) et `lo eval`
(<dossier downstream>/eval/<split>-<date>/), voir latentopt/hub.py.

    <run_dir>/
    ├── config.yaml     config complète résolue (se suffit à elle-même)
    ├── git.txt         commit, branche, fichiers non commités
    ├── lineage.json    empreintes de uv.lock (environnement) et des splits figés (données)
    ├── run.log         avancement (handler fichier de Hydra + LogProgress) ; pas pour eval
    ├── metrics.csv     métriques par pas et par epoch (CSVLogger de Lightning) ; pas pour eval
    ├── checkpoints/    last.ckpt : reprise d'un run interrompu
    ├── backbone.pt     state_dict du backbone (save_backbone) ; neck.pt et head.pt pour `lo train`
    ├── README.md       fiche du dossier (hub.write_card), pour backbone/ et downstream/<…>
    ├── results.json    empreintes, paramètres, résultats finaux ; écrit en dernier
    └── mlflow.db       suivi MLflow du run : config, courbes, résultats, tags (mlflow_logger)

`backbone.pt` et `results.json` forment le format relu par `from_run` : le modifier
casse le rechargement des runs existants. Un dossier de run n'est écrit que par son run,
puis plus jamais modifié : c'est ce qui permet de publier sans rien écraser (hub.plan).

MLflow : une base SQLite par dossier de run (`mlflow_logger`, `mlflow_finish`). Un seul
programme y écrit, donc aucun conflit de verrou, même sur le disque réseau du DCE où
tournent des jobs simultanés. La base est publiée sur le Hub avec le run, et `lo mlflow sync`
(commande manuelle) fusionne ensuite les runs terminés (Hub et results/ locaux) dans la base
d'équipe data/mlflow.db. Le dossier de run reste la référence (poids,
checkpoints) : MLflow l'indexe, il ne le remplace pas.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from latentopt import hub
from latentopt.data.paths import PROJECT_ROOT
from latentopt.data.splits import SPLITS_DIR
from latentopt.models.contract import count_params, weights_hash

log = logging.getLogger(__name__)

ACCELERATORS = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}
PRECISIONS = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    # git absent ou hors dépôt : les champs valent None, le run ne plante pas
    changed = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "changed": changed.splitlines() if changed else [],
        "author": git("config", "user.name"),
    }


def _sha256_text(path: Path) -> str:
    """Empreinte d'un fichier texte, fins de ligne normalisées : même valeur sous Windows
    (CRLF après écriture locale) et sous Linux."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def lineage() -> dict[str, Any]:
    """Environnement et données du run : empreinte de uv.lock et de chaque split figé.

    Le commit les fixe déjà quand l'arbre est propre ; les empreintes restent justes avec un
    arbre modifié et permettent de comparer deux runs sans remonter l'historique git.
    """
    lock = PROJECT_ROOT / "uv.lock"
    return {
        "uv_lock_sha256": _sha256_text(lock) if lock.exists() else None,
        "splits_sha256": {p.stem: _sha256_text(p) for p in sorted(SPLITS_DIR.glob("*.json"))},
    }


def check_launch(cfg: DictConfig, git: dict[str, Any]) -> None:
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


def lightning_settings(device: str, precision: str) -> dict[str, str]:
    """device (cluster) et precision (méthode) de la config -> arguments du Trainer."""
    accelerator = ACCELERATORS[device]
    on_cpu = accelerator == "cpu" or (accelerator == "auto" and not _cuda())
    if on_cpu and precision == "fp16":
        raise ValueError("fp16 n'est pas disponible sur CPU : precision=fp32 (ou bf16)")
    return {"accelerator": accelerator, "precision": PRECISIONS[precision]}


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def prepare(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    """Vérifie le lancement, écrit config.yaml (résolue), git.txt et lineage.json.
    Renvoie l'état git, avec la lignée sous la clé `lineage`."""
    git = git_info()
    check_launch(cfg, git)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    lines = [f"commit: {git['commit']}", f"branch: {git['branch']}", f"dirty: {git['dirty']}"]
    (run_dir / "git.txt").write_text("\n".join(lines + git["changed"]) + "\n", encoding="utf-8")
    git = {**git, "lineage": lineage()}
    (run_dir / "lineage.json").write_text(
        json.dumps(git["lineage"], indent=2) + "\n", encoding="utf-8"
    )
    log.info("run dir %s | commit %s%s", run_dir, git["commit"], " (dirty)" if git["dirty"] else "")
    return git


def save_backbone(run_dir: Path, backbone: nn.Module, **results: Any) -> dict[str, Any]:
    """backbone.pt, README.md puis results.json (qui marque le run terminé), avec
    l'empreinte des poids enregistrés."""
    import torch

    backbone = backbone.cpu().eval()
    torch.save(backbone.state_dict(), run_dir / "backbone.pt")
    payload = {
        "backbone_sha256": weights_hash(backbone),
        "backbone_params": count_params(backbone),
        **results,
    }
    hub.write_card(run_dir, payload)
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("backbone.pt sauvegardé, sha256 %s", payload["backbone_sha256"][:12])
    return payload


MLFLOW_DB = "mlflow.db"
# fichiers légers du run, joints au run dans la base centrale par `lo mlflow sync`
MLFLOW_ARTIFACTS = ("config.yaml", "git.txt", "lineage.json", "results.json", "run.log")


def mlflow_uri(cfg: DictConfig, run_dir: Path) -> str | None:
    """MLFLOW_TRACKING_URI (serveur) s'il est défini, sinon la base du run ; None : désactivé."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # message destiné aux agents IA
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    if not cfg.cluster.get("mlflow", False):
        return None
    return "sqlite:///" + (run_dir.resolve() / MLFLOW_DB).as_posix()


def mlflow_run_name(cfg: DictConfig, run_dir: Path) -> str:
    """Chemin du dossier sous results/ : `<run>/backbone` (pretrain),
    `<run>/downstream/<tâche>-<neck>-<tête>-<date>` (train), le même suivi de
    `/eval/<split>-<date>` (eval). Hors de results_root : son nom."""
    try:
        root = Path(cfg.cluster.results_root).resolve()
        return run_dir.resolve().relative_to(root).as_posix()
    except ValueError:  # dossier de run hors de results_root (override de hydra.run.dir)
        return run_dir.name


def name_tags(run_dir: Path) -> dict[str, str]:
    """Parties des noms (docs/HUB.md) en tags filtrables, et dépôt Hugging Face de destination :
    celles du backbone, plus, pour un entraînement downstream/<…> ou l'une de ses
    évaluations, la tâche, le neck, la tête et le nom du dossier d'entraînement (`downstream`,
    commun à l'entraînement et à ses évaluations)."""
    try:
        root = hub.run_root(run_dir)
    except ValueError:  # dossier hors convention
        return {}
    name = hub.parse(root.name)
    tags = {"data": name.data, "arch": name.arch, "method": name.method,
            "backbone_date": name.date, "backbone_run": root.name,
            "hub_repo": hub.repo_id(root.name)}  # fmt: skip
    trained = run_dir.parent.parent if run_dir.parent.name == hub.EVAL else run_dir
    if trained.parent.name == hub.DOWNSTREAM:
        ds = hub.parse_downstream(trained.name)
        tags |= {"task": ds.task, "neck": ds.neck, "head": ds.head, "downstream": trained.name}
    return tags


def env_tags() -> dict[str, str]:
    """Machine et versions : expliquent un écart de vitesse ou de résultat entre deux runs."""
    import torch

    tags = {
        "host": platform.node(),
        "env/python": platform.python_version(),
        "env/torch": torch.__version__,
        "env/cuda": str(torch.version.cuda),
    }
    if torch.cuda.is_available():
        tags["env/gpu"] = torch.cuda.get_device_name(0)
    return tags


def mlflow_logger(
    cfg: DictConfig, stage: str, run_dir: Path, git: dict[str, Any]
) -> MLFlowLogger | None:
    """Run MLflow du run : expérience `<stage>/<experiment_name>`, nom = chemin du dossier de
    run, paramètres = config résolue complète (aplatie), tags = parties du nom, dépôt Hugging
    Face, auteur, état git, lignée, machine et versions. None si le suivi est désactivé.

    N'y entrent jamais : les poids (dossier de run, Hub) ni les variables d'environnement."""
    uri = mlflow_uri(cfg, run_dir)
    if uri is None:
        return None
    tags = {
        "stage": stage,
        **name_tags(run_dir),
        "author": str(git.get("author")),
        "git_commit": str(git["commit"]),
        "git_branch": str(git["branch"]),
        "git_dirty": str(git["dirty"]),
        "run_dir": run_dir.resolve().as_posix(),
        "cluster": cfg.cluster.name,
        **env_tags(),
    }
    if "SLURM_JOB_ID" in os.environ:
        tags["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    if stage == "eval" and cfg.get("split"):
        tags["eval_split"] = str(cfg.split)
    lin = git.get("lineage", {})
    if lin.get("uv_lock_sha256"):
        tags["uv_lock_sha256"] = lin["uv_lock_sha256"]
    for dataset, sha in lin.get("splits_sha256", {}).items():
        tags[f"splits/{dataset}_sha256"] = sha
    logger = MLFlowLogger(
        experiment_name=f"{stage}/{cfg.experiment_name}",
        run_name=mlflow_run_name(cfg, run_dir),
        tracking_uri=uri,
        tags=tags,
        # base du run : aucun artefact n'y est écrit, ses fichiers sont déjà à côté ;
        # serveur (MLFLOW_TRACKING_URI) : son propre emplacement
        artifact_location=(run_dir.resolve() / "mlartifacts").as_uri()
        if uri.startswith("sqlite:")
        else None,
    )
    # La création du run (premier accès à logger.experiment) appelle mlflow.set_tracking_uri,
    # qui écrit MLFLOW_TRACKING_URI dans l'environnement : sans restauration, le run suivant
    # du même processus (balayage -m local) écrirait dans la base de celui-ci, et les
    # sous-processus en hériteraient.
    before = os.environ.get("MLFLOW_TRACKING_URI")
    try:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    finally:
        if before is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = before
    return logger


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        out.update(_flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def mlflow_finish(logger: MLFlowLogger | None, results: dict[str, Any]) -> None:
    """Résultats finaux : nombres -> métriques `results/...`, textes et booléens -> tags."""
    if logger is None:
        return
    client, run_id = logger.experiment, logger.run_id
    metrics, tags = {}, {}
    for k, v in _flatten(results).items():
        if isinstance(v, bool) or isinstance(v, str):
            tags[f"results/{k}"] = str(v)[:500]
        elif isinstance(v, int | float):
            metrics[f"results/{k}"] = float(v)
    logger.log_metrics(metrics)
    for k, v in tags.items():
        client.set_tag(run_id, k, v)


class LogProgress(L.Callback):
    """Une ligne par epoch dans run.log : métriques d'epoch, lr, temps écoulé."""

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()}
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        elapsed = (time.time() - self.t0) / 60
        log.info(
            "epoch %d/%d | %s | lr %.2e | %.1f min",
            trainer.current_epoch + 1, trainer.max_epochs, metrics, lr, elapsed,
        )  # fmt: skip
