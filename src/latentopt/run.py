"""Dossier de run, commun a `lo pretrain` (results/<run>/backbone/), `lo train`
(results/<run>/downstream/<tache>-<neck>-<tete>-<date>/) et `lo eval`
(<dossier downstream>/eval/<split>-<date>/), voir latentopt/hub.py.

    <run_dir>/
    |-- config.yaml     config complete resolue (se suffit a elle-meme)
    |-- git.txt         commit, branche, fichiers non commites
    |-- lineage.json    empreintes de uv.lock (environnement) et des splits figes (donnees)
    |-- run.log         avancement (handler fichier de Hydra + LogProgress) ; pas pour eval
    |-- metrics.csv     metriques par pas et par epoch (CSVLogger de Lightning) ; pas pour eval
    |-- checkpoints/    last.ckpt : reprise d'un run interrompu
    |-- backbone.pt     state_dict du backbone (save_backbone) ; neck.pt et head.pt pour `lo train`
    |-- README.md       fiche du dossier (hub.write_card), pour backbone/ et downstream/<...>
    |-- results.json    empreintes, parametres, resultats finaux ; ecrit en dernier
    `-- mlflow.db       suivi MLflow du run : config, courbes, resultats, tags (mlflow_logger)

`backbone.pt` et `results.json` forment le format relu par `from_run` : le modifier
casse le rechargement des runs existants. Un dossier de run n'est ecrit que par son run,
puis plus jamais modifie : c'est ce qui permet de publier sans rien ecraser (hub.plan).

MLflow : une base SQLite par dossier de run (`mlflow_logger`, `mlflow_finish`). Un seul
programme y ecrit, donc aucun conflit de verrou, meme sur le disque reseau du DCE ou
tournent des jobs simultanes. La base est publiee sur le Hub avec le run, et `lo mlflow sync`
(commande manuelle) fusionne ensuite les runs termines (Hub et results/ locaux) dans la base
d'equipe data/mlflow.db. Le dossier de run reste la reference (poids,
checkpoints) : MLflow l'indexe, il ne le remplace pas.
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from functools import cache
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

DEVICES = ("auto", "cpu", "cuda")  # cluster.device
ACCELERATORS = {"cpu": "cpu", "cuda": "gpu"}  # device resolu -> accelerator de Lightning
PRECISIONS = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def _git(*args: str) -> str | None:
    """Sortie d'une commande git dans le depot ; None si git est absent ou echoue."""
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict[str, Any]:
    # git absent ou hors depot : les champs valent None, le run ne plante pas
    changed = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "changed": changed.splitlines() if changed else [],
        "author": _git("config", "user.name"),
    }


def unpushed_commit() -> str | None:
    """HEAD s'il n'est sur aucune branche distante connue (`git push` pas fait), sinon None."""
    head = _git("rev-parse", "HEAD")
    if head is None:
        return None
    return head if _git("rev-list", "-n1", "HEAD", "--not", "--remotes") else None


def confirm_pushed(ask: Callable[[str], str] = input) -> bool:
    """Avant un run (`lo pretrain|train|eval`) : le commit doit etre sur GitHub.

    git.txt et MLflow enregistrent le commit du run ; s'il n'est pas pousse, personne d'autre
    ne peut retrouver le code du run. En terminal, demande s'il faut continuer (y/n, non par
    defaut) ; sans terminal (job Slurm, CI, tests), avertit seulement.
    """
    commit = unpushed_commit()
    if commit is None:
        return True
    warning = (
        f"Le commit {commit[:7]} n'est pas pousse sur GitHub : le run l'enregistrera, mais "
        "personne d'autre ne pourra retrouver son code. Sans `git push`, la tracabilite du run "
        "est cassee."
    )
    if not sys.stdin.isatty():
        print(f"ATTENTION : {warning}", file=sys.stderr)
        return True
    answer = ask(f"{warning}\nPousser d'abord (git push). Continuer quand meme ? [y/n] ")
    return answer.strip().lower() in ("y", "yes")


def _sha256_text(path: Path) -> str:
    """Empreinte d'un fichier texte, fins de ligne normalisees : meme valeur sous Windows
    (CRLF apres ecriture locale) et sous Linux."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def lineage() -> dict[str, Any]:
    """Environnement et donnees du run : empreinte de uv.lock et de chaque split fige.

    Le commit les fixe deja quand l'arbre est propre ; les empreintes restent justes avec un
    arbre modifie et permettent de comparer deux runs sans remonter l'historique git.
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
            "job long avec un arbre git non commite (CONTRIBUTING.md) : commiter, ou "
            "allow_dirty=true en connaissance de cause. Fichiers : " + ", ".join(git["changed"][:5])
        )


def lightning_settings(device: str, precision: str) -> dict[str, str]:
    """device (cluster) et precision (methode) de la config -> arguments du Trainer."""
    accelerator = ACCELERATORS[resolve_device(device)]
    if accelerator == "cpu" and precision == "fp16":
        raise ValueError("fp16 n'est pas disponible sur CPU : precision=fp32 (ou bf16)")
    return {"accelerator": accelerator, "precision": PRECISIONS[precision]}


def resolve_device(device: str) -> str:
    """cluster.device (auto, cpu, cuda) -> cpu ou cuda, le GPU seulement s'il est utilisable.

    auto : cuda si cuda_problem() ne trouve rien, sinon cpu (avertissement si un GPU NVIDIA est
    present mais inutilisable). cuda : erreur si le GPU est inutilisable, plutot qu'un job long
    sur CPU.
    """
    if device not in DEVICES:
        raise KeyError(device)
    if device == "cpu":
        return "cpu"
    problem = cuda_problem()
    if problem is None:
        return "cuda"
    if device == "cuda":
        raise RuntimeError(f"device=cuda, mais le GPU est inutilisable : {problem}")
    if shutil.which("nvidia-smi"):
        log.warning("GPU NVIDIA present mais inutilisable, run sur CPU : %s", problem)
    return "cpu"


@cache
def cuda_problem() -> str | None:
    """None si torch calcule sur le GPU 0, sinon la raison : pas de GPU, pilote trop ancien
    pour le CUDA de torch, ou carte que ses wheels ne prennent plus en charge (INSTALL.md)."""
    import torch

    if not torch.cuda.is_available():
        return (
            f"torch {torch.__version__} ne voit aucun GPU (pas de GPU NVIDIA, ou pilote trop "
            f"ancien pour CUDA {torch.version.cuda})"
        )
    try:  # une vraie operation : une carte trop ancienne passe is_available() puis echoue ici
        (torch.ones(1, device="cuda") + 1).item()
    except RuntimeError as e:
        return f"{torch.cuda.get_device_name(0)} : {e}"
    return None


def prepare(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    """Verifie le lancement, ecrit config.yaml (resolue), git.txt et lineage.json.
    Renvoie l'etat git, avec la lignee sous la cle `lineage`."""
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
    """backbone.pt, README.md puis results.json (qui marque le run termine), avec
    l'empreinte des poids enregistres."""
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
    log.info("backbone.pt sauvegarde, sha256 %s", payload["backbone_sha256"][:12])
    return payload


MLFLOW_DB = "mlflow.db"
# fichiers legers du run, joints au run dans la base centrale par `lo mlflow sync`
MLFLOW_ARTIFACTS = ("config.yaml", "git.txt", "lineage.json", "results.json", "run.log")


def mlflow_uri(cfg: DictConfig, run_dir: Path) -> str | None:
    """MLFLOW_TRACKING_URI (serveur) s'il est defini, sinon la base du run ; None : desactive."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # message destine aux agents IA
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    if not cfg.cluster.get("mlflow", False):
        return None
    return "sqlite:///" + (run_dir.resolve() / MLFLOW_DB).as_posix()


def mlflow_run_name(cfg: DictConfig, run_dir: Path) -> str:
    """Chemin du dossier sous results/ : `<run>/backbone` (pretrain),
    `<run>/downstream/<tache>-<neck>-<tete>-<date>` (train), le meme suivi de
    `/eval/<split>-<date>` (eval). Hors de results_root : son nom."""
    try:
        root = Path(cfg.cluster.results_root).resolve()
        return run_dir.resolve().relative_to(root).as_posix()
    except ValueError:  # dossier de run hors de results_root (override de hydra.run.dir)
        return run_dir.name


def name_tags(run_dir: Path) -> dict[str, str]:
    """Parties des noms (docs/HUB.md) en tags filtrables, et depot Hugging Face de destination :
    celles du backbone, plus, pour un entrainement downstream/<...> ou l'une de ses
    evaluations, la tache, le neck, la tete et le nom du dossier d'entrainement (`downstream`,
    commun a l'entrainement et a ses evaluations)."""
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
    """Machine et versions : expliquent un ecart de vitesse ou de resultat entre deux runs."""
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
    """Run MLflow du run : experience `<stage>/<experiment_name>`, nom = chemin du dossier de
    run, parametres = config resolue complete (aplatie), tags = parties du nom, depot Hugging
    Face, auteur, etat git, lignee, machine et versions. None si le suivi est desactive.

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
        # base du run : aucun artefact n'y est ecrit, ses fichiers sont deja a cote ;
        # serveur (MLFLOW_TRACKING_URI) : son propre emplacement
        artifact_location=(run_dir.resolve() / "mlartifacts").as_uri()
        if uri.startswith("sqlite:")
        else None,
    )
    # La creation du run (premier acces a logger.experiment) appelle mlflow.set_tracking_uri,
    # qui ecrit MLFLOW_TRACKING_URI dans l'environnement : sans restauration, le run suivant
    # du meme processus (balayage -m local) ecrirait dans la base de celui-ci, et les
    # sous-processus en heriteraient.
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
    """Resultats finaux : nombres -> metriques `results/...`, textes et booleens -> tags."""
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
    """Une ligne par epoch dans run.log : metriques d'epoch, lr, temps ecoule."""

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
