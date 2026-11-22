"""Evaluation des necks et des tetes entraines par `lo train`.

Entree : des dossiers d'entrainement (results/<run>/downstream/<tache>-<neck>-<tete>-<date>),
sur un ou plusieurs backbones. Chaque tete est reconstruite avec son neck (le neck.pt qu'elle
designe, appris avec elle ou repris gele) et son backbone, puis evaluee sur le split demande.
Sortie : une entree par entrainement, ecrite par `write` dans le results.json d'un nouveau
dossier <entrainement>/eval/<split>-<date>/ (voir cli/run/eval.py).

Violations (signalees, pas bloquantes) : backbone > 30M parametres, neck ou tete appris sur
d'autres poids que ceux charges (empreintes). Les parametres du backbone, du neck et de la
tete sont rapportes separement, sans limite pour le neck et la tete.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from latentopt import hub
from latentopt.data.datasets.base import abbrev
from latentopt.evaluation.features import AUTOCAST, CachedFeatures, collate, load_or_extract
from latentopt.evaluation.tasks import Task, get_task, to_device, to_float
from latentopt.models import recipe
from latentopt.models.backbones.checkpoint import resolve_run
from latentopt.models.contract import (
    BACKBONE_MAX_PARAMS,
    Downstream,
    count_params,
    load_head,
    load_neck,
    weights_hash,
)

log = logging.getLogger(__name__)


@torch.no_grad()
def score(
    backbone: nn.Module,
    head: nn.Module,
    task: Task,
    loader: DataLoader,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, float]:
    """Metriques de la tache sur `loader` (images, ou features en cache)."""
    dtype = AUTOCAST[precision]
    backbone, head = backbone.to(device).eval(), head.to(device).eval()
    metric = task.metric()
    for x, y in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            feats = to_device(x, device) if isinstance(x, dict) else backbone(x.to(device))
            out = head({k: v.float() for k, v in feats.items()})
        metric.update(to_float(out), to_device(y, device))
    return metric.compute()


def backbone_info(run_cfg: Any) -> dict[str, Any]:
    """Origine du backbone : nos runs (temps GPU, donnees), poids externes (reference)
    ou poids aleatoires."""
    target = run_cfg.backbone._target_
    if target.endswith("checkpoint.from_run"):
        pre_dir = resolve_run(run_cfg.backbone.run)
        pre = json.loads((pre_dir / "results.json").read_text(encoding="utf-8"))
        pre_cfg = OmegaConf.load(pre_dir / "config.yaml")
        # nom court, comme les noms de runs et le tag data
        data = abbrev(pre_cfg.data.source._target_)
        return {"reference": False, "gpu_hours": pre.get("gpu_hours"), "pretraining_data": data}
    if ".external." in target:
        return {"reference": True, "gpu_hours": None, "pretraining_data": target.split(".")[-1]}
    return {"reference": False, "gpu_hours": None, "pretraining_data": "aucun (poids aleatoires)"}


def neck_folder(head_dir: Path, ref: str) -> Path:
    """Dossier du neck d'une tete : `ref` (neck_dir de head.pt), relatif au dossier du backbone
    (downstream/<...> : le sien, ou celui du neck repris)."""
    path = Path(ref)
    return path if path.is_absolute() else hub.run_root(head_dir) / path


def load_trained(
    head_dir: Path, cfg: DictConfig, backbone: nn.Module
) -> tuple[Downstream, dict[str, Any], list[str]]:
    """Neck + tete d'un dossier d'entrainement, poids charges ; renvoie aussi le checkpoint de la
    tete et les violations de provenance (empreintes du backbone et du neck)."""
    task = get_task(cfg.task)
    ckpt = torch.load(head_dir / "head.pt", map_location="cpu", weights_only=False)
    neck_pt = neck_folder(head_dir, ckpt["neck_dir"]) / "neck.pt"
    neck_ckpt = torch.load(neck_pt, map_location="cpu", weights_only=False)
    model = recipe.build(
        OmegaConf.create(neck_ckpt["config"]), cfg.head, backbone, task.num_classes
    )
    issues = []
    try:
        load_neck(model.neck, neck_pt, backbone)
    except ValueError as e:
        issues.append(str(e))
    try:
        load_head(model.head, head_dir / "head.pt", backbone, model.neck)
    except ValueError as e:
        issues.append(str(e))
    return model, ckpt, issues


def evaluate_downstream(
    trained: list[str | Path],
    split: str,
    device: torch.device,
    precision: str = "fp32",
    cache_features: bool = True,
    num_workers: int = 0,
    batch_size: int = 64,
) -> dict[Path, dict[str, Any]]:
    """Evalue chaque entrainement sur `split` ; renvoie, par dossier, son entree d'evaluation.

    Un backbone partage par plusieurs tetes n'est construit qu'une fois. Avec
    `cache_features`, les features sont extraites une fois (cache de features.py, batch 256).
    """
    backbones: dict[str, tuple[nn.Module, str, int, dict]] = {}
    out = {}
    for head_dir in (hub.locate(t) for t in trained):
        cfg = OmegaConf.load(head_dir / "config.yaml")
        key = json.dumps([OmegaConf.to_container(cfg.backbone, resolve=True), cfg.seed])
        if key not in backbones:
            torch.manual_seed(cfg.seed)  # comme `lo train` : backbone a poids aleatoires
            backbone = instantiate(cfg.backbone).eval()
            backbones[key] = (backbone, weights_hash(backbone), count_params(backbone),
                              backbone_info(cfg))  # fmt: skip
        backbone, sha, n_params, info = backbones[key]
        task = get_task(cfg.task)
        model, ckpt, issues = load_trained(head_dir, cfg, backbone)
        if n_params > BACKBONE_MAX_PARAMS:
            issues.append(f"backbone : {n_params:,} parametres > {BACKBONE_MAX_PARAMS:,}")

        if cache_features:
            feats_info = {"backbone": OmegaConf.to_container(cfg.backbone, resolve=True)}
            entries = load_or_extract(backbone, sha, task, split, task.eval_transform,
                                      model.in_keys, 256, num_workers, device, precision,
                                      feats_info)  # fmt: skip
            native = task.collate_fn is not None
            loader = DataLoader(CachedFeatures(entries), batch_size=batch_size,
                                collate_fn=collate(native_targets=native))  # fmt: skip
        else:
            loader = DataLoader(task.dataset(split), batch_size=batch_size,
                                num_workers=num_workers, collate_fn=task.collate_fn)  # fmt: skip
        metrics = score(backbone, model, task, loader, device, precision)
        out[head_dir] = {
            "split": split,
            "score": metrics[task.primary_metric],
            "metric": task.primary_metric,
            "metrics": metrics,
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "params": {
                "backbone": n_params,
                "neck": count_params(model.neck),
                "head": count_params(model.head),
            },  # fmt: skip
            "backbone": {"sha256": sha, **info},
            "seed": ckpt.get("seed"),
            "violations": issues,
        }
        flag = "" if not issues else "   VIOLATION : " + "; ".join(issues)
        log.info("%s : %s %s = %.4f%s", head_dir.name, split, task.primary_metric,
                 metrics[task.primary_metric], flag)  # fmt: skip
    return out


def write(eval_dir: Path, entry: dict[str, Any]) -> None:
    """results.json d'une evaluation, dans son propre dossier eval/<split>-<date>/ : le
    dossier d'entrainement n'est jamais modifie."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "results.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
