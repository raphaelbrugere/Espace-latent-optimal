"""Évaluation d'un backbone et de ses têtes (evaluate.py de projects/lejepa_arena).

Entrée : des runs de tools/train.py (une tête par tâche), tous sur le même backbone.
Sortie : le format JSON de l'arène (team, backbone, violations, tasks), lu par
reporting/leaderboard.py.

Violations (signalées, pas bloquantes, comme dans l'arène) : backbone > 30M paramètres,
tête hors budget, tête entraînée sur un autre backbone que celui évalué.
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from latentopt.evaluation.features import AUTOCAST, CachedFeatures, collate, load_or_extract
from latentopt.evaluation.tasks import Task, get_task, to_device, to_float
from latentopt.models.backbones.checkpoint import resolve_run
from latentopt.models.contract import BACKBONE_MAX_PARAMS, count_params, make_head, weights_hash

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
    """Métriques de la tâche sur `loader` (images, ou features en cache)."""
    dtype = AUTOCAST[precision]
    backbone, head = backbone.to(device).eval(), head.to(device).eval()
    metric = task.metric()
    for x, y in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            feats = to_device(x, device) if isinstance(x, dict) else backbone(x.to(device))
            out = head({k: v.float() for k, v in feats.items()})
        metric.update(to_float(out), to_device(y, device))
    return metric.compute()


def backbone_info(run_cfg) -> dict:
    """Origine du backbone : nos runs (temps GPU, données), poids externes (référence,
    affichée sans rang) ou poids aléatoires."""
    target = run_cfg.backbone._target_
    if target.endswith("checkpoint.from_run"):
        pre_dir = resolve_run(run_cfg.backbone.run)
        pre = json.loads((pre_dir / "results.json").read_text(encoding="utf-8"))
        pre_cfg = OmegaConf.load(pre_dir / "config.yaml")
        data = pre_cfg.data.source._target_.split(".")[-2]
        return {"reference": False, "gpu_hours": pre.get("gpu_hours"), "pretraining_data": data}
    if ".external." in target:
        return {"reference": True, "gpu_hours": None, "pretraining_data": target.split(".")[-1]}
    return {"reference": False, "gpu_hours": None, "pretraining_data": "aucun (poids aléatoires)"}


def evaluate_runs(
    runs: list[str],
    split: str,
    team: str,
    device: torch.device,
    precision: str = "fp32",
    cache_features: bool = True,
    num_workers: int = 0,
    batch_size: int = 64,
) -> dict:
    run_dirs = [resolve_run(r) for r in runs]
    cfgs = [OmegaConf.load(d / "config.yaml") for d in run_dirs]
    torch.manual_seed(cfgs[0].seed)  # comme tools/train.py : backbone à poids aléatoires
    backbone = instantiate(cfgs[0].backbone).eval()
    sha, n_params = weights_hash(backbone), count_params(backbone)
    info = backbone_info(cfgs[0])
    result = {
        "team": team,
        "split": split,
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "reference": info["reference"],
        "backbone": {
            "sha256": sha,
            "params": n_params,
            "gpu_hours": info["gpu_hours"],
            "pretraining_data": info["pretraining_data"],
        },
        "violations": [],
        "tasks": {},
    }
    if n_params > BACKBONE_MAX_PARAMS:
        result["violations"].append(f"backbone has {n_params:,} params > {BACKBONE_MAX_PARAMS:,}")

    for run_dir, cfg in zip(run_dirs, cfgs, strict=True):
        if OmegaConf.to_container(cfg.backbone) != OmegaConf.to_container(cfgs[0].backbone):
            raise ValueError(f"{run_dir} : backbone différent des autres runs évalués ensemble")
        task = get_task(cfg.task)
        ckpt = torch.load(run_dir / "head.pt", map_location="cpu", weights_only=False)
        head = make_head(instantiate(cfg.head), backbone, task.num_classes)
        head.load_state_dict(ckpt["state_dict"])
        issues = []
        if ckpt.get("backbone_sha256") != sha:
            issues.append("head was trained on a different backbone than backbone.pt")
        if count_params(head) > task.head_max_params:
            issues.append(f"head has {count_params(head):,} params > {task.head_max_params:,}")

        if cache_features:
            info = {"backbone": OmegaConf.to_container(cfg.backbone, resolve=True)}
            entries = load_or_extract(backbone, sha, task, split, task.eval_transform, head.in_keys,
                                      256, num_workers, device, precision, info)  # fmt: skip
            native = task.collate_fn is not None
            loader = DataLoader(CachedFeatures(entries), batch_size=batch_size,
                                collate_fn=collate(native_targets=native))  # fmt: skip
        else:
            loader = DataLoader(task.dataset(split), batch_size=batch_size,
                                num_workers=num_workers, collate_fn=task.collate_fn)  # fmt: skip
        metrics = score(backbone, head, task, loader, device, precision)
        result["tasks"][task.name] = {
            "score": metrics[task.primary_metric],
            "metric": task.primary_metric,
            "metrics": metrics,
            "head_params": count_params(head),
            "seed": ckpt.get("seed"),
            "violations": issues,
        }
        flag = "" if not issues else "   VIOLATION: " + "; ".join(issues)
        log.info("[%s] %s: %s = %.4f%s", team, task.name, task.primary_metric,
                 metrics[task.primary_metric], flag)  # fmt: skip
    return result


def write(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
