"""Évaluation d'un backbone et de ses têtes (configs/eval.yaml).

    python tools/eval.py name=<nom> runs='[results/train/<exp>/<date>/imagenette_cls,...]'
    python tools/eval.py name=<nom> runs=... split=test

Écrit results/eval/<name>/<split>/<date>/results.json, lu par reporting/leaderboard.py.
"""

import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from latentopt import run
from latentopt.config import register_resolvers
from latentopt.evaluation.evaluate import evaluate_runs, write

register_resolvers()
log = logging.getLogger("eval")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    os.environ["DATA_ROOT"] = cfg.cluster.data_root
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run.prepare(cfg, run_dir)
    run.lightning_settings(cfg.cluster.device, cfg.precision)  # vérifie fp16 sur CPU
    use_cuda = cfg.cluster.device != "cpu" and torch.cuda.is_available()
    result = evaluate_runs(
        list(cfg.runs),
        split=cfg.split,
        team=cfg.name,
        device=torch.device("cuda" if use_cuda else "cpu"),
        precision=cfg.precision,
        cache_features=cfg.cache_features,
        num_workers=cfg.cluster.num_workers,
        batch_size=cfg.batch_size,
    )
    write(result, run_dir / "results.json")
    log.info("[%s] -> %s", cfg.name, run_dir / "results.json")


if __name__ == "__main__":
    main()
