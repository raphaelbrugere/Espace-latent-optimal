"""Évaluation d'entraînements de `lo train` : neck + tête sur leur backbone (configs/eval.yaml).

    lo eval downstream='[<run>/downstream/pets_seg-fpn-conv_seg-<date>,<run>/downstream/<…>]'
    lo eval downstream=... split=test

Chaque entraînement évalué reçoit un nouveau dossier eval/<split>-<date>/ : results.json
(score, métriques, violations), config.yaml, git.txt, lineage.json et un run MLflow
eval/<experiment_name> dans son mlflow.db. Le dossier d'entraînement n'est jamais modifié.
Seul le journal du lancement va dans results/.eval/<date>/.
"""

import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from latentopt import hub, run
from latentopt.config import CONFIGS_DIR, register_resolvers
from latentopt.evaluation.evaluate import evaluate_downstream, write

register_resolvers()
log = logging.getLogger("eval")


@hydra.main(version_base="1.3", config_path=CONFIGS_DIR, config_name="eval")
def main(cfg: DictConfig) -> None:
    os.environ["DATA_ROOT"] = cfg.cluster.data_root
    os.environ["OUTPUT_ROOT"] = cfg.cluster.results_root  # désignés par nom (hub.locate)
    run.prepare(cfg, Path(HydraConfig.get().runtime.output_dir))
    run.lightning_settings(cfg.cluster.device, cfg.precision)  # vérifie fp16 sur CPU
    use_cuda = run.resolve_device(cfg.cluster.device) == "cuda"
    results = evaluate_downstream(
        list(cfg.downstream),
        split=cfg.split,
        device=torch.device("cuda" if use_cuda else "cpu"),
        precision=cfg.precision,
        cache_features=cfg.cache_features,
        num_workers=cfg.cluster.num_workers,
        batch_size=cfg.batch_size,
    )
    for trained, entry in results.items():
        eval_dir = trained / hub.EVAL / hub.eval_name(cfg.split, hub.stamp())
        git = run.prepare(cfg, eval_dir)
        mlflow = run.mlflow_logger(cfg, "eval", eval_dir, git)
        run.mlflow_finish(mlflow, entry)
        if mlflow is not None:
            mlflow.finalize("success")  # pas de Trainer pour le faire
        write(eval_dir, entry)  # en dernier : marque l'évaluation terminée (hub.parts)
        log.info("%s -> %s", trained.name, eval_dir / "results.json")


if __name__ == "__main__":
    main()
