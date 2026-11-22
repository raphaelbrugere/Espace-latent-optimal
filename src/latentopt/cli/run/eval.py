"""Evaluation d'entrainements de `lo train` : neck + tete sur leur backbone (configs/eval.yaml).

    lo eval downstream='[<run>/downstream/pets_seg-fpn-conv_seg-<date>,<run>/downstream/<...>]'
    lo eval downstream=... split=test

Chaque entrainement evalue recoit un nouveau dossier eval/<split>-<date>/ : results.json
(score, metriques, violations), config.yaml, git.txt, lineage.json et un run MLflow
eval/<experiment_name> dans son mlflow.db. Le dossier d'entrainement n'est jamais modifie.
Seul le journal du lancement (run.log) va dans results/.eval/<date>/.
"""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from latentopt import hub, run
from latentopt.config import CONFIGS_DIR, register_resolvers
from latentopt.evaluation.evaluate import evaluate_downstream, write

register_resolvers()
log = logging.getLogger("eval")


@hydra.main(version_base="1.3", config_path=CONFIGS_DIR, config_name="eval")
def main(cfg: DictConfig) -> None:
    os.environ["DATA_ROOT"] = cfg.env.data_root
    os.environ["OUTPUT_ROOT"] = cfg.env.results_root  # designes par nom (hub.locate)
    # Garde-fous appeles deux fois, volontairement : ici, pour echouer AVANT de calculer les
    # scores, sans rien ecrire (results/.eval/<date>/ ne garde que run.log) ; run.prepare(), plus
    # bas, les rappelle pour chaque evaluation et ecrit config/git/lineage dans son dossier eval/.
    run.quiet_libraries()
    run.check_launch(cfg, run.git_info())
    # lightning_settings ne sert qu'a sa validation (precision connue, pas fp16 sur CPU, pas bf16
    # sans bf16 materiel) ; son resultat est ignore, l'evaluation n'utilise pas de Trainer.
    run.lightning_settings(cfg.env.device, cfg.precision)
    use_cuda = run.resolve_device(cfg.env.device) == "cuda"
    results = evaluate_downstream(
        list(cfg.downstream),
        split=cfg.split,
        device=torch.device("cuda" if use_cuda else "cpu"),
        precision=cfg.precision,
        cache_features=cfg.cache_features,
        num_workers=cfg.env.num_workers,
        batch_size=cfg.batch_size,
    )
    for trained, entry in results.items():
        eval_dir = trained / hub.EVAL / hub.eval_name(cfg.split, hub.stamp())
        git = run.prepare(cfg, eval_dir)
        mlflow = run.mlflow_logger(cfg, "eval", eval_dir, git)
        run.mlflow_finish(mlflow, entry)
        if mlflow is not None:
            mlflow.finalize("success")  # pas de Trainer pour le faire
        write(eval_dir, entry)  # en dernier : marque l'evaluation terminee (hub.parts)
        log.info("%s -> %s", trained.name, eval_dir / "results.json")


if __name__ == "__main__":
    main()
