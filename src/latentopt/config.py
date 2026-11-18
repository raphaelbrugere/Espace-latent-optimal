"""Resolvers OmegaConf du projet, à enregistrer avant la composition Hydra.

${project_root:}                        racine du dépôt (absolue, séparateurs /)
${run_name:<données>,<archi>,<méthode>[,<job>]}   nom d'un backbone (latentopt/hub.py)
${train_dir:...}                                 dossier d'un run de `lo train` (<run>/downstream/…)
${run_stamp:[<job>]}                             date du lancement, AAAAMMJJ_HHMMSS[_<job>]
${data_abbrev:<_target_ du pool>}                nom court du dataset (ABBREV de son module)
"""

from omegaconf import OmegaConf

from latentopt import hub
from latentopt.data.datasets.base import abbrev
from latentopt.data.paths import PROJECT_ROOT

# configs/ en chemin absolu pour @hydra.main : les commandes vivent dans le paquet, loin de configs/
CONFIGS_DIR = (PROJECT_ROOT / "configs").as_posix()


def register_resolvers() -> None:
    resolvers = {
        "project_root": lambda: PROJECT_ROOT.as_posix(),
        "run_name": hub.run_name,
        "train_dir": hub.train_dir,
        "run_stamp": hub.stamp,
        "data_abbrev": abbrev,
    }
    for name, fn in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
