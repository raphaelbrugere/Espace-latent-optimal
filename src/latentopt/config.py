"""Resolvers OmegaConf du projet, à enregistrer avant la composition Hydra.

${project_root:}   racine du dépôt (chemin absolu, séparateurs /), quel que soit le cwd
"""

from omegaconf import OmegaConf

from latentopt.data.paths import PROJECT_ROOT


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("project_root"):
        OmegaConf.register_new_resolver("project_root", lambda: PROJECT_ROOT.as_posix())
