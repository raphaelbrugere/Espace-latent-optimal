"""Emplacement des donnees, resolu depuis la racine du projet et non depuis le cwd.

    DATA_ROOT        defaut <racine>/data, ignore par git
    |-- datasets/    sources telechargees par `lo data prepare`
    |-- features/    features mises en cache    (FEATURES_ROOT pour le deplacer)
    |-- models/      poids externes telecharges (MODELS_ROOT pour le deplacer)
    |-- mlflow.db    base MLflow d'equipe, remplie par `lo mlflow sync`
    `-- mlflow_hub/  bases MLflow des depots du Hub, telechargees par `lo mlflow sync`

La racine est deduite de l'emplacement du paquet : valable en installation editable
(`uv sync`), qui est le mode d'installation du projet.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/latentopt/data/paths.py -> racine


def data_root() -> Path:
    """`DATA_ROOT` s'il est defini, sinon `<racine>/data` ; chemin absolu."""
    return Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).resolve()


def datasets_root() -> Path:
    return data_root() / "datasets"


def features_root() -> Path:
    return Path(os.environ.get("FEATURES_ROOT", data_root() / "features")).resolve()


def models_root() -> Path:
    return Path(os.environ.get("MODELS_ROOT", data_root() / "models")).resolve()
