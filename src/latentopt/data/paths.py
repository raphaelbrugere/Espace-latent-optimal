"""Emplacement des données, résolu depuis la racine du projet et non depuis le cwd.

    DATA_ROOT        défaut <racine>/data, ignoré par git
    ├── datasets/    sources téléchargées par tools/prepare_data.py
    ├── features/    features mises en cache    (FEATURES_ROOT pour le déplacer)
    └── models/      poids externes téléchargés (MODELS_ROOT pour le déplacer)

La racine est déduite de l'emplacement du paquet : valable en installation éditable
(`uv sync`), qui est le mode d'installation du projet.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/latentopt/data/paths.py -> racine


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).resolve()


def datasets_root() -> Path:
    return data_root() / "datasets"


def features_root() -> Path:
    return Path(os.environ.get("FEATURES_ROOT", data_root() / "features")).resolve()


def models_root() -> Path:
    return Path(os.environ.get("MODELS_ROOT", data_root() / "models")).resolve()
