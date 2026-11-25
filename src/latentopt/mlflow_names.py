"""Noms et reglages MLflow partages, sans rien importer de lourd (ni lightning ni mlflow)."""

import os

MLFLOW_DB = "mlflow.db"
# fichiers legers du run, joints au run dans la base centrale par `lo mlflow sync`
MLFLOW_ARTIFACTS = ("config.yaml", "git.txt", "lineage.json", "results.json", "run.log")


def quiet_mlflow_env() -> None:
    """Reglages d'environnement de MLflow : a poser AVANT `import mlflow`."""
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # message destine aux agents IA
