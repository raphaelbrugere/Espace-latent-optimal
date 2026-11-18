"""Fusionne les bases MLflow des runs (une `mlflow.db` par dossier de run) dans la base
d'équipe `data/mlflow.db`, pour comparer tous nos modèles dans une seule interface.

    lo mlflow sync                                  # Hub + results/ -> data/mlflow.db
    lo mlflow sync --results results <home collègue>/Espace-latent-optimal/results
    lo mlflow sync --no-hub                         # results/ locaux seulement (hors ligne)
    lo mlflow sync --include-running               # aussi les runs non terminés

Deux sources :
- le Hub Hugging Face : chaque dépôt de l'organisation contient une `mlflow.db` par run
  (backbone, entraînements, évaluations ; `lo hub push`). Seules les bases et les fichiers
  légers sont téléchargés, dans data/mlflow_hub/<dépôt>/, jamais les poids ;
- les dossiers `--results` (défaut : results/) : les runs pas encore publiés. Ils ne sont
  que lus (droit de lecture de groupe suffisant pour ceux d'un collègue).

Idempotent : un run est reconnu par son identifiant MLflow d'origine (tag `source_run_id`),
le même qu'il vienne du Hub ou d'un dossier local. Relancer n'importe que les nouveaux runs.
Copié : paramètres, métriques avec tout leur historique, tags, nom, dates, statut, et les
fichiers légers du dossier de run (config.yaml, git.txt, lineage.json, results.json, run.log).

Les runs encore en cours (RUNNING) sont ignorés par défaut : un job tué par Slurm (limite
de temps) peut rester RUNNING ; `--include-running` les importe tels quels (sans les
réimporter ensuite).

Commande manuelle, lancée par une seule personne à la fois : les jobs n'écrivent jamais dans
la base d'équipe, donc aucun conflit sur le disque réseau du DCE. Dépôts privés : `hf auth
login` avant, sinon seuls les dépôts publics sont vus.

Interface : MLFLOW_DISABLE_AGENT_HINT=1 uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # avant l'import de mlflow

import mlflow  # noqa: E402
from mlflow.entities import Metric, Param, ViewType  # noqa: E402

from latentopt import hub  # noqa: E402
from latentopt.data.paths import PROJECT_ROOT, data_root  # noqa: E402
from latentopt.run import MLFLOW_ARTIFACTS, MLFLOW_DB  # noqa: E402

TERMINATED = {"FINISHED", "FAILED", "KILLED"}
PARAMS_PER_BATCH, METRICS_PER_BATCH = 100, 1000  # limites de l'API MLflow


def sqlite_uri(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


def fetch_hub(dest: Path, org: str = hub.ORG, api=None) -> list[str]:
    """Télécharge la `mlflow.db` et les fichiers légers de chaque dépôt de `org` dans
    dest/<dépôt>/ (à jour : rien n'est retéléchargé) ; renvoie les dépôts parcourus."""
    from huggingface_hub import HfApi, snapshot_download

    api = api or HfApi()
    patterns = [f"*{name}" for name in (MLFLOW_DB, *MLFLOW_ARTIFACTS)]  # à toute profondeur
    repos = sorted(m.id for m in api.list_models(author=org))
    for repo in repos:
        snapshot_download(repo, local_dir=dest / repo.split("/", 1)[1], allow_patterns=patterns)
    return repos


def source_dbs(dirs: list[Path], central: Path) -> list[Path]:
    """Bases `mlflow.db` trouvées sous ces dossiers, hors la base centrale."""
    found = {p.resolve() for d in dirs for p in d.rglob(MLFLOW_DB)}
    return sorted(found - {central.resolve()})


def imported(client: mlflow.MlflowClient) -> set[str]:
    """Identifiants d'origine (source_run_id) des runs déjà présents dans la base centrale."""
    done, token = set(), None
    exps = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ALL)]
    while exps:
        page = client.search_runs(exps, run_view_type=ViewType.ALL, max_results=1000,
                                  page_token=token)  # fmt: skip
        done |= {r.data.tags["source_run_id"] for r in page if "source_run_id" in r.data.tags}
        token = page.token
        if not token:
            break
    return done


def target_experiment(client: mlflow.MlflowClient, name: str, artifacts: Path) -> str:
    """Identifiant de l'expérience `name` dans la base centrale, créée au besoin."""
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(name, artifact_location=(artifacts / name).resolve().as_uri())


def copy_run(
    src: mlflow.MlflowClient,
    dst: mlflow.MlflowClient,
    run: mlflow.entities.Run,
    exp_name: str,
    db: Path,
    artifacts: Path,
) -> str:
    """Recopie un run de `src` dans `dst` ; renvoie le nouvel identifiant. Les fichiers
    légers sont lus à côté de la base (dossier de run, ou sa copie téléchargée du Hub)."""
    info, data = run.info, run.data
    tags = {k: v for k, v in data.tags.items() if k != "mlflow.runName"}
    tags |= {"source_db": db.as_posix(), "source_run_id": info.run_id}
    new = dst.create_run(
        target_experiment(dst, exp_name, artifacts),
        start_time=info.start_time,
        tags=tags,
        run_name=info.run_name,
    )
    rid = new.info.run_id

    params = [Param(k, v) for k, v in data.params.items()]
    for i in range(0, len(params), PARAMS_PER_BATCH):
        dst.log_batch(rid, params=params[i : i + PARAMS_PER_BATCH])
    metrics = [
        Metric(m.key, m.value, m.timestamp, m.step)
        for key in data.metrics
        for m in src.get_metric_history(info.run_id, key)
    ]
    for i in range(0, len(metrics), METRICS_PER_BATCH):
        dst.log_batch(rid, metrics=metrics[i : i + METRICS_PER_BATCH])

    for name in MLFLOW_ARTIFACTS:
        if (db.parent / name).exists():
            dst.log_artifact(rid, str(db.parent / name))
    dst.set_terminated(rid, status=info.status, end_time=info.end_time)
    return rid


def sync(dirs: list[Path], central: Path, include_running: bool = False) -> dict:
    """Importe les nouveaux runs des bases trouvées sous `dirs` ; renvoie les compteurs
    (imported, present, running). Un run présent dans plusieurs bases n'est importé qu'une fois."""
    central.parent.mkdir(parents=True, exist_ok=True)
    artifacts = central.with_name(central.stem + "_artifacts")
    counts = {"imported": 0, "present": 0, "running": 0}
    dst = mlflow.MlflowClient(sqlite_uri(central))
    done = imported(dst)
    for db in source_dbs(dirs, central):
        src = mlflow.MlflowClient(sqlite_uri(db))
        for exp in src.search_experiments(view_type=ViewType.ALL):
            for run in src.search_runs([exp.experiment_id], run_view_type=ViewType.ALL):
                if run.info.run_id in done:
                    counts["present"] += 1
                elif run.info.status not in TERMINATED and not include_running:
                    counts["running"] += 1
                    print(f"en cours, ignoré : {exp.name} · {run.info.run_name} ({db})")
                else:
                    copy_run(src, dst, run, exp.name, db, artifacts)
                    done.add(run.info.run_id)
                    counts["imported"] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, nargs="+", default=[PROJECT_ROOT / "results"],
                   help="dossiers de résultats locaux à parcourir (défaut : results/)")  # fmt: skip
    p.add_argument("--no-hub", action="store_true", help="ne lit pas les dépôts du Hub")
    p.add_argument("--db", type=Path, default=None,
                   help="base d'équipe (défaut : <DATA_ROOT>/mlflow.db)")  # fmt: skip
    p.add_argument("--include-running", action="store_true")
    args = p.parse_args(argv)
    args.db = args.db or data_root() / MLFLOW_DB
    dirs = list(args.results)
    if not args.no_hub:
        cache = data_root() / "mlflow_hub"
        repos = fetch_hub(cache)
        print(f"Hub : {len(repos)} dépôts de {hub.ORG} -> {cache}")
        dirs.append(cache)
    counts = sync(dirs, args.db, args.include_running)
    print(f"{counts['imported']} importés, {counts['present']} déjà présents, "
          f"{counts['running']} en cours ignorés -> {args.db}")  # fmt: skip


if __name__ == "__main__":
    main()
