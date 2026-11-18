"""lo mlflow sync : recopie fidèle des runs dans la base d'équipe, sans doublon, depuis
plusieurs dossiers de résultats (un par membre de l'équipe) et les dépôts du Hub."""

import shutil
from types import SimpleNamespace

import mlflow
import pytest
from omegaconf import OmegaConf

from latentopt import run
from latentopt.cli.store import mlflow_sync
from latentopt.data.paths import PROJECT_ROOT

GIT = {"commit": "abc", "branch": "main", "dirty": False, "changed": []}


def make_run(results, stage, exp, name, status="success", losses=(1.0, 0.5, 0.25)):
    """Un dossier de run complet : fichiers légers + base MLflow du run."""
    cfg = OmegaConf.create(
        {"cluster": {"name": "local", "results_root": results.as_posix(), "mlflow": True},
         "experiment_name": exp, "allow_dirty": False, "seed": 0}
    )  # fmt: skip
    run_dir = results / stage / exp / name
    run_dir.mkdir(parents=True)
    for f in run.MLFLOW_ARTIFACTS:
        (run_dir / f).write_text(f"{f} de {name}", encoding="utf-8")
    (run_dir / "backbone.pt").write_bytes(b"poids")
    logger = run.mlflow_logger(cfg, stage, run_dir, GIT)
    for step, loss in enumerate(losses):
        logger.log_metrics({"train/loss": loss}, step=step)
    run.mlflow_finish(logger, {"backbone_sha256": "ab" * 32, "score": 0.7})
    if status is not None:
        logger.finalize(status)
    return run_dir


def central_runs(db):
    client = mlflow.MlflowClient(mlflow_sync.sqlite_uri(db))
    exps = {e.experiment_id: e.name for e in client.search_experiments()}
    return client, [(exps[r.info.experiment_id], r) for r in client.search_runs(list(exps))]


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results"


def test_sync_copies_runs_faithfully(results):
    run_dir = make_run(results, "pretrain", "exp", "2026-01-01")
    db = results / "mlflow.db"
    assert mlflow_sync.sync([results], db) == {"imported": 1, "present": 0, "running": 0}

    client, runs = central_runs(db)
    ((exp, r),) = runs
    assert exp == "pretrain/exp" and r.info.run_name == "pretrain/exp/2026-01-01"
    assert r.info.status == "FINISHED"
    assert r.data.params["seed"] == "0" and r.data.params["experiment_name"] == "exp"
    assert r.data.tags["git_commit"] == "abc"
    assert r.data.tags["results/backbone_sha256"] == "ab" * 32
    assert r.data.tags["source_db"] == (run_dir / "mlflow.db").resolve().as_posix()
    history = client.get_metric_history(r.info.run_id, "train/loss")
    assert [(m.step, m.value) for m in history] == [(0, 1.0), (1, 0.5), (2, 0.25)]
    assert r.data.metrics["results/score"] == 0.7
    files = sorted(a.path for a in client.list_artifacts(r.info.run_id))
    assert files == sorted(run.MLFLOW_ARTIFACTS)  # jamais les poids
    local = client.download_artifacts(r.info.run_id, "results.json")
    assert open(local, encoding="utf-8").read() == "results.json de 2026-01-01"


def test_sync_keeps_original_dates_and_failed_status(results):
    run_dir = make_run(results, "train", "exp", "crash", status="failed")
    src = mlflow.MlflowClient("sqlite:///" + (run_dir / "mlflow.db").as_posix())
    (orig,) = src.search_runs([e.experiment_id for e in src.search_experiments()])
    mlflow_sync.sync([results], results / "mlflow.db")
    ((_, r),) = central_runs(results / "mlflow.db")[1]
    assert r.info.status == "FAILED"
    assert (r.info.start_time, r.info.end_time) == (orig.info.start_time, orig.info.end_time)


def test_sync_is_idempotent(results):
    make_run(results, "pretrain", "exp", "a")
    db = results / "mlflow.db"
    mlflow_sync.sync([results], db)
    assert mlflow_sync.sync([results], db) == {"imported": 0, "present": 1, "running": 0}
    make_run(results, "pretrain", "exp", "b")
    assert mlflow_sync.sync([results], db) == {"imported": 1, "present": 1, "running": 0}
    assert len(central_runs(db)[1]) == 2


def test_running_runs_are_skipped_unless_asked(results):
    make_run(results, "pretrain", "exp", "en_cours", status=None)
    db = results / "mlflow.db"
    assert mlflow_sync.sync([results], db)["running"] == 1
    assert central_runs(db)[1] == []
    assert mlflow_sync.sync([results], db, include_running=True)["imported"] == 1
    assert mlflow_sync.sync([results], db, include_running=True)["present"] == 1


def test_sync_merges_several_results_dirs_into_one_experiment(tmp_path):
    """Les runs de deux membres de l'équipe, même expérience : une seule expérience centrale."""
    mine, theirs = tmp_path / "moi" / "results", tmp_path / "collegue" / "results"
    make_run(mine, "pretrain", "lejepa", "2026-01-01")
    make_run(theirs, "pretrain", "lejepa", "2026-01-02")
    make_run(theirs, "train", "lejepa", "pets_seg/2026-01-03")
    db = mine / "mlflow.db"
    assert mlflow_sync.sync([mine, theirs], db)["imported"] == 3
    names = sorted(e for e, _ in central_runs(db)[1])
    assert names == ["pretrain/lejepa", "pretrain/lejepa", "train/lejepa"]
    client = mlflow.MlflowClient(mlflow_sync.sqlite_uri(db))
    assert len([e for e in client.search_experiments() if e.name == "pretrain/lejepa"]) == 1


def test_central_database_is_never_treated_as_a_source(results):
    make_run(results, "pretrain", "exp", "a")
    db = results / "mlflow.db"
    mlflow_sync.sync([results], db)
    assert db.resolve() not in mlflow_sync.source_dbs([results], db)
    assert mlflow_sync.sync([results], db)["imported"] == 0


def test_default_team_database_is_in_data_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    mine, theirs = tmp_path / "moi" / "results", tmp_path / "collegue" / "results"
    make_run(mine, "pretrain", "exp", "a")
    make_run(theirs, "pretrain", "exp", "b")
    repo_db = PROJECT_ROOT / "data" / "mlflow.db"
    existed = repo_db.exists()
    mlflow_sync.main(["--results", str(mine), str(theirs), "--no-hub"])
    assert (tmp_path / "data" / "mlflow.db").exists()
    assert not (mine / "mlflow.db").exists() and not (theirs / "mlflow.db").exists()
    assert repo_db.exists() == existed  # la base du dépôt n'est jamais touchée
    assert "2 importés" in capsys.readouterr().out


def test_main_prints_summary(results, capsys):
    make_run(results, "eval", "smoke", "val/2026-01-01")
    mlflow_sync.main(["--results", str(results), "--db", str(results / "central.db"), "--no-hub"])
    assert "1 importés, 0 déjà présents, 0 en cours ignorés" in capsys.readouterr().out


def test_a_run_both_local_and_on_the_hub_is_imported_once(tmp_path):
    """Même base à deux endroits (dossier de run et sa copie du Hub) : un seul run central,
    y compris dans une même synchronisation."""
    results = tmp_path / "results"
    run_dir = make_run(results, "pretrain", "exp", "a")
    copy = tmp_path / "mlflow_hub" / "a"
    shutil.copytree(run_dir, copy)
    db = tmp_path / "data" / "mlflow.db"
    assert mlflow_sync.sync([results, copy.parent], db) == {
        "imported": 1, "present": 1, "running": 0}  # fmt: skip
    assert mlflow_sync.sync([copy.parent], db)["present"] == 1
    assert len(central_runs(db)[1]) == 1


def test_fetch_hub_downloads_only_databases_and_light_files(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(repo, local_dir, allow_patterns):
        calls.append((repo, local_dir, allow_patterns))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    models = [SimpleNamespace(id="org/b"), SimpleNamespace(id="org/a")]
    api = SimpleNamespace(list_models=lambda author: models if author == "org" else [])
    repos = mlflow_sync.fetch_hub(tmp_path, org="org", api=api)
    assert repos == ["org/a", "org/b"]
    assert [(r, d) for r, d, _ in calls] == [("org/a", tmp_path / "a"), ("org/b", tmp_path / "b")]
    patterns = calls[0][2]
    assert "*mlflow.db" in patterns and "*results.json" in patterns
    assert not any(p.endswith(".pt") for p in patterns)  # jamais les poids
