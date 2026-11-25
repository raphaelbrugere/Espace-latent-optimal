"""`lo eval` signale une tete entrainee avec un autre traitement des images que le code actuel."""

import json

from latentopt.evaluation.evaluate import pipeline_issues
from latentopt.evaluation.tasks import get_task


def write_results(folder, pipelines):
    results = {"task": "synthetic_det"} | ({"pipelines": pipelines} if pipelines else {})
    (folder / "results.json").write_text(json.dumps(results), encoding="utf-8")


def test_same_pipelines_raise_nothing(tmp_path):
    task = get_task("synthetic_det")
    current = {
        "train": task.train_pipeline.fingerprint(),
        "eval": task.eval_pipeline.fingerprint(),
    }
    write_results(tmp_path, current)
    assert pipeline_issues(tmp_path, task) == []


def test_a_changed_pipeline_is_reported(tmp_path):
    task = get_task("synthetic_det")
    write_results(tmp_path, {"train": "ancien", "eval": task.eval_pipeline.fingerprint()})
    issues = pipeline_issues(tmp_path, task)
    assert len(issues) == 1 and "pipeline train" in issues[0] and "ancien" in issues[0]


def test_runs_without_fingerprints_are_not_flagged(tmp_path):
    write_results(tmp_path, None)
    assert pipeline_issues(tmp_path, get_task("synthetic_det")) == []
