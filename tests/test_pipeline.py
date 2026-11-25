"""Smoke test de bout en bout, par les vraies commandes lo (python -m latentopt.cli, en
sous-processus, comme en production) : pretrain -> train (neck + tete, puis tete sur un
neck gele) -> eval -> mlflow sync, le tout dans un seul dossier de backbone results/<run>/.

Remplace " lancer experiment=smoke avant tout job long " par une verification
automatique a chaque PR. Donnees synthetiques, CPU, resultats dans un dossier temporaire.

    pytest -m slow        # seulement ces tests
    pytest -m "not slow"  # tout sauf ces tests (iteration rapide en local)
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from latentopt import hub
from latentopt.data.paths import PROJECT_ROOT
from latentopt.evaluation.evaluate import evaluate_downstream
from latentopt.models.backbones.checkpoint import from_run
from latentopt.models.contract import weights_hash

pytestmark = pytest.mark.slow
TASKS = ("synthetic_cls", "synthetic_seg", "synthetic_det")


def tool(name: str, *overrides: str, root) -> None:
    root = root.as_posix()
    args = [
        sys.executable,
        "-m",
        "latentopt.cli",
        *name.split(),
        *overrides,
        "env=ci",
        f"env.data_root='{root}/data'",
        f"env.results_root='{root}/results'",
    ]
    # stdin ferme : pas de terminal, donc pas de question " commit non pousse " (run.confirm_pushed)
    proc = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"{name} a echoue :\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"


def only(pattern: str, root):
    found = sorted(root.glob(pattern))
    assert len(found) == 1, f"{pattern} : {found}"
    return found[0]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Les etapes, une fois pour tous les tests du module."""
    root = tmp_path_factory.mktemp("pipeline")
    tool("pretrain", "experiment=smoke", root=root)
    run = only("results/synth-vit_s8-lejepa-*", root)

    # backbone designe par son nom, pas son chemin : retrouve dans results/ par hub.locate
    # (telecharge depuis le Hub s'il manquait)
    for task in TASKS:
        tool("train", "experiment=smoke_train", f"task={task}", "backbone=from_run",
             f"backbone.run={run.name}", "cache_features=false", root=root)  # fmt: skip
    trained = {t: only(f"downstream/{t}-*", run) for t in TASKS}

    # tete neuve sur le neck fpn de synthetic_seg, gele
    source = trained["synthetic_seg"]
    tool("train", "experiment=smoke_train", "task=synthetic_seg", "backbone=from_run",
         f"backbone.run={run.name}", f"neck_from={source.name}", "cache_features=false",
         root=root)  # fmt: skip
    frozen = next(p for p in (run / "downstream").glob("synthetic_seg-*") if p != source)

    every = [*trained.values(), frozen]
    refs = ",".join(f"{run.name}/downstream/{p.name}" for p in every)
    before = {p: sorted(f.name for f in p.iterdir()) for p in every}
    tool("eval", f"downstream=[{refs}]", "precision=fp32", "cache_features=false", root=root)
    return {"root": root, "run": run, "trained": trained, "frozen": frozen, "before": before}


def test_backbone_folder_is_complete(pipeline):
    pre = pipeline["run"] / "backbone"
    for name in ("config.yaml", "git.txt", "lineage.json", "run.log", "metrics.csv", "backbone.pt",
                 "results.json", "mlflow.db", "README.md", "checkpoints/last.ckpt"):  # fmt: skip
        assert (pre / name).exists(), name
    results = read(pre / "results.json")
    assert results["method"] == "lejepa" and results["steps"] > 0
    assert all(isinstance(v, float) for v in results["final_metrics"].values())
    assert "git_commit" in results and "git_dirty" in results


def test_pretrained_backbone_reloads_with_its_hash(pipeline):
    backbone = from_run(str(pipeline["run"]))
    sha = read(pipeline["run"] / "backbone" / "results.json")["backbone_sha256"]
    assert weights_hash(backbone) == sha


def test_each_training_writes_its_neck_and_its_head_in_one_folder(pipeline):
    run = pipeline["run"]
    sha = read(run / "backbone" / "results.json")["backbone_sha256"]
    necks = {hub.parse_downstream(p.name).neck for p in pipeline["trained"].values()}
    assert necks == {"embed_norm", "fpn", "conv_trunk"}
    for task, folder in pipeline["trained"].items():
        results = read(folder / "results.json")
        assert results["task"] == task and results["backbone_sha256"] == sha
        assert results["metric"] in results["val"]
        assert not results["neck_frozen"] and results["neck"] == f"downstream/{folder.name}"
        assert results["params"]["neck"] > 0 and results["params"]["head"] > 0
        assert (folder / "README.md").exists() and (folder / "mlflow.db").exists()
        head_pt = torch.load(folder / "head.pt", map_location="cpu", weights_only=False)
        neck_pt = torch.load(folder / "neck.pt", map_location="cpu", weights_only=False)
        assert head_pt["backbone_sha256"] == neck_pt["backbone_sha256"] == sha
        assert head_pt["neck_sha256"] == neck_pt["neck_sha256"] == results["neck_sha256"]
        assert head_pt["seed"] == 0 and head_pt["params"] == results["params"]["head"]
        assert neck_pt["params"] == results["params"]["neck"]


def test_a_head_on_a_frozen_neck_reuses_it_unchanged(pipeline):
    source, frozen = pipeline["trained"]["synthetic_seg"], pipeline["frozen"]
    results = read(frozen / "results.json")
    assert hub.parse_downstream(frozen.name).neck == hub.parse_downstream(source.name).neck
    assert results["neck_frozen"] and results["neck"] == f"downstream/{source.name}"
    assert results["neck_sha256"] == read(source / "results.json")["neck_sha256"]
    assert not (frozen / "neck.pt").exists()  # pas de nouveau neck


def test_eval_writes_a_new_folder_and_leaves_the_training_untouched(pipeline):
    sha = read(pipeline["run"] / "backbone" / "results.json")["backbone_sha256"]
    for folder, files in pipeline["before"].items():
        assert sorted(f.name for f in folder.iterdir()) == sorted([*files, "eval"]), folder.name
        assert "eval" not in read(folder / "results.json")
        (eval_dir,) = (folder / "eval").iterdir()
        assert hub.parse_eval(eval_dir.name).split == "val"
        for name in ("results.json", "config.yaml", "git.txt", "lineage.json", "mlflow.db"):
            assert (eval_dir / name).exists(), name
        entry = read(eval_dir / "results.json")
        assert entry["violations"] == [], folder.name
        assert entry["split"] == "val" and 0.0 <= entry["score"] <= 1.0
        assert entry["metric"] in entry["metrics"]
        assert entry["backbone"]["sha256"] == sha and entry["params"]["neck"] > 0


def test_eval_flags_a_head_trained_on_another_backbone(pipeline, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(pipeline["root"] / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(pipeline["root"] / "results"))  # backbone par son nom
    source = pipeline["trained"]["synthetic_cls"]
    tampered = source.parent / "synthetic_cls-embed_norm-linear-20990101_000000"
    shutil.copytree(source, tampered)
    try:
        ckpt = torch.load(tampered / "head.pt", map_location="cpu", weights_only=False)
        ckpt["backbone_sha256"] = "0" * 64
        torch.save(ckpt, tampered / "head.pt")
        (entry,) = evaluate_downstream([tampered], "val", torch.device("cpu"),
                                       cache_features=False).values()  # fmt: skip
        assert any("autre backbone" in v for v in entry["violations"])
    finally:
        shutil.rmtree(tampered)


def test_the_backbone_folder_is_published_as_one_repo(pipeline):
    up = hub.plan(pipeline["run"])
    assert up.repo_id == hub.repo_id(pipeline["run"].name)
    assert "backbone/backbone.pt" in up.files and "README.md" in up.files
    for folder in pipeline["trained"].values():
        assert f"downstream/{folder.name}/neck.pt" in up.files
        assert f"downstream/{folder.name}/head.pt" in up.files
    assert any(f.startswith(f"downstream/{pipeline['frozen'].name}/eval/") for f in up.files)
    assert not any("checkpoints" in f for f in up.files)


def test_every_run_has_its_own_mlflow_database(pipeline):
    run = pipeline["run"]
    trained = [*pipeline["trained"].values(), pipeline["frozen"]]
    evals = [e for t in trained for e in (t / "eval").iterdir()]
    run_dirs = [run / "backbone", *trained, *evals]
    assert sorted(run.rglob("mlflow.db")) == sorted(d / "mlflow.db" for d in run_dirs)


def test_mlflow_sync_gathers_every_stage(pipeline):
    """Apres synchronisation : pretrain, 4 entrainements et 4 evaluations dans la base
    d'equipe, termines, relies au meme backbone ; une seconde synchronisation n'ajoute rien."""
    import mlflow

    results = pipeline["root"] / "results"
    sync = [sys.executable, "-m", "latentopt.cli", "mlflow", "sync",
            "--db", str(results / "mlflow.db"), "--no-hub"]  # fmt: skip
    env = {**os.environ, "OUTPUT_ROOT": str(results)}  # results/ de la machine
    for expected in ("9 importes, 0 deja presents", "0 importes, 9 deja presents"):
        proc = subprocess.run(sync, cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=300, env=env)  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        assert expected in proc.stdout

    client = mlflow.MlflowClient("sqlite:///" + (results / "mlflow.db").as_posix())
    sha = read(pipeline["run"] / "backbone" / "results.json")["backbone_sha256"]

    def runs(experiment):
        exp = client.get_experiment_by_name(experiment)
        assert exp is not None, experiment
        return client.search_runs([exp.experiment_id])

    (pre,) = runs("pretrain/smoke")
    assert pre.data.tags["results/backbone_sha256"] == sha
    assert len(client.get_metric_history(pre.info.run_id, "train/loss")) > 0

    trained = runs("train/smoke")
    assert sorted(r.data.params["task"] for r in trained) == sorted([*TASKS, "synthetic_seg"])
    assert all(r.data.tags["results/backbone_sha256"] == sha for r in trained)

    evals = runs("eval/default")
    assert len(evals) == 4 and all(r.data.tags["eval_split"] == "val" for r in evals)
    assert all(r.data.tags["results/backbone/sha256"] == sha for r in evals)
    # chaque eval est reliee a son entrainement par le tag `downstream`
    links = {r.data.tags["downstream"] for r in evals}
    assert links == {r.data.tags["downstream"] for r in trained}

    for r in [pre, *trained]:
        assert r.info.status == "FINISHED"
        assert r.data.tags["git_commit"] and r.data.tags["env"] == "ci"
        assert {a.path for a in client.list_artifacts(r.info.run_id)} >= {
            "config.yaml",
            "results.json",
        }
