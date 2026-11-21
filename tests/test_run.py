"""Dossier de run commun et garde-fous de lancement (latentopt/run.py)."""

import json

import pytest
from omegaconf import OmegaConf

from latentopt import run
from latentopt.models.contract import count_params, weights_hash


def cfg(cluster="local", experiment="exp", allow_dirty=False):
    return OmegaConf.create(
        {"cluster": {"name": cluster}, "experiment_name": experiment, "allow_dirty": allow_dirty}
    )


CLEAN = {"commit": "abc", "branch": "main", "dirty": False, "changed": []}
DIRTY = {"commit": "abc", "branch": "main", "dirty": True, "changed": [" M src/x.py"]}

# --- check_launch ------------------------------------------------------------------------


def test_local_runs_are_always_allowed():
    run.check_launch(cfg("local"), DIRTY)


def test_dce_outside_slurm_is_refused(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="hors d'un job Slurm"):
        run.check_launch(cfg("dce"), CLEAN)


def test_dce_long_job_with_dirty_tree_is_refused(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(RuntimeError, match="src/x.py"):
        run.check_launch(cfg("dce"), DIRTY)


def test_dce_dirty_tree_allowed_explicitly_or_for_smoke(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    run.check_launch(cfg("dce", allow_dirty=True), DIRTY)
    run.check_launch(cfg("dce", experiment="smoke"), DIRTY)
    run.check_launch(cfg("dce"), CLEAN)


# --- confirm_pushed ----------------------------------------------------------------------


class Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def test_pushed_commit_runs_without_question(monkeypatch):
    monkeypatch.setattr(run, "unpushed_commit", lambda: None)
    assert run.confirm_pushed(ask=lambda _: pytest.fail("question posee"))


@pytest.mark.parametrize("answer,ok", [("y", True), ("Y", True), ("n", False), ("", False)])
def test_unpushed_commit_asks_in_a_terminal(monkeypatch, answer, ok):
    monkeypatch.setattr(run, "unpushed_commit", lambda: "abc1234def")
    monkeypatch.setattr(run.sys, "stdin", Stdin(tty=True))
    asked = []
    assert run.confirm_pushed(ask=lambda q: asked.append(q) or answer) is ok
    assert "abc1234" in asked[0] and "tracabilite" in asked[0]


def test_unpushed_commit_only_warns_without_terminal(monkeypatch, capsys):
    monkeypatch.setattr(run, "unpushed_commit", lambda: "abc1234def")
    monkeypatch.setattr(run.sys, "stdin", Stdin(tty=False))
    assert run.confirm_pushed(ask=lambda _: pytest.fail("question posee"))
    assert "tracabilite" in capsys.readouterr().err


# --- lightning_settings ------------------------------------------------------------------


@pytest.fixture
def gpu_ok(monkeypatch):
    monkeypatch.setattr(run, "cuda_problem", lambda: None)


@pytest.fixture
def gpu_broken(monkeypatch):
    monkeypatch.setattr(run, "cuda_problem", lambda: "pilote trop ancien")


@pytest.mark.parametrize(
    "device,precision,expected",
    [
        ("cpu", "fp32", {"accelerator": "cpu", "precision": "32-true"}),
        ("cpu", "bf16", {"accelerator": "cpu", "precision": "bf16-mixed"}),
        ("cuda", "fp16", {"accelerator": "gpu", "precision": "16-mixed"}),
        ("auto", "fp16", {"accelerator": "gpu", "precision": "16-mixed"}),
    ],
)
def test_lightning_settings_mapping(gpu_ok, device, precision, expected):
    assert run.lightning_settings(device, precision) == expected


def test_fp16_on_cpu_is_refused(gpu_broken):
    with pytest.raises(ValueError, match="fp16"):
        run.lightning_settings("cpu", "fp16")
    with pytest.raises(ValueError, match="fp16"):
        run.lightning_settings("auto", "fp16")


# --- resolve_device ----------------------------------------------------------------------


def test_cpu_never_touches_the_gpu(monkeypatch):
    monkeypatch.setattr(run, "cuda_problem", lambda: pytest.fail("GPU interroge"))
    assert run.resolve_device("cpu") == "cpu"


def test_auto_uses_the_gpu_only_if_usable(monkeypatch):
    monkeypatch.setattr(run, "cuda_problem", lambda: None)
    assert run.resolve_device("auto") == "cuda"
    monkeypatch.setattr(run, "cuda_problem", lambda: "pilote trop ancien")
    assert run.resolve_device("auto") == "cpu"


def test_cuda_with_unusable_gpu_is_refused(gpu_broken):
    with pytest.raises(RuntimeError, match="pilote trop ancien"):
        run.resolve_device("cuda")


@pytest.mark.parametrize("bad", [("tpu", "fp32"), ("cpu", "fp8")])
def test_unknown_device_or_precision_fails(bad):
    with pytest.raises(KeyError):
        run.lightning_settings(*bad)


# --- resolve_compile ---------------------------------------------------------------------


def test_compile_auto_follows_triton_on_gpu(monkeypatch):
    monkeypatch.setattr("torch.utils._triton.has_triton", lambda: True)
    assert run.resolve_compile("auto", "gpu") is True
    monkeypatch.setattr("torch.utils._triton.has_triton", lambda: False)
    assert run.resolve_compile("auto", "gpu") is False


def test_compile_auto_is_eager_on_cpu_and_false_never_compiles():
    assert run.resolve_compile("auto", "cpu") is False
    assert run.resolve_compile(False, "gpu") is False


def test_compile_true_without_triton_is_refused(monkeypatch):
    monkeypatch.setattr("torch.utils._triton.has_triton", lambda: False)
    with pytest.raises(RuntimeError, match="Triton"):
        run.resolve_compile(True, "gpu")
    with pytest.raises(RuntimeError, match="CPU"):
        run.resolve_compile(True, "cpu")


def test_compile_needs_a_c_compiler_outside_windows(monkeypatch):
    monkeypatch.setattr("torch.utils._triton.has_triton", lambda: True)
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    monkeypatch.delenv("CC", raising=False)
    assert run.resolve_compile("auto", "gpu") is False
    with pytest.raises(RuntimeError, match="compilateur C"):
        run.resolve_compile(True, "gpu")
    monkeypatch.setenv("CC", "/opt/cc")
    assert run.resolve_compile(True, "gpu") is True


def test_unknown_compile_mode_fails():
    with pytest.raises(ValueError):
        run.resolve_compile("yes", "gpu")


# --- git_info / prepare / save_backbone ---------------------------------------------------


def test_git_info_structure():
    info = run.git_info()
    assert set(info) == {"commit", "branch", "dirty", "changed", "author"}
    assert info["commit"] is None or len(info["commit"]) == 40
    assert info["dirty"] == bool(info["changed"])


def test_prepare_writes_resolved_config_and_git(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "git_info", lambda: DIRTY)
    c = OmegaConf.create(
        {"cluster": {"name": "local"}, "experiment_name": "e", "allow_dirty": False,
         "a": 2, "b": "${a}"}
    )  # fmt: skip
    git = run.prepare(c, tmp_path / "r")
    assert git == {**DIRTY, "lineage": run.lineage()}
    saved = OmegaConf.load(tmp_path / "r" / "config.yaml")
    assert saved.b == 2  # resolue : se suffit a elle-meme
    text = (tmp_path / "r" / "git.txt").read_text(encoding="utf-8")
    assert "commit: abc" in text and "dirty: True" in text and "src/x.py" in text
    on_disk = json.loads((tmp_path / "r" / "lineage.json").read_text(encoding="utf-8"))
    assert on_disk == git["lineage"]


def test_lineage_hashes_lockfile_and_each_frozen_split():
    lin = run.lineage()
    assert len(lin["uv_lock_sha256"]) == 64
    assert "imagenette" in lin["splits_sha256"]
    assert all(len(sha) == 64 for sha in lin["splits_sha256"].values())


def test_lineage_hash_ignores_line_endings(tmp_path):
    (tmp_path / "lf").write_bytes(b"a\nb\n")
    (tmp_path / "crlf").write_bytes(b"a\r\nb\r\n")
    assert run._sha256_text(tmp_path / "lf") == run._sha256_text(tmp_path / "crlf")


def test_prepare_checks_launch_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "git_info", lambda: CLEAN)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError):
        run.prepare(cfg("dce"), tmp_path / "r")
    assert not (tmp_path / "r").exists()


def test_save_backbone_records_hash_and_results(tmp_path, pyramid):
    pyramid.train()
    payload = run.save_backbone(tmp_path, pyramid, epochs=3, final_metrics={"x": 1.0})
    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload == on_disk
    assert on_disk["backbone_sha256"] == weights_hash(pyramid)
    assert on_disk["backbone_params"] == count_params(pyramid)
    assert on_disk["epochs"] == 3 and on_disk["final_metrics"] == {"x": 1.0}
    assert (tmp_path / "backbone.pt").exists()
    assert not pyramid.training  # sauvegarde en eval, sur CPU


# --- MLflow ---------------------------------------------------------------------------------


def mlflow_cfg(tmp_path, enabled=True):
    return OmegaConf.create(
        {
            "cluster": {
                "name": "local",
                "results_root": (tmp_path / "results").as_posix(),
                "mlflow": enabled,
            },
            "experiment_name": "exp",
            "allow_dirty": False,
            "method": {"lr": 0.002, "blocks": [2, 5]},
        }
    )


def run_dir_of(tmp_path, *parts):
    path = tmp_path.joinpath("results", *parts)
    path.mkdir(parents=True)
    return path


def only_run(run_dir):
    import mlflow

    client = mlflow.MlflowClient("sqlite:///" + (run_dir / run.MLFLOW_DB).as_posix())
    (exp,) = [e for e in client.search_experiments() if e.name != "Default"]
    (r,) = client.search_runs([exp.experiment_id])
    return client, exp, r


def test_mlflow_disabled(tmp_path):
    cfg = mlflow_cfg(tmp_path, enabled=False)
    assert run.mlflow_uri(cfg, tmp_path) is None
    assert run.mlflow_logger(cfg, "pretrain", tmp_path / "r", CLEAN) is None
    run.mlflow_finish(None, {"x": 1})  # sans effet


def test_mlflow_uses_one_database_per_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cfg = mlflow_cfg(tmp_path)
    assert run.mlflow_uri(cfg, a) == "sqlite:///" + (a.resolve() / "mlflow.db").as_posix()
    assert run.mlflow_uri(cfg, a) != run.mlflow_uri(cfg, b)


def test_mlflow_env_uri_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://serveur:5000")
    assert run.mlflow_uri(mlflow_cfg(tmp_path), tmp_path) == "http://serveur:5000"


def test_mlflow_records_config_git_and_results(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    run_dir = run_dir_of(
        tmp_path, BB, "downstream", "synthetic_cls-embed_norm-linear-20260101_000000"
    )
    lineage = {"uv_lock_sha256": "11" * 32, "splits_sha256": {"imagenette": "22" * 32}}
    git = {**DIRTY, "lineage": lineage}
    logger = run.mlflow_logger(mlflow_cfg(tmp_path), "train", run_dir, git)
    logger.log_metrics({"train/loss": 1.0}, step=0)
    logger.log_metrics({"train/loss": 0.5}, step=1)
    results = {"val": {"top1": 0.5}, "neck_frozen": True, "backbone_sha256": "ab" * 32,
               "head_params": 1000, "changed": ["a", "b"]}  # fmt: skip
    run.mlflow_finish(logger, results)
    logger.finalize("success")

    assert (run_dir / "mlflow.db").exists()
    client, exp, r = only_run(run_dir)
    assert exp.name == "train/exp"
    assert r.info.run_name == f"{BB}/downstream/synthetic_cls-embed_norm-linear-20260101_000000"
    assert r.info.status == "FINISHED"
    assert r.data.params["method/lr"] == "0.002" and r.data.params["experiment_name"] == "exp"
    tags = r.data.tags
    assert tags["stage"] == "train" and tags["git_commit"] == "abc" and tags["git_dirty"] == "True"
    assert tags["slurm_job_id"] == "123" and tags["run_dir"] == run_dir.resolve().as_posix()
    assert tags["uv_lock_sha256"] == "11" * 32
    assert tags["splits/imagenette_sha256"] == "22" * 32
    assert tags["results/backbone_sha256"] == "ab" * 32 and tags["results/neck_frozen"] == "True"
    assert r.data.metrics["results/val/top1"] == 0.5
    assert r.data.metrics["results/head_params"] == 1000
    assert [m.value for m in client.get_metric_history(r.info.run_id, "train/loss")] == [1.0, 0.5]
    assert not any("changed" in k for k in {**r.data.metrics, **tags})  # listes ignorees
    assert client.list_artifacts(r.info.run_id) == []  # les fichiers restent dans le dossier


def test_mlflow_run_name_falls_back_outside_results_root(tmp_path):
    run_dir = tmp_path / "ailleurs" / "2026-01-01"
    run.mlflow_logger(mlflow_cfg(tmp_path), "eval", run_dir, CLEAN).finalize("success")
    assert only_run(run_dir)[2].info.run_name == "2026-01-01"


def test_mlflow_marks_crashed_runs_as_failed(tmp_path):
    run_dir = run_dir_of(tmp_path, "pretrain", "exp", "d")
    logger = run.mlflow_logger(mlflow_cfg(tmp_path), "pretrain", run_dir, CLEAN)
    logger.finalize("failed")  # ce que fait Lightning quand fit() leve une exception
    assert only_run(run_dir)[2].info.status == "FAILED"


def test_successive_runs_in_one_process_keep_their_own_database(tmp_path):
    """Balayage -m local : plusieurs runs dans le meme processus. Lightning appelle
    mlflow.set_tracking_uri, qui ecrit MLFLOW_TRACKING_URI : il ne doit pas fuir."""
    import os

    cfg = mlflow_cfg(tmp_path)
    dirs = [run_dir_of(tmp_path, "train", "exp", t, "d") for t in ("cls", "seg", "det")]
    for d in dirs:
        run.mlflow_logger(cfg, "train", d, CLEAN).finalize("success")
        assert "MLFLOW_TRACKING_URI" not in os.environ
    for d in dirs:
        assert only_run(d)[2].info.run_name == f"train/exp/{d.parent.name}/d"  # un run par base


def test_user_tracking_uri_is_preserved(tmp_path, monkeypatch):
    import os

    db = tmp_path / "serveur.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + db.as_posix())
    run_dir = run_dir_of(tmp_path, "pretrain", "exp", "d")
    run.mlflow_logger(mlflow_cfg(tmp_path), "pretrain", run_dir, CLEAN).finalize("success")
    assert os.environ["MLFLOW_TRACKING_URI"] == "sqlite:///" + db.as_posix()
    assert db.exists() and not (run_dir / "mlflow.db").exists()


BB = "imnt160-vit_s8-lejepa-20261116_183012"


DS = "pets_seg-fpn-conv_seg-20261117_090000"


def test_mlflow_run_name_is_the_run_folder_and_name_parts_become_tags(tmp_path):
    cfg = mlflow_cfg(tmp_path)
    pre = run_dir_of(tmp_path, BB, "backbone")
    run.mlflow_logger(cfg, "pretrain", pre, {**CLEAN, "author": "rapha"}).finalize("success")
    r = only_run(pre)[2]
    assert r.info.run_name == f"{BB}/backbone"
    tags = r.data.tags
    assert (tags["data"], tags["arch"], tags["method"]) == ("imnt160", "vit_s8", "lejepa")
    assert tags["backbone_run"] == BB and tags["hub_repo"] == f"Espace-latent-optimal/{BB}"
    assert tags["author"] == "rapha" and tags["env/torch"] and tags["host"]
    assert "task" not in tags

    trained = run_dir_of(tmp_path, BB, "downstream", DS)
    run.mlflow_logger(cfg, "train", trained, CLEAN).finalize("success")
    tags = only_run(trained)[2].data.tags
    assert only_run(trained)[2].info.run_name == f"{BB}/downstream/{DS}"
    assert (tags["task"], tags["neck"], tags["head"]) == ("pets_seg", "fpn", "conv_seg")
    assert tags["downstream"] == DS and tags["backbone_run"] == BB
    assert tags["hub_repo"] == f"Espace-latent-optimal/{BB}"


def test_mlflow_eval_has_its_own_database_tagged_like_its_training(tmp_path):
    """L'evaluation a sa base dans eval/<split>-<date>/ : celle de l'entrainement n'est pas
    modifiee, et le tag `downstream` relie les deux."""
    cfg = mlflow_cfg(tmp_path)
    trained = run_dir_of(tmp_path, BB, "downstream", DS)
    run.mlflow_logger(cfg, "train", trained, CLEAN).finalize("success")
    before = (trained / run.MLFLOW_DB).read_bytes()
    cfg.split = "test"
    eval_dir = run_dir_of(tmp_path, BB, "downstream", DS, "eval", "test-20261118_100000")
    run.mlflow_logger(cfg, "eval", eval_dir, CLEAN).finalize("success")
    assert (trained / run.MLFLOW_DB).read_bytes() == before
    _, exp, r = only_run(eval_dir)
    assert exp.name == "eval/exp"
    assert r.info.run_name == f"{BB}/downstream/{DS}/eval/test-20261118_100000"
    tags = r.data.tags
    assert tags["eval_split"] == "test" and tags["downstream"] == DS and tags["neck"] == "fpn"
