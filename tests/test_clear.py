"""lo clear : cibles, confirmation obligatoire, fichiers suivis par git epargnes."""

import pytest

from latentopt.cli.data import clear


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """DATA_ROOT et OUTPUT_ROOT temporaires, remplis comme apres quelques runs."""
    data, results = tmp_path / "data", tmp_path / "results"
    for sub in ("datasets", "features", "models"):
        (data / sub).mkdir(parents=True)
        (data / sub / ".gitkeep").touch()
    (data / "datasets" / "imagenette").mkdir()
    (data / "datasets" / "imagenette" / "img.jpg").write_bytes(b"x" * 10)
    (data / "features" / "run1").mkdir()
    (data / "mlflow.db").write_bytes(b"db")
    (data / "mlflow_artifacts" / "eval").mkdir(parents=True)
    (results / "synth-run").mkdir(parents=True)
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.delenv("FEATURES_ROOT", raising=False)
    monkeypatch.delenv("MODELS_ROOT", raising=False)
    monkeypatch.setenv("OUTPUT_ROOT", str(results))
    monkeypatch.setattr(clear, "tracked", set)
    # sinon les cibles mlflow et cache visent la vraie racine du depot
    monkeypatch.setattr(clear, "PROJECT_ROOT", tmp_path / "project")
    (tmp_path / "project").mkdir()
    return data, results


def answer(monkeypatch, text):
    monkeypatch.setattr("builtins.input", lambda _: text)


def test_target_clears_only_the_given_targets(roots, monkeypatch):
    data, results = roots
    answer(monkeypatch, "y")
    assert clear.main(["--target", "datasets", "mlflow"]) == 0
    assert [p.name for p in (data / "datasets").iterdir()] == [".gitkeep"]
    assert not (data / "mlflow.db").exists() and not (data / "mlflow_artifacts").exists()
    assert (data / "features" / "run1").exists() and (results / "synth-run").exists()


def test_mlflow_also_clears_mlflow_ui_leftovers_at_root(roots, monkeypatch):
    project = clear.PROJECT_ROOT
    (project / "mlruns").mkdir()
    (project / "mlflow.db").write_bytes(b"db")
    (project / ".ruff_cache").mkdir()
    answer(monkeypatch, "y")
    assert clear.main(["--target", "mlflow"]) == 0
    assert not (project / "mlruns").exists() and not (project / "mlflow.db").exists()
    assert (project / ".ruff_cache").exists()


def test_all_clears_every_target(roots, monkeypatch):
    data, results = roots
    monkeypatch.setattr(clear, "TARGETS", {k: v for k, v in clear.TARGETS.items() if k != "cache"})
    answer(monkeypatch, "y")
    assert clear.main(["--all"]) == 0
    assert not any(results.iterdir()) and not (data / "mlflow.db").exists()
    for sub in ("datasets", "features"):
        assert [p.name for p in (data / sub).iterdir()] == [".gitkeep"]


@pytest.mark.parametrize("reply", ["n", "", "oui"])
def test_anything_but_y_deletes_nothing(roots, monkeypatch, reply):
    data, _ = roots
    answer(monkeypatch, reply)
    assert clear.main(["--target", "datasets"]) == 1
    assert (data / "datasets" / "imagenette" / "img.jpg").exists()


def test_closed_stdin_deletes_nothing(roots, monkeypatch):
    data, _ = roots

    def eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert clear.main(["--target", "features"]) == 1
    assert (data / "features" / "run1").exists()


def test_paths_tracked_by_git_are_never_deleted(roots, monkeypatch):
    data, _ = roots
    keep = data / "features" / "run1"
    monkeypatch.setattr(clear, "tracked", lambda: {keep.resolve()})
    answer(monkeypatch, "y")
    assert clear.main(["--target", "features"]) == 0
    assert keep.exists()


def test_nothing_to_clear_asks_nothing(roots, monkeypatch):
    def never(_):
        raise AssertionError("confirmation demandee sans rien a supprimer")

    monkeypatch.setattr("builtins.input", never)
    assert clear.main(["--target", "models"]) == 0


@pytest.mark.parametrize("argv", [[], ["--all", "--target", "results"], ["--target", "nope"]])
def test_needs_exactly_one_valid_choice(argv):
    with pytest.raises(SystemExit) as exc:
        clear.main(argv)
    assert exc.value.code == 2


def test_real_repo_tracked_files_are_protected():
    """Sur le vrai depot : les .gitkeep de data/ et le code sont bien vus comme suivis."""
    protected = clear.tracked()
    if not protected:
        pytest.skip("pas de depot git")
    assert (clear.PROJECT_ROOT / "data" / "datasets" / ".gitkeep").resolve() in protected
    assert (clear.PROJECT_ROOT / "src").resolve() in protected
