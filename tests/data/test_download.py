"""`download(force=True)` supprime l'ancien telechargement avant de recommencer."""

import pytest

from latentopt.cli.data import prepare
from latentopt.data.datasets import flowers102, imagenette, oxford_pets
from latentopt.data.datasets.base import reset


class FakeTorchvisionDataset:
    """Remplace le constructeur torchvision : note si l'ancien dossier existait encore."""

    calls: list[bool] = []
    folder = None

    def __init__(self, root, *args, **kwargs):
        self.calls.append(self.folder.exists())


@pytest.fixture
def fake(monkeypatch):
    def install(module, attr, folder):
        cls = type("Fake", (FakeTorchvisionDataset,), {"calls": [], "folder": folder})
        monkeypatch.setattr(module, attr, cls)
        return cls

    return install


def make_download(root, name):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "ancien.bin").write_text("x")
    return folder


def test_reset_removes_folders_and_files_and_ignores_missing(tmp_path):
    folder, file = tmp_path / "dossier", tmp_path / "archive.tgz"
    (folder / "sous").mkdir(parents=True)
    file.write_text("x")
    reset(folder, file, tmp_path / "absent")
    assert not folder.exists() and not file.exists()


@pytest.mark.parametrize(
    "module, attr",
    [(flowers102, "TvFlowers102"), (imagenette, "TvImagenette"), (oxford_pets, "TvOxfordIIITPet")],
)
def test_force_download_removes_the_old_folder_first(tmp_path, fake, module, attr):
    folder = make_download(tmp_path, module.FOLDER)
    cls = fake(module, attr, folder)
    module.DATASET.download(tmp_path, force=True)
    assert cls.calls and not any(cls.calls)  # le dossier n'existait plus a l'appel


def test_imagenette_force_download_removes_the_archive_too(tmp_path, fake):
    make_download(tmp_path, imagenette.FOLDER)
    archive = tmp_path / f"{imagenette.FOLDER}.tgz"
    archive.write_text("x")
    fake(imagenette, "TvImagenette", tmp_path / imagenette.FOLDER)
    imagenette.DATASET.download(tmp_path, force=True)
    assert not archive.exists()


def test_download_without_force_keeps_existing_files(tmp_path, fake):
    folder = make_download(tmp_path, flowers102.FOLDER)
    fake(flowers102, "TvFlowers102", folder)
    flowers102.DATASET.download(tmp_path)
    assert (folder / "ancien.bin").exists()


def toy_module(splits_module=None, present=False):
    """Faux dataset : note les appels de `download` et, si demande, fige un split `toy`."""
    calls = {"download": [], "splits": []}

    class Module:
        @staticmethod
        def present(root):
            return present

        @staticmethod
        def download(root, force=False):
            calls["download"].append(force)

        @staticmethod
        def freeze_splits(root, force=False):
            calls["splits"].append(force)
            ids = [str(i) for i in range(10)]
            compute = lambda: (splits_module.holdout(10), ids, {})  # noqa: E731
            return splits_module.freeze("toy", compute, force)

    return Module, calls


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from latentopt.data import splits

    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(prepare, "WEIGHTS", {})

    def run(*args, present=False):
        module, calls = toy_module(splits, present)
        monkeypatch.setattr(prepare, "DATASETS", {"toy": module})
        prepare.main(["toy", *args])
        return calls

    return run


def test_prepare_passes_the_force_flags(prepared):
    assert prepared("--force-download") == {"download": [True], "splits": [False]}
    calls = prepared("--force-splits")
    assert calls == {"download": [False], "splits": [True]}


def test_prepare_says_when_the_dataset_is_already_there(prepared, capsys):
    calls = prepared(present=True)
    out = capsys.readouterr().out
    assert calls["download"] == [] and "[toy] deja present dans" in out and "tel quel" in out


def test_force_download_downloads_even_if_present(prepared, capsys):
    calls = prepared("--force-download", present=True)
    assert calls["download"] == [True]
    assert "deja present" not in capsys.readouterr().out


def test_prepare_says_what_happens_to_the_splits(prepared, capsys):
    prepared()
    assert "splits toy : crees (train 9, val 1)" in capsys.readouterr().out
    prepared()
    assert "splits toy : deja figes, conserves" in capsys.readouterr().out
    prepared("--force-splits")
    assert "splits toy : reecrits" in capsys.readouterr().out


def test_download_time_is_only_shown_with_verbose(prepared, capsys):
    prepared()
    assert "telecharge en" not in capsys.readouterr().out
    prepared("--verbose")
    assert "[toy] telecharge en" in capsys.readouterr().out


def test_progress_bars_are_hidden_unless_verbose_then_restored():
    from torchvision.datasets import utils

    original = utils.tqdm
    with prepare.progress_bars(False):
        with utils.tqdm(total=1) as bar:  # meme appel que `_urlretrieve`
            assert bar.disable
    assert utils.tqdm is original

    with prepare.progress_bars(True):
        assert utils.tqdm is original  # --verbose : rien n'est masque
