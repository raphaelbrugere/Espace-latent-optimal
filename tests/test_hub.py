"""Noms, dossiers de backbone, publication sur le Hub et telechargement (latentopt/hub.py),
sans reseau."""

import json

import pytest

from latentopt import hub

BB = "imnt160-vit_s8-lejepa-20261116_183012"
DS = "pets_seg-fpn-conv_seg-20261117_090000"
FROZEN = "pets_det-fpn-soft_argmax-20261118_090000"
EVAL = "test-20261119_100000"


@pytest.fixture(autouse=True)
def fixed_stamp(monkeypatch):
    monkeypatch.setattr(hub, "_process_stamp", lambda: "20261116_183012")


# --- noms -------------------------------------------------------------------------------


def test_normalize_turns_dashes_into_underscores():
    assert hub.normalize("vit-s8 dino") == "vit_s8_dino"
    assert hub.normalize("pets_seg") == "pets_seg"


@pytest.mark.parametrize("bad", ["", "vit/s8", "a..b", "\u00e9t\u00e9"])
def test_normalize_rejects_invalid_characters(bad):
    with pytest.raises(ValueError):
        hub.normalize(bad)


def test_names_follow_the_convention():
    assert hub.run_name("imnt160", "vit-s8", "lejepa") == BB
    assert hub.run_name("imnt160", "vit_s8", "lejepa", 0) == BB
    assert hub.run_name("imnt160", "vit_s8", "lejepa", "2") == BB + "_2"
    assert hub.run_name(None, "vit_s8", "") == "none-vit_s8-none-20261116_183012"
    assert hub.downstream_name("pets-seg", "fpn", "conv_seg", "20261117_090000") == DS
    assert hub.eval_name("test", "20261119_100000") == EVAL


def test_parse_round_trips():
    assert hub.parse(BB) == ("imnt160", "vit_s8", "lejepa", "20261116_183012")
    assert hub.parse_downstream(DS) == ("pets_seg", "fpn", "conv_seg", "20261117_090000")
    assert hub.parse_eval(EVAL) == ("test", "20261119_100000")


@pytest.mark.parametrize(
    "bad",
    [
        "vit_s8-20261116_183012",
        "imnt160-vit_s8-lejepa",
        "imnt160-vit_s8-lejepa-2026_11_16",
        f"{BB}-pets_seg-fpn",  # ancien nom de tete a 6 parties
    ],
)
def test_parse_rejects_malformed_names(bad):
    with pytest.raises(ValueError):
        hub.parse(bad)


def test_parse_downstream_and_eval_reject_malformed_names():
    with pytest.raises(ValueError):
        hub.parse_downstream("pets_seg-conv_seg-20261117_090000")  # ancien nom heads/, sans neck
    with pytest.raises(ValueError):
        hub.parse_eval(DS)


def test_repo_id_checks_the_hub_length_limit():
    assert hub.repo_id(BB) == f"Espace-latent-optimal/{BB}"
    with pytest.raises(ValueError, match="caracteres"):
        hub.repo_id(f"{'d' * 80}-vit_s8-lejepa-20261116_183012")


# --- dossiers ---------------------------------------------------------------------------


def test_run_root_finds_the_backbone_folder(tmp_path):
    root = tmp_path / "results" / BB
    ds = root / "downstream" / DS
    for inside in (root, root / "backbone", ds, ds / "eval" / EVAL):
        assert hub.run_root(inside) == root
    with pytest.raises(ValueError):
        hub.run_root(tmp_path / "ailleurs")


def test_a_downstream_folder_is_never_taken_for_a_backbone(tmp_path):
    """<tache>-<neck>-<tete>-<date> a 4 parties, comme un nom de backbone."""
    assert hub.parse_or_none(DS) is not None
    assert hub.run_root(tmp_path / BB / "downstream" / DS) == tmp_path / BB
    with pytest.raises(ValueError):
        hub.run_root(tmp_path / "ailleurs" / "downstream" / DS)


def test_backbone_run_of_a_train_run(tmp_path):
    assert hub.backbone_run("vit_s8") == "none-vit_s8-none-20261116_183012"
    assert hub.backbone_run("vit_s8_dino", data="imagenet1k", method="dino") == (
        "imagenet1k-vit_s8_dino-dino-20261116_183012"
    )
    assert hub.backbone_run("from_run", str(tmp_path / BB)) == BB
    assert hub.backbone_run("from_run", str(tmp_path / BB / "backbone")) == BB
    assert hub.backbone_run("vit_s8_dino", "", BB) == BB
    with pytest.raises(ValueError):
        hub.backbone_run("vit_s8_dino", "", "pets_seg-conv_seg-20261117_090000")


def test_train_dir_names_the_task_the_neck_and_the_head(tmp_path):
    run = str(tmp_path / BB)
    args = ("from_run", run, "", "none", "none")
    date = "20261116_183012"
    assert hub.train_dir(*args, "pets_seg", "embed_norm", "linear") == (
        f"{BB}/downstream/pets_seg-embed_norm-linear-{date}"
    )
    # neck et tete non choisis : ceux de la baseline de la tache
    assert hub.train_dir(*args, "pets_seg", None, "null") == (
        f"{BB}/downstream/pets_seg-fpn-conv_seg-{date}"
    )
    assert hub.train_dir(*args, "pets_det", None, None, None, 2).endswith(
        f"pets_det-conv_trunk-soft_argmax-{date}_2"
    )


def test_train_dir_takes_the_neck_name_of_a_reused_neck(tmp_path):
    args = ("from_run", str(tmp_path / BB), "", "none", "none", "pets_det", None, "soft_argmax")
    expected = "pets_det-fpn-soft_argmax-20261116_183012"
    assert hub.train_dir(*args, DS).endswith(expected)
    assert hub.train_dir(*args, f"/autre/{BB}/downstream/{DS}").endswith(expected)


# --- fiches ------------------------------------------------------------------------------


DS_RESULTS = {
    "task": "pets_seg",
    "metric": "miou",
    "val": {"miou": 0.5},
    "neck": f"downstream/{DS}",
    "neck_frozen": False,
    "params": {"neck": 1000, "head": 200},
    "seed": 0,
    "epochs": 3,
    "git_commit": "0123456789abcdef",
    "git_dirty": False,
}


def test_each_trained_folder_gets_its_own_card(tmp_path):
    root = tmp_path / BB
    (root / "backbone").mkdir(parents=True)
    hub.write_card(root / "backbone", {"backbone_params": 21_000_000, "backbone_sha256": "abc"})
    card = (root / "backbone" / "README.md").read_text(encoding="utf-8")
    assert f"# {BB} - backbone" in card and "21,000,000" in card

    for name, res in ((DS, DS_RESULTS), (FROZEN, {**DS_RESULTS, "neck_frozen": True})):
        (root / "downstream" / name).mkdir(parents=True)
        hub.write_card(root / "downstream" / name, res)
    card = (root / "downstream" / DS / "README.md").read_text(encoding="utf-8")
    assert "| 0.5000 |" in card and "| 1,000 / 200 |" in card and "`01234567`" in card
    assert f"neck_from={DS}" in card and "repris" not in card
    frozen = (root / "downstream" / FROZEN / "README.md").read_text(encoding="utf-8")
    assert f"repris gele de `downstream/{DS}`" in frozen


def test_the_repo_card_does_not_list_the_runs():
    card = hub.repo_card(BB)
    assert card.startswith("---\nlibrary_name: pytorch") and "- lejepa" in card
    assert DS not in card  # fixe : jamais reecrite quand d'autres ajoutent des runs


def test_no_card_outside_the_results_layout(tmp_path):
    hub.write_card(tmp_path / "backbone", {})
    assert not (tmp_path / "backbone" / "README.md").exists()


# --- publication ------------------------------------------------------------------------


def write(path, **content):
    """write(dir, head_pt="w") ecrit dir/head.pt (dernier " _ " = extension)."""
    path.mkdir(parents=True, exist_ok=True)
    for name, value in content.items():
        name = ".".join(name.rsplit("_", 1))
        text = json.dumps(value) if name.endswith(".json") else str(value)
        (path / name).write_text(text, encoding="utf-8")


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Un backbone avec deux entrainements termines (le second sur le neck du premier, gele),
    une evaluation terminee, une en cours et un entrainement en cours."""
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "results"))
    root = tmp_path / "results" / BB
    backbone = {"backbone_sha256": "abc", "backbone_params": 21_000_000, "epochs": 1}
    write(root / "backbone", backbone_pt="w", config_yaml="a: 1", mlflow_db="x", README_md="b",
          results_json=backbone)  # fmt: skip
    write(root / "backbone" / "checkpoints", last_ckpt="x")
    ds = root / "downstream" / DS
    write(ds, neck_pt="w", head_pt="w", config_yaml="a: 1", mlflow_db="x", README_md="d",
          results_json=DS_RESULTS)  # fmt: skip
    write(ds / "eval" / EVAL, config_yaml="a: 1", mlflow_db="x", results_json={"score": 0.4})
    write(ds / "eval" / "val-20261119_110000", config_yaml="a: 1")  # eval en cours
    write(root / "downstream" / FROZEN, head_pt="w", config_yaml="a: 1",
          results_json={**DS_RESULTS, "neck_frozen": True})  # fmt: skip
    write(root / "downstream" / "pets_seg-fpn-conv_seg-20261120_090000", config_yaml="a: 1")
    return root


def test_parts_are_the_finished_runs(results):
    assert [p.relative_to(results).as_posix() for p in hub.parts(results)] == [
        "backbone",
        f"downstream/{FROZEN}",
        f"downstream/{DS}",
        f"downstream/{DS}/eval/{EVAL}",
    ]


def test_plan_publishes_the_finished_runs_of_the_folder(results):
    up = hub.plan(results)
    assert up.repo_id == f"Espace-latent-optimal/{BB}" and up.folder == results
    assert up.files == [
        "backbone/README.md",
        "backbone/backbone.pt",
        "backbone/config.yaml",
        "backbone/mlflow.db",
        "backbone/results.json",
        f"downstream/{FROZEN}/config.yaml",
        f"downstream/{FROZEN}/head.pt",
        f"downstream/{FROZEN}/results.json",
        f"downstream/{DS}/README.md",
        f"downstream/{DS}/config.yaml",
        f"downstream/{DS}/eval/{EVAL}/config.yaml",
        f"downstream/{DS}/eval/{EVAL}/mlflow.db",
        f"downstream/{DS}/eval/{EVAL}/results.json",
        f"downstream/{DS}/head.pt",
        f"downstream/{DS}/mlflow.db",
        f"downstream/{DS}/neck.pt",
        f"downstream/{DS}/results.json",
        "README.md",
    ]
    assert up.present == []


def test_plan_never_resends_what_is_already_published(results):
    """Une copie locale d'un run deja publie (telechargee par `pull`) n'est jamais renvoyee :
    un push ne peut pas ecraser ce que d'autres ont publie."""
    published = {f"downstream/{DS}/results.json", f"downstream/{DS}/mlflow.db", "README.md"}
    up = hub.plan(results, published)
    assert not published & set(up.files)
    assert sorted(up.present) == sorted(published)
    assert f"downstream/{DS}/eval/{EVAL}/results.json" in up.files  # nouvelle eval : envoyee


def test_plan_accepts_any_folder_inside_the_backbone(results):
    assert hub.plan(results / "downstream" / DS / "eval" / EVAL).folder == results


def test_plan_refuses_a_folder_without_finished_parts(tmp_path):
    with pytest.raises(FileNotFoundError):
        hub.plan(tmp_path / BB)
    with pytest.raises(ValueError):
        hub.plan(tmp_path)


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, private, exist_ok):
        self.calls.append(("create", repo_id, private))

    def upload_folder(self, repo_id, folder_path, allow_patterns, commit_message, parent_commit):
        self.calls.append(("upload", repo_id, folder_path, tuple(allow_patterns), parent_commit))


def test_push_writes_the_repo_card_once_and_uploads_the_new_files(results):
    api = FakeApi()
    up = hub.plan(results)
    hub.push(up, parent="c0ffee", api=api)
    assert api.calls == [
        ("create", f"Espace-latent-optimal/{BB}", True),
        ("upload", f"Espace-latent-optimal/{BB}", results, tuple(up.files), "c0ffee"),
    ]
    assert (results / "README.md").read_text(encoding="utf-8") == hub.repo_card(BB)
    assert "README.md" not in hub.files(results)  # republier ne la compte pas deux fois

    (results / "README.md").write_text("modifiee", encoding="utf-8")
    hub.push(hub.plan(results, {"README.md"}), api=FakeApi())  # deja sur le Hub : pas reecrite
    assert (results / "README.md").read_text(encoding="utf-8") == "modifiee"


# --- emplacement local et telechargement ------------------------------------------------


def test_locate_keeps_existing_paths(results):
    assert hub.locate(results) == results
    assert hub.locate(BB, pull=False) == results


def test_locate_maps_names_and_foreign_paths_to_the_local_folder(results):
    ds = results / "downstream" / DS
    assert hub.locate(f"{BB}/downstream/{DS}") == ds
    assert hub.locate(f"/usr/users/x/results/{BB}") == results
    assert hub.locate(f"/autre/machine/results/{BB}/downstream/{DS}") == ds


def test_locate_downloads_a_missing_backbone_to_its_local_folder(results, monkeypatch):
    fetched = []

    def fake_fetch(name, org):
        fetched.append(name)
        write(hub.local_dir(name) / "downstream" / DS, head_pt="w")

    monkeypatch.setattr(hub, "fetch", fake_fetch)
    name = "imnt160-vit_s8-lejepa-20261120_000000"
    ds = hub.locate(f"{name}/downstream/{DS}")
    assert fetched == [name] and ds == results.parent / name / "downstream" / DS
    assert hub.locate(f"{name}/downstream/{DS}") == ds and fetched == [name]  # deja la


def test_locate_reports_a_path_missing_from_the_repo(results, monkeypatch):
    monkeypatch.setattr(hub, "fetch", lambda name, org: None)
    with pytest.raises(FileNotFoundError, match="absent"):
        hub.locate(f"{BB}/downstream/pets_seg-fpn-conv_seg-20990101_000000")
