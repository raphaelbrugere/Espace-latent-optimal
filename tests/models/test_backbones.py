"""Backbones : TimmViT au contrat, rechargement d'un run (`from_run`) vérifié par empreinte."""

import json

import pytest
import torch
from omegaconf import OmegaConf

from latentopt import run
from latentopt.models.backbones.archs.vit import TimmViT
from latentopt.models.backbones.checkpoint import from_run, resolve_run
from latentopt.models.contract import check_backbone, weights_hash


@pytest.mark.parametrize("size", [32, 64, 96])
def test_vit_respects_contract_at_several_resolutions(tiny_vit, size):
    check_backbone(tiny_vit, size=size)  # dynamic_img_size


def test_vit_feature_info_and_outputs(tiny_vit):
    assert tiny_vit.embed_dim == 24
    assert tiny_vit.feature_info == [
        {"name": "block1", "channels": 192, "stride": 16},
        {"name": "block2", "channels": 192, "stride": 16},
    ]
    out = tiny_vit(torch.randn(2, 3, 32, 32))
    assert set(out) == {"block1", "block2", "embed"}
    assert out["embed"].shape == (2, 24)
    assert out["block1"].shape == (2, 192, 2, 2)


def test_vit_without_projection_exposes_cls_width():
    torch.manual_seed(0)
    vit = TimmViT("vit_tiny_patch16_224", img_size=32, embed_dim=None, blocks=(0,))
    assert vit.embed_dim == 192
    assert vit(torch.randn(1, 3, 32, 32))["embed"].shape == (1, 192)


def test_vit_is_deterministic_in_eval():
    torch.manual_seed(0)
    vit = TimmViT("vit_tiny_patch16_224", img_size=32, embed_dim=8, blocks=(0,), drop_path_rate=0.5)
    vit.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(vit(x)["embed"], vit(x)["embed"])


def test_vit_rebuilt_from_config_and_state_dict_is_identical(tiny_vit, tiny_vit_cfg):
    kwargs = {k: v for k, v in tiny_vit_cfg.items() if k != "_target_"}
    torch.manual_seed(123)  # autre init, écrasée par le state_dict
    clone = TimmViT(**kwargs).eval()
    clone.load_state_dict(tiny_vit.state_dict())
    assert weights_hash(clone) == weights_hash(tiny_vit)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(clone(x)["embed"], tiny_vit(x)["embed"])


def test_vit_grad_checkpointing_gives_same_output():
    torch.manual_seed(0)
    a = TimmViT("vit_tiny_patch16_224", img_size=32, embed_dim=8, blocks=(0, 1))
    b = TimmViT("vit_tiny_patch16_224", img_size=32, embed_dim=8, blocks=(0, 1),
                grad_checkpointing=True)  # fmt: skip
    b.load_state_dict(a.state_dict())
    a.eval(), b.eval()
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(a(x)["embed"], b(x)["embed"], atol=1e-6)


# --- from_run ----------------------------------------------------------------------------

BB = "synth-vit_tiny-test-20261116_183012"


@pytest.fixture
def pretrain_run(tmp_path, tiny_vit, tiny_vit_cfg):
    """Dossier backbone/ d'un run, au format de latentopt/run.py (config.yaml, backbone.pt,
    results.json), à l'emplacement de latentopt/hub.py (results/<run>/backbone/)."""
    run_dir = tmp_path / "results" / BB / "backbone"
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"backbone": tiny_vit_cfg}), run_dir / "config.yaml")
    run.save_backbone(run_dir, tiny_vit, method="test")
    return run_dir


def test_from_run_reloads_identical_backbone(pretrain_run, tiny_vit):
    backbone = from_run(str(pretrain_run))
    assert weights_hash(backbone) == weights_hash(tiny_vit)
    check_backbone(backbone, size=32)


def test_from_run_detects_tampered_weights(pretrain_run):
    sd = torch.load(pretrain_run / "backbone.pt")
    key = next(iter(sd))
    sd[key] = sd[key] + 1
    torch.save(sd, pretrain_run / "backbone.pt")
    with pytest.raises(ValueError, match="empreinte"):
        from_run(str(pretrain_run))


def test_from_run_detects_wrong_recorded_hash(pretrain_run):
    results = json.loads((pretrain_run / "results.json").read_text(encoding="utf-8"))
    results["backbone_sha256"] = "0" * 64
    (pretrain_run / "results.json").write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(ValueError, match="empreinte"):
        from_run(str(pretrain_run))


@pytest.mark.parametrize("missing", ["config.yaml", "backbone.pt", "results.json"])
def test_from_run_requires_a_finished_run(pretrain_run, missing):
    (pretrain_run / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        from_run(str(pretrain_run))


def test_from_run_refuses_weights_in_another_architecture(pretrain_run):
    cfg = OmegaConf.load(pretrain_run / "config.yaml")
    cfg.backbone.embed_dim = 32  # projection de taille différente
    OmegaConf.save(cfg, pretrain_run / "config.yaml")
    with pytest.raises(RuntimeError):
        from_run(str(pretrain_run))


def test_resolve_run_accepts_the_run_folder_its_name_or_a_folder_inside(pretrain_run, monkeypatch):
    monkeypatch.setenv("OUTPUT_ROOT", str(pretrain_run.parents[1]))
    root = pretrain_run.parent
    for run_ref in (root, pretrain_run, BB, f"{BB}/backbone"):  # existants : sans réseau
        assert resolve_run(run_ref) == pretrain_run, run_ref
