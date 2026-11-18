"""Contrat Backbone / Head : empreinte des poids, vérifications, sauvegarde des têtes."""

import copy

import pytest
import torch
import torch.nn as nn

from latentopt.models import contract
from latentopt.models.heads import ConvSegHead, LinearHead, SoftArgmaxDecoder
from latentopt.models.necks import ConvTrunk, EmbedNorm, FPNNeck

# --- weights_hash ---------------------------------------------------------------------


def test_weights_hash_is_stable_and_hex(pyramid):
    h = contract.weights_hash(pyramid)
    assert h == contract.weights_hash(pyramid)
    assert len(h) == 64 and int(h, 16) >= 0


def test_weights_hash_equal_for_copies(pyramid):
    assert contract.weights_hash(copy.deepcopy(pyramid)) == contract.weights_hash(pyramid)


def test_weights_hash_changes_with_any_parameter(pyramid):
    before = contract.weights_hash(pyramid)
    with torch.no_grad():
        pyramid.proj.bias[0] += 1e-6
    assert contract.weights_hash(pyramid) != before


def test_weights_hash_covers_buffers():
    m = nn.BatchNorm1d(4)
    before = contract.weights_hash(m)
    m.running_mean += 1.0
    assert contract.weights_hash(m) != before


def test_weights_hash_independent_of_training_mode(pyramid):
    before = contract.weights_hash(pyramid)
    pyramid.train()
    assert contract.weights_hash(pyramid) == before


def test_weights_hash_depends_on_key_names():
    a, b = nn.Sequential(nn.Linear(2, 2)), nn.Sequential()
    b.add_module("other", copy.deepcopy(a[0]))
    assert contract.weights_hash(a) != contract.weights_hash(b)


def test_count_params(pyramid):
    expected = (3 * 8 * 16 + 8) + (8 * 12 * 4 + 12) + (12 * 16 + 16)
    assert contract.count_params(pyramid) == expected


def test_freeze_disables_grads_and_sets_eval(pyramid):
    pyramid.train()
    frozen = contract.freeze(pyramid)
    assert frozen is pyramid and not frozen.training
    assert not any(p.requires_grad for p in frozen.parameters())


# --- check_backbone --------------------------------------------------------------------


def test_check_backbone_accepts_contract_backbones(pyramid, tiny_vit):
    contract.check_backbone(pyramid, size=64)
    contract.check_backbone(tiny_vit, size=64)


def test_check_backbone_restores_training_mode(pyramid):
    pyramid.train()
    contract.check_backbone(pyramid, size=32)
    assert pyramid.training


@pytest.mark.parametrize("attr", ["feature_info", "embed_dim"])
def test_check_backbone_requires_attributes(pyramid, attr):
    delattr(pyramid, attr)
    with pytest.raises(AttributeError, match=attr):
        contract.check_backbone(pyramid)


def test_check_backbone_requires_increasing_strides(pyramid):
    pyramid.feature_info = list(reversed(pyramid.feature_info))
    with pytest.raises(ValueError, match="stride le plus fin"):
        contract.check_backbone(pyramid)


def test_check_backbone_rejects_wrong_declared_channels(pyramid):
    pyramid.feature_info[0]["channels"] = 99
    with pytest.raises(ValueError, match="c4"):
        contract.check_backbone(pyramid, size=32)


def test_check_backbone_rejects_wrong_embed_dim(pyramid):
    pyramid.embed_dim = 3
    with pytest.raises(ValueError, match="embed"):
        contract.check_backbone(pyramid, size=32)


def test_check_backbone_rejects_missing_map(pyramid):
    pyramid.feature_info.append({"name": "c16", "channels": 12, "stride": 16})
    with pytest.raises(ValueError, match="c16"):
        contract.check_backbone(pyramid, size=32)


def test_check_backbone_enforces_param_budget(pyramid, monkeypatch):
    monkeypatch.setattr(contract, "BACKBONE_MAX_PARAMS", 10)
    with pytest.raises(ValueError, match="paramètres"):
        contract.check_backbone(pyramid)


# --- save_neck / save_head / load_neck / load_head --------------------------------------


def saved(pyramid, tmp_path, recipe_head):
    """Neck et tête d'un même entraînement, sauvegardés séparément."""
    model = recipe_head("fpn", pyramid, 3, channels=8)
    neck_sha = contract.save_neck(model.neck, tmp_path / "neck.pt", pyramid, seed=3,
                                  config={"x": 1})  # fmt: skip
    sha = contract.save_head(model.head, tmp_path / "head.pt", pyramid, model.neck, seed=3,
                             task="toy")  # fmt: skip
    return model, neck_sha, sha


def test_neck_and_head_are_saved_separately_with_their_params(pyramid, tmp_path, recipe_head):
    model, neck_sha, sha = saved(pyramid, tmp_path, recipe_head)
    assert sha == contract.weights_hash(pyramid) and neck_sha == contract.weights_hash(model.neck)
    neck = torch.load(tmp_path / "neck.pt", weights_only=False)
    head = torch.load(tmp_path / "head.pt", weights_only=False)
    assert neck["params"] == contract.count_params(model.neck) and neck["config"] == {"x": 1}
    assert head["params"] == contract.count_params(model.head)
    assert head["neck_sha256"] == neck_sha and head["backbone_sha256"] == sha
    assert not any(k.startswith("neck") for k in head["state_dict"])  # la tête seule


def test_save_and_load_roundtrip(pyramid, tmp_path, recipe_head):
    model, _, _ = saved(pyramid, tmp_path, recipe_head)
    fresh = recipe_head("fpn", pyramid, 3, channels=8)
    contract.load_neck(fresh.neck, tmp_path / "neck.pt", pyramid)
    ckpt = contract.load_head(fresh.head, tmp_path / "head.pt", pyramid, fresh.neck)
    assert ckpt["seed"] == 3 and ckpt["task"] == "toy" and "argv" in ckpt
    for k, v in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)


def test_loading_refuses_another_backbone(pyramid, tmp_path, recipe_head):
    saved(pyramid, tmp_path, recipe_head)
    other = copy.deepcopy(pyramid)
    with torch.no_grad():
        other.proj.weight.mul_(2)
    fresh = recipe_head("fpn", pyramid, 3, channels=8)
    with pytest.raises(ValueError, match="autre backbone"):
        contract.load_neck(fresh.neck, tmp_path / "neck.pt", other)
    with pytest.raises(ValueError, match="autre backbone"):
        contract.load_head(fresh.head, tmp_path / "head.pt", other, fresh.neck)


def test_head_refuses_another_neck(pyramid, tmp_path, recipe_head):
    saved(pyramid, tmp_path, recipe_head)
    fresh = recipe_head("fpn", pyramid, 3, channels=8)  # neck neuf : autres poids
    with pytest.raises(ValueError, match="autre neck"):
        contract.load_head(fresh.head, tmp_path / "head.pt", pyramid, fresh.neck)


# --- make_neck / make_head / make_downstream ---------------------------------------------


def test_make_downstream_passes_only_requested_arguments(pyramid):
    probe = contract.make_downstream(EmbedNorm, LinearHead, pyramid, num_classes=4)
    assert probe.head.fc.in_features == pyramid.embed_dim and probe.head.fc.out_features == 4

    fpn = contract.make_downstream(FPNNeck, ConvSegHead, pyramid, num_classes=3)
    assert fpn.neck.names == ["c4", "c8"]

    det = contract.make_downstream(ConvTrunk, SoftArgmaxDecoder, pyramid, num_classes=2)
    assert det.head.classifier.out_features == 2


def test_make_downstream_works_with_hydra_partials(pyramid):
    from functools import partial

    neck = partial(FPNNeck, channels=8, levels=["c8"])
    head = contract.make_downstream(neck, ConvSegHead, pyramid, 3)
    assert head.neck.names == ["c8"] and head.head.logits.in_channels == 8


def test_make_head_reads_the_neck_output_width(pyramid):
    neck = contract.make_neck(FPNNeck, pyramid)
    assert neck.out_dim == 64
    assert contract.make_head(ConvSegHead, neck, 3).conv.in_channels == 64
