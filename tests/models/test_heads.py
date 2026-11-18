"""Neck + tete par defaut : formes, cles lues, gradients."""

import pytest
import torch


def feats_of(backbone, size=64, batch=2):
    with torch.no_grad():
        return backbone(torch.randn(batch, 3, size, size))


# --- linear -------------------------------------------------------------------------------


@pytest.mark.parametrize("norm", [True, False])
def test_linear_shapes(pyramid, recipe_head, norm):
    head = recipe_head("linear", pyramid, 7, norm=norm)
    assert head.in_keys == ["embed"]
    assert head({"embed": torch.randn(5, pyramid.embed_dim)}).shape == (5, 7)
    assert isinstance(head.neck.norm, torch.nn.LayerNorm) is norm


def test_linear_with_norm_is_scale_invariant(pyramid, recipe_head):
    head = recipe_head("linear", pyramid, 3).eval()
    x = torch.randn(4, pyramid.embed_dim)
    assert torch.allclose(head({"embed": x}), head({"embed": 100 * x}), atol=1e-4)


def test_linear_ignores_other_keys(pyramid, recipe_head):
    head = recipe_head("linear", pyramid, 2)
    out = head({"embed": torch.randn(1, pyramid.embed_dim), "c4": torch.randn(1, 8, 2, 2)})
    assert out.shape == (1, 2)


# --- fpn ----------------------------------------------------------------------------------


def test_fpn_outputs_at_finest_stride(pyramid, recipe_head):
    head = recipe_head("fpn", pyramid, 3)
    assert head.in_keys == ["c4", "c8"]
    out = head(feats_of(pyramid, 64))
    assert out.shape == (2, 3, 16, 16)  # stride 4


def test_fpn_level_selection(pyramid, recipe_head):
    head = recipe_head("fpn", pyramid, 3, levels=["c8"])
    assert head.in_keys == ["c8"]
    assert head({"c8": torch.randn(1, 12, 8, 8)}).shape == (1, 3, 8, 8)


def test_fpn_rejects_unknown_level(pyramid, recipe_head):
    with pytest.raises(ValueError, match="block99"):
        recipe_head("fpn", pyramid, 3, levels=["block99"])


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_fpn_upsample_modes_and_non_square_inputs(pyramid, recipe_head, mode):
    head = recipe_head("fpn", pyramid, 2, upsample=mode)
    with torch.no_grad():
        feats = pyramid(torch.randn(1, 3, 64, 96))
    assert head(feats).shape == (1, 2, 16, 24)


def test_fpn_on_iso_stride_vit(tiny_vit, recipe_head):
    head = recipe_head("fpn", tiny_vit, 3)
    assert head(feats_of(tiny_vit, 64)).shape == (2, 3, 4, 4)


# --- soft_argmax --------------------------------------------------------------------------


def test_soft_argmax_picks_map_closest_to_target_stride(pyramid, recipe_head):
    assert recipe_head("soft_argmax", pyramid, 2).in_keys == ["c8"]
    assert recipe_head("soft_argmax", pyramid, 2, target_stride=4).in_keys == ["c4"]


def test_soft_argmax_prefers_deepest_map_on_ties(tiny_vit, recipe_head):
    assert recipe_head("soft_argmax", tiny_vit, 2).in_keys == ["block2"]


def test_soft_argmax_output_is_valid_box(pyramid, recipe_head):
    head = recipe_head("soft_argmax", pyramid, 2)
    out = head(feats_of(pyramid, 64, batch=4))
    assert out["box"].shape == (4, 4) and out["logits"].shape == (4, 2)
    box = out["box"]
    assert ((box >= 0) & (box <= 1)).all()
    assert (box[:, 0] <= box[:, 2]).all() and (box[:, 1] <= box[:, 3]).all()


def test_soft_argmax_is_differentiable_end_to_end(pyramid, recipe_head):
    head = recipe_head("soft_argmax", pyramid, 2)
    out = head(feats_of(pyramid, 64))
    (out["box"].sum() + out["logits"].sum()).backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
