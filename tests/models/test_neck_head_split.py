"""La coupe neck / tête ne change rien au calcul : les couples livrés (configs/neck/, configs/head/)
reproduisent les anciennes têtes monolithiques, mêmes poids initiaux et mêmes sorties."""

import pytest
import torch
from legacy_heads import FPNHead as LegacyFPN
from legacy_heads import LinearProbe as LegacyLinear
from legacy_heads import SoftArgmaxHead as LegacySoftArgmax
from omegaconf import OmegaConf

from latentopt.models import contract, recipe
from latentopt.models.heads import ConvSegHead, LinearHead
from latentopt.models.necks import EmbedNorm, FPNNeck

PAIRS = {"linear": ("embed_norm", "linear"), "fpn": ("fpn", "conv_seg"),
         "soft_argmax": ("conv_trunk", "soft_argmax")}  # fmt: skip


def from_recipe(name, backbone, num_classes, **neck):
    neck_name, head_name = PAIRS[name]
    cfg = OmegaConf.merge(recipe.load("neck", neck_name), neck)
    return recipe.build(cfg, recipe.load("head", head_name), backbone, num_classes)


def strip(key: str) -> str:
    """`neck.lateral.0.weight` -> `lateral.0.weight` : clé de l'ancienne tête monolithique."""
    return key.removeprefix("neck.").removeprefix("head.")


def feats(pyramid, batch=2):
    torch.manual_seed(1)
    return pyramid(torch.rand(batch, 3, 16, 16))


def build_pair(make_legacy, make_new, seed=7):
    torch.manual_seed(seed)
    old = make_legacy()
    torch.manual_seed(seed)  # même graine : l'ordre de création des couches doit être le même
    return old, make_new()


CASES = {
    "linear": (
        lambda p: LegacyLinear(p.embed_dim, 5),
        lambda p: from_recipe("linear", p, 5),
    ),
    "linear_strict": (
        lambda p: LegacyLinear(p.embed_dim, 5, norm=False),
        lambda p: from_recipe("linear", p, 5, norm=False),
    ),
    "fpn": (
        lambda p: LegacyFPN(p.feature_info, 3, channels=8),
        lambda p: from_recipe("fpn", p, 3, channels=8),
    ),
    "fpn_one_level": (
        lambda p: LegacyFPN(p.feature_info, 3, channels=8, levels=["c8"], upsample="bilinear"),
        lambda p: from_recipe("fpn", p, 3, channels=8, levels=["c8"], upsample="bilinear"),
    ),
    "soft_argmax": (
        lambda p: LegacySoftArgmax(p.feature_info, 2, channels=8),
        lambda p: from_recipe("soft_argmax", p, 2, channels=8),
    ),
    "soft_argmax_fine": (
        lambda p: LegacySoftArgmax(p.feature_info, 2, channels=8, target_stride=4),
        lambda p: from_recipe("soft_argmax", p, 2, channels=8, target_stride=4),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_same_initial_weights_under_the_same_seed(pyramid, case):
    legacy, new = CASES[case]
    old, composed = build_pair(lambda: legacy(pyramid), lambda: new(pyramid))
    new_state = {strip(k): v for k, v in composed.state_dict().items()}
    assert list(new_state) == list(old.state_dict())  # mêmes clés, même ordre
    for key, value in old.state_dict().items():
        assert torch.equal(new_state[key], value), key


@pytest.mark.parametrize("case", CASES)
def test_same_outputs_and_params(pyramid, case):
    legacy, new = CASES[case]
    old, composed = build_pair(lambda: legacy(pyramid), lambda: new(pyramid))
    assert contract.count_params(old) == contract.count_params(composed)
    assert old.in_keys == composed.in_keys
    x = feats(pyramid)
    for module in (old, composed):
        module.eval()
    out_old, out_new = old(x), composed(x)
    if isinstance(out_old, dict):
        assert out_old.keys() == out_new.keys()
        for k in out_old:
            assert torch.equal(out_old[k], out_new[k])
    else:
        assert torch.equal(out_old, out_new)


def test_same_defaults_as_the_legacy_heads(pyramid):
    """Recettes sans surcharge = anciennes têtes avec leurs valeurs par défaut."""
    for legacy, name, k in (
        (lambda: LegacyLinear(pyramid.embed_dim, 5), "linear", 5),
        (lambda: LegacyFPN(pyramid.feature_info, 3), "fpn", 3),
        (lambda: LegacySoftArgmax(pyramid.feature_info, 2), "soft_argmax", 2),
    ):
        old, composed = build_pair(legacy, lambda n=name, k=k: from_recipe(n, pyramid, k))
        assert contract.count_params(old) == contract.count_params(composed), name


def test_same_training_step_for_a_head_with_batchnorm(pyramid):
    """Mode train : les statistiques BatchNorm du neck évoluent comme dans l'ancienne tête."""
    old, composed = build_pair(
        lambda: LegacySoftArgmax(pyramid.feature_info, 2, channels=8),
        lambda: from_recipe("soft_argmax", pyramid, 2, channels=8),
    )
    x = feats(pyramid, batch=4)
    out_old, out_new = old.train()(x), composed.train()(x)
    assert torch.equal(out_old["box"], out_new["box"])
    assert torch.equal(old.bn1.running_mean, composed.neck.bn1.running_mean)


# --- Downstream et compatibilité neck / tête -----------------------------------------------


def test_downstream_exposes_the_neck_keys_and_one_state_dict(pyramid):
    model = from_recipe("fpn", pyramid, 3, channels=8)
    assert model.in_keys == model.neck.in_keys == ["c4", "c8"]
    assert {k.split(".")[0] for k in model.state_dict()} == {"neck", "head"}


def test_incompatible_neck_and_head_fail_early(pyramid):
    with pytest.raises(ValueError, match="produit un 'map', LinearHead attend un 'vector'"):
        contract.Downstream(FPNNeck(pyramid.feature_info, 8), LinearHead(8, 3))
    with pytest.raises(ValueError, match="produit un 'vector', ConvSegHead attend un 'map'"):
        contract.Downstream(EmbedNorm(pyramid.embed_dim), ConvSegHead(pyramid.embed_dim, 3))


def test_embed_norm_without_norm_is_the_identity_on_embed(pyramid):
    """Le cas « pas de neck » d'origine : sonde linéaire stricte, `embed` rendu tel quel."""
    x = feats(pyramid)
    neck = EmbedNorm(pyramid.embed_dim, norm=False)
    assert neck.in_keys == ["embed"] and neck.out_kind == "vector"
    assert torch.equal(neck(x), x["embed"]) and not list(neck.parameters())
