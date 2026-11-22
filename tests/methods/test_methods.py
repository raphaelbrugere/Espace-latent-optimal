"""Methodes d'entrainement : LeJEPA (losses, sonde detachee, scheduler) et tete sur
backbone gele (le backbone ne bouge jamais)."""

import pytest
import torch
from PIL import Image

from latentopt.evaluation.tasks import get_task
from latentopt.methods import FrozenBackboneModule, LeJEPA, MultiViewTransform, build_optimizer
from latentopt.models.contract import weights_hash

# self.log() hors Trainer : sans effet, avertissement attendu
pytestmark = pytest.mark.filterwarnings("ignore:.*self.log.*")

# --- MultiViewTransform ------------------------------------------------------------------


def test_multiview_transform_shapes_and_label():
    torch.manual_seed(0)
    img = Image.new("RGB", (80, 60), (120, 30, 200))
    views, label = MultiViewTransform(size=32, num_views=3)(img, 4)
    assert views.shape == (3, 3, 32, 32) and views.dtype == torch.float32
    assert label == 4


def test_multiview_views_are_independent_draws():
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    arr = (torch.rand(64, 64, 3, generator=rng) * 255).to(torch.uint8).numpy()
    views, _ = MultiViewTransform(size=32, num_views=4)(Image.fromarray(arr), 0)
    assert not torch.equal(views[0], views[1])


# --- LeJEPA ------------------------------------------------------------------------------


@pytest.fixture
def lejepa(pyramid):
    torch.manual_seed(0)
    return LeJEPA(pyramid.train(), proj_dim=4, hidden_dim=32, lamb=0.1, num_probe_classes=3)


def views(n=6, v=2, size=32):
    torch.manual_seed(1)
    return torch.randn(n, v, 3, size, size)


def test_lejepa_loss_combines_sigreg_and_invariance(pyramid):
    model = LeJEPA(pyramid.train(), proj_dim=4, hidden_dim=32, lamb=0.1)
    out = model(views())
    assert set(out) == {"loss", "inv_loss", "sigreg_loss"}
    expected = 0.1 * out["sigreg_loss"] + 0.9 * out["inv_loss"]
    assert torch.allclose(out["loss"], expected)
    assert all(torch.isfinite(v) for v in out.values())


def test_lejepa_invariance_is_zero_for_identical_views(lejepa):
    x = views(v=1).repeat(1, 3, 1, 1, 1)
    assert lejepa(x)["inv_loss"].item() == pytest.approx(0.0, abs=1e-10)


def test_lejepa_probe_only_with_labels(lejepa):
    x, y = views(), torch.tensor([0, 1, 2, 0, 1, 2])
    assert "probe_loss" not in lejepa(x)
    out = lejepa(x, y)
    ssl = 0.1 * out["sigreg_loss"] + 0.9 * out["inv_loss"]
    assert torch.allclose(out["loss"], ssl + out["probe_loss"])


def test_lejepa_probe_does_not_leak_labels_into_backbone(lejepa):
    out = lejepa(views(), torch.tensor([0, 1, 2, 0, 1, 2]))
    out["probe_loss"].backward()
    assert all(p.grad is None for p in lejepa.backbone.parameters())
    assert all(p.grad is None for p in lejepa.projector.parameters())
    assert all(p.grad is not None for p in lejepa.probe.parameters())


def test_lejepa_ssl_loss_trains_backbone_and_projector(lejepa):
    lejepa(views())["loss"].backward()
    assert all(p.grad is not None for p in lejepa.backbone.parameters())
    assert all(p.grad is not None for p in lejepa.projector.parameters())


def test_ssl_parameters_exclude_probe(lejepa):
    ssl = {id(p) for p in lejepa.ssl_parameters()}
    assert not ssl & {id(p) for p in lejepa.probe.parameters()}
    assert len(ssl) == len(list(lejepa.backbone.parameters())) + len(
        list(lejepa.projector.parameters())
    )


def test_build_optimizer_warmup_then_cosine(lejepa):
    opt, sched = build_optimizer(lejepa, total_steps=100, warmup_steps=10, lr=2e-3, eta_min=1e-3)
    assert len(opt.param_groups) == 2  # ssl + sonde
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(2e-5)  # 1 % du lr
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:11], strict=True))  # warmup croissant
    assert lrs[10] == pytest.approx(2e-3)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))  # cosinus decroissant
    assert lrs[-1] == pytest.approx(1e-3, rel=1e-3)  # eta_min absolu


def test_build_optimizer_weight_decay_per_group(lejepa):
    opt, _ = build_optimizer(lejepa, 10, 1, weight_decay=0.05, probe_weight_decay=1e-7)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.05, 1e-7]


def test_build_optimizer_without_probe(pyramid):
    model = LeJEPA(pyramid, proj_dim=4, hidden_dim=8)
    opt, _ = build_optimizer(model, 10, 2)
    assert len(opt.param_groups) == 1


# --- FrozenBackboneModule -----------------------------------------------------------------


@pytest.fixture
def frozen(pyramid, recipe_head):
    torch.manual_seed(0)
    head = recipe_head("linear", pyramid, 10)
    return FrozenBackboneModule(pyramid, head, get_task("synthetic_cls"))


def test_frozen_backbone_stays_in_eval_and_without_grad(frozen):
    frozen.train()
    assert not frozen.backbone.training
    assert frozen.downstream.training
    assert not any(p.requires_grad for p in frozen.backbone.parameters())


def test_frozen_training_step_never_updates_backbone(frozen):
    before = weights_hash(frozen.backbone)
    opt = torch.optim.SGD(frozen.parameters(), lr=1.0)
    frozen.train()
    x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    for _ in range(3):
        opt.zero_grad()
        frozen.training_step((x, y), 0).backward()
        opt.step()
    assert weights_hash(frozen.backbone) == before


def test_frozen_neck_stays_fixed_and_only_the_head_learns(pyramid, recipe_head):
    """Neck repris d'un entrainement precedent : eval(), sans gradient, hors de l'optimiseur."""
    torch.manual_seed(0)
    head = recipe_head("linear", pyramid, 10)
    module = FrozenBackboneModule(pyramid, head, get_task("synthetic_cls"), freeze_neck=True)
    module.train()
    assert not module.downstream.neck.training and module.downstream.head.training
    assert not any(p.requires_grad for p in module.downstream.neck.parameters())
    opt = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=1.0)
    neck_before, head_before = (
        weights_hash(module.downstream.neck),
        weights_hash(module.downstream.head),
    )
    x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    opt.zero_grad()
    module.training_step((x, y), 0).backward()
    opt.step()
    assert weights_hash(module.downstream.neck) == neck_before
    assert weights_hash(module.downstream.head) != head_before


def test_frozen_same_loss_on_images_and_cached_features(frozen):
    x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        cached = frozen.backbone(x)
        a = frozen.training_step((x, y), 0)
        b = frozen.training_step((cached, y), 0)
    assert torch.allclose(a, b)


def test_frozen_validation_accumulates_task_metric(frozen):
    frozen.on_validation_epoch_start()
    for _ in range(2):
        frozen.validation_step((torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])), 0)
    scores = frozen.metric.compute()
    assert set(scores) == {"top1"} and 0 <= scores["top1"] <= 1
    assert frozen.metric.total == 8
