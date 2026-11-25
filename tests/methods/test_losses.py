"""Losses : SIGReg distingue N(0, I) d'un collapse ; losses supervisees coherentes."""

import pytest
import torch
import torch.nn.functional as F

from latentopt.losses import SIGReg, detection_loss, resize_logits, segmentation_loss

# --- SIGReg ------------------------------------------------------------------------------


@pytest.fixture
def sigreg():
    torch.manual_seed(0)
    return SIGReg()


def test_sigreg_returns_finite_nonnegative_scalar(sigreg):
    loss = sigreg(torch.randn(4, 256, 16))
    assert loss.shape == () and torch.isfinite(loss) and loss >= 0


def test_sigreg_is_small_for_isotropic_gaussian(sigreg):
    gauss = sigreg(torch.randn(2, 512, 16))
    assert gauss < 5


@pytest.mark.parametrize(
    "bad",
    [
        lambda z: torch.zeros_like(z),  # collapse
        lambda z: 3 * z,  # trop disperse
        lambda z: 0.3 * z,  # trop concentre
        lambda z: z + 1,  # decentre
    ],
    ids=["collapse", "scale_x3", "scale_x0.3", "shift"],
)
def test_sigreg_penalizes_non_gaussian_embeddings(sigreg, bad):
    z = torch.randn(2, 512, 16)
    assert sigreg(bad(z)) > 20 * sigreg(z)


def test_sigreg_gradient_is_finite_and_pushes_collapse_away(sigreg):
    z = (0.01 * torch.randn(2, 128, 8)).requires_grad_()
    sigreg(z).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_sigreg_quadrature_buffers():
    s = SIGReg(knots=5, domain=2.0)
    assert torch.allclose(s.t, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]))
    trapezoid = torch.tensor([0.5, 1.0, 1.0, 1.0, 0.5])
    assert torch.allclose(s.weights, trapezoid * torch.exp(-(s.t**2) / 2))
    # buffers : suivent .to(device) et le state_dict
    assert {"t", "phi", "weights"} <= set(s.state_dict())


def test_sigreg_draws_new_directions_at_each_call():
    s = SIGReg(num_slices=4)
    z = torch.randn(1, 64, 8)
    torch.manual_seed(1)
    a = s(z)
    torch.manual_seed(1)
    assert s(z) == a  # meme graine : meme resultat
    assert s(z) != a  # tirage suivant : autres directions


# --- Losses supervisees --------------------------------------------------------------------


def test_resize_logits_is_noop_at_target_size():
    x = torch.randn(1, 3, 8, 8, dtype=torch.float16)
    assert resize_logits(x, (8, 8)) is x


def test_resize_logits_upsamples_in_fp32():
    x = torch.randn(2, 3, 4, 4, dtype=torch.float16)
    y = resize_logits(x, (10, 6))
    assert y.shape == (2, 3, 10, 6) and y.dtype == torch.float32


def test_segmentation_loss_is_pixel_ce_at_mask_resolution():
    logits, target = torch.randn(2, 3, 8, 8), torch.randint(0, 3, (2, 8, 8))
    assert torch.allclose(segmentation_loss(logits, target), F.cross_entropy(logits, target))
    coarse = torch.randn(2, 3, 4, 4)
    loss = segmentation_loss(coarse, target)
    assert torch.allclose(loss, F.cross_entropy(resize_logits(coarse, (8, 8)), target))


def test_detection_loss_reduces_to_ce_for_perfect_boxes():
    box = torch.tensor([[0.1, 0.2, 0.5, 0.6], [0.3, 0.3, 0.9, 0.8]])
    logits, label = torch.randn(2, 4), torch.tensor([1, 3])
    out = {"box": box.clone(), "logits": logits}
    loss = detection_loss(out, {"box": box, "label": label})
    assert torch.allclose(loss, F.cross_entropy(logits, label), atol=1e-6)


def test_detection_loss_grows_with_box_error_and_has_gradients():
    target = {"box": torch.tensor([[0.2, 0.2, 0.6, 0.6]]), "label": torch.tensor([0])}
    logits = torch.tensor([[5.0, -5.0]])
    near = detection_loss({"box": torch.tensor([[0.21, 0.2, 0.6, 0.6]]), "logits": logits}, target)
    far_box = torch.tensor([[0.7, 0.7, 0.9, 0.9]], requires_grad=True)
    far = detection_loss({"box": far_box, "logits": logits}, target)
    assert far > near
    far.backward()
    assert far_box.grad.abs().sum() > 0
