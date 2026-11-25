"""Registre des taches : coherence de chaque tache, et chaine complete
dataset -> tete par defaut -> loss -> metrique sur les taches synthetiques."""

import pytest
import torch
from torch.utils.data import DataLoader

from latentopt.data.paths import PROJECT_ROOT
from latentopt.evaluation import tasks
from latentopt.models import recipe

PROTOCOL_TASKS = {"imagenette_cls", "flowers102_cls", "galaxy10_cls", "pets_seg", "pets_det"}
SYNTHETIC = ["synthetic_cls", "synthetic_seg", "synthetic_det"]


def test_registry_contains_protocol_and_smoke_tasks():
    assert set(tasks.TASKS) == PROTOCOL_TASKS | set(SYNTHETIC)
    assert all(name == t.name for name, t in tasks.TASKS.items())


def test_get_task_unknown_lists_available():
    with pytest.raises(KeyError, match="imagenette_cls"):
        tasks.get_task("nope")


@pytest.mark.parametrize("name", sorted(tasks.TASKS))
def test_task_is_consistent(name):
    t = tasks.get_task(name)
    assert (PROJECT_ROOT / "configs" / "neck" / f"{t.default_neck}.yaml").exists()
    assert (PROJECT_ROOT / "configs" / "head" / f"{t.default_head}.yaml").exists()
    assert t.epochs > 0 and t.num_classes > 1
    # les transforms d'eval sont deterministes
    assert getattr(t.eval_pipeline, "hflip", 0.0) == 0.0
    # Task est fige : le protocole ne se modifie pas a l'execution
    with pytest.raises(AttributeError):
        t.epochs = 1


@pytest.mark.parametrize(
    "real,synthetic",
    [("imagenette_cls", "synthetic_cls"), ("pets_seg", "synthetic_seg"),
     ("pets_det", "synthetic_det")],
)  # fmt: skip
def test_synthetic_tasks_mirror_the_real_protocol(real, synthetic):
    r, s = tasks.get_task(real), tasks.get_task(synthetic)
    for field in ("primary_metric", "loss", "default_neck", "default_head"):
        assert getattr(r, field) == getattr(s, field), field
    assert type(r.eval_pipeline) is type(s.eval_pipeline)
    assert (r.collate_fn is None) == (s.collate_fn is None)


def test_galaxy_uses_balanced_accuracy():
    assert tasks.get_task("galaxy10_cls").primary_metric == "balanced_acc"


@pytest.mark.parametrize("name", SYNTHETIC)
def test_synthetic_task_end_to_end(name, pyramid):
    """Dataset -> backbone -> tete par defaut -> loss (gradient) -> metrique principale."""
    t = tasks.get_task(name)
    neck_cfg, head_cfg = recipe.load("neck", t.default_neck), recipe.load("head", t.default_head)
    head = recipe.build(neck_cfg, head_cfg, pyramid, t.num_classes)

    x, y = next(iter(DataLoader(t.dataset("train", train=True), batch_size=4)))
    with torch.no_grad():
        feats = pyramid(x)
    loss = t.loss(tasks.to_float(head(feats)), y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())

    metric = t.metric()
    val = DataLoader(t.dataset("val"), batch_size=4, collate_fn=t.collate_fn)
    xv, yv = next(iter(val))
    with torch.no_grad():
        metric.update(tasks.to_float(head(pyramid(xv))), yv)
    scores = metric.compute()
    assert t.primary_metric in scores and 0.0 <= scores[t.primary_metric] <= 1.0


def test_to_device_handles_nested_structures():
    y = {"box": torch.zeros(2, 4), "label": [torch.ones(1), 3]}
    out = tasks.to_device(y, torch.device("cpu"))
    assert torch.equal(out["box"], y["box"]) and out["label"][1] == 3


def test_to_float_handles_tensors_and_dicts():
    assert tasks.to_float(torch.ones(2, dtype=torch.float16)).dtype == torch.float32
    out = tasks.to_float({"box": torch.ones(1, dtype=torch.bfloat16)})
    assert out["box"].dtype == torch.float32
    assert tasks.to_float("x") == "x"
