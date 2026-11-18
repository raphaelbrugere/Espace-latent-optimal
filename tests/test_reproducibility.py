"""Reproductibilite (P3 du cours) : meme config + meme graine => memes poids, sur CPU.

Les runs passent par les fonctions `build` des commandes lo (celles des vrais runs), avec un
petit ViT et quelques pas sur donnees synthetiques. On compare les poids, pas un score :
deux modeles differents peuvent avoir la meme metrique.

Sur GPU, l'egalite exacte n'est pas garantie (sommes en ordre variable) : ces tests
fixent la reference CPU.
"""

import importlib

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir

from latentopt.config import register_resolvers
from latentopt.data.paths import PROJECT_ROOT
from latentopt.models.contract import weights_hash

register_resolvers()

TINY = [
    "backbone.arch=vit_tiny_patch16_224",
    "backbone.embed_dim=32",
    "backbone.blocks=[0,1]",
    "cluster=ci",
]


def load_tool(name: str):
    return importlib.import_module(f"latentopt.cli.run.{name}")


def config(name: str, overrides: list[str], data_root):
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name=name, overrides=overrides)
    cfg.cluster.data_root = str(data_root)
    return cfg


def fit(module, train, val, steps: int):
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=steps,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train, val)
    return trainer


def train_head(isolated_data, *extra):
    tool = load_tool("train")
    cfg = config("train", ["experiment=smoke_train", *TINY, *extra], isolated_data)
    tool.fill_defaults(cfg)
    module, train, val = tool.build(cfg)
    fit(module, train, val, steps=3)
    return module


def same_weights(a: torch.nn.Module, b: torch.nn.Module, atol: float = 0.0) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(
        torch.allclose(sa[k].float(), sb[k].float(), atol=atol, rtol=0) for k in sa
    )


@pytest.mark.parametrize("task", ["synthetic_cls", "synthetic_seg", "synthetic_det"])
def test_head_training_is_reproducible(isolated_data, task):
    a = train_head(isolated_data, f"task={task}")
    b = train_head(isolated_data, f"task={task}")
    assert same_weights(a.head, b.head)
    assert weights_hash(a.backbone) == weights_hash(b.backbone)


def test_head_training_depends_on_seed(isolated_data):
    a = train_head(isolated_data, "seed=0")
    b = train_head(isolated_data, "seed=1")
    assert not same_weights(a.head, b.head)


def test_feature_cache_gives_same_head_as_on_the_fly(isolated_data):
    """Le cache ne change pas l'entrainement : meme init, meme ordre, memes features."""
    cached = train_head(isolated_data, "cache_features=true")
    live = train_head(isolated_data, "cache_features=false")
    assert same_weights(cached.head, live.head, atol=1e-5)


def test_backbone_seed_fixes_random_init(isolated_data):
    """Backbone a poids aleatoires : meme graine, meme backbone (evalue avec ses tetes)."""
    tool = load_tool("train")
    cfg = config("train", ["experiment=smoke_train", *TINY, "cache_features=false"], isolated_data)
    tool.fill_defaults(cfg)
    a, _, _ = tool.build(cfg)
    b, _, _ = tool.build(cfg)
    assert weights_hash(a.backbone) == weights_hash(b.backbone)


def pretrain(isolated_data, seed: int):
    tool = load_tool("pretrain")
    cfg = config(
        "pretrain",
        [
            "experiment=smoke",
            *TINY,
            "data.img_size=32",
            "method.model.hidden_dim=64",
            f"seed={seed}",
        ],
        isolated_data,
    )
    L.seed_everything(cfg.seed, workers=False)  # comme lo pretrain (cli/run/pretrain.py:main)
    module, train, val = tool.build(cfg)
    fit(module, train, val, steps=2)
    return module.model.backbone


def test_pretraining_is_reproducible(isolated_data):
    a, b = pretrain(isolated_data, 0), pretrain(isolated_data, 0)
    assert weights_hash(a) == weights_hash(b)
    assert weights_hash(pretrain(isolated_data, 1)) != weights_hash(a)
