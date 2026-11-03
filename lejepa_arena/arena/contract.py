"""THE CONTRACT.

A submission is a folder:

    submissions/<team>/
        backbone.py          def build_backbone() -> nn.Module
        backbone.pt          its state_dict
        heads/<task>.py      def build_head(feature_info, embed_dim, num_classes) -> nn.Module
        heads/<task>.pt      saved by save_head() below
        train/               all your training code (reviewed, and re-run for top teams)
        REPORT.md            pretraining data, GPU-hours, what you tried

Backbone rules
    forward(x) returns a dict:
        "embed": (B, D) pooled vector
        one entry per item of `backbone.feature_info`, a list of
        {"name": str, "channels": int, "stride": int}, high to low resolution:
        (B, channels, H / stride, W / stride)
    `backbone.embed_dim` = D.
    Trained from scratch, without labels. Anything else is up to you!

Head rules
    forward(feats) takes the backbone's dict and returns the task's output
    (see arena/tasks.py). The backbone is frozen, so a head only ever sees those exact
    weights. Architecture, loss, augmentation, optimizer and schedule are up to you.
"""
import hashlib
import importlib.util
import pathlib
import sys

import torch

BACKBONE_MAX_PARAMS = 30_000_000 # Note: Could be updated in the future


def import_file(path, name):
    path = pathlib.Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def weights_hash(module):
    """sha256 over parameters and buffers."""
    h = hashlib.sha256()
    for k, t in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def load_backbone(submission, device="cpu", weights=True):
    sub = pathlib.Path(submission)
    module = import_file(sub / "backbone.py", f"backbone_{sub.name}")
    backbone = module.build_backbone()
    if weights:
        backbone.load_state_dict(torch.load(sub / "backbone.pt", map_location="cpu", weights_only=True))
    check_backbone(backbone)
    return backbone.to(device).eval()


def freeze(backbone):
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.eval()


def build_head(submission, task, backbone):
    sub = pathlib.Path(submission)
    module = import_file(sub / "heads" / f"{task.name}.py", f"head_{sub.name}_{task.name}")
    return module.build_head(backbone.feature_info, backbone.embed_dim, task.num_classes)


def save_head(head, path, backbone, seed, **extra):
    """Store the head with the proof of which frozen backbone it was trained on."""
    torch.save({"state_dict": head.state_dict(), "backbone_sha256": weights_hash(backbone),
                "seed": seed, "argv": sys.argv, **extra}, path)


@torch.no_grad()
def check_backbone(backbone, size=64):
    """Fail early, with a clear message, if the contract is not respected."""
    for attr in ("feature_info", "embed_dim"):
        if not hasattr(backbone, attr):
            raise AttributeError(f"backbone must define `{attr}` (see arena/contract.py)")
    was_training = backbone.training
    backbone.eval()
    out = backbone(torch.zeros(1, 3, size, size, device=next(backbone.parameters()).device))
    backbone.train(was_training)
    if out.get("embed") is None or tuple(out["embed"].shape) != (1, backbone.embed_dim):
        raise ValueError(f"'embed' must have shape (B, {backbone.embed_dim})")
    for f in backbone.feature_info:
        t = out.get(f["name"])
        expected = (1, f["channels"], size // f["stride"], size // f["stride"])
        if t is None or tuple(t.shape) != expected:
            raise ValueError(f"feature '{f['name']}': expected shape {expected}, got "
                             f"{None if t is None else tuple(t.shape)}")
