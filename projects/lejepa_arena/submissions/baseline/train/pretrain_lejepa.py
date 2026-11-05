"""Baseline LeJEPA pretraining (the LeJEPA minimal example: ViT-S/8 on Imagenette).

Defaults reproduce the authors' recommended run (lamb 0.02, V 4, proj_dim 16,
lr 2e-3, bs 256, 800 epochs, cosine floor 1e-3, SIGReg on [0, 3]), reported at
90.7% linear-probe top-1 on Imagenette in about 200 minutes on one GPU.

Standalone on purpose: read it top to bottom, copy it, change anything.
Only the output must respect the contract: ../backbone.pt loadable by
../backbone.py's build_backbone().

    python submissions/baseline/train/pretrain_lejepa.py               # 800 epochs
    python submissions/baseline/train/pretrain_lejepa.py --epochs 100  # quicker, weaker
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import MLP
from torchvision.transforms import v2

SUB = pathlib.Path(__file__).resolve().parents[1]      # the submission folder
sys.path.insert(0, str(SUB.parents[1]))                  # repo root, for `arena`
from arena.contract import check_backbone, import_file, weights_hash  # noqa: E402
from arena.tasks import MEAN, STD, get_task, pretraining_images  # noqa: E402

build_backbone = import_file(SUB / "backbone.py", "baseline_backbone").build_backbone


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (unchanged from train_lejepa.py)."""

    def __init__(self, knots=17, num_slice=256, domain=3.0):
        super().__init__()
        self.num_slice = num_slice
        t = torch.linspace(0, domain, knots)
        dt = domain / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)

    def forward(self, proj):  # proj: (V, N, D)
        A = torch.randn(proj.size(-1), self.num_slice, device=proj.device)
        A = A / A.norm(dim=0)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


class Views(torch.utils.data.Dataset):
    """V augmented views of each image, LeJEPA augmentations."""

    def __init__(self, raw, V, size):
        self.raw, self.V = raw, V
        self.aug = v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        img, label = self.raw[i]
        return torch.stack([self.aug(img) for _ in range(self.V)]), label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--V", type=int, default=4, help="views per image")
    p.add_argument("--lamb", type=float, default=0.02, help="SIGReg weight")
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--eta_min", type=float, default=1e-3, help="cosine floor (minimal example: 1e-3)")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_steps_per_epoch", type=int, default=None, help="for quick tests")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    torch.manual_seed(0)

    train = DataLoader(Views(pretraining_images(), args.V, args.img_size), batch_size=args.bs, shuffle=True,
                       drop_last=True, num_workers=args.workers, persistent_workers=args.workers > 0)
    probe_task = get_task("imagenette_cls")
    val = DataLoader(probe_task.dataset("val"), batch_size=256, num_workers=args.workers)

    backbone = build_backbone().to(device)
    check_backbone(backbone)
    proj = MLP(backbone.embed_dim, [2048, 2048, args.proj_dim], norm_layer=nn.BatchNorm1d).to(device)
    # Online probe: trained on DETACHED embeddings, so labels never reach the backbone.
    # It is only a progress monitor; the real evaluation uses a fresh head on the frozen backbone.
    probe = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, 10)).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW([
        {"params": [*backbone.parameters(), *proj.parameters()], "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}])
    steps = min(len(train), args.max_steps_per_epoch or len(train))
    warmup, total = steps, steps * args.epochs
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, 0.01, total_iters=warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, total - warmup), eta_min=args.eta_min)], [warmup])

    t0, probe_history = time.time(), []
    for epoch in range(args.epochs):
        backbone.train(); proj.train(); probe.train()
        for i, (views, y) in enumerate(train):
            if i >= steps:
                break
            views, y = views.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N, V = views.shape[:2]
            with torch.autocast(**amp):
                emb = backbone(views.flatten(0, 1))["embed"]                  # (N*V, D)
                z = proj(emb).reshape(N, V, -1).transpose(0, 1)                # (V, N, d)
                inv_loss = (z.mean(0) - z).square().mean()                     # views agree
                sig_loss = sigreg(z)                                           # no collapse
                loss = args.lamb * sig_loss + (1 - args.lamb) * inv_loss
                loss = loss + F.cross_entropy(probe(emb.detach()), y.repeat_interleave(V))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        backbone.eval(); probe.eval()
        correct = total_n = 0
        with torch.no_grad(), torch.autocast(**amp):
            for x, y in val:
                correct += (probe(backbone(x.to(device))["embed"]).argmax(1).cpu() == y).sum().item()
                total_n += len(y)
        print(f"epoch {epoch:>3}  inv {inv_loss.item():.4f}  sigreg {sig_loss.item():.4f}  "
              f"online-probe val acc {correct / total_n:.4f}", flush=True)
        probe_history.append(round(correct / total_n, 4))

    backbone.cpu().eval()
    torch.save(backbone.state_dict(), SUB / "backbone.pt")
    log = {"method": "lejepa", "pretraining_data": "imagenette train (unlabeled)",
           "online_probe_val_acc": probe_history[-1] if probe_history else None,
           "online_probe_history": probe_history,
           "gpu_hours": round((time.time() - t0) / 3600, 3) if device == "cuda" else 0.0,
           "backbone_sha256": weights_hash(backbone), "args": vars(args)}
    (SUB / "train" / "pretrain_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved {SUB / 'backbone.pt'}  sha256 {log['backbone_sha256'][:12]}")


if __name__ == "__main__":
    main()
