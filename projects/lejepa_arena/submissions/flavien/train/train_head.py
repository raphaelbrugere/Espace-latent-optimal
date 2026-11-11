"""Baseline head training on the frozen backbone.

    python submissions/baseline/train/train_head.py --task pets_seg --epochs 30
    python submissions/baseline/train/train_head.py --task imagenette_cls --epochs 20

Standalone on purpose: change the loop, the loss, the augmentation, anything.
Two things to keep:
  * the backbone stays frozen (the evaluator uses backbone.pt anyway);
  * save with arena.contract.save_head, which records the seed and the backbone
    hash so organizers can re-run this script and reproduce your score.

Easy speedup to try: with a deterministic train transform, compute the
backbone features once and train the head on the cached tensors.
"""
import argparse
import math
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

SUB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB.parents[1]))
from arena.contract import build_head, count_params, freeze, load_backbone, save_head  # noqa: E402
from arena.tasks import get_task, to_device, to_float  # noqa: E402

DEFAULT_EPOCHS = {"imagenette_cls": 20, "flowers102_cls": 30, "galaxy10_cls": 20, "pets_seg": 30, "pets_det": 30}


def evaluate(backbone, head, loader, task, device, amp):
    head.eval()
    metric = task.metric()
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(**amp):
                out = head(backbone(x.to(device)))
            metric.update(to_float(out), to_device(y, device))
    return metric.compute()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="recorded, used by organizers to reproduce")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    task = get_task(args.task)
    epochs = args.epochs or DEFAULT_EPOCHS.get(args.task, 20)

    backbone = freeze(load_backbone(SUB, device))
    torch.manual_seed(args.seed)                      # head init and data order depend on the seed
    head = build_head(SUB, task, backbone).to(device)
    print(f"{task.name}: head {count_params(head):,} params (budget {task.head_max_params:,})")

    train_tf = getattr(task, "flip_transform", None)  # simple augmentation when the task offers one
    train = DataLoader(task.dataset("train", transform=train_tf), batch_size=args.bs, shuffle=True,
                       drop_last=True, num_workers=args.workers, persistent_workers=args.workers > 0)
    val = DataLoader(task.eval_dataset("val"), batch_size=args.bs, num_workers=args.workers,
                     collate_fn=task.collate_fn)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))

    for epoch in range(epochs):
        head.train()
        running = 0.0
        for x, y in train:
            x, y = x.to(device, non_blocking=True), to_device(y, device)
            with torch.no_grad(), torch.autocast(**amp):
                feats = backbone(x)
            with torch.autocast(**amp):
                loss = task.loss(to_float(head({k: v.float() for k, v in feats.items()})), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        scores = evaluate(backbone, head, val, task, device, amp)
        print(f"epoch {epoch:>3}  loss {running / len(train):.4f}  val {scores}", flush=True)

    out = SUB / "heads" / f"{task.name}.pt"
    save_head(head.cpu(), out, backbone.cpu(), seed=args.seed, val=scores, epochs=epochs)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
