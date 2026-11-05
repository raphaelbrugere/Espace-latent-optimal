"""Evaluate a submission folder. Students: run it on "val" to check your
submission before sending it. Organizers: run it on "test" for the leaderboard.

    python -m arena.evaluate submissions/baseline               # val
    python -m arena.evaluate submissions/baseline --split test  # leaderboard
"""
import argparse
import json
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from arena.contract import BACKBONE_MAX_PARAMS, build_head, count_params, load_backbone, weights_hash
from arena.tasks import TASKS, get_task, to_device, to_float


@torch.no_grad()
def score(backbone, head, task, split, device, bs=64, workers=4):
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    metric = task.metric()
    for x, y in DataLoader(task.eval_dataset(split), batch_size=bs, num_workers=workers,
                           collate_fn=task.collate_fn):
        with torch.autocast(**amp):
            feats = backbone(x.to(device))
            out = head({k: v.float() for k, v in feats.items()})
        metric.update(to_float(out), to_device(y, device))
    return metric.compute()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("submission")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default="results")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sub = pathlib.Path(args.submission)

    backbone = load_backbone(sub, device)           # contract checked here
    sha = weights_hash(backbone)
    n_params = count_params(backbone)
    log_file = sub / "train" / "pretrain_log.json"
    pretrain = json.loads(log_file.read_text()) if log_file.exists() else {}
    result = {"team": sub.name, "split": args.split, "time": time.strftime("%Y-%m-%d %H:%M"),
              "reference": bool(pretrain.get("reference", False)),   # external weights: shown, not ranked
              "backbone": {"sha256": sha, "params": n_params, "gpu_hours": pretrain.get("gpu_hours"),
                           "pretraining_data": pretrain.get("pretraining_data")},
              "violations": [], "tasks": {}}
    if n_params > BACKBONE_MAX_PARAMS:
        result["violations"].append(f"backbone has {n_params:,} params > {BACKBONE_MAX_PARAMS:,}")
    print(f"[{sub.name}] backbone {n_params / 1e6:.1f}M params, sha {sha[:12]}")

    for head_file in sorted((sub / "heads").glob("*.py")):
        name, weights = head_file.stem, head_file.with_suffix(".pt")
        if name not in TASKS or not weights.exists():
            print(f"[{sub.name}] skip heads/{head_file.name} (unknown task or no .pt)")
            continue
        task = get_task(name)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        head = build_head(sub, task, backbone)
        head.load_state_dict(ckpt["state_dict"])
        head = head.to(device).eval()
        issues = []
        if ckpt.get("backbone_sha256") != sha:
            issues.append("head was trained on a different backbone than backbone.pt")
        if count_params(head) > task.head_max_params:
            issues.append(f"head has {count_params(head):,} params > {task.head_max_params:,}")
        metrics = score(backbone, head, task, args.split, device, workers=args.workers)
        result["tasks"][name] = {"score": metrics[task.primary_metric], "metric": task.primary_metric,
                                 "metrics": metrics, "head_params": count_params(head),
                                 "seed": ckpt.get("seed"), "violations": issues}
        flag = "" if not issues else "   VIOLATION: " + "; ".join(issues)
        print(f"[{sub.name}] {name}: {task.primary_metric} = {metrics[task.primary_metric]:.4f}{flag}")

    out = pathlib.Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sub.name}.json").write_text(json.dumps(result, indent=2))
    print(f"[{sub.name}] -> {out / (sub.name + '.json')}")


if __name__ == "__main__":
    main()
