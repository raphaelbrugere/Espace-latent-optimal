"""Download the timm weights and save them as ../backbone.pt (replaces pretraining).

    python submissions/ref_dino_vits8/train/export_timm_weights.py
"""
import argparse
import json
import pathlib
import sys

import torch

SUB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB.parents[1]))
from arena.contract import check_backbone, import_file, weights_hash  # noqa: E402

module = import_file(SUB / "backbone.py", "ref_dino_backbone")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true", help="random weights, no download (tests only)")
    args = p.parse_args()
    backbone = module.TimmViTBackbone(pretrained=not args.dry_run).eval()
    check_backbone(backbone)
    torch.save(backbone.state_dict(), SUB / "backbone.pt")
    log = {"method": "dino", "reference": True, "external_weights": f"timm/{module.ARCH}",
           "pretraining_data": "ImageNet-1k (unlabeled, external weights)", "gpu_hours": None,
           "dry_run": args.dry_run, "backbone_sha256": weights_hash(backbone)}
    (SUB / "train" / "pretrain_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved {SUB / 'backbone.pt'}  ({module.ARCH}, sha256 {log['backbone_sha256'][:12]})")


if __name__ == "__main__":
    main()
