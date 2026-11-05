# LeJEPA Arena

Pretrain a backbone without labels, freeze it, train heads on top, compete.
You are free in **how** you train. The only fixed parts are the **contract**
(what a submission looks like) and the **evaluation** (how it is scored).

```
arena/                     shared, do not modify
  contract.py              the contract: read this first
  tasks.py                 datasets, splits, losses, metrics
  evaluate.py              scores a submission folder
  leaderboard.py           organizers only
calibration/               organizers only: reference heads on known backbones
submissions/baseline/      the naive baseline: copy it to start
submissions/ref_dino_vits8/ reference: DINO ViT-S/8 from timm (external weights, unranked)
  backbone.py              ViT-S/8, LeJEPA minimal example
  heads/*_cls.py           linear probes (Imagenette, Flowers-102, Galaxy10)
  heads/pets_seg.py        FPN decoder
  heads/pets_det.py        soft-argmax localization head
  train/                   pretraining and head training scripts
  REPORT.md
```

## Quick start

```bash
pip install -r requirements.txt
cp -r submissions/baseline submissions/<team>

python submissions/<team>/train/pretrain_lejepa.py                    # writes backbone.pt
python submissions/<team>/train/train_head.py --task imagenette_cls   # writes heads/*.pt
python submissions/<team>/train/train_head.py --task flowers102_cls
python submissions/<team>/train/train_head.py --task galaxy10_cls
python submissions/<team>/train/train_head.py --task pets_seg
python submissions/<team>/train/train_head.py --task pets_det
python -m arena.evaluate submissions/<team>                           # check on val
```

Everything in your folder is yours to change: architecture, LeJEPA variant,
augmentations, head design, losses, schedules. All datasets download on first use (Imagenette, Flowers-102 and Pets through torchvision
into `./data`, or `$ARENA_DATA`; Galaxy10 from the Hugging Face Hub). `ARENA_SYNTHETIC=1` swaps every
dataset for a tiny generated one, so you can test changes on a laptop in a minute.
## Small-league tasks

| task | data | input | metric |
|---|---|---|---|
| `imagenette_cls` | Imagenette | 128px | top-1 |
| `flowers102_cls` | Oxford Flowers-102 | 128px | top-1 |
| `galaxy10_cls` | Galaxy10 DECaLS | 128px | balanced accuracy |
| `pets_seg` | Oxford-IIIT Pet | 128px | mIoU at native mask resolution |
| `pets_det` | Oxford-IIIT Pet head boxes | 128px | localization accuracy (class correct and IoU above threshold, averaged over IoU 0.5 to 0.95) |

Output formats: logits `(B, K)` for classification, per-pixel logits `(B, K, h, w)` at any resolution for segmentation, and `{"box": (B, 4) xyxy in [0, 1], "logits": (B, 2)}` for detection.

## Rules

- Backbone trained from scratch, no labels, on the official pool
  (`arena.tasks.pretraining_images()`, Imagenette train), at most 30M parameters (may be flexible in the future).
- Backbone frozen for all heads. The evaluator rebuilds it from `backbone.pt` and
  rejects a head whose recorded backbone hash does not match.
- Heads within the task's parameter budget (`head_max_params` in `arena/tasks.py`).
- Never train or tune on `test`. Save heads with `save_head()`, which records the
  seed. A re-run of the training scripts expect a similar score.
- (Optional) Report the GPU-hours you used in `REPORT.md` (`train/pretrain_log.json` records
  pretraining time automatically).

## Leaderboard

Run `python -m arena.evaluate submissions/<team> --split test`, then `python -m arena.leaderboard`.
Each task is ranked by its metric, and the overall rank is the mean gain over the baseline across tasks (a task not submitted counts as zero gain).
