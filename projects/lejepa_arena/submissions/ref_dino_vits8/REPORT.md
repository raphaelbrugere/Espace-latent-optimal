# Reference: DINO ViT-S/8 (timm)

**Not a competing submission.** It uses external pretrained weights, which the rules forbid for
students. It is shown unranked on the leaderboard, as a reference point.

**Backbone.** `vit_small_patch8_224.dino` from timm: the same ViT-S/8 as the baseline, pretrained
with DINO (self-supervised, no labels) on ImageNet-1k at 224px, about 130 times more images than
the Imagenette pool. Run at 128px here, as every backbone in the small league.

**Heads.** Identical to the baseline (linear probes, FPN decoder, soft-argmax detection head),
trained with the baseline's `train_head.py`.

**What it measures.** Against the baseline, the effect of pretraining data scale and method on
the same architecture. The embedding is DINO's CLS token (384-d) instead of the baseline's
512-d projection.

    python submissions/ref_dino_vits8/train/export_timm_weights.py
    for t in imagenette_cls flowers102_cls galaxy10_cls pets_seg pets_det; do
      python submissions/ref_dino_vits8/train/train_head.py --task $t
    done
    python -m arena.evaluate submissions/ref_dino_vits8
