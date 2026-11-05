# Baseline (organizers)

**Backbone.** ViT-S/8, LeJEPA with the minimal example's recommended settings (SIGReg weight
0.02, 4 views, projector output 16, lr 2e-3, batch 256, 800 epochs), pretrained from scratch on
Imagenette train images without labels, 128px. The authors report 90.7% linear-probe top-1.

**Heads.** Linear probe (LayerNorm + Linear on the embedding) for `imagenette_cls`,
the FPN decoder of `train_seg_ablation.py` for `pets_seg`.

**Pretraining compute.** See `train/pretrain_log.json`.

**Known weaknesses, on purpose.** A plain ViT gives four maps at the same stride
(8), so the FPN has no real pyramid and segmentation is predicted at 1/8
resolution. The linear probe only reads the final embedding. No augmentation for
classification heads, no feature caching, no hyperparameter search.
