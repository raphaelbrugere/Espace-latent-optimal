# flavien : ConvNeXt-Tiny + LeJEPA (première soumission)

## Résumé

Le ViT-S/8 de la baseline donne 4 cartes au même stride 8 : il n'a pas de vraie pyramide,
ce qui le désavantage sur les tâches denses. Cette soumission teste si un **ConvNeXt-Tiny**
(pyramide aux strides 4, 8, 16 et 32) fait mieux **à méthode, données, hyperparamètres et
têtes identiques**. Seule l'architecture du backbone change.

**Conclusion.** À 800 epochs, le ConvNeXt gagne légèrement en classification (+0,9 à +2,2)
mais **perd sur les tâches denses** (seg −3,0, det −4,0). Le gain moyen sur les 5 tâches est
d'environ **−0,8 point**, pour **43 % de calcul de pretraining en moins**. L'hypothèse est
réfutée pour les tâches denses : une pyramide dans le backbone ne suffit pas si les têtes et
le pretraining ne l'exploitent pas.

## Backbone

`timm.create_model("convnext_tiny", pretrained=False, num_classes=0, drop_path_rate=0.1)`,
entraîné from scratch, **27,8 M paramètres**.

| sortie | channels | stride | taille à 128 px |
|---|---|---|---|
| `stage1` | 96 | 4 | 32×32 |
| `stage2` | 192 | 8 | 16×16 |
| `stage3` | 384 | 16 | 8×8 |
| `stage4` | 768 | 32 | 4×4 |
| `embed` | 768 | – | moyenne de `stage4` + LayerNorm, sans couche ajoutée |

## Pretraining

LeJEPA, script de la baseline sans modification de la méthode : Imagenette train
**sans labels**, 128 px, 4 vues, SIGReg 0,02, projecteur de sortie 16, AdamW lr 2e-3,
weight decay 0,05, batch 128, cosinus avec plancher 1e-3, **800 epochs**. Les valeurs sont
les défauts du script ; le `REPORT.md` de la baseline indique un batch de 256, mais le
script utilise 128.

Seul ajout au script : une sauvegarde de l'état complet toutes les 5 epochs et une
option `--resume`, pour survivre à l'annulation d'un job sur le cluster. La reprise est
exacte : un run coupé puis repris donne le même backbone (même hash) qu'un run d'un seul
tenant.

## Têtes

Les 5 têtes de la baseline, **sans modification**, entraînées avec `train/train_head.py`
(AdamW lr 1e-3, batch 64, 20 à 30 epochs, seed 0). Elles s'adaptent seules au backbone via
`feature_info` :

- **classification** : LayerNorm + Linear sur `embed` ;
- **`pets_seg`** : FPN sur les 4 cartes, qui prédit sur une grille 32×32 (16×16 pour la baseline) ;
- **`pets_det`** : soft-argmax sur la carte la plus proche du stride 8, donc `stage2` (192 ch).

## Calcul

| | ConvNeXt-T | ViT-S/8 (baseline) |
|---|---|---|
| pretraining 800 epochs | **5,4 h GPU** | 9,6 h GPU |
| temps par epoch | ~24 s | ~43 s |
| 5 têtes + évaluation | ~10 min | ~10 min |

Un RTX 3090 (24 Go) au DCE de CentraleSupélec. Un pas de pretraining (128 images × 4 vues)
ne tient pas dans les 10 Go d'un RTX 3080, et les cartes sans bf16 (RTX 2080 Ti,
GTX 1080 Ti) ne peuvent pas exécuter les convolutions en bf16.

## Résultats (val, 800 epochs)

La baseline a été réentraînée par nous dans les mêmes conditions (même script, mêmes
hyperparamètres). Son online probe atteint 90,9 %, contre 90,7 % annoncé par les auteurs :
la reproduction est fiable.

| tâche | métrique | ConvNeXt-T | baseline | Δ | params tête (C / B) |
|---|---|---|---|---|---|
| `imagenette_cls` | top-1 | **0,926** | 0,918 | +0,9 | 9 226 / 6 154 |
| `flowers102_cls` | top-1 | **0,428** | 0,406 | +2,2 | 79 974 / 53 350 |
| `galaxy10_cls` | balanced acc | 0,454 | 0,453 | +0,0 | 9 226 / 6 154 |
| `pets_seg` | mIoU | 0,658 | **0,688** | −3,0 | 166 467 / 172 611 |
| `pets_det` | loc_acc | 0,372 | **0,412** | −4,0 | 370 311 / 591 495 |
| **moyenne** | | | | **−0,8** | |

Détail de `pets_det` (ConvNeXt / baseline) : acc à IoU 0,5 = 0,774 / 0,817 ;
à IoU 0,75 = 0,311 / 0,360 ; IoU moyen = 0,647 / 0,670 ; chat ou chien = 0,902 / 0,935.

**Online probe Imagenette pendant le pretraining** (top-1 sur val) :

| epoch | 50 | 100 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|
| ConvNeXt-T | 0,797 | 0,882 | 0,911 | 0,932 | 0,947 | 0,939 |
| baseline | 0,543 | 0,683 | 0,829 | 0,877 | 0,913 | 0,909 |

**Runs courts de 100 epochs** (pretraining séparé, cosinus complet sur 100 epochs) :

| | ConvNeXt-T | baseline | Δ |
|---|---|---|---|
| online probe | 0,886 | 0,759 | +12,7 |
| `pets_seg` mIoU | 0,634 | 0,610 | +2,4 |
| `pets_det` loc_acc | 0,252 | 0,319 | −6,7 |
| heures GPU | 0,67 | 1,19 | |

## Analyse

- **Le ConvNeXt apprend beaucoup plus vite** : il est très bon dès 100 epochs, alors que le
  ViT y est encore loin. Avec un pretraining long, le ViT le rattrape en classification et le
  dépasse sur les tâches denses. Sur la segmentation, l'avantage du ConvNeXt à 100 epochs
  (+2,4) s'inverse à 800 (−3,0).
- **Hypothèse 1, la perte LeJEPA ne vise que `embed`.** Les cartes intermédiaires du
  ConvNeXt (`stage1` à `stage3`), celles que lisent les têtes denses, ne sont entraînées
  qu'indirectement. Chez le ViT, les 4 cartes (blocs 3, 6, 9 et 12) restent à une
  résolution de 16×16 jusqu'au dernier bloc, donc les têtes denses disposent d'une carte
  à la fois fine et profonde.
- **Hypothèse 2, les têtes de la baseline exploitent mal la pyramide.** La détection ne lit
  que `stage2`, qui n'a que 6 blocs de profondeur ; chez le ViT, elle lit la sortie du
  dernier bloc. Le FPN de segmentation additionne des niveaux sans normalisation, alors que
  leurs échelles diffèrent fortement chez ConvNeXt.
- **Limites** : un seul seed, et des val petits (184 images en seg, 367 en det). Les écarts
  de 2 à 3 points sont à confirmer.

## Pistes

1. Têtes qui exploitent la pyramide : détection multi-échelle (`stage2` + `stage3` + `stage4`),
   FPN avec normalisation des niveaux. Le backbone reste figé, donc c'est peu coûteux.
2. Perte dense pendant le pretraining : appliquer LeJEPA aussi aux cartes intermédiaires.
3. Plusieurs seeds par tête pour avoir des barres d'erreur.

## Reproduire

Depuis la racine du dépôt, sur le DCE :

```bash
sbatch slurm/arena_pretrain.sbatch flavien --resume      # écrit backbone.pt et train/pretrain_log.json
sbatch --export=ALL,TAG=800ep slurm/arena_heads.sbatch flavien   # 5 têtes puis evaluate sur val
```

Backbone : hash `1df9b38a4311…` (complet dans `train/pretrain_log.json`). Poids : sur
Hugging Face Hub (lien à ajouter), jamais dans git.
