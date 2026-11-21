# Configs

Les configs Hydra sont dans [`configs/`](../configs/). Un dossier est un **groupe** : on choisit un fichier
avec `groupe=fichier`, et on surcharge une clé avec `groupe.clé=valeur`.

```bash
lo pretrain backbone=vit_s8 data=imagenette method.epochs=100
lo train backbone=from_run backbone.run=<run> task=pets_seg neck=fpn neck.channels=128 head=conv_seg
```

Règle : ce qui change le résultat d'un run est dans `method/`, `data/`, `backbone/`, `neck/` ou `head/`, jamais dans
`cluster/`.

## Premier niveau : un fichier par commande

| Fichier | Commande | Groupes choisis | Clés propres |
|---|---|---|---|
| `pretrain.yaml` | `lo pretrain` | `backbone` (obligatoire), `method` (`lejepa`), `data` (`imagenette`), `cluster` (`local`), `experiment` | `seed`, `resume`, `allow_dirty`, `trainer.*` |
| `train.yaml` | `lo train` | `backbone` (obligatoire), `neck`, `head` (ceux de la tâche), `cluster`, `experiment` | `task` (obligatoire), `neck_from`, `seed`, `epochs`, `max_steps_per_epoch`, `batch_size`, `precision`, `optimizer.*`, `cache_features`, `backbone_run`, `backbone_data`, `backbone_method`, `resume`, `allow_dirty`, `trainer.*` |
| `eval.yaml` | `lo eval` | `cluster` | `downstream` (obligatoire), `split`, `cache_features`, `precision`, `batch_size`, `experiment_name`, `allow_dirty` |

## `backbone/` : l'extracteur de features

Exemple, `vit_s8.yaml` :

```yaml
_target_: latentopt.models.backbones.archs.vit.TimmViT
arch: vit_small_patch8_224 # architecture timm
img_size: ${oc.select:data.img_size,128} # celle du pool en pré-entraînement, 128 sinon
embed_dim: 512 # projection après le pooling ; null = token CLS
blocks: [2, 5, 8, 11] # blocs exposés comme cartes de features
drop_path_rate: 0.1
grad_checkpointing: ${oc.select:cluster.grad_checkpointing,false} # réglé dans cluster/
```

| Fichier | Rôle | Clés |
|---|---|---|
| `vit_s8.yaml` | ViT-S/8 à poids aléatoires, baseline | celles ci-dessus |
| `vit_s8_dino.yaml` | ViT-S/8 aux poids DINO (référence externe) | `img_size`, `blocks`, `drop_path_rate` ; `grad_checkpointing` lu dans `cluster/` |
| `from_run.yaml` | backbone d'un de nos pré-entraînements | `run` : dossier du run, absolu ou relatif à la racine (obligatoire) |

## `method/` : la méthode de pré-entraînement

| Clé (`lejepa.yaml`) | Rôle |
|---|---|
| `model.proj_dim`, `model.hidden_dim` | projecteur MLP (sortie, couches cachées) |
| `model.lamb` | poids de SIGReg ; `1 - lamb` pour l'invariance |
| `num_views` | vues augmentées par image |
| `epochs`, `batch_size`, `warmup_epochs`, `max_steps_per_epoch` | durée et taille des pas |
| `precision` | `fp16`, `bf16` ou `fp32` |
| `optimizer.*` | `lr`, `weight_decay`, `eta_min`, et ceux de la sonde (`probe_lr`, `probe_weight_decay`) |

## `data/` : le pool de pré-entraînement

Un fichier par dataset ; `pretrain.yaml` en choisit un (`data=…`, défaut `imagenette`). Les datasets des tâches
aval ne passent pas par ce groupe : ils sont choisis par `task=`.

| Clé | Rôle |
|---|---|
| `source._target_` | `pretraining_images` du module du dataset : les images du pool |
| `img_size` | résolution du pré-entraînement |
| `num_probe_classes` | classes de la sonde en ligne ; `null` = pas de sonde |
| `probe_task` | tâche dont le val sert à la sonde ; `null` = pas de validation |

| Fichier | Pool | Sonde |
|---|---|---|
| `imagenette.yaml` | train officiel (pool du protocole) | `imagenette_cls` |
| `flowers102.yaml` | train officiel | `flowers102_cls` |
| `galaxy10.yaml` | split train figé (80 %) | `galaxy10_cls` |
| `oxford_pets.yaml` | trainval sans le test de `pets_det` | aucune |
| `synthetic.yaml` | images générées (smoke run, CI) | `synthetic_cls` |

Seul `imagenette` suit le protocole ; aucun pool ne contient de test (voir [DATASETS.md](DATASETS.md)).

## `neck/` et `head/` : ce qu'on entraîne sur le backbone gelé

Un neck adapte les features du backbone (vecteur ou carte), une tête produit la sortie de la tâche. Chaque
config porte un nom : c'est la partie `<neck>` ou `<tête>` du dossier `downstream/<tâche>-<neck>-<tête>-<date>/`
([HUB.md](HUB.md)). Sans `neck=` ni `head=`, `lo train` prend ceux de la baseline de la tâche ;
`neck_from=<tâche>-<neck>-<tête>-<date>` reprend le neck d'un entraînement précédent, gelé, et seule la tête apprend.

```yaml
# neck/fpn.yaml
_target_: latentopt.models.necks.FPNNeck
_partial_: true
channels: 64
levels: null
upsample: nearest
```

| Neck | Classe (clés) | Sortie |
|---|---|---|
| `embed_norm` | `EmbedNorm` (`norm` : LayerNorm, ou `embed` tel quel) | vecteur |
| `fpn` | `FPNNeck` (`channels`, `levels`, `upsample`) | carte |
| `conv_trunk` | `ConvTrunk` (`channels`, `target_stride`) | carte |

| Tête | Classe | Entrée |
|---|---|---|
| `linear` | `LinearHead` | vecteur |
| `conv_seg` | `ConvSegHead` | carte |
| `soft_argmax` | `SoftArgmaxDecoder` | carte |

| Tâches | Neck et tête par défaut |
|---|---|
| classification | `embed_norm` + `linear` |
| segmentation | `fpn` + `conv_seg` |
| détection | `conv_trunk` + `soft_argmax` |

Les dimensions (`feature_info`, `embed_dim`, `in_dim`, `num_classes`) sont fournies à l'exécution. Une variante
(`neck.channels=128`) garde le nom de sa config ; une variante retenue mérite son propre fichier. Les paramètres
du neck et de la tête sont comptés séparément dans `results.json`.

## `cluster/` : l'environnement d'exécution

| Clé | Rôle |
|---|---|
| `data_root`, `results_root` | emplacement des données et des résultats |
| `mlflow` | base MLflow par run ; `false` = désactivé |
| `num_workers`, `device`, `progress_bar` | DataLoader, matériel, affichage |
| `grad_checkpointing` | recalcule les activations du backbone au backward : ~5x moins de mémoire GPU, ~25 % plus lent, même résultat. `false` par défaut, `true` sur `dce` (sans lui, OOM sur une RTX 3080) ; lu par les configs de `backbone/` |
| `compile` | `torch.compile` du backbone dans `lo pretrain` : `auto` (défaut) s'il tourne sur GPU avec Triton et un compilateur C, `true` l'exige (erreur au démarrage sinon ; défaut de `dce`), `false` jamais. Au bruit numérique près, sans effet sur le résultat ; l'état réel est noté dans `results.json` (`compiled`) |
| `hydra.launcher.*` (`dce.yaml`) | job Slurm : partition, durée, CPU, mémoire, contrainte de nœud |

Fichiers : `local.yaml` (machine personnelle), `ci.yaml` (GitHub Actions, CPU), `dce.yaml` (cluster, avec `-m`).

## `experiment/` : une expérience complète

Un fichier qui fixe les groupes (`override /data: …`) et surcharge les clés d'une expérience, choisi par
`experiment=…`. Il définit aussi `experiment_name`, le nom de l'expérience dans MLflow.

| Fichier | Rôle |
|---|---|
| `vit_lejepa_imagenette.yaml` | baseline, référence de toutes les expériences (800 epochs, bf16) |
| `smoke.yaml` | `lo pretrain` minuscule, données synthétiques, avant un job long |
| `smoke_train.yaml` | `lo train` minuscule sur une tâche synthétique |
