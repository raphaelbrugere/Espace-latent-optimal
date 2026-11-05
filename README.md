# Espace-latent-optimal
Évaluer et améliorer la capacité de transfert de tâche de réseaux embarquables, en mutualisant un même backbone entre plusieurs têtes spécialisées.

## Contexte

Beaucoup d’applications embarquées font tourner plusieurs réseaux en parallèle : en automobile par exemple, segmentation sémantique, détection d’objets et estimation de profondeur. Leur cumul est difficile à embarquer sur un système contraint.

Un réseau se décompose en deux parties :

- l’**extracteur de features (backbone)**, qui projette le signal brut (image, son, texte…) dans un espace latent ne conservant que sa sémantique ;
- la **tête applicative**, qui utilise ces features pour produire la prédiction.

On mutualise donc le backbone et on ajoute une tête par tâche, comme le font VampNet [0], NeuroCorgi [1] ou CLIP [2]. Cela réduit la charge de calcul et ouvre la voie à des implémentations très spécialisées et très efficaces énergétiquement (cf. NeuroCorgi [1]).

## Objectifs

1. **Évaluer les capacités de transfert de tâche de réseaux classiques et embarquables**
   - modèles : MobileNetV2 [4], ConvNeXt [5] ;
   - tâches : segmentation, classification, détection, décodage.
2. **Étudier des méthodes pour construire un espace latent optimal pour le multi-tâche**
   - LeJEPA [3] et son régulariseur SIGReg ;
   - apprentissage par tâches auxiliaires.

## Références

- [0] VampNet
- [1] NeuroCorgi
- [2] CLIP — Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021
- [3] LeJEPA — Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, 2025
- [4] MobileNetV2 — Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2018
- [5] ConvNeXt — Liu et al., *A ConvNet for the 2020s*, 2022

## Structure du dépôt

```
Espace-latent-optimal/
├── README.md                   # présentation, installation, démarrage rapide
├── AUTHORS.md                  # noms des contributeurs
├── CONTRIBUTING.md             # règles d'équipe : branches, PR, relecture, smoke test
├── INSTALL.md                  # installation de l'environnement et de pre-commit
├── pyproject.toml              # métadonnées, dépendances, config ruff et pytest
├── uv.lock                     # versions exactes des dépendances (généré par uv)
├── .python-version             # version de Python (3.12)
├── .pre-commit-config.yaml     # nettoyage automatique à chaque commit
├── .gitignore                  # fichiers jamais commités (données, poids, sorties)
├── .gitattributes              # fins de ligne LF pour tout le monde
│
├── src/latentopt/              # bibliothèque, identique en local et sur Slurm
│   ├── config.py               # resolvers Hydra (${project_root:})
│   ├── run.py                  # dossier de run commun : config résolue, git, log, sauvegardes, garde-fous
│   ├── data/                   # chargement des données (lecture seule, jamais de téléchargement)
│   │   ├── datasets/           # un fichier par dataset (Imagenette, Flowers-102, Galaxy10, Pets, synthétiques)
│   │   ├── splits/             # splits figés, un JSON par dataset (commités)
│   │   ├── paths.py            # DATA_ROOT, FEATURES_ROOT, MODELS_ROOT
│   │   └── transforms.py       # transforms du protocole d'évaluation
│   ├── models/                 # composants des réseaux
│   │   ├── contract.py         # contrat Backbone / Head : vérifications, empreinte des poids, save_head
│   │   ├── backbones/          # extracteurs de features
│   │   │   ├── archs/          # architectures au contrat, poids aléatoires (ViT ; plus tard MobileNetV2, ConvNeXt)
│   │   │   ├── external/       # poids externes (DINO…) : téléchargement et chargement dans une archi d'archs/
│   │   │   └── checkpoint.py   # from_run : backbone d'un de nos runs de pré-entraînement
│   │   ├── necks/              # passage features → vecteur (vide : embed est calculé par le backbone)
│   │   └── heads/              # têtes downstream : LinearProbe, FPNHead (seg), SoftArgmaxHead (det)
│   ├── methods/                # procédures d'entraînement Lightning : LeJEPA, tête sur backbone gelé
│   ├── losses/                 # fonctions de perte : SIGReg, segmentation, détection
│   ├── evaluation/             # tâches (registre, budgets), métriques, cache de features, évaluation
│   └── deploy/                 # export ONNX, quantization INT8 (à venir)
│
├── tools/                      # points d'entrée : pretrain, train, eval, prepare_data
├── configs/                    # configs Hydra
│   ├── pretrain.yaml, train.yaml, eval.yaml   # un fichier de premier niveau par point d'entrée
│   ├── cluster/                # local, ci, dce : seul dossier qui change entre CI, local et DCE
│   ├── data/                   # pool de pré-entraînement (imagenette, synthetic)
│   ├── backbone/               # vit_s8, vit_s8_dino, from_run
│   ├── neck/                   # choix du feature_mode (vide pour l'instant)
│   ├── head/                   # linear, fpn, soft_argmax et leurs hyperparamètres
│   ├── method/                 # lejepa
│   └── experiment/             # une config complète par expérience, dont smoke et smoke_train
│
├── slurm/                      # COMMANDS.md : toutes les commandes (local, smoke, DCE) ; jobs hors submitit
├── bench/                      # latence, mémoire, énergie sur la cible embarquée
├── reporting/                  # leaderboard.py, figures pour le rapport
├── projects/                   # exploration libre, hors du code principal
│
├── data/                       # DATA_ROOT : dossiers suivis par git (.gitkeep), contenu ignoré
│   ├── datasets/               # sources téléchargées par tools/prepare_data.py
│   ├── features/               # features mises en cache (FEATURES_ROOT)
│   └── models/                 # poids externes téléchargés (MODELS_ROOT)
├── results/                    # un dossier par run : config, log, métriques, checkpoints (ignoré)
│
├── tests/                      # tests unitaires et smoke test
│   ├── data/                   # tests des datasets et de l'anti-fuite
│   ├── models/                 # conformité au contrat Backbone / Head
│   └── methods/                # 2 steps de chaque méthode
│
├── docs/                       # documentation
│   ├── DATASETS.md             # source, licence, checksum, taille, split de chaque dataset
│   └── CLUSTER.md              # mode d'emploi du DCE : quotas, partitions, Internet
│
├── .vscode/                    # réglages VS Code partagés (voir plus bas)
│
└── .github/                    # CI, CODEOWNERS
    ├── workflows/              # CI : pre-commit + tests à chaque PR et push sur main
    └── ISSUE_TEMPLATE/         # templates experiment, bug, task
```

## Installation

Voir [INSTALL.md](INSTALL.md) : installation de uv, création de l'environnement, activation de pre-commit et détail de ce qu'il fait.

## Démarrage rapide

Le pipeline : pré-entraîner un backbone, entraîner une tête par tâche sur ce backbone gelé, puis évaluer.
Chaque run écrit un dossier complet dans `results/<étape>/<expérience>/<date>/` : config résolue, `run.log`,
métriques, poids et `results.json`.

```bash
# vérifier le pipeline en ~1 min, sur CPU et données synthétiques
uv run python tools/pretrain.py experiment=smoke
uv run python tools/train.py experiment=smoke_train task=synthetic_cls,synthetic_seg,synthetic_det -m

# vraies données (une fois)
uv run python tools/prepare_data.py --freeze-splits

# pré-entraînement, têtes, évaluation
uv run python tools/pretrain.py experiment=vit_lejepa_imagenette cluster=dce -m
uv run python tools/train.py backbone=from_run backbone.run=results/pretrain/<exp>/<date> \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det cluster=dce -m
uv run python tools/eval.py name=<nom> runs='[results/train/<exp>/<date>/imagenette_cls,...]'
uv run python reporting/leaderboard.py --split val
```

Toutes les commandes, avec leurs variantes et pièges : [slurm/COMMANDS.md](slurm/COMMANDS.md).
Le DCE : [docs/CLUSTER.md](docs/CLUSTER.md). Les données : [docs/DATASETS.md](docs/DATASETS.md).

## Réglages VS Code (`.vscode/`)

Le dossier `.vscode/` est commité pour que toute l'équipe ait le même comportement :

- `settings.json` :
  - formatage avec ruff et tri des imports à chaque enregistrement ;
  - fins de ligne LF, suppression des espaces en fin de ligne ;
  - interpréteur Python du `.venv`, tests pytest dans l'onglet Tests ;
  - masque les fichiers sans intérêt (`.gitkeep`, caches, `.venv`, `wandb/`) et regroupe les fichiers de config sous `pyproject.toml`, `README.md` et `.gitignore` (flèche pour les déplier) ;
  - exclut les sorties d'expériences de la surveillance et de la recherche, pour que VS Code reste rapide.
- `extensions.json` : extensions proposées à l'ouverture du dépôt (Python et Ruff).

Les préférences personnelles (thème, police, raccourcis) restent dans les paramètres utilisateur de chacun.
