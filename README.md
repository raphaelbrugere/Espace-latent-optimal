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
│   ├── data/                   # chargement des données
│   │   └── datasets/           # un fichier par dataset
│   ├── models/                 # composants des réseaux
│   │   ├── backbones/          # extracteurs de features
│   │   │   ├── archs/          # nos architectures (MobileNetV2, ConvNeXt)
│   │   │   └── external/       # wrappers timm, hf, torch_hub
│   │   ├── necks/              # passage features → vecteur (CLS, mean, concat ; plus tard FPN)
│   │   └── heads/              # têtes downstream (linear, mlp ; plus tard det/seg)
│   ├── methods/                # procédures d'entraînement Lightning (LeJEPA, supervisé)
│   ├── losses/                 # fonctions de perte (SIGReg…)
│   ├── evaluation/             # kNN, linear probe, few-shot, cache de features
│   └── deploy/                 # export ONNX, quantization INT8
│
├── tools/                      # points d'entrée : pretrain, train, eval, export, prepare_data
├── configs/                    # configs Hydra
│   ├── cluster/                # seul dossier qui change entre CI, local et DCE
│   ├── data/                   # choix du dataset
│   ├── backbone/               # choix du backbone
│   ├── neck/                   # choix du feature_mode
│   ├── head/                   # choix de la tête
│   ├── method/                 # LeJEPA, supervised (frozen | finetune | scratch)
│   └── experiment/             # une config complète par expérience, dont smoke
│
├── slurm/                      # jobs interactifs et sync W&B (le reste passe par submitit)
├── bench/                      # latence, mémoire, énergie sur la cible embarquée
├── reporting/                  # leaderboard et figures pour le rapport
├── projects/                   # exploration libre, hors du code principal
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
