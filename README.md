# Espace-latent-optimal
Évaluer et améliorer la capacité de transfert de tâche de réseaux embarquables, en mutualisant un même backbone entre plusieurs têtes spécialisées.

## Contexte

Beaucoup d’applications embarquées font tourner plusieurs réseaux en parallèle : en automobile par exemple, segmentation sémantique, détection d’objets et estimation de profondeur. Leur cumul est difficile à embarquer sur un système contraint.

Un réseau se décompose en deux parties :

- l’**extracteur de features (backbone)**, qui projette le signal brut (image, son, texte…) dans un espace latent ne conservant que sa sémantique ;
- la **tête applicative**, qui utilise ces features pour produire la prédiction.

On mutualise donc le backbone et on ajoute une tête par tâche, comme le font NeuroCorgi [1] ou CLIP [2] ; VampNet [0] pousse l'idée plus loin avec un seul modèle qui change de tâche par masquage, sans tête dédiée. Cela réduit la charge de calcul et ouvre la voie à des implémentations très spécialisées et très efficaces énergétiquement (cf. NeuroCorgi [1]).

## Objectifs

1. **Évaluer les capacités de transfert de tâche de réseaux classiques et embarquables**
   - modèles : MobileNetV2 [4], ConvNeXt [5] ;
   - tâches : segmentation, classification, détection, décodage.
2. **Étudier des méthodes pour construire un espace latent optimal pour le multi-tâche**
   - LeJEPA [3] et son régulariseur SIGReg ;
   - apprentissage par tâches auxiliaires.

## Références

- [0] VampNet — Flores García et al., *VampNet: Music Generation via Masked Acoustic Token Modeling*, ISMIR 2023
- [1] NeuroCorgi — Miro-Panades et al. (CEA-List), *A 772μJ/frame ImageNet Feature Extractor Accelerator on HD Images at 30FPS*, APCCAS 2024
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
│   ├── cli/                    # commande `lo` : un module par commande, groupés par famille
│   │   ├── run/                # lo pretrain, lo train, lo eval (Hydra)
│   │   ├── data/               # lo data prepare, lo clear
│   │   └── store/              # lo hub push|pull, lo mlflow sync
│   ├── config.py               # resolvers Hydra (${project_root:}, noms des dossiers de run)
│   ├── hub.py                  # dossier d'un backbone (backbone/, downstream/) et son dépôt Hugging Face (docs/HUB.md)
│   ├── run.py                  # dossier de run commun : config résolue, git, log, sauvegardes, garde-fous, suivi MLflow
│   ├── data/                   # chargement des données (lecture seule, jamais de téléchargement)
│   │   ├── datasets/           # un fichier par dataset (Imagenette, Flowers-102, Galaxy10, Pets, synthétiques)
│   │   ├── splits/             # splits figés, un JSON par dataset (commités)
│   │   ├── splits.py           # calcul et lecture des splits figés
│   │   ├── paths.py            # DATA_ROOT, FEATURES_ROOT, MODELS_ROOT
│   │   └── transforms.py       # transforms du protocole d'évaluation
│   ├── models/                 # composants des réseaux
│   │   ├── contract.py         # contrat Backbone / Neck / Head : vérifications, empreintes, neck.pt et head.pt
│   │   ├── backbones/          # extracteurs de features
│   │   │   ├── archs/          # architectures au contrat, poids aléatoires (ViT ; plus tard MobileNetV2, ConvNeXt)
│   │   │   ├── external/       # poids externes (DINO…) : téléchargement et chargement dans une archi d'archs/
│   │   │   └── checkpoint.py   # from_run : backbone d'un de nos runs de pré-entraînement
│   │   ├── necks/              # du dict du backbone à ce que lit la tête : EmbedNorm, FPNNeck, ConvTrunk
│   │   ├── heads/              # têtes : LinearHead, ConvSegHead, SoftArgmaxDecoder
│   │   └── recipe.py           # configs nommées des necks et des têtes (configs/neck/, configs/head/)
│   ├── methods/                # procédures d'entraînement Lightning : LeJEPA, tête sur backbone gelé
│   ├── losses/                 # fonctions de perte : SIGReg, segmentation, détection
│   └── evaluation/             # registre des tâches, métriques, cache de features, évaluation
│
├── configs/                    # configs Hydra : ce que calcule un run, et sur quelle machine
│   ├── pretrain.yaml, train.yaml, eval.yaml   # un fichier de premier niveau par commande Hydra
│   ├── env/                    # local, ci, dce : la machine (chemins, workers, device), sans effet sur le résultat
│   ├── data/                   # pool de pré-entraînement : imagenette (protocole), flowers102, galaxy10, oxford_pets, synthetic
│   ├── backbone/               # vit_s8, vit_s8_dino, from_run
│   ├── neck/                   # necks : embed_norm, fpn, conv_trunk et leurs hyperparamètres
│   ├── head/                   # têtes : linear, conv_seg, soft_argmax
│   ├── method/                 # lejepa
│   └── experiment/             # une config complète par expérience, dont smoke et smoke_train
│
├── slurm/                      # tout ce qui est propre au DCE et à Slurm, et rien d'autre
│   ├── README.md               # compte, partitions, nœuds, lancer un job, balayages, pièges
│   └── launcher/               # profils de job Slurm (long, short) : launcher=long -m
├── bench/                      # latence, mémoire, énergie sur la cible embarquée
├── projects/                   # exploration libre, hors du code principal
│
├── data/                       # DATA_ROOT : dossiers suivis par git (.gitkeep), contenu ignoré
│   ├── datasets/               # sources téléchargées par `lo data prepare`
│   ├── features/               # features mises en cache (FEATURES_ROOT)
│   └── models/                 # poids externes téléchargés (MODELS_ROOT)
├── results/                    # un dossier par backbone : backbone/, downstream/ (ignoré)
│
├── tests/                      # tests unitaires et smoke test
│   ├── data/                   # tests des datasets et de l'anti-fuite
│   ├── models/                 # conformité au contrat Backbone / Neck / Head, recettes
│   ├── methods/                # 2 steps de chaque méthode, pertes
│   └── evaluation/             # tâches, métriques, cache de features
│
├── docs/                       # documentation
│   ├── COMMANDS.md             # toutes les commandes (local, smoke, DCE), avec variantes et pièges
│   ├── CONFIGS.md              # les groupes de configs Hydra et leurs clés
│   ├── DATASETS.md             # source, licence, checksum, taille, split de chaque dataset
│   ├── HUB.md                  # noms des runs, organisation Hugging Face, publication
│   └── MLFLOW.md               # comparer les runs : base d'équipe, interface, filtres
│
├── .vscode/                    # réglages VS Code partagés (voir plus bas)
│
└── .github/                    # CI et templates GitHub
    ├── workflows/              # lint.yml : pre-commit à chaque PR et push sur main ;
    │                           # ci.yml : tests et smoke, seulement si le code change
    ├── ISSUE_TEMPLATE/         # templates experiment, bug, task
    └── pull_request_template.md
```

## Installation

Voir [INSTALL.md](INSTALL.md) : installation de uv, création de l'environnement, activation de pre-commit et détail de ce qu'il fait.

## Démarrage rapide

Le pipeline : pré-entraîner un backbone, entraîner un neck et une tête par tâche sur ce backbone gelé, puis évaluer.
Tout passe par une seule commande, `lo` (`uv run lo --help` liste les commandes par famille).
Chaque backbone a son dossier `results/<run>/`, avec `backbone/` et un dossier `downstream/<…>/` par
entraînement de neck + tête (ses évaluations dans `eval/`) : config résolue, `run.log`, métriques, poids et
`results.json` ([docs/HUB.md](docs/HUB.md) pour les noms).

```bash
# vérifier le pipeline sur CPU et données synthétiques
uv run lo pretrain experiment=smoke
uv run lo train experiment=smoke_train task=synthetic_cls,synthetic_seg,synthetic_det -m

# vraies données (une fois)
uv run lo data prepare

# pré-entraînement, têtes, évaluation
uv run lo pretrain experiment=vit_lejepa_imagenette launcher=long -m
uv run lo train backbone=from_run backbone.run=<run> \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det launcher=long -m
# une tête de plus sur un neck déjà entraîné (gelé)
uv run lo train backbone=from_run backbone.run=<run> task=pets_det neck_from=pets_seg-fpn-conv_seg-<date> launcher=long -m
uv run lo eval downstream='[<run>/downstream/imagenette_cls-embed_norm-linear-<date>,...]'

# publication sur Hugging Face (un dépôt privé par backbone, avec ses entraînements et évaluations)
uv run lo hub push <run>
uv run lo hub pull <run>          # le récupérer ailleurs (ou donner simplement son nom)

# comparer tous les runs (Hub + results/) dans la base d'équipe data/mlflow.db (docs/MLFLOW.md)
uv run lo mlflow sync
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db   # http://localhost:5000
```

Toutes les commandes, avec leurs variantes et pièges : [docs/COMMANDS.md](docs/COMMANDS.md).
Le DCE et Slurm : [slurm/README.md](slurm/README.md). Les données : [docs/DATASETS.md](docs/DATASETS.md).
Noms des runs et Hugging Face : [docs/HUB.md](docs/HUB.md). Les configs : [docs/CONFIGS.md](docs/CONFIGS.md).

## Réglages VS Code (`.vscode/`)

Le dossier `.vscode/` est commité pour que toute l'équipe ait le même comportement :

- `settings.json` :
  - formatage avec ruff et tri des imports à chaque enregistrement ;
  - fins de ligne LF, suppression des espaces en fin de ligne ;
  - interpréteur Python du `.venv`, tests pytest dans l'onglet Tests ;
  - masque les fichiers sans intérêt (`.gitkeep`, caches, `.venv`) et regroupe les fichiers de config sous `pyproject.toml`, `README.md` et `.gitignore` (flèche pour les déplier) ;
  - exclut les sorties d'expériences de la surveillance et de la recherche, pour que VS Code reste rapide.
- `extensions.json` : extensions proposées à l'ouverture du dépôt (Python et Ruff).

Les préférences personnelles (thème, police, raccourcis) restent dans les paramètres utilisateur de chacun.
