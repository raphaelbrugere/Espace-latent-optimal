# Datasets

Règle : aucun set de test downstream dans le pré-entraînement.

Les tâches, les splits figés et les transforms d'évaluation forment le **protocole** : ils ne changent
pas, pour que nos scores restent comparables entre eux. Le code est dans
[`src/latentopt/data/datasets/`](../src/latentopt/data/datasets/), les tâches dans
[`src/latentopt/evaluation/tasks.py`](../src/latentopt/evaluation/tasks.py).

| Dataset | Usage | Source | Licence | Checksum (MD5 des archives) | Taille | Splits |
|---|---|---|---|---|---|---|
| Imagenette 160px | pool du protocole (`data=imagenette`) ; `imagenette_cls` | [fast.ai](https://github.com/fastai/imagenette), `imagenette2-160.tgz` | Apache-2.0 (dépôt fast.ai) ; images issues d'ImageNet, soumises à ses conditions d'usage | `e793b78cc4c9e9a4ccc0c1155377a412` | 99 Mo | train 9 469 / val 3 925 officiels → train / val = holdout 95/5 du train (**figé**), test = val officiel |
| Oxford Flowers-102 | pool `data=flowers102` ; `flowers102_cls` | [VGG Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | non précisée sur la page officielle | images `52808999861908f626f3c1f4e79d11fa`, labels `e0620be6f572b9609742df49c70aed4d`, splits `a5357ecc9cb78c4bef273ce3793fc85c` | 345 Mo | officiels : train 1 020 / val 1 020 / test 6 149 |
| Galaxy10 DECaLS | pool `data=galaxy10` ; `galaxy10_cls` | [Hugging Face `matthieulel/galaxy10_decals`](https://huggingface.co/datasets/matthieulel/galaxy10_decals), d'après [Galaxy10](https://github.com/henrysky/Galaxy10) | MIT (dépôt Galaxy10) ; images DESI Legacy Imaging Surveys, labels Galaxy Zoo | révision HF épinglée `d1cfafdcce69ca7cd25aecce1a0b578cf07e6a30` | 2,5 Go | 17 736 images (train HF 15 962 + test HF 1 774, concaténés) → stratifié 80/10/10 (**figé**) |
| Oxford-IIIT Pet (segmentation) | pool `data=oxford_pets` ; `pets_seg` | [VGG Oxford](https://www.robots.ox.ac.uk/~vgg/data/pets/) | CC BY-SA 4.0 | images `5c4f3ee8e5d25df40f4fd59a7f44e54c`, annotations `95a8c909bbe2e81eed6a22bccdf3f68f` | 792 Mo + 19 Mo | trainval 3 680 / test 3 669 officiels → train / val = holdout 95/5 de trainval (**figé**), test = test officiel |
| Oxford-IIIT Pet (détection) | `pets_det` | idem, boîtes de tête dans `annotations/xmls` | CC BY-SA 4.0 | idem | idem | images de trainval ayant une boîte → stratifié 80/10/10 par espèce (**figé**) |

Les MD5 sont ceux que torchvision vérifie au téléchargement. Les tailles sont celles des archives.

## Un dataset dans le code

Chaque module de dataset de `src/latentopt/data/datasets/` (hors `base.py`, `registry.py` et `__init__.py`) définit une
classe et son instance `DATASET`. Un dataset, c'est une classe qui donne :

- son nom (`NAME`), son nom court (`ABBREV`, voir plus bas) ;
- son pool de pré-entraînement (`pretraining_images`) ;
- ses tâches (`tasks()`) : pour chaque type (`"cls"`, `"seg"`, `"det"`), le constructeur qui fabrique les données d'un
  split. Le protocole de la tâche (transforms, métrique) est dans `evaluation/tasks.py`.

Un dataset à télécharger (`DownloadableDataset`) ajoute `present`, `download` et `freeze_splits`, appelés par
`lo data prepare`. `synthetic`, généré à la volée, n'en est pas un : `lo data prepare` ne le propose pas.

Le `DATASET` ne garde aucune donnée : il fabrique des données brutes (`source[i]` donne `(image, cible)`, `ids()` les
identifiants des splits figés), que la tâche filtre ensuite par split et transforme pour le DataLoader. Ces données
brutes ne sont pas toujours un split officiel : Galaxy10 concatène le train et le test de Hugging Face, et les pools de
pré-entraînement de Galaxy10 et de Pets sont déjà filtrés.

Règle : la classe porte ce que le reste du projet lit (`NAME`, `ABBREV`, méthodes) ; les constantes propres au
dataset (dossier, révision…) restent dans son module.

Une classe incomplète ne peut pas être instanciée : l'oubli se voit dès l'import. Ajouter un dataset :

1. écrire sa classe et son `DATASET`, et l'ajouter à `ALL` dans `registry.py` ;
2. créer `configs/data/<NAME>.yaml` (son pool) ;
3. déclarer ses tâches dans `evaluation/tasks.py` ;
4. figer son nom court dans `PINNED`, `tests/data/test_abbreviations.py`.

Les tests du registre et des pools signalent un oubli.

## Noms courts

Chaque dataset définit un nom court, `ABBREV`. C'est la partie `<données>` du nom d'un backbone pré-entraîné, donc de
son dossier `results/<run>/` et de son dépôt Hugging Face ([HUB.md](HUB.md)). Il est lu sur le dataset désigné par
`data.source._target_` (`<module>.DATASET.pretraining_images`), jamais dans les configs : renommer un fichier de
`configs/data/` ne renomme pas les runs.

| Dataset | `ABBREV` |
|---|---|
| `imagenette` | `imnt160` (suit la variante chargée : 160px) |
| `flowers102` | `fl102` |
| `galaxy10` | `gal10d` (DECaLS, à distinguer de Galaxy10 SDSS) |
| `oxford_pets` | `pets` |
| `synthetic` | `synth` |

Un nom court ne se modifie jamais : il renommerait les runs existants. Il est unique, fait de minuscules et de
chiffres, et figé par `tests/data/test_abbreviations.py`. Un nouveau dataset de pré-entraînement doit en définir un.

## Tâches

| Tâche | Entrée | Classes | Métrique |
|---|---|---|---|
| `imagenette_cls` | 128px, petit côté + crop central | 10 | top-1 |
| `flowers102_cls` | 128px, petit côté + crop central | 102 | top-1 |
| `galaxy10_cls` | 128px, resize carré sans crop | 10 (déséquilibrées) | balanced accuracy |
| `pets_seg` | 128px carré ; masque à la résolution native | 3 (animal, fond, contour) | mIoU à la résolution native |
| `pets_det` | 128px carré ; boîte normalisée dans [0, 1] | 2 (chat, chien) | localisation : classe juste et IoU ≥ seuil, moyenné sur IoU 0.5 à 0.95 |

Chaque tâche déclare dans `evaluation/tasks.py` le traitement de ses images, en deux parties
(détail dans `data/transforms.py`) :

- **`prepare`** (la colonne « Entrée ») : taille et conversion, appliquée à l'évaluation comme à l'entraînement.
  Elle fait partie du protocole : la modifier rend les scores incomparables.
- **`augment`** : l'augmentation à l'entraînement des têtes, jamais à l'évaluation. Aujourd'hui, un
  retournement horizontal une fois sur deux pour `pets_seg` et `pets_det`, aucune pour la classification. Les
  augmentations sont celles de torchvision : on peut les combiner, la cible (masque, boîte) suit l'image.

Le cache de features calcule chaque version d'une augmentation (image normale et retournée), puis en tire une
au hasard à chaque passage. Une empreinte du traitement est enregistrée avec les features et dans les résultats
de `lo train` : modifier le traitement ne réutilise jamais d'anciennes features, et `lo eval` signale une tête
entraînée avec un autre traitement que le code actuel.

Les images sortent dans [0, 1] : la normalisation est faite par le backbone (contrat dans `models/contract.py`).

## Respect de la règle

Le pool de pré-entraînement est le **train officiel d'Imagenette**. Il contient le val d'`imagenette_cls`,
puisque ce val est tiré de ce même train, mais aucun test downstream : le test d'`imagenette_cls` est le val
officiel d'Imagenette, jamais vu en pré-entraînement.

Les autres datasets ont aussi un pool (`configs/data/`), hors protocole : leurs scores ne sont pas
comparables à ceux de la baseline. Ils ne contiennent aucun test : Flowers-102 et Galaxy10 se limitent à leur
split train ; Pets prend trainval sans le test de `pets_det`, tiré de trainval. Pets n'a pas de sonde en
ligne (pas de tâche de classification). Le pool utilisé est choisi par `data=` (défaut : `imagenette`).

Tout nouveau pool de pré-entraînement (ImageNet-100, par exemple) doit être vérifié contre les tests ci-dessus
avant usage, en particulier contre le val officiel d'Imagenette, qui est un sous-ensemble du val d'ImageNet.

## Splits figés

Les splits sont écrits une fois dans [`src/latentopt/data/splits/`](../src/latentopt/data/splits/), un JSON par
dataset, deux pour Pets (segmentation et détection). Chacun fait quelques centaines de Ko, sous la limite de 1000 Ko du hook `check-added-large-files`. Ils sont commités : c'est ce qui rend les scores comparables d'une
machine à l'autre. Ils sont stockés par identifiant stable, jamais par indice :

| Dataset | Identifiant |
|---|---|
| Imagenette | chemin relatif de l'image dans son dossier officiel : `<wnid>/<fichier>.JPEG` |
| Flowers-102 | nom du fichier : `image_00001.jpg` |
| Galaxy10 | numéro de ligne dans la concaténation train + test de la révision HF épinglée |
| Pets (seg, det) | nom de l'image, ex. `Abyssinian_100` |

Deux origines, dans les mêmes JSON :

- **Dérivés** (train / val d'Imagenette et de Pets segmentation, tout Galaxy10 et Pets détection) : calculés,
  holdout 95/5 ou stratifié 80/10/10, avec `seed=0`.
- **Officiels** (Flowers-102, tests d'Imagenette et de Pets) : non calculés. Le JSON liste les identifiants que
  donne la définition officielle du dataset, lue par torchvision.

Chaque dataset définit son split dans `freeze_splits`. `lo data prepare` l'appelle pour chaque
dataset préparé. Les constructeurs de splits des tâches et les pools de pré-entraînement l'appellent aussi avant de
lire, en filet de sécurité : un JSON existant est lu tel quel, sinon il est créé (écriture atomique). Un JSON
existant n'est réécrit que par `--force-splits` : ni un téléchargement, ni `--force-download` ne le touchent. Un JSON généré doit être commité : sans lui, deux machines dont l'ordre de
listage diffère n'auraient pas les mêmes splits.

Les images sont lues par identifiant, donc dans l'ordre trié des identifiants et non dans l'ordre officiel de
torchvision. L'évaluation n'en dépend pas, mais un entraînement avec la même graine mélange différemment de ce qu'il
faisait avant cet ordre. Un identifiant manquant au chargement lève une erreur.

Galaxy10 demande une attention particulière : ses identifiants sont des numéros de ligne, valables pour la seule
révision HF épinglée (`REVISION` dans `galaxy10.py`). Le JSON enregistre cette révision, et la tâche
`galaxy10_cls` comme le pool de pré-entraînement (`data=galaxy10`) vérifient qu'elle est celle du code avant toute
lecture. En revanche, les données sur disque ne gardent pas trace de la leur : changer `REVISION` sans
re-télécharger passerait inaperçu, et les numéros de ligne désigneraient d'autres images. Changer `REVISION` impose
donc `lo data prepare galaxy10 --force-download --force-splits`, et rend les scores incomparables aux anciens.

## Téléchargement et emplacement

Le téléchargement et le chargement sont séparés. Seul `lo data prepare` accède au réseau ; les
datasets ne font que lire, et échouent immédiatement, avec la commande à lancer, si les données manquent.

```bash
uv run lo data prepare                          # tout (tailles : tableau ci-dessus)
uv run lo data prepare imagenette oxford_pets   # une partie
uv run lo data prepare dino_vits8               # poids DINO du ViT-S/8 -> data/models/ (83 Mo)
uv run lo data prepare oxford_pets --force-download --force-splits --verbose   # tout refaire, avec les barres
```

| Option | Effet |
|---|---|
| `--force-splits` | réécrit aussi les JSON de splits qui existent ; les scores ne sont plus comparables aux anciens |
| `--force-download` | supprime le dataset, son archive et le cache temporaire, puis le re-télécharge ; ne concerne ni les splits ni les poids |
| `--verbose` | affiche les barres de progression et la durée des téléchargements ; masquées par défaut, pour ne pas remplir les logs Slurm |

Chaque dataset annonce ce qui se passe : `déjà présent dans …, utilisé tel quel` ou `téléchargement dans …`, puis
pour ses splits `déjà figés, conservés`, `créés` ou `réécrits`. « Déjà présent » est décidé ainsi : pour Imagenette
et Galaxy10, le dossier existe ; pour Flowers-102 et Pets, torchvision vérifie les fichiers attendus. Un
téléchargement interrompu qui a laissé un dossier incomplet n'est donc pas toujours détecté pour les deux premiers :
`--force-download` repart de zéro.

Sur le DCE, lancer cette commande depuis la passerelle, pour ne pas occuper un GPU pendant les
téléchargements (voir [slurm/README.md](../slurm/README.md)).

`DATA_ROOT` contient les données (variable et défaut : [Variables d'environnement](#variables-denvironnement)).
Les trois sous-dossiers sont suivis par git grâce à un `.gitkeep`, leur contenu est ignoré :

```
data/                         DATA_ROOT
├── datasets/                 sources téléchargées par `lo data prepare`
│   ├── imagenette2-160/
│   ├── flowers-102/
│   ├── oxford-iiit-pet/
│   └── galaxy10_decals/      copie save_to_disk, relue sans réseau
├── features/                 features mises en cache   (FEATURES_ROOT pour le déplacer)
├── models/                   poids externes téléchargés (MODELS_ROOT pour le déplacer)
├── mlflow.db                 base MLflow d'équipe, remplie par `lo mlflow sync`
├── mlflow_artifacts/         fichiers légers des runs (config, results…), copiés par `lo mlflow sync`
└── mlflow_hub/               bases MLflow des dépôts du Hub, téléchargées par `lo mlflow sync`
```

Les sous-dossiers de `datasets/` sont créés par le téléchargement, avec les noms imposés par torchvision : ne
pas les créer à la main, torchvision prendrait un dossier vide pour un téléchargement déjà fait.

Galaxy10 transite par un cache HF temporaire (`data/datasets/.hf_cache/`, ~2,5 Go) supprimé après la copie (et par `--force-download`) :
prévoir la place du cache en plus de celle de la copie pendant le téléchargement.

## Variables d'environnement

Seule liste des variables lues par le code ; les autres documents y renvoient.

| Variable | Défaut | Rôle |
|---|---|---|
| `DATA_ROOT` | `<racine du dépôt>/data` | données (arborescence ci-dessus) |
| `FEATURES_ROOT` | `DATA_ROOT/features` | features mises en cache |
| `MODELS_ROOT` | `DATA_ROOT/models` | poids externes téléchargés |
| `OUTPUT_ROOT` | `<racine du dépôt>/results` | dossiers de runs (`results/`, [HUB.md](HUB.md)) ; les runs désignés par leur nom y sont cherchés, ou téléchargés |
| `MLFLOW_TRACKING_URI` | aucun : une base MLflow par run | serveur MLflow ([MLFLOW.md](MLFLOW.md#limites)) |

Pour `lo pretrain`, `lo train` et `lo eval`, c'est la config qui fait foi : `env.data_root` et
`env.results_root` reprennent `DATA_ROOT` et `OUTPUT_ROOT` s'ils sont définis (dans les trois
environnements, `local`, `ci` et `dce`), puis sont recopiés dans ces variables au lancement. Les autres
commandes (`lo data prepare`, `lo clear`, `lo mlflow sync`…) lisent les variables directement.

## Données synthétiques

`synthetic_cls`, `synthetic_seg` et `synthetic_det` (64px), et le pool `data=synthetic`, sont des images
générées, mêmes formats de cible que les vrais datasets : smoke run, CI et tests. Aucun fichier, aucun réseau,
aucun JSON de splits : chaque split a sa propre graine. Les scores n'ont aucun sens.
