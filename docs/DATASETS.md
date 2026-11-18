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

## Noms courts

Chaque module de dataset définit un nom court, `ABBREV`. C'est la partie `<données>` du nom d'un backbone
pré-entraîné, donc de son dossier `results/<run>/` et de son dépôt Hugging Face ([HUB.md](HUB.md)). Il est lu dans
le module désigné par `data.source._target_`, jamais dans les configs : renommer un fichier de `configs/data/` ne
renomme pas les runs.

| Module | `ABBREV` |
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

Les transforms d'évaluation font partie du protocole : les modifier rend les scores incomparables.

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
dataset, deux pour Pets (segmentation, détection) (de 85 à 510 Ko, environ 1,2 Mo au total), et commités : c'est ce qui rend les scores comparables d'une
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

Chaque module de `datasets/` définit son split dans `freeze_splits`. `lo data prepare` l'appelle pour chaque
dataset préparé, et `build` aussi avant de lire, en filet de sécurité : un JSON existant est lu tel quel, sinon il
est créé (écriture atomique). Un JSON existant n'est réécrit que par `--force-splits` : ni un téléchargement, ni
`--force-download` ne le touchent. Un JSON généré doit être commité : sans lui, deux machines dont l'ordre de
listage diffère n'auraient pas les mêmes splits.

Les images sont lues par identifiant, donc dans l'ordre trié des identifiants et non dans l'ordre officiel de
torchvision. L'évaluation n'en dépend pas, mais un entraînement avec la même graine mélange différemment de ce qu'il
faisait avant cet ordre. Un identifiant manquant au chargement lève une erreur.

Galaxy10 demande une attention particulière : ses identifiants sont des numéros de ligne, valables pour la seule
révision HF épinglée (`REVISION` dans `galaxy10.py`). Le JSON enregistre cette révision, et `build` vérifie qu'elle
est celle du code. En revanche, les données sur disque ne gardent pas trace de la leur : changer `REVISION` sans
re-télécharger passerait inaperçu, et les numéros de ligne désigneraient d'autres images. Changer `REVISION` impose
donc `lo data prepare galaxy10 --force-download --force-splits`, et rend les scores incomparables aux anciens.

## Téléchargement et emplacement

Le téléchargement et le chargement sont séparés. Seul `lo data prepare` accède au réseau ; les
datasets ne font que lire, et échouent immédiatement, avec la commande à lancer, si les données manquent.

```bash
uv run lo data prepare                          # tout (~3,8 Go)
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
téléchargements (voir [CLUSTER.md](CLUSTER.md)).

`DATA_ROOT` (variable d'environnement, défaut `<racine du projet>/data/`) contient les données. Pour
`lo pretrain`, `lo train` et `lo eval`, c'est la config qui fait foi : `cluster.data_root` est recopié dans
`DATA_ROOT` au lancement (`local` et `dce` reprennent la variable d'environnement si elle est définie, `ci`
l'ignore). Les trois sous-dossiers sont suivis par git grâce à un `.gitkeep`, leur contenu est ignoré :

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
└── mlflow_hub/               bases MLflow des dépôts du Hub, téléchargées par `lo mlflow sync`
```

Les sous-dossiers de `datasets/` sont créés par le téléchargement, avec les noms imposés par torchvision : ne
pas les créer à la main, torchvision prendrait un dossier vide pour un téléchargement déjà fait.

Galaxy10 transite par un cache HF temporaire (`data/datasets/.hf_cache/`, ~2,5 Go) supprimé après la copie (et par `--force-download`) :
prévoir ~5 Go libres pendant le téléchargement.

## Données synthétiques

`synthetic_cls`, `synthetic_seg` et `synthetic_det` (64px), et le pool `data=synthetic`, sont des images
générées, mêmes formats de cible que les vrais datasets : smoke run, CI et tests. Aucun fichier, aucun réseau,
aucun JSON de splits : chaque split a sa propre graine. Les scores n'ont aucun sens.
