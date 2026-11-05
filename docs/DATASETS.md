# Datasets

Règle : aucun set de test downstream dans le pré-entraînement.

Les tâches et les splits reprennent la small league de [`projects/lejepa_arena`](../projects/lejepa_arena/README.md),
pour que nos scores restent comparables aux siens. Le code est dans
[`src/latentopt/data/datasets/`](../src/latentopt/data/datasets/), les tâches dans
[`src/latentopt/evaluation/tasks.py`](../src/latentopt/evaluation/tasks.py).

| Dataset | Usage | Source | Licence | Checksum (MD5 des archives) | Taille | Splits |
|---|---|---|---|---|---|---|
| Imagenette 160px | pré-entraînement (train, sans labels) ; `imagenette_cls` | [fast.ai](https://github.com/fastai/imagenette), `imagenette2-160.tgz` | Apache-2.0 (dépôt fast.ai) ; images issues d'ImageNet, soumises à ses conditions d'usage | `e793b78cc4c9e9a4ccc0c1155377a412` | 99 Mo | train 9 469 / val 3 925 officiels → train / val = holdout 95/5 du train (**figé**), test = val officiel |
| Oxford Flowers-102 | `flowers102_cls` | [VGG Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | non précisée sur la page officielle | images `52808999861908f626f3c1f4e79d11fa`, labels `e0620be6f572b9609742df49c70aed4d`, splits `a5357ecc9cb78c4bef273ce3793fc85c` | 345 Mo | officiels : train 1 020 / val 1 020 / test 6 149 |
| Galaxy10 DECaLS | `galaxy10_cls` | [Hugging Face `matthieulel/galaxy10_decals`](https://huggingface.co/datasets/matthieulel/galaxy10_decals), d'après [Galaxy10](https://github.com/henrysky/Galaxy10) | MIT (dépôt Galaxy10) ; images DESI Legacy Imaging Surveys, labels Galaxy Zoo | révision HF épinglée `d1cfafdcce69ca7cd25aecce1a0b578cf07e6a30` | 2,5 Go | 17 736 images (train HF 15 962 + test HF 1 774, concaténés) → stratifié 80/10/10 (**figé**) |
| Oxford-IIIT Pet (segmentation) | `pets_seg` | [VGG Oxford](https://www.robots.ox.ac.uk/~vgg/data/pets/) | CC BY-SA 4.0 | images `5c4f3ee8e5d25df40f4fd59a7f44e54c`, annotations `95a8c909bbe2e81eed6a22bccdf3f68f` | 792 Mo + 19 Mo | trainval 3 680 / test 3 669 officiels → train / val = holdout 95/5 de trainval (**figé**), test = test officiel |
| Oxford-IIIT Pet (détection) | `pets_det` | idem, boîtes de tête dans `annotations/xmls` | CC BY-SA 4.0 | idem | idem | images de trainval ayant une boîte → stratifié 80/10/10 par espèce (**figé**) |

Les MD5 sont ceux que torchvision vérifie au téléchargement. Les tailles sont celles des archives.

## Tâches

| Tâche | Entrée | Classes | Métrique | Budget de tête |
|---|---|---|---|---|
| `imagenette_cls` | 128px, petit côté + crop central | 10 | top-1 | 2M paramètres |
| `flowers102_cls` | 128px, petit côté + crop central | 102 | top-1 | 2M |
| `galaxy10_cls` | 128px, resize carré sans crop | 10 (déséquilibrées) | balanced accuracy | 2M |
| `pets_seg` | 128px carré ; masque à la résolution native | 3 (animal, fond, contour) | mIoU à la résolution native | 3M |
| `pets_det` | 128px carré ; boîte normalisée dans [0, 1] | 2 (chat, chien) | localisation : classe juste et IoU ≥ seuil, moyenné sur IoU 0.5 à 0.95 | 3M |

Les transforms d'évaluation font partie du protocole : les modifier rend les scores incomparables.

## Respect de la règle

Le pool de pré-entraînement est le **train officiel d'Imagenette**. Il contient le val d'`imagenette_cls`,
puisque ce val est tiré de ce même train, mais aucun test downstream : le test d'`imagenette_cls` est le val
officiel d'Imagenette, jamais vu en pré-entraînement. Les autres datasets n'entrent pas dans le pré-entraînement.

Tout nouveau pool de pré-entraînement (ImageNet-100, par exemple) doit être vérifié contre les tests ci-dessus
avant usage, en particulier contre le val officiel d'Imagenette, qui est un sous-ensemble du val d'ImageNet.

## Splits figés

Les splits dérivés (holdout, stratifié) sont calculés une fois et commités dans
[`src/latentopt/data/splits/`](../src/latentopt/data/splits/), un JSON par dataset. Ils sont stockés par
identifiant stable, jamais par indice :

| Dataset | Identifiant |
|---|---|
| Imagenette | chemin relatif de l'image dans le train officiel : `<wnid>/<fichier>.JPEG` |
| Galaxy10 | numéro de ligne dans la concaténation train + test de la révision HF épinglée |
| Pets (seg, det) | nom de l'image, ex. `Abyssinian_100` |

Algorithmes et graine (`seed=0`) sont ceux de l'arène : les ensembles obtenus sont identiques aux siens.
Un changement d'ordre de listage côté torchvision ou HF ne peut pas déplacer une image d'un split à l'autre.
Un identifiant manquant au chargement lève une erreur.

Les splits officiels (Flowers-102, tests d'Imagenette et de Pets) ne sont pas figés : ils sont définis par
les fichiers du dataset eux-mêmes.

## Téléchargement et emplacement

Le téléchargement et le chargement sont séparés. Seul `tools/prepare_data.py` accède au réseau ; les
datasets ne font que lire, et échouent immédiatement, avec la commande à lancer, si les données manquent.

```bash
python tools/prepare_data.py                          # tout (~3,8 Go)
python tools/prepare_data.py imagenette oxford_pets   # une partie
python tools/prepare_data.py dino_vits8               # poids DINO du ViT-S/8 -> data/models/ (83 Mo)
python tools/prepare_data.py --freeze-splits          # + recalcule les splits figés (une seule fois)
```

Sur le DCE, lancer cette commande depuis la passerelle, pour ne pas occuper un GPU pendant les
téléchargements (voir [CLUSTER.md](CLUSTER.md)).

`DATA_ROOT` (variable d'environnement, défaut `<racine du projet>/data/`) contient trois sous-dossiers.
Les dossiers sont suivis par git grâce à un `.gitkeep`, leur contenu est ignoré :

```
data/                         DATA_ROOT
├── datasets/                 sources téléchargées par prepare_data
│   ├── imagenette2-160/
│   ├── flowers-102/
│   ├── oxford-iiit-pet/
│   └── galaxy10_decals/      copie save_to_disk, relue sans réseau
├── features/                 features mises en cache   (FEATURES_ROOT pour le déplacer)
└── models/                   poids externes téléchargés (MODELS_ROOT pour le déplacer)
```

Les sous-dossiers de `datasets/` sont créés par le téléchargement, avec les noms imposés par torchvision : ne
pas les créer à la main, torchvision prendrait un dossier vide pour un téléchargement déjà fait.

Galaxy10 transite par un cache HF temporaire (`data/datasets/.hf_cache/`, ~2,5 Go) supprimé après la copie :
prévoir ~5 Go libres pendant le téléchargement.

## Données synthétiques

`synthetic_cls`, `synthetic_seg` et `synthetic_det` (64px) sont des images générées, mêmes formats de cible
que les vrais datasets : smoke run, CI et tests. Aucun fichier, aucun réseau. Les scores n'ont aucun sens.
