# Noms des runs et Hugging Face

Chaque backbone pré-entraîné a **un seul nom**. Ce nom est celui de son dossier dans `results/`
et de son dépôt dans l'organisation Hugging Face
[`Espace-latent-optimal`](https://huggingface.co/Espace-latent-optimal). Les necks et les têtes
entraînés sur ce backbone, et leurs évaluations, sont rangés **dans ce même dossier**, donc dans ce
même dépôt. Le dépôt est **la copie du dossier local** : télécharger un backbone le recrée à
l'identique, au même endroit. Un run se désigne donc indifféremment par son chemin ou par son nom,
qu'il soit sur la machine ou seulement sur le Hub. Le code est dans
[`src/latentopt/hub.py`](../src/latentopt/hub.py).

## Un dossier par backbone

```
results/<données>-<archi>-<méthode>-<AAAAMMJJ_HHMMSS>/   ⇄  Espace-latent-optimal/<run>
├── README.md                              fiche du dépôt, écrite par le premier lo hub push
├── backbone/                              lo pretrain
│   ├── backbone.pt, README.md
│   ├── config.yaml, results.json, git.txt, lineage.json, metrics.csv, run.log, mlflow.db
│   └── checkpoints/                       reprise d'un run interrompu (jamais publié)
└── downstream/<tâche>-<neck>-<tête>-<AAAAMMJJ_HHMMSS>/
    │                                      lo train : neck.pt, head.pt, README.md, results.json,
    │                                      config.yaml, git.txt, lineage.json, metrics.csv,
    │                                      run.log, mlflow.db
    └── eval/<split>-<AAAAMMJJ_HHMMSS>/    lo eval : results.json, config.yaml, git.txt,
                                           lineage.json, mlflow.db
```

```
imnt160-vit_s8-lejepa-20261116_183012/
├── backbone/
├── downstream/pets_seg-fpn-conv_seg-20261117_090000/
│   └── eval/val-20261118_100000/
├── downstream/pets_det-fpn-soft_argmax-20261118_142530/       neck de pets_seg-fpn-conv_seg-… repris, gelé
└── downstream/imagenette_cls-embed_norm-linear-20261117_091500/
imagenet1k-vit_s8_dino-dino-20261117_100000/                     backbone externe : pas de backbone/
```

| Partie | Origine |
|---|---|
| `<données>` | nom court (`ABBREV`) du module du dataset de pré-entraînement : `imnt160` (Imagenette 160px), `fl102`, `gal10d`, `pets`, `synth` ; défini dans le code, pas dans les configs, et figé |
| `<archi>` | groupe `backbone` (`vit_s8`, `vit_s8_dino`…) |
| `<méthode>` | groupe `method` (`lejepa`…) |
| `<AAAAMMJJ_HHMMSS>` | date du lancement ; `_1`, `_2`… pour les jobs 1, 2… d'un balayage `-m` |
| `<tâche>` | tâche entraînée (`evaluation/tasks.py`) |
| `<neck>` | config du groupe `neck` (`embed_norm`, `fpn`, `conv_trunk`) ; sans `neck=`, celui de la baseline de la tâche ; avec `neck_from`, celui du neck repris |
| `<tête>` | config du groupe `head` (`linear`, `conv_seg`, `soft_argmax`) ; sans `head=`, celle de la baseline de la tâche |
| `<split>` | split évalué : `val` ou `test` |

Dans la doc, **`<run>`** désigne le nom complet d'un backbone (`<données>-<archi>-<méthode>-<date>`).

- **Un run = un dossier, écrit par ce run seul, puis plus jamais modifié.** Un entraînement de
  `lo train` écrit son neck et sa tête dans le même dossier `downstream/<…>/`, avec un seul
  `mlflow.db`. Une évaluation ne modifie pas le dossier évalué : elle crée `eval/<split>-<date>/`.
- **Chaque run a son dossier daté** : réentraîner (autre graine, autre `lr`) ou réévaluer ajoute
  un dossier, sans écraser le précédent.
- **Plusieurs personnes peuvent ajouter des entraînements et des évaluations au même backbone** :
  leurs dossiers datés ne se recouvrent jamais, et la publication n'envoie que des fichiers nouveaux.

Les parties sont séparées par `-` et n'en contiennent jamais : un `-` dans une partie devient `_`
(`hub.normalize`). Un nom se découpe donc toujours en 4 (backbone, downstream) ou 2 (éval)
morceaux (`hub.parse`, `parse_downstream`, `parse_eval`). Un dossier `downstream/<…>` a 4 parties
comme un backbone, mais n'est jamais pris pour un backbone (`hub.run_root`). Les variantes en ligne
de commande (`neck.channels=128`) gardent le nom de leur config : une variante retenue mérite sa
propre config, donc son propre nom. Les hyperparamètres, le commit et l'expérience ne sont pas dans
le nom : ils sont dans `config.yaml`, `results.json` et MLflow.

**Limites du Hub** : nom de dépôt de 96 caractères au plus, sans `--` (`hub.repo_id` refuse un nom
invalide) ; 10 000 entrées au plus par dossier, et moins de 100 000 fichiers par dépôt
recommandés. Les exemples ci-dessus font 40 et 43 caractères.

## Ce que contient chaque dossier

| Fichier | Contenu |
|---|---|
| `neck.pt` | `state_dict`, `params`, `seed`, `backbone_sha256`, `neck_sha256`, `config` ; absent avec `neck_from` |
| `head.pt` | `state_dict` (la tête seule), `params`, `seed`, `backbone_sha256`, `neck_sha256`, `neck_dir` (dossier du neck, relatif au dossier du backbone) |
| `downstream/…/results.json` | scores val, `params` : `{"neck", "head"}` comptés séparément, `backbone_params`, `neck` (dossier du neck : le sien, ou celui repris), `neck_frozen`, `names`, `classes`, commit |
| `eval/…/results.json` | `split`, score, métriques, paramètres du backbone, du neck et de la tête, violations |
| `README.md` | fiche du dossier (`hub.write_card`), écrite à la fin du run : `backbone/` et `downstream/<…>/` |

`results.json` est écrit **en dernier** : sa présence marque le run terminé (`hub.parts`). Au
chargement, chaque `.pt` est vérifié contre l'empreinte des poids qu'il attend : un neck ou une
tête appris sur un autre backbone (ou un autre neck) est signalé.

## Désigner un run : chemin ou nom

Partout où un run est attendu (`backbone.run=`, `neck_from=`, `downstream=[...]` de `lo eval`,
`lo hub push`), on peut donner :

| Valeur | Résultat |
|---|---|
| un chemin existant | utilisé tel quel |
| `<run>` | `results/<run>/` |
| `<run>/downstream/pets_seg-fpn-conv_seg-20261117_090000` | `results/<run>/downstream/pets_seg-fpn-conv_seg-20261117_090000/` |
| le chemin d'une autre machine (`/usr/users/…/results/<run>/…`) | le même chemin sous `results/` de cette machine |

`neck_from=` accepte aussi le seul nom du dossier (`pets_seg-fpn-conv_seg-20261117_090000`), pris
dans `downstream/` du backbone du run. Ce dossier doit contenir `neck.pt` : un entraînement qui a
lui-même repris un neck n'en a pas, et l'erreur indique le dossier d'origine à désigner.

Si le dossier n'existe pas en local, **le dépôt est téléchargé depuis le Hub** dans `results/`, puis
utilisé (`hub.locate`). Un run entraîné sur le DCE peut donc être évalué sur un PC sans rien copier :

```bash
uv run lo train backbone=from_run backbone.run=imnt160-vit_s8-lejepa-20261116_183012 task=pets_seg
uv run lo eval downstream='[imnt160-vit_s8-lejepa-20261116_183012/downstream/pets_seg-fpn-conv_seg-20261117_090000,...]'
```

`results/` est `cluster.results_root` (variable `OUTPUT_ROOT`, sinon `<projet>/results`).

## Dossiers de sortie

| Script | Dossier |
|---|---|
| `lo pretrain` | `results/<run>/backbone/` |
| `lo train` | `results/<run>/downstream/<tâche>-<neck>-<tête>-<date>/` : neck et tête |
| `lo train neck_from=<…>` | `results/<run>/downstream/<tâche>-<neck repris>-<tête>-<date>/` : tête seule, le neck repris reste gelé |
| `lo eval` | `<dossier évalué>/eval/<split>-<date>/` pour chaque dossier ; journal du lancement dans `results/.eval/<date>/` |

Pour `lo train`, le `<run>` du backbone vient :

1. de `backbone_run=<run>` s'il est donné ;
2. sinon, avec `backbone=from_run`, du run de pré-entraînement ;
3. sinon (backbone externe comme `vit_s8_dino`, ou à poids aléatoires), de
   `<backbone_data>-<choix de backbone>-<backbone_method>-<date du lancement>`. `backbone_data` et
   `backbone_method` valent `none` par défaut ; pour DINO, `backbone_data=imagenet1k
   backbone_method=dino`. Dans un balayage `-m` sur le DCE, chaque job a sa propre date : fixer
   `backbone_run=imagenet1k-vit_s8_dino-dino-<AAAAMMJJ_HHMMSS>` pour que les entraînements
   partagent le même dossier.

## Publier

Prérequis, une fois par machine (PC et DCE) : un token **Write** créé sur
<https://huggingface.co/settings/tokens>, puis `uv run hf auth login`. Le token est rangé dans
`~/.cache/huggingface/token`, hors du dépôt. Chacun utilise son propre token.

```bash
# runs terminés du backbone : backbone/, downstream/<…>/, eval/<…>/ (un seul dépôt)
uv run lo hub push <run>

# afficher ce qui serait publié, sans rien envoyer (ni lire le Hub : tout paraît nouveau)
uv run lo hub push <run> --dry-run
```

- **Seuls les runs terminés sont publiés**, c'est-à-dire ceux qui ont leur `results.json` (plus
  `backbone.pt` ou `head.pt`). Un run en cours attend la publication suivante.
- **Le push n'envoie que les fichiers absents du dépôt**, en un commit. Un dossier de run n'étant
  jamais modifié, une copie locale d'un run déjà publié (obtenue par `pull`) n'est jamais renvoyée :
  **un push ne peut pas écraser ce que d'autres ont publié**, quel que soit l'ordre des push.
- **Si quelqu'un publie pendant le push**, le Hub refuse le commit (`parent_commit`, comme un
  `git push` refusé) : relancer `lo hub push`, qui recalcule ce qui manque.
- **Rien n'est supprimé du dépôt.**
- **Fiches** : chaque dossier qui contient des poids a son `README.md`, écrit à la fin de son run.
  La fiche du dépôt (`README.md` à la racine) ne dépend que du nom du backbone ; elle est écrite au
  premier push et jamais réécrite. La vue d'ensemble des runs d'un backbone est dans MLflow.
- Les dépôts sont **privés** par défaut ; `--public` les rend publics.
- Un run lancé avec un arbre git non commité est refusé, sauf avec `--allow-dirty`.

**Jamais publiés** : `checkpoints/` (reprise, reste sur le DCE), `mlartifacts/`, `.submitit/`,
`.cache/`. Les datasets ne sont pas publiés non plus : ils se téléchargent depuis leur source,
avec vérification des MD5 ([DATASETS.md](DATASETS.md)), et certains ne peuvent pas être republiés
(licence non précisée pour Flowers, CC BY-SA pour Pets).

Les bases MLflow (une `mlflow.db` par dossier de run : backbone, entraînements, évaluations) sont
publiées : `lo mlflow sync` les relit sur le Hub pour construire la base d'équipe `data/mlflow.db`
([slurm/COMMANDS.md](../slurm/COMMANDS.md), section 9).

## Télécharger

Le plus souvent, rien à faire : donner le nom suffit (voir plus haut). Pour télécharger
explicitement :

```bash
uv run lo hub pull <run>                                     # → results/<run>/ (tout le dépôt)
uv run lo hub pull <run> --only "downstream/pets_seg-*"      # une partie seulement
uv run lo hub pull <run> --only "backbone/*"
```

Les fichiers déjà présents et à jour sont conservés. Les poids du backbone sont vérifiés contre
l'empreinte de `results.json` au chargement. Le téléchargement ajoute un dossier caché
`.cache/huggingface/` (métadonnées du Hub), qui n'est jamais republié.

## Travailler à plusieurs sur un backbone

```bash
uv run lo hub pull <run>                                              # ce que les autres ont publié
uv run lo train backbone=from_run backbone.run=<run> task=pets_seg    # nouvel entraînement
uv run lo train backbone=from_run backbone.run=<run> task=pets_det \
    neck_from=pets_seg-fpn-conv_seg-20261117_090000                   # tête sur un neck repris
uv run lo eval downstream='[<run>/downstream/pets_seg-fpn-conv_seg-20261117_090000]'
uv run lo hub push <run>                                              # ajoute ses dossiers datés
```

Un push n'écrase jamais le travail d'un autre : chaque run a son dossier daté, jamais modifié
ensuite, et le push n'envoie que ce qui manque sur le Hub. Deux personnes qui évaluent le même
entraînement créent deux dossiers `eval/` distincts.

## Organisation Hugging Face

- **Rôles** : `admin` pour un responsable, `write` pour les autres membres (créer et pousser des
  dépôts). Il n'y a pas de token d'organisation à partager : chaque token Write agit avec le rôle
  de son propriétaire.
- **Visibilité** : privé tant que les résultats ne sont pas publiés.
- **Avatar** : [`docs/assets/icon.jpg`](assets/icon.jpg).
