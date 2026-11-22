# MLflow

MLflow sert à **comparer les runs** : paramètres, courbes et scores de tous nos pré-entraînements,
entraînements et évaluations, dans une seule interface. Il ne remplace pas le dossier de run
(`results/<run>/`), qui garde les poids et reste la référence ([HUB.md](HUB.md)) : MLflow l'indexe.

## Où sont les bases

- **Une base par run** : `<dossier du run>/mlflow.db`, écrite par ce run seul (`backbone/`,
  `downstream/<…>/`, `downstream/<…>/eval/<…>/`). Aucun conflit entre jobs simultanés sur le DCE.
  Publiée sur le Hub avec le run.
- **La base d'équipe** : `data/mlflow.db`, construite à la main par `lo mlflow sync`, qui fusionne
  les bases des runs dont le statut MLflow est terminé (réussi, échoué ou tué). Aucun job n'y écrit.

## Construire la base d'équipe

```bash
uv run hf auth login                     # une fois : les dépôts du Hub sont privés
uv run lo mlflow sync                    # Hub + results/ -> data/mlflow.db
uv run lo mlflow sync --no-hub           # results/ locaux seulement, sans réseau
uv run lo mlflow sync --include-running  # aussi les runs non terminés (job tué par Slurm)
```

Sources : le Hub et son propre `results/`. Les runs des autres arrivent par le Hub, une fois
publiés : le sync ne lit jamais le home d'un autre. Du Hub, seuls les bases et les fichiers légers
(`config.yaml`, `git.txt`, `lineage.json`, `results.json`, `run.log`) sont téléchargés (dans
`data/mlflow_hub/`), jamais les poids.

- **Commande manuelle**, lancée par une seule personne à la fois.
- Un run est reconnu par son identifiant MLflow d'origine : relancer n'importe que les nouveaux runs.
- Les runs en cours sont ignorés, importés au sync suivant. Un job tué par Slurm (limite de temps)
  reste `RUNNING` : `--include-running`. Un run importé ainsi est remplacé par sa version complète
  au sync qui suit sa fin (compteur « mis à jour » ; l'ancienne copie va dans la corbeille MLflow).

## Ouvrir l'interface

Sur la machine où `lo mlflow sync` a été lancé (son PC, ou la passerelle du DCE). Les fichiers
légers sont copiés dans `data/mlflow_artifacts/` et la base y renvoie par chemin absolu : une
`mlflow.db` copiée ailleurs garde paramètres et métriques mais perd les fichiers.

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000   # http://localhost:5000
```

Sur le DCE, lancer la même commande sur la passerelle, puis un tunnel depuis sa machine :
`ssh -N -L 5000:localhost:5000 dce`. Changer de port si 5000 est pris.

Suivre un run en cours, sans sync : `--backend-store-uri sqlite:///<dossier du run>/mlflow.db`,
ou `tail -f <dossier du run>/run.log`.

## Ce qu'on y trouve

| | |
|---|---|
| Expériences | `pretrain/<experiment_name>`, `train/<…>`, `eval/<…>` |
| Nom d'un run | son chemin sous `results/` : `<run>/downstream/pets_seg-fpn-conv_seg-<date>` |
| Paramètres | la config résolue complète, aplatie (`optimizer/lr`, `neck/channels`…) |
| Métriques | courbes `train/*`, `val/*` ; résultats finaux `results/*` |
| Tags | `task`, `neck`, `head`, `downstream`, `backbone_run`, `git_commit`, `host`… |

Le détail des tags : [COMMANDS.md](COMMANDS.md), section 9.

## Comparer

Filtres utiles dans la barre de recherche :

```
tags.task = "pets_seg"
tags.backbone_run = "imnt160-vit_s8-lejepa-20261116_183012"
tags.downstream = "pets_seg-fpn-conv_seg-20261117_090000"     # un entraînement et ses évaluations
metrics.`results/score` > 0.5
```

- **Même tâche** : filtrer sur `tags.task`, puis onglet *Chart* (barres, coordonnées parallèles,
  nuages de points) ou *Compare*. « Group by » `neck`, `head` ou `backbone_run` regroupe les runs et affiche
  la moyenne de chaque groupe (min ou max au choix) : à moyenner sur des runs qui ne diffèrent que
  par la graine.
- **Tâches différentes** : utiliser les runs `eval/…`, dont le score principal s'appelle toujours
  `results/score`. Côté entraînement, la métrique dépend de la tâche (`results/val/miou`, `…/top1`).
- **Entraînement et évaluations** : runs séparés, reliés par le tag `downstream`. Cocher `train/…`
  et `eval/…` dans la liste des expériences pour les voir ensemble.

## Limites

- Colonnes, graphiques et groupements choisis dans l'interface sont gardés **dans le navigateur**
  (*Share* les enregistre dans la base, perdus à sa reconstruction). Seuls filtre et tri passent dans l'URL.
- L'interface ne joint pas deux runs : pas de tableau « une ligne par entraînement, val et test ».
- **`env.mlflow` et `MLFLOW_TRACKING_URI`** : `env.mlflow=false` désactive la base par run. Si
  `MLFLOW_TRACKING_URI` est défini, le suivi va quand même vers ce serveur (la variable passe
  avant `env.mlflow`), avec un avertissement au démarrage.
