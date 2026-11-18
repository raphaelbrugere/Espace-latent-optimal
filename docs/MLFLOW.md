# MLflow

MLflow sert à **comparer les runs** : paramètres, courbes et scores de tous nos pré-entraînements,
entraînements et évaluations, dans une seule interface. Il ne remplace pas le dossier de run
(`results/<run>/`), qui garde les poids et reste la référence ([HUB.md](HUB.md)) : MLflow l'indexe.

## Où sont les bases

- **Une base par run** : `<dossier du run>/mlflow.db`, écrite par ce run seul (`backbone/`,
  `downstream/<…>/`, `downstream/<…>/eval/<…>/`). Aucun conflit entre jobs simultanés sur le DCE.
  Publiée sur le Hub avec le run.
- **La base d'équipe** : `data/mlflow.db`, construite à la main par `lo mlflow sync`, qui fusionne
  les bases des runs terminés. Aucun job n'y écrit.

## Construire la base d'équipe

```bash
uv run hf auth login                     # une fois : les dépôts du Hub sont privés
uv run lo mlflow sync                    # Hub + results/ -> data/mlflow.db
uv run lo mlflow sync --no-hub           # results/ locaux seulement, sans réseau
uv run lo mlflow sync --include-running  # aussi les runs non terminés (job tué par Slurm)
```

Sources : le Hub et son propre `results/`. Les runs des autres arrivent par le Hub, une fois
publiés : le sync ne lit jamais le home d'un autre. Le relancer n'importe que les nouveaux runs.

## Ouvrir l'interface

Sur son PC :

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000   # http://localhost:5000
```

Sur le DCE, lancer la même commande sur la passerelle, puis un tunnel depuis sa machine :
`ssh -N -L 5000:localhost:5000 dce`. Changer de port si 5000 est pris.

Suivre un run en cours, sans sync : `--backend-store-uri sqlite:///<dossier du run>/mlflow.db`.

## Ce qu'on y trouve

| | |
|---|---|
| Expériences | `pretrain/<experiment_name>`, `train/<…>`, `eval/<…>` |
| Nom d'un run | son chemin sous `results/` : `<run>/downstream/pets_seg-fpn-conv_seg-<date>` |
| Paramètres | la config résolue complète, aplatie (`optimizer/lr`, `neck/channels`…) |
| Métriques | courbes `train/*`, `val/*` ; résultats finaux `results/*` |
| Tags | `task`, `neck`, `head`, `downstream`, `backbone_run`, `git_commit`, `host`… |

Le détail des tags : [slurm/COMMANDS.md](../slurm/COMMANDS.md), section 9.

## Comparer

Filtres utiles dans la barre de recherche :

```
tags.task = "pets_seg"
tags.backbone_run = "imnt160-vit_s8-lejepa-20261116_183012"
tags.downstream = "pets_seg-fpn-conv_seg-20261117_090000"     # un entraînement et ses évaluations
metrics.`results/score` > 0.5
```

- **Même tâche** : filtrer sur `tags.task`, puis onglet *Chart* (barres, coordonnées parallèles,
  nuages de points) ou *Compare*. « Group by » `neck`, `head` ou `backbone_run` moyenne les graines.
- **Tâches différentes** : utiliser les runs `eval/…`, dont le score principal s'appelle toujours
  `results/score`. Côté entraînement, la métrique dépend de la tâche (`results/val/miou`, `…/top1`).
- **Entraînement et évaluations** : runs séparés, reliés par le tag `downstream`. Cocher `train/…`
  et `eval/…` dans la liste des expériences pour les voir ensemble.

## Limites

- Colonnes, graphiques et groupements choisis dans l'interface sont gardés **dans le navigateur** :
  ni partagés, ni conservés après une reconstruction de la base. Seuls filtre et tri passent dans l'URL.
- L'interface ne joint pas deux runs : pas de tableau « une ligne par entraînement, val et test ».
- `cluster.mlflow=false` désactive le suivi d'un run.
