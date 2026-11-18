# Commandes du projet

Tout passe par la commande `lo`, préfixée par `uv run` (`uv run lo --help` liste les commandes
par famille ; `latentopt` est son nom long). Elle se lance depuis n'importe quel dossier du dépôt.
Installation du compte DCE, partitions et stockage : voir [docs/CLUSTER.md](../docs/CLUSTER.md).

Sur le DCE, `cluster=dce` soumet un job Slurm via submitit, à condition d'ajouter **`-m`**. Sans
`-m`, le script refuse de tourner sur la passerelle. `-m` se place **après tous les overrides**.

## 1. Après chaque `git pull`

```bash
git pull && uv sync
```

## 2. Une fois : données et poids externes (sur la passerelle)

```bash
uv run lo data prepare imagenette   # pool de pré-entraînement (~100 Mo)
uv run lo data prepare              # tout (~3,8 Go)
uv run lo data prepare dino_vits8   # poids DINO (83 Mo)
```

Chaque dataset crée ses `src/latentopt/data/splits/*.json` manquants et garde les autres (`--force-splits`
les réécrit). Sur une installation à jour, le diff git doit rester vide.

## 3. Vérifier un nœud GPU (session interactive)

```bash
srun -p gpu_inter -t 00:30:00 --pty bash      # attendre que l'invite change de @chome au nom du nœud
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); torch.zeros(1).cuda()"
uv run pytest
exit
```

Sur une RTX 2080 Ti (nœuds `tx`), `sm_75` doit figurer dans la liste.

## 4. Smoke tests (avant tout job long)

| où | commande |
|---|---|
| local, CPU | `uv run lo pretrain experiment=smoke` |
| DCE, dans une session `srun` | `uv run lo pretrain experiment=smoke cluster=dce method.precision=fp16` |
| DCE, job Slurm | `uv run lo pretrain experiment=smoke cluster=dce method.precision=fp16 hydra.launcher.partition=gpu_inter hydra.launcher.timeout_min=30 -m` |
| têtes, local | `uv run lo train experiment=smoke_train task=synthetic_cls,synthetic_seg,synthetic_det -m` |

Dans une session `srun`, `cluster=dce` fonctionne sans `-m`, puisqu'on est déjà dans un job Slurm.
`method.precision=fp16` sert à tester le GradScaler sur GPU. Le smoke run est en fp32 par défaut,
pour pouvoir tourner sur CPU.

## 5. Pré-entraînement d'un backbone

```bash
# baseline (bf16, gpu_prod_long, 800 epochs)
uv run lo pretrain experiment=vit_lejepa_imagenette cluster=dce -m

# changer un réglage
uv run lo pretrain experiment=vit_lejepa_imagenette method.model.lamb=0.05 cluster=dce -m

# balayer plusieurs valeurs : un job par valeur
uv run lo pretrain experiment=vit_lejepa_imagenette method.model.lamb=0.01,0.02,0.05 cluster=dce -m

# manque de mémoire GPU : backbone.grad_checkpointing=true (même résultat, plus lent).
# Déjà activé dans vit_lejepa_imagenette : sans lui, OOM sur une RTX 3080 (10 Go).
uv run lo pretrain experiment=<autre> backbone.grad_checkpointing=true cluster=dce -m

# reprendre un run interrompu
uv run lo pretrain experiment=vit_lejepa_imagenette resume=results/<run>/backbone/checkpoints/last.ckpt cluster=dce -m
```

Sortie : `results/<données>-<archi>-<méthode>-<AAAAMMJJ_HHMMSS>/backbone/` (`backbone.pt`, `results.json`,
`README.md`, `run.log`, `metrics.csv`, `config.yaml`, `git.txt`, `lineage.json`, `mlflow.db`). Dans un balayage `-m`, les
jobs 1, 2… ajoutent `_1`, `_2`… à la date. `<données>-<archi>-<méthode>-<date>` est le dossier du backbone, de
ses entraînements downstream et de leurs évaluations, et son dépôt Hugging Face ([docs/HUB.md](../docs/HUB.md)).

## 6. Entraînement des necks et des têtes (backbone gelé)

```bash
# une tâche, sur un de nos backbones
uv run lo train backbone=from_run backbone.run=<run> task=pets_seg cluster=dce -m

# les 5 tâches, un job chacune
uv run lo train backbone=from_run backbone.run=<run> \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det cluster=dce -m

# référence DINO (poids externes), mêmes 5 tâches ; backbone_run fixé pour que les 5 jobs
# rangent leurs têtes sous le même nom de backbone (sinon chaque job date le sien)
uv run lo train backbone=vit_s8_dino experiment_name=dino backbone_data=imagenet1k backbone_method=dino \
    backbone_run=imagenet1k-vit_s8_dino-dino-<AAAAMMJJ_HHMMSS> \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det cluster=dce -m

# un autre neck ou une autre tête, ou d'autres réglages
uv run lo train backbone=from_run backbone.run=<run> task=pets_seg neck=fpn neck.channels=128 head=conv_seg epochs=50 cluster=dce -m

# une tête sur un neck déjà entraîné (gelé) : seule la tête apprend
uv run lo train backbone=from_run backbone.run=<run> task=pets_det head=soft_argmax neck_from=pets_seg-fpn-conv_seg-<AAAAMMJJ_HHMMSS> cluster=dce -m

# sans cache de features : backbone appliqué à la volée (voir plus bas)
uv run lo train backbone=from_run backbone.run=<run> task=pets_seg cache_features=false cluster=dce -m
```

Sans précision, le neck, la tête et le nombre d'epochs sont ceux de la baseline de chaque tâche.

Sortie, dans le dossier du backbone `results/<run>/` ([docs/HUB.md](../docs/HUB.md)) :
`downstream/<tâche>-<neck>-<tête>-<AAAAMMJJ_HHMMSS>/` : `neck.pt`, `head.pt`, `README.md`, `results.json`
(score val, paramètres du neck et de la tête séparés, du backbone), `config.yaml`, `run.log`, `mlflow.db`.
Avec `neck_from`, pas de `neck.pt` : le neck repris reste gelé, et `results.json` désigne son dossier.

**Cache de features.** Il est activé par défaut. Les features du backbone sont calculées une
fois dans `data/features/<date>_<empreinte>/`, puis relues. L'algorithme est le même que sans cache,
mais les tirages aléatoires (ordre des images, retournements) diffèrent, comme avec une autre
graine. `cache_features=false` applique le backbone à la volée à chaque epoch.

## 7. Évaluation

```bash
# évaluer des entraînements sur val (un ou plusieurs backbones)
uv run lo eval downstream='[<run>/downstream/imagenette_cls-embed_norm-linear-<date>,<run>/downstream/pets_seg-fpn-conv_seg-<date>]'

# score final sur test : une seule fois, à la fin
uv run lo eval downstream='[...]' split=test
```

Sortie : un nouveau dossier `eval/<split>-<AAAAMMJJ_HHMMSS>/` dans chaque dossier évalué, avec
`results.json` (score, métriques, paramètres du backbone, du neck et de la tête, violations), `config.yaml`,
`git.txt`, `lineage.json` et un run MLflow `eval/<experiment_name>` dans son `mlflow.db`. Le dossier
d'entraînement n'est pas modifié. Seul le journal du lancement va dans `results/.eval/<date>/`. Violations signalées :
backbone de plus de 30M paramètres, neck ou tête appris sur d'autres poids que ceux chargés.

## 8. Suivre les jobs

```bash
squeue -u <login>                                   # jobs en cours ou en attente
scancel <job_id>                                    # annuler
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
tail -f results/<run>/backbone/run.log             # avancement, une ligne par epoch
ls results/.submitit/<job_id>/                      # sorties brutes de Slurm (stdout, stderr)
```

## 9. Comparer les runs (MLflow)

Chaque run (`pretrain`, `train`, et chaque évaluation de `eval`) a sa propre base MLflow,
`<dossier du run>/mlflow.db`.
Un seul programme écrit dans chaque base, donc aucun conflit entre jobs simultanés sur le disque
réseau. Un run MLflow contient :

| | Contenu |
|---|---|
| Expérience | `pretrain/<experiment_name>`, `train/<experiment_name>`, `eval/<experiment_name>` |
| Nom | chemin du dossier sous `results/` : `<run>/backbone`, `<run>/downstream/<tâche>-<neck>-<tête>-<date>` (train), suivi de `/eval/<split>-<date>` (eval) |
| Paramètres | config résolue complète, aplatie (`method/lr`, `backbone/arch`…), y compris `cluster/*` (propre à la machine) |
| Métriques | courbes par pas et par epoch (`train/*`, `val/*`) ; résultats finaux `results/*` (scores, paramètres, heures GPU) |
| Tags : nom | `data`, `arch`, `method`, `backbone_date`, `backbone_run`, `hub_repo` (dépôt de destination, [docs/HUB.md](../docs/HUB.md)), plus `task`, `neck`, `head` et `downstream` (nom du dossier d'entraînement, commun à l'entraînement et à ses évaluations) |
| Tags : traçabilité | `author`, `git_commit`, `git_branch`, `git_dirty`, `uv_lock_sha256`, `splits/<dataset>_sha256`, `results/backbone_sha256` |
| Tags : machine | `host`, `cluster`, `slurm_job_id`, `env/python`, `env/torch`, `env/cuda`, `env/gpu`, `run_dir` |
| Tags : évaluation | `eval_split` : `val` ou `test` |

N'y entrent jamais : les poids (dossier du run, Hub) et les variables d'environnement (tokens).

Pour comparer tous nos modèles, `lo mlflow sync` fusionne les bases des runs terminés dans la base
d'équipe `data/mlflow.db`, avec leurs fichiers légers (`config.yaml`, `git.txt`, `lineage.json`,
`results.json`, `run.log`). Expériences : `pretrain/<experiment_name>`, `train/<experiment_name>`,
`eval/<experiment_name>`.

Deux sources :
- **le Hub** : chaque dépôt publié par `lo hub push` contient les `mlflow.db` de son backbone, de ses
  entraînements et de leurs évaluations. Seules
  les bases et les fichiers légers sont téléchargés (dans `data/mlflow_hub/`), jamais les poids ;
- **les `results/` locaux** (`--results`) : les runs et les évaluations pas encore publiés.

```bash
hf auth login                            # une fois : les dépôts sont privés
uv run lo mlflow sync                    # Hub + results/ -> data/mlflow.db
uv run lo mlflow sync --results results <home d'un collègue>/Espace-latent-optimal/results
uv run lo mlflow sync --no-hub           # results/ locaux seulement (hors ligne)
uv run lo mlflow sync --include-running  # aussi les runs RUNNING (job tué par Slurm)
```

- **Commande manuelle**, lancée par une seule personne à la fois.
- **Relancer n'importe que les nouveaux runs** : un run est reconnu par son identifiant MLflow
  d'origine, qu'il vienne du Hub ou d'un dossier local ; rien n'est dupliqué.
- **Les runs en cours sont ignorés** : ils seront importés à la synchronisation suivante, une fois
  terminés. Un job tué par Slurm (limite de temps) reste `RUNNING` : `--include-running`.
- **Runs des collègues non publiés** : passer leurs `results/` à `--results`. Il suffit de pouvoir
  les lire (voir docs/CLUSTER.md, « Stockage »).

Interface, sur la passerelle du DCE (là où sont les runs : les fichiers y sont référencés par
chemin absolu) :

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000
```

puis, sur sa machine, un tunnel SSH et http://localhost:5000 dans le navigateur :

```bash
ssh -N -L 5000:localhost:5000 dce
```

Choisir un autre port si 5000 est pris (la passerelle est partagée). Suivre un run en cours :
`--backend-store-uri sqlite:///<dossier du run>/mlflow.db`, ou `tail -f <dossier du run>/run.log`.
`cluster.mlflow=false` désactive le suivi ; `MLFLOW_TRACKING_URI` l'envoie vers un serveur.

## 10. Récupérer des résultats (sur sa machine)

```bash
scp -r dce:~/Espace-latent-optimal/results/<run> .   # ou lo hub push <run> sur le DCE, puis lo hub pull <run>
```

## 11. Repartir de zéro (sur la passerelle)

```bash
rm -rf data/datasets/* data/features/* data/models/* results/*   # irréversible : récupérer les résultats avant
```

Les fichiers cachés (`.gitkeep`) sont conservés. Il faut ensuite relancer la section 2.

## Pièges

- **Ne pas mettre `-m` au milieu de la commande.** On le met après tous les overrides, ou avant tous.
- **Arbre git non commité.** Un job long sur le DCE refuse de partir si des fichiers ne sont pas
  commités (CONTRIBUTING.md). `allow_dirty=true` lève ce garde-fou, en connaissance de cause. Les smoke runs
  ne sont pas concernés.
- **La précision.** fp16 par défaut, car un job peut tomber sur une 2080 Ti, qui n'a pas de bf16. Le bf16
  n'est possible que sur `gpu_prod_long`, qui n'a que des nœuds `sh`. Le fp16 n'est pas disponible sur CPU.
- **Ne pas demander de GPU à Slurm** (`--gres`, `--gpus-per-node`). Le DCE ne déclare pas de gres :
  `sbatch` échoue avec « Invalid generic resource (gres) specification ». Le GPU vient avec le nœud
  d'une partition `gpu_*`.
- **Le compte est partagé à trois** (fairshare commun). Chaque heure de GPU fait baisser la priorité
  de toute l'équipe : faire un smoke test d'abord.
- **Windows, en local.** Avec beaucoup de workers, Windows peut manquer de mémoire virtuelle
  (`WinError 1455`). Dans ce cas, `cluster.num_workers=2`.
