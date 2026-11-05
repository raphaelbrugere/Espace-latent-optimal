# Commandes du projet

Toutes les commandes se lancent depuis la racine du dépôt, préfixées par `uv run`. Installation
du compte DCE, partitions et stockage : voir [docs/CLUSTER.md](../docs/CLUSTER.md).

Sur le DCE, `cluster=dce` soumet un job Slurm via submitit, à condition d'ajouter **`-m`**. Sans
`-m`, le script refuse de tourner sur la passerelle. `-m` se place **après tous les overrides**.

## 1. Après chaque `git pull`

```bash
git pull && uv sync
```

## 2. Une fois : données et poids externes (sur la passerelle)

```bash
uv run python tools/prepare_data.py imagenette --freeze-splits   # pool de pré-entraînement (~100 Mo)
uv run python tools/prepare_data.py --freeze-splits              # tout (~3,8 Go)
uv run python tools/prepare_data.py dino_vits8                   # poids DINO (83 Mo)
```

`--freeze-splits` réécrit `src/latentopt/data/splits/*.json`. Sur une installation à jour, le
diff git doit rester vide.

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
| local, CPU | `uv run python tools/pretrain.py experiment=smoke` |
| DCE, dans une session `srun` | `uv run python tools/pretrain.py experiment=smoke cluster=dce method.precision=fp16` |
| DCE, job Slurm | `uv run python tools/pretrain.py experiment=smoke cluster=dce method.precision=fp16 hydra.launcher.partition=gpu_inter hydra.launcher.timeout_min=30 -m` |
| têtes, local | `uv run python tools/train.py experiment=smoke_train task=synthetic_cls,synthetic_seg,synthetic_det -m` |

Dans une session `srun`, `cluster=dce` fonctionne sans `-m`, puisqu'on est déjà dans un job Slurm.
`method.precision=fp16` sert à tester le GradScaler sur GPU. Le smoke run est en fp32 par défaut,
pour pouvoir tourner sur CPU.

## 5. Pré-entraînement d'un backbone

```bash
# reproduction de la baseline de l'arène (bf16, gpu_prod_long, 800 epochs)
uv run python tools/pretrain.py experiment=vit_lejepa_imagenette cluster=dce -m

# changer un réglage
uv run python tools/pretrain.py experiment=vit_lejepa_imagenette method.model.lamb=0.05 cluster=dce -m

# balayer plusieurs valeurs : un job par valeur
uv run python tools/pretrain.py experiment=vit_lejepa_imagenette method.model.lamb=0.01,0.02,0.05 cluster=dce -m

# manque de mémoire GPU : backbone.grad_checkpointing=true (même résultat, plus lent).
# Déjà activé dans vit_lejepa_imagenette : sans lui, OOM sur une RTX 3080 (10 Go).
uv run python tools/pretrain.py experiment=<autre> backbone.grad_checkpointing=true cluster=dce -m

# reprendre un run interrompu
uv run python tools/pretrain.py experiment=vit_lejepa_imagenette resume=results/pretrain/<exp>/<date>/checkpoints/last.ckpt cluster=dce -m
```

Sortie : `results/pretrain/<experiment_name>/<date>/` (`backbone.pt`, `results.json`, `run.log`,
`metrics.csv`, `config.yaml`, `git.txt`). Avec `-m`, il y a un sous-dossier par job :
`…/<date>/0/`, `…/<date>/1/`…

## 6. Entraînement des têtes (backbone gelé)

```bash
# une tâche, sur un de nos backbones
uv run python tools/train.py backbone=from_run backbone.run=results/pretrain/<exp>/<date> task=pets_seg cluster=dce -m

# les 5 tâches, un job chacune
uv run python tools/train.py backbone=from_run backbone.run=results/pretrain/<exp>/<date> \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det cluster=dce -m

# référence DINO (poids externes), mêmes 5 tâches
uv run python tools/train.py backbone=vit_s8_dino experiment_name=dino \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det cluster=dce -m

# une autre tête, ou d'autres réglages
uv run python tools/train.py backbone=from_run backbone.run=<run> task=pets_seg head=fpn head.channels=128 epochs=50 cluster=dce -m

# reproduction exacte de l'arène : sans cache de features (voir plus bas)
uv run python tools/train.py backbone=from_run backbone.run=<run> task=pets_seg cache_features=false cluster=dce -m
```

Sans précision, la tête et le nombre d'epochs sont ceux de la baseline de chaque tâche.

Sortie : `results/train/<exp>/<task>/<date>/` pour un run seul, et `results/train/<exp>/<date>/<task>/`
avec `-m`. Chaque dossier contient `head.pt` et `results.json` (score val, budget).

**Cache de features.** Il est activé par défaut. Les features du backbone sont calculées une
fois dans `data/features/<date>_<empreinte>/`, puis relues. L'algorithme est le même que sans cache,
mais les tirages aléatoires (ordre des images, retournements) diffèrent, comme avec une autre
graine. `cache_features=false` reproduit l'arène au bit près.

## 7. Évaluation et leaderboard

```bash
# évaluer un backbone et ses têtes sur val
uv run python tools/eval.py name=lejepa_vits8 \
    runs='[results/train/<exp>/<date>/imagenette_cls,results/train/<exp>/<date>/pets_seg]'

# score final sur test : une seule fois, à la fin
uv run python tools/eval.py name=lejepa_vits8 runs='[...]' split=test

# leaderboard (gain moyen par rapport à la baseline)
uv run python reporting/leaderboard.py --split val
uv run python reporting/leaderboard.py --split test --baseline baseline
```

Pour servir de référence aux gains, la baseline doit être évaluée avec `name=baseline`.

Sortie : `results/eval/<name>/<split>/<date>/results.json` et `results/eval/leaderboard_<split>.md`.

## 8. Suivre les jobs

```bash
squeue -u <login>                                   # jobs en cours ou en attente
scancel <job_id>                                    # annuler
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
tail -f results/pretrain/<exp>/<date>/0/run.log     # avancement, une ligne par epoch
ls results/pretrain/<exp>/<date>/.submitit/         # sorties brutes de Slurm (stdout, stderr)
```

## 9. Récupérer des résultats (sur sa machine)

```bash
scp -r dce:~/Espace-latent-optimal/results/<étape>/<expérience>/<date> .
```

## 10. Repartir de zéro (sur la passerelle)

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
