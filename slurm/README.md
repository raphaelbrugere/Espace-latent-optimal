# DCE et Slurm

Data Center d'Enseignement de CentraleSupélec : cluster Slurm avec nœuds CPU et GPU.

Ce dossier contient tout ce qui est propre au DCE et à Slurm, et rien d'autre :
- ce mode d'emploi (compte, partitions, nœuds, pièges) ;
- [`launcher/`](launcher/) : les profils de job (`long`, `short`), lus par Hydra et submitit.

Règle du dépôt : ce qui change le résultat d'un run est dans `configs/` (hors `env/`) ; la
machine (chemins, workers, device) est dans `configs/env/` ; les ressources demandées à Slurm
(partition, durée, nœuds) sont ici. Aucun réglage Slurm (launcher, partition, sbatch/srun) dans
`configs/` : `tests/test_configs.py` le vérifie.
Tout se fait à la main, dans l'ordre ci-dessous. `<login>` est l'identifiant reçu par mail.

- Documentation officielle : https://dce.pages.centralesupelec.fr/
- Passerelle SSH : `dce.metz.centralesupelec.fr`
- Portail web (Open OnDemand) : https://portail.dce.metz.centralesupelec.fr/
- Support : dce_support@groupes.renater.fr

## Une seule fois

### 1. Première connexion

```bash
ssh <login>@dce.metz.centralesupelec.fr
```

Se connecter avec le mot de passe temporaire reçu par mail, puis le remplacer par un mot de passe personnel.
Ne jamais mettre ces identifiants dans le dépôt.

À la première connexion, le DCE affiche « Creating ssh key in ~/.ssh/id_rsa… ». C'est normal : cette clé reste sur le DCE et sert aux connexions entre ses nœuds. Ne pas y toucher. Elle ne remplace pas l'étape 2.

### 2. Clé SSH (sur sa machine)

Elle évite de taper le mot de passe à chaque connexion et sert aussi à VS Code. Ces commandes se lancent sur sa propre machine, pas sur le DCE.

Windows (PowerShell) :

```powershell
# Créer une clé dédiée au DCE
ssh-keygen -t ed25519 -f $HOME\.ssh\dce

# L'installer sur le DCE (demande le mot de passe une dernière fois)
type $HOME\.ssh\dce.pub | ssh <login>@dce.metz.centralesupelec.fr "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Linux / macOS :

```bash
ssh-keygen -t ed25519 -f ~/.ssh/dce
cat ~/.ssh/dce.pub | ssh <login>@dce.metz.centralesupelec.fr "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Ajouter ensuite ce bloc à `~/.ssh/config`. Sous Windows, ce fichier est `C:\Users\<nom>\.ssh\config` ; on peut l'ouvrir avec `code $HOME\.ssh\config` :

```
Host dce
    HostName dce.metz.centralesupelec.fr
    User <login>
    IdentityFile ~/.ssh/dce
```

Désormais `ssh dce` suffit pour se connecter.

### 3. VS Code (facultatif)

Installer l'extension `ms-vscode-remote.remote-ssh`, puis lancer « Remote-SSH: Connect to Host… » et choisir `dce`.

### 4. Installer uv (sur la passerelle)

```bash
curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh    # version de [tool.uv] required-version
source ~/.local/bin/env    # ou se déconnecter et se reconnecter
```

### 5. Cloner le dépôt et créer l'environnement (sur la passerelle)

La passerelle et les nœuds de calcul ont tous accès à Internet. On installe quand même depuis la passerelle, pour ne pas occuper un GPU pendant les téléchargements.

```bash
git clone https://github.com/raphaelbrugere/Espace-latent-optimal.git
cd Espace-latent-optimal
uv sync                      # télécharge aussi Python 3.12.16 (le 3.12.3 du système n'est pas utilisé)
uv run pre-commit install    # les commits faits depuis le DCE passent aussi par les hooks
```

### 6. Git : identité et connexion à GitHub (sur la passerelle)

Sans identité, `git commit` échoue (« Author identity unknown »). Mettre la même que sur sa machine :

```bash
git config --global user.name "<nom>"
git config --global user.email "<email>"
```

Pour pousser depuis le DCE (ou cloner si le dépôt est privé), GitHub refuse le mot de passe du compte. Le plus simple est GitHub CLI, avec une connexion par lien. `gh` s'installe dans le home, sans droits admin :

```bash
cd ~
curl -LO https://github.com/cli/cli/releases/download/v2.62.0/gh_2.62.0_linux_amd64.tar.gz
tar xzf gh_2.62.0_linux_amd64.tar.gz
mv gh_2.62.0_linux_amd64/bin/gh ~/.local/bin/
rm -rf gh_2.62.0_linux_amd64*
gh auth login    # GitHub.com, HTTPS, « Authenticate Git » : Yes, « Login with a web browser »
```

`gh` affiche un code. On l'entre sur https://github.com/login/device depuis le navigateur de sa machine. `git push` fonctionne ensuite sans mot de passe.

### 7. Télécharger les données (sur la passerelle)

```bash
uv run lo data prepare               # tous les datasets
uv run lo data prepare imagenette    # ou une partie seulement
```

Les datasets vont dans `DATA_ROOT/datasets` (défaut : [docs/DATASETS.md](../docs/DATASETS.md#variables-denvironnement)). Par défaut, c'est dans le home, qui est accessible depuis tous les nœuds.

### 8. Vérifier sur un nœud GPU

```bash
srun -p gpu_inter -t 00:30:00 --pty bash    # session interactive de 30 min
cd ~/Espace-latent-optimal
nvidia-smi
uv run pytest
exit                                        # libère le nœud
```

Après `srun`, attendre que l'invite passe de `@chome` au nom du nœud avant de taper la suite. Sinon, les commandes s'exécutent sur la passerelle une fois le job terminé. Le démarrage peut prendre quelques minutes si le nœud était éteint (`idle~` dans `sinfo`).

## Au quotidien

| Étape | Où | Commande |
|---|---|---|
| Se connecter | sa machine | `ssh dce` |
| Mettre à jour le code | passerelle | `git pull && uv sync` |
| Session interactive GPU | passerelle | `srun -p gpu_inter -t 00:30:00 --pty bash` |
| Soumettre un job | passerelle, dans `tmux` | `uv run lo pretrain experiment=<nom> launcher=long -m` (voir [Lancer un job](#lancer-un-job)) |
| Voir ses jobs | passerelle | `squeue -u <login>` |
| Annuler un job | passerelle | `scancel <job_id>` |
| Récupérer des résultats | sa machine | `scp -r dce:~/Espace-latent-optimal/results/<run> .` (ou `lo hub push` puis `lo hub pull`) |

Rappels (voir aussi [CONTRIBUTING.md](../CONTRIBUTING.md)) :

- Toujours faire un smoke test avant un job long.
- Ne jamais lancer de job long avec un arbre git non commité.

## Lancer un job

Un job Slurm se lance avec un profil de [`launcher/`](launcher/) et **`-m`** :

| Profil | Partition | Durée max | Nœuds | Pour |
|---|---|---|---|---|
| `launcher=long` | `gpu_prod_long` | 2 jours | RTX 3090 ou 3080 (GTX 1080 Ti exclues) : bf16 possible | pré-entraînements, têtes |
| `launcher=short` | `gpu_inter` | 30 min (2 h max) | tous sauf GTX 1080 Ti : RTX 3090, 3080, 2080 Ti, 5080 ; bf16 non garanti (2080 Ti) | smoke runs |

```bash
uv run lo pretrain experiment=vit_lejepa_imagenette launcher=long -m
uv run lo pretrain experiment=smoke launcher=short method.precision=fp16 -m
uv run lo pretrain experiment=smoke launcher=short hydra.launcher.timeout_min=120 -m   # autre durée
```

- **Le profil choisit `env=dce`** : inutile de l'écrire. Dans une session `srun`, on est déjà sur
  un nœud : `env=dce` seul, sans profil ni `-m`.
- **Le terminal reste bloqué jusqu'à la fin des jobs** (submitit attend leurs résultats). Lancer
  la commande dans `tmux` (`tmux new -s lo`, puis `Ctrl-b d` pour se détacher, `tmux attach -t lo`
  pour revenir), ou avec `nohup ... &`. Vérifier avec `squeue -u <login>` que les jobs tournent
  bien, avant de fermer quoi que ce soit.
- **Sorties brutes de Slurm** : `results/.submitit/<job_id>/` (stdout, stderr). Utiles quand le job
  meurt avant d'écrire son `run.log` (limite de temps, mémoire, erreur de démarrage).
- **Changer une ressource pour un job** : `hydra.launcher.<clé>=<valeur>`
  (`timeout_min`, `mem_gb`, `exclude`, `partition`…). Un réglage qu'on change souvent mérite son
  propre profil dans `launcher/`.

### Balayages

`-m` lance un job par combinaison des valeurs séparées par des virgules, en parallèle
(`array_parallelism` : 4 à la fois avec `long`, la limite par utilisateur de `gpu_prod_long`).

```bash
# 3 valeurs de lamb x 3 graines = 9 jobs
uv run lo pretrain experiment=vit_lejepa_imagenette method.model.lamb=0.01,0.02,0.05 seed=0,1,2 launcher=long -m

# les 5 tâches d'un backbone, un job chacune
uv run lo train backbone=from_run backbone.run=<run> \
    task=imagenette_cls,flowers102_cls,galaxy10_cls,pets_seg,pets_det launcher=long -m
```

Une étude qu'on relance ou qu'on veut garder se décrit dans une expérience, avec
`hydra.sweeper.params` : elle est versionnée avec le code, dans `configs/experiment/`, puisque
c'est de la recette.

```yaml
# configs/experiment/lamb_x_seed.yaml
# @package _global_
defaults:
  - vit_lejepa_imagenette
hydra:
  sweeper:
    params:
      method.model.lamb: 0.01,0.02,0.05
      seed: 0,1,2
```

`uv run lo pretrain experiment=lamb_x_seed launcher=long -m` lance alors les 9 jobs.

- **Coût** : les heures GPU d'un balayage se multiplient par le nombre de jobs, sur le fairshare
  commun (voir [Partitions](#partitions)). Faire un smoke test d'abord.
- **Noms** : sur Slurm, chaque job prend la date de son propre démarrage (le nom est résolu sur
  le nœud), suivie de `_1`, `_2`… pour les jobs 1, 2… En local, les jobs d'un `-m` partagent la
  même date. Les valeurs balayées ne sont pas dans le nom : les retrouver dans MLflow (config
  complète). Pour que des `lo train` rangent leurs têtes sous le même backbone, fixer
  `backbone_run` ([docs/HUB.md](../docs/HUB.md)).
- **Têtes en parallèle sur un même backbone** : sans risque, le cache de features est partagé
  entre jobs simultanés (`evaluation/features.py`).

### Réglages de `env` et ressources

Ils ne changent pas le résultat, seulement les ressources. Valeur de `env=dce` entre parenthèses ;
override : `env.<clé>=<valeur>`.

| Réglage | CPU | GPU | RAM | Quand le changer |
|---|---|---|---|---|
| `num_workers` (6) | + 1 processus par worker | mieux alimenté si les données le font attendre | + par worker | GPU sous 100 % : en ajouter (cœurs du nœud − 1 au plus) ; RAM pleine : en retirer |
| `grad_checkpointing` (`auto`) | — | mémoire ↓↓, calcul ↑ (forward refait au backward) : plus lent | — | `true` si la mémoire GPU manque, `false` si elle suffit. `auto` : actif sous 20 Go (seuil choisi pour la baseline) |
| `compile` (`true`) | compilation au démarrage | plus rapide ensuite ; 1re epoch plus longue | compilation au démarrage | `false` pour déboguer ; `true` arrête le job si Triton manque |
| `device` (`cuda`) | calcul si `cpu` | calcul si `cuda` | — | `cpu` : smoke run sans GPU |
| `data_root` (`<dépôt>/data`) | — | — | — | données lentes à lire (home réseau) : copier dans `$TMPDIR` et y pointer `DATA_ROOT` |
| `progress_bar` (`false`) | — | — | — | `true` en `srun` pour voir les it/s |
| `mlflow` (`true`) | — | — | — | `false` : pas de `mlflow.db` |

La précision joue aussi sur la vitesse et la mémoire GPU, mais elle est dans la recette
(`method.precision`) : elle peut changer le résultat.

**Avant un run long, évaluer les meilleurs réglages pour ce type de nœud et ce modèle** : ils
dépendent du GPU, des cœurs, du modèle et du batch. Sur un nœud du type visé (`srun -w <nœud>`),
lancer ~10 epochs (`method.epochs=10 env.progress_bar=true`) en changeant un réglage à la fois,
et comparer la durée par epoch à partir de la 3e. Depuis un second terminal sur le nœud
(`srun --jobid=<job_id> --overlap --pty bash`) :

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 2
```

- GPU sous 100 % : les données freinent (`num_workers`, `$TMPDIR`) ;
- mémoire libre avec `grad_checkpointing` : essayer sans.

Garder la combinaison la plus rapide qui tient en mémoire.

### Pièges

- **Ne pas demander de GPU à Slurm** (`--gres`, `--gpus-per-node`). Le DCE ne déclare pas de gres :
  `sbatch` échoue avec « Invalid generic resource (gres) specification ». Le GPU vient avec le nœud
  d'une partition `gpu_*` ; les profils mettent `gpus_per_node: null`.
- **Sans `-m`**, Hydra n'utilise pas le launcher : `launcher=long` sans `-m` lance le run sur la
  passerelle, et le garde-fou de `lo` l'arrête (« env=dce hors d'un job Slurm »).
- **Le compte est partagé à trois** (fairshare commun). Chaque heure de GPU fait baisser la priorité
  de toute l'équipe : smoke test d'abord, et compter les heures d'un balayage avant de le lancer.

## Stockage

| Emplacement | Usage | Remarques |
|---|---|---|
| `~` (home) | code, environnement, données, sorties | stockage réseau partagé entre les nœuds, permanent, mais lent |
| `$TMPDIR` | fichiers temporaires d'un job | `/tmp/<login>-<job_id>` sur le disque local du nœud : rapide, effacé à la fin de l'allocation |
| `/mounts/datasets` | datasets, modèles et venvs déposés par les admins (sous-dossiers `datasets`, `models`, `venvs`…) | lecture seule, 6 To ; regarder s'il contient déjà un dataset avant de le télécharger |

L'espace du home est partagé entre tous les comptes du DCE : faire le ménage régulièrement.

```bash
du -ahd1 ~ | sort -hr | head    # ce qui prend de la place
uv cache clean                  # le cache uv (~/.cache/uv) peut être vidé sans risque
```

On ne peut rien écrire dans `/mounts/datasets`. Chacun garde ses données dans son propre home ; on ne lit jamais le home d'un autre.

## Partitions

Relevé fait avec `sinfo` et `sacctmgr` le 2026-11-05. Les partitions `*_tp` et `*_tp_resa` sont réservées aux TP.

| Partition | Durée max | Limite par utilisateur | Usage |
|---|---|---|---|
| `gpu_inter` | 2 h | 1 job, 1 nœud | session interactive, débogage, smoke test |
| `gpu_prod_night` | 12 h | 4 jobs | entraînements de nuit (partition `down` lors du relevé) |
| `gpu_prod_long` | 2 jours | 4 jobs | entraînements longs |
| `cpu_inter` | 2 h | 1 job, 1 nœud | session interactive CPU |
| `cpu_prod` | 12 h | 1 nœud | prétraitement CPU |
| `cpu_prod_long` | 1 jour | | prétraitement CPU long |

Compte Slurm du projet : `3a_cs_ia_n40_safran`.
- **Partagé par 3 utilisateurs** (`sshare -U` affiche `NormShares` = 1/3).
- **Priorité commune** : le temps GPU consommé par l'un fait baisser la priorité (fairshare) des jobs de tous, et elle remonte avec le temps.
- **Partitions testées** : `gpu_inter` et `gpu_prod_long` (jobs exécutés). Les partitions `*_tp` et `*_tp_resa` servent aux TP.

Internet : la passerelle et les nœuds GPU y ont accès (testé). Python système sur la passerelle : 3.12.3, non utilisé par le projet (uv installe le Python de `.python-version`).

### Nœuds GPU

Chaque nœud a 1 GPU, qui vient avec le nœud : Slurm ne déclare pas de gres (`Gres=(null)` dans `scontrol show node`), donc on ne demande ni `--gres` ni `--gpus-per-node`, sinon `sbatch` refuse le job. Driver : 610 (CUDA 13.3) sur tx00 le 2026-11-05, 615 (CUDA 13.4) sur dani02 le 2026-11-21 ; tous deux suffisent pour `torch 2.15` (CUDA 13.2).

`torch.cuda.get_arch_list()` (propre au paquet torch, pas au nœud) : `sm_75` à `sm_120`. RTX 2080 Ti, 3090, 3080 et 5080 sont donc prises en charge, pas les GTX 1080 Ti (`sm_61`).

GPU d'après la documentation officielle, partitions d'après le relevé. CPU : 1 socket, 2 threads par cœur ; « cœurs » compte les threads vus par Slurm. RAM : 32 Go (30 Go utilisables).

| Nœuds | GPU | Mémoire GPU | CPU | Partitions |
|---|---|---|---|---|
| `sh01`–`sh09`, `sh20`–`sh22` | RTX 3090 (Ampere) | 24 Go | Xeon W-2125, 8 cœurs | `gpu_inter`, `gpu_prod_long` |
| `sh00`, `sh17`–`sh19` | RTX 3080 (Ampere) | 10 Go | Xeon W-2225, 8 cœurs | `gpu_inter`, `gpu_prod_long` |
| `sh10`–`sh16` | GTX 1080 Ti (Pascal) | 11 Go | Xeon W-2125, 8 cœurs | `gpu_inter`, `gpu_prod_long` |
| `tx00`–`tx16` | RTX 2080 Ti (Turing) | 11 Go | Xeon W-2125 (W-2225 pour `tx05`), 8 cœurs | `gpu_inter`, et `gpu_prod_night` pour `tx09`–`tx16` |
| `dani01`–`dani17` | RTX 5080 (Blackwell) | 16 Go | Xeon w3-2423, 12 cœurs | `gpu_inter`, `gpu_prod_night` |

Lors du relevé, `sh01`–`sh16` étaient en maintenance (`drain`) : seuls les nœuds `sh00` et `sh17`–`sh22` servaient `gpu_prod_long`.

Les deux profils excluent donc les GTX 1080 Ti (`exclude: sh[10-16]`).

Choix de la précision :
- **`fp16` par défaut** : marche sur tous les nœuds.
- **`bf16`** exige une Ampere ou une Blackwell (capacité de calcul ≥ 8.0) : RTX 3090, 3080, 5080. Avec `launcher=long`, c'est garanti (3090 et 3080 seulement). Sur un autre GPU, `lo` s'arrête au démarrage, plutôt qu'un bf16 émulé, beaucoup plus lent.

Mémoire GPU : 24 Go sur une RTX 3090, 10 Go sur une RTX 3080, toutes deux servies par `gpu_prod_long`. Sans `grad_checkpointing`, la baseline ne tient pas sur une RTX 3080 (plus de 9,6 Go dès le premier pas sur `sh00`). `env=dce` met `grad_checkpointing: auto` : activé sur un GPU de moins de 20 Go (3080, 2080 Ti, 5080), désactivé au-delà (3090). Effet sur les ressources : voir [Réglages de `env`](#réglages-de-env-et-ressources).

`torch.compile` : `env=dce` l'exige (`compile: true`). Il faut Triton (installé avec torch) et un compilateur C : `/usr/bin/gcc` est présent et `has_triton()` vrai (vérifié sur tx08 et dani02 le 2026-11-21).

Chargement des données :
- **`num_workers`** : 6 dans `env=dce`, ce qui laisse de la marge sur tous les nœuds (8 cœurs sur `sh` et `tx`, 12 sur `dani`).
- **Si le chargement ralentit l'entraînement**, copier le dataset dans `$TMPDIR` au début du job : le home est un disque réseau, plus lent.

`nvidia-smi` n'existe que sur les nœuds GPU, pas sur la passerelle. Pour l'exécuter sans session interactive :

```bash
sbatch -p gpu_inter -t 00:05:00 -o gpu_info.out --wrap 'hostname; nvidia-smi'
cat gpu_info.out    # une fois le job terminé (squeue vide)
```

### Autres machines

D'après la documentation officielle (OS : Ubuntu 18.04 ou 20.04 ; disque local SSD 256 Go / SATA 1 To ; home sur NAS) :

| Machine | Rôle | CPU | RAM |
|---|---|---|---|
| `chome` | passerelle | 2 × Xeon Silver 4214R (12 cœurs, Cascade Lake) | 64 Go |
| `kyle01`–`kyle68` | calcul CPU | 2 × Xeon Silver 4110 (8 cœurs, Skylake) | 64 Go |
| `sar01`–`sar32` | calcul CPU | 2 × Xeon E5-2637 v3 (4 cœurs, Haswell) | 32 Go |

## Citation

Si des résultats publiés ont été obtenus sur le DCE, citer :

J. Fix, S. Vialle, R. Hellequin, C. Mercier, P. Mercier et J.-B. Tavernier, *Feedback from a data center for education at CentraleSupélec engineering school*, IPDPSW 2022, pp. 330-337, doi: 10.1109/IPDPSW55747.2022.00065.
