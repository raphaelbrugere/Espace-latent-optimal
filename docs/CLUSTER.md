# Cluster DCE

Data Center d'Enseignement de CentraleSupélec : cluster Slurm avec nœuds CPU et GPU.
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
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env    # ou se déconnecter et se reconnecter
```

### 5. Cloner le dépôt et créer l'environnement (sur la passerelle)

La passerelle et les nœuds de calcul ont tous accès à Internet. On installe quand même depuis la passerelle, pour ne pas occuper un GPU pendant les téléchargements.

```bash
git clone https://github.com/raphaelbrugere/Espace-latent-optimal.git
cd Espace-latent-optimal
uv sync
```

Si le dépôt est privé, il faut s'authentifier auprès de GitHub depuis le DCE : soit un token personnel, soit une clé SSH créée sur le DCE et ajoutée à son compte GitHub.

### 6. Télécharger les données (sur la passerelle)

```bash
uv run python tools/prepare_data.py               # tous les datasets
uv run python tools/prepare_data.py imagenette    # ou une partie seulement
```

Les datasets vont dans `DATA_ROOT/datasets`. Par défaut `DATA_ROOT` est `<racine du dépôt>/data`, dans le home, qui est accessible depuis tous les nœuds.

### 7. Vérifier sur un nœud GPU

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
| Soumettre un job | passerelle | `uv run python tools/pretrain.py experiment=<nom> cluster=dce -m` (voir [slurm/COMMANDS.md](../slurm/COMMANDS.md)) |
| Voir ses jobs | passerelle | `squeue -u <login>` |
| Annuler un job | passerelle | `scancel <job_id>` |
| Récupérer des résultats | sa machine | `scp -r dce:~/Espace-latent-optimal/results/<étape>/<expérience>/<date> .` |

Rappels (voir aussi [CONTRIBUTING.md](../CONTRIBUTING.md)) :

- Toujours faire un smoke test avant un job long.
- Ne jamais lancer de job long avec un arbre git non commité.

## Stockage

| Emplacement | Usage | Remarques |
|---|---|---|
| `~` (home) | code, environnement, données, sorties | stockage réseau partagé entre les nœuds, permanent, mais lent |
| `$TMPDIR` | fichiers temporaires d'un job | `/tmp/<login>-<job_id>` sur le disque local du nœud (environ 200 Go libres) : rapide, effacé à la fin de l'allocation |
| `/mounts/datasets` | datasets, modèles et venvs déposés par les admins (sous-dossiers `datasets`, `models`, `venvs`…) | lecture seule, 6 To ; regarder s'il contient déjà un dataset avant de le télécharger |

Aucun quota n'est imposé pour l'instant, mais l'espace est partagé entre environ 600 comptes. Il faut rester sous 100 Go et faire le ménage au-delà de 200 Go.

```bash
du -ahd1 ~ | sort -hr | head    # ce qui prend de la place
uv cache clean                  # le cache uv (~/.cache/uv) peut être vidé sans risque
```

On ne peut rien écrire dans `/mounts/datasets`. Pour éviter que chaque membre de l'équipe télécharge sa propre copie des données, deux options :
- **Ouvrir son dossier `data` au groupe Unix commun.** Le propriétaire lance `chmod g+x ~ ~/Espace-latent-optimal` et `chmod -R g+rX ~/Espace-latent-optimal/data`. Les autres mettent `DATA_ROOT` sur ce chemin dans leur `~/.bashrc`. À vérifier : `id` doit afficher le même groupe pour tous.
- **Demander au support un dossier partagé en écriture.** C'est la solution si les données grossissent : features en cache, checkpoints, ImageNet-100.

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

Internet : la passerelle et les nœuds GPU y ont accès (testé). Python système sur la passerelle : 3.12.3.

### Nœuds GPU

Chaque nœud a 1 GPU. Le driver est le 610 (CUDA 13.3), ce qui suffit pour `torch 2.15` (CUDA 13.2). Prise en charge des RTX 2080 Ti (`sm_75`) à vérifier : voir [slurm/COMMANDS.md](../slurm/COMMANDS.md), section 3.

| Famille | GPU | Mémoire GPU | CPU / RAM | Partitions |
|---|---|---|---|---|
| `sh00`–`sh22` | RTX 3080 (Ampere) | 10 Go | 8 cœurs / 30 Go | `gpu_inter`, `gpu_prod_long` |
| `tx00`–`tx16` | RTX 2080 Ti (Turing) | 11 Go | 8 cœurs / 30 Go | `gpu_inter`, et `gpu_prod_night` pour `tx09`–`tx16` |
| `dani01`–`dani17` | non relevé (nœuds éteints lors du relevé) | | 12 cœurs / 30 Go | `gpu_inter`, `gpu_prod_night` |

Lors du relevé, `sh01`–`sh16` étaient en maintenance (`drain`) : seuls 6 nœuds servaient `gpu_prod_long`.

Choix de la précision :
- **`16-mixed` (fp16) par défaut.** Un job `gpu_inter` peut tomber sur un nœud `tx` : la 2080 Ti n'a pas de bf16 matériel.
- **`bf16-mixed` possible** seulement si le job est sûr de tourner sur un `sh` (`gpu_prod_long`, ou `-C sh`).
- **Mémoire** : prévoir 10 Go de mémoire GPU au plus.

Chargement des données :
- **`num_workers` à 6 au plus** : un nœud n'a que 8 cœurs.
- **Si le chargement ralentit l'entraînement**, copier le dataset dans `$TMPDIR` au début du job : le home est un disque réseau, plus lent.

`nvidia-smi` n'existe que sur les nœuds GPU, pas sur la passerelle. Pour l'exécuter sans session interactive :

```bash
sbatch -p gpu_inter -t 00:05:00 -o gpu_info.out --wrap 'hostname; nvidia-smi'
cat gpu_info.out    # une fois le job terminé (squeue vide)
```

## À compléter

- GPU des nœuds `dani*`.
- Variables d'environnement lues par le code : `DATA_ROOT` (défaut `<racine du dépôt>/data`), `FEATURES_ROOT` (défaut `DATA_ROOT/features`), `MODELS_ROOT` (défaut `DATA_ROOT/models`). Pas encore utilisées : `OUTPUT_ROOT`, `ENV_ROOT`.

## Citation

Si des résultats publiés ont été obtenus sur le DCE, citer :

J. Fix, S. Vialle, R. Hellequin, C. Mercier, P. Mercier et J.-B. Tavernier, *Feedback from a data center for education at CentraleSupélec engineering school*, IPDPSW 2022, pp. 330-337, doi: 10.1109/IPDPSW55747.2022.00065.
