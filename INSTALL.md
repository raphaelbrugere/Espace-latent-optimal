# Installation

## Prérequis

[git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/) et VS Code.

## Étapes

À faire une seule fois par machine :

```bash
# 1. Installer uv, à la version figée dans pyproject.toml ([tool.uv] required-version)
#    Windows (PowerShell) :
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.13.7/install.ps1 | iex"
#    Linux / macOS / DCE :
curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh
#    uv déjà installé : `uv self update 0.13.7` (ou `pip install uv==0.13.7` s'il vient de pip)

# 2. Cloner le dépôt
git clone https://github.com/raphaelbrugere/Espace-latent-optimal.git
cd Espace-latent-optimal

# 3. Créer l'environnement (.venv) avec les versions exactes de uv.lock. uv télécharge
#    lui-même le Python de .python-version (3.12.16) : aucun Python à installer.
uv sync

# 4. Activer le nettoyage automatique à chaque commit
uv run pre-commit install

# 5. Vérifier que tout fonctionne
uv run pre-commit run --all-files
uv run pytest -m "not slow"

# 6. Installer les extensions VS Code du projet
code --install-extension ms-python.python
code --install-extension charliermarsh.ruff
```

Ces commandes fonctionnent dans PowerShell comme dans un terminal Linux. Alternative dans VS Code : onglet Extensions, taper `@recommended`, puis « Install Workspace Recommended Extensions ».

La liste des extensions est celle de `.vscode/extensions.json` : si on en ajoute une, mettre à jour les deux.

## Versions figées

Tout le monde (local, CI, DCE) utilise exactement les mêmes versions :

| Quoi | Où c'est figé |
|---|---|
| uv | `pyproject.toml`, `[tool.uv] required-version` : avec une autre version, uv refuse de tourner. La CI lit la version au même endroit |
| Python | `.python-version` ; `python-preference = "only-managed"` : uv n'utilise jamais le Python du système |
| Bibliothèques Python et outils de dev (pytest, ruff, pre-commit) | `uv.lock` (versions exactes et empreintes) ; les `>=` de `pyproject.toml` ne sont que des minimums |
| Hooks pre-commit | `.pre-commit-config.yaml` (`rev`) |
| Actions GitHub | `.github/workflows/lint.yml` et `ci.yml` |
| gh (DCE) | `docs/CLUSTER.md`, étape 6 |

Pour changer une version : modifier le fichier concerné dans une PR, puis `uv self update <version>` (uv) ou `uv sync` (Python, bibliothèques) sur chaque machine.

Deux versions sont écrites à deux endroits, à garder identiques à la main :

- **ruff** : dans `uv.lock` (`uv run ruff`, et l'extension VS Code grâce à `"ruff.importStrategy": "fromEnvironment"` dans `.vscode/settings.json` ; sans ce réglage, elle utiliserait sa propre copie) et dans `.pre-commit-config.yaml` (hooks, CI). Si elles divergent, le formatage de l'éditeur et celui de la CI peuvent différer. Pour changer : `uv add --dev "ruff==<version>"` et le `rev` du hook `ruff-pre-commit`.
- **uv** : dans `pyproject.toml` et dans les commandes d'installation de ce fichier (étape 1) et de `docs/CLUSTER.md` (étape 4).

## Ce que font les extensions VS Code

| Extension | Ce qu'elle fait |
|---|---|
| `ms-python.python` | support Python : choix de l'interpréteur (`.venv`), débogueur, onglet Tests (pytest). Installe aussi Pylance (autocomplétion, aller à la définition, erreurs de type) |
| `charliermarsh.ruff` | formate le code et trie les imports à chaque enregistrement, souligne les erreurs de lint en direct. Mêmes règles que pre-commit |

## Ce que fait pre-commit

À chaque `git commit`, pre-commit lance des vérifications sur les fichiers modifiés (configurées dans `.pre-commit-config.yaml`). La CI GitHub relance exactement les mêmes vérifications sur tout le dépôt.

Corrections automatiques :

| Hook | Ce qu'il fait |
|---|---|
| `trailing-whitespace` | supprime les espaces en fin de ligne |
| `end-of-file-fixer` | garantit un seul saut de ligne en fin de fichier |
| `mixed-line-ending` | uniformise les fins de ligne d'un fichier |
| `ruff-check --fix` | lint Python : trie les imports, supprime les imports inutilisés, modernise la syntaxe, signale les erreurs probables (variable non définie, etc.) |
| `ruff-format` | formate le code Python : espaces, virgules, indentation, guillemets, lignes de 100 caractères max |

Vérifications bloquantes (rien n'est corrigé, le commit est refusé) :

| Hook | Ce qu'il vérifie |
|---|---|
| `check-yaml`, `check-toml` | les configs Hydra, `pyproject.toml` et les workflows sont syntaxiquement valides |
| `check-added-large-files` | aucun fichier de plus de 1 Mo (checkpoint, dataset, PDF ajouté par erreur) |
| `gitleaks` | aucun secret (token Hugging Face, mot de passe) dans le code |

Si un hook a modifié des fichiers, le commit est interrompu pour que tu puisses voir les changements : refaire `git add` puis `git commit`. Si une erreur ne peut pas être corrigée automatiquement (une variable non définie par exemple), ruff indique le fichier et la ligne à corriger à la main.

Pour lancer les vérifications sans committer : `uv run pre-commit run --all-files`.

## Au quotidien

Après chaque `git pull` qui modifie `uv.lock`, relancer `uv sync` pour mettre l'environnement à jour.

Pour ajouter une dépendance : `uv add <paquet>` (ou `uv add --dev <paquet>` pour un outil de développement), puis commiter `pyproject.toml` et `uv.lock`.

PyTorch : `uv sync` installe la version CUDA 13.2 partout (Windows, Linux, DCE). Le GPU n'est
utilisé que s'il est compatible : pilote NVIDIA qui accepte CUDA 13.2 (`nvidia-smi` affiche la
« CUDA Version » maximale du pilote) et carte prise en charge par torch (RTX 20xx et plus récentes ;
pas les GTX 10xx). Avant chaque run, `lo` fait un calcul d'essai sur le GPU (`run.cuda_problem`) :

- `cluster=local` (`device: auto`) : GPU compatible, sinon CPU, avec un avertissement qui donne la
  raison si un GPU NVIDIA est présent ;
- `cluster=dce` (`device: cuda`) : GPU incompatible, le run s'arrête avec la raison, au lieu de
  tourner des heures sur CPU.

Pour vérifier sans lancer de run :

```bash
uv run python -c "from latentopt.run import cuda_problem; print(cuda_problem() or 'GPU OK')"
```

Un smoke run (`uv run lo pretrain experiment=smoke`) vérifie le reste de l'installation.

Données : rien n'est téléchargé à l'installation. `uv run lo data prepare` les place dans `data/`
(voir [docs/DATASETS.md](docs/DATASETS.md)).

Pour le cluster DCE, voir [docs/CLUSTER.md](docs/CLUSTER.md), et [slurm/COMMANDS.md](slurm/COMMANDS.md) pour
toutes les commandes du projet.
