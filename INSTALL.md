# Installation

## Prérequis

[git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/) et VS Code.

## Étapes

À faire une seule fois par machine :

```bash
# 1. Installer uv (une seule fois par machine)
#    Windows (PowerShell) :
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Linux / macOS / DCE :
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Cloner le dépôt
git clone https://github.com/raphaelbrugere/Espace-latent-optimal.git
cd Espace-latent-optimal

# 3. Créer l'environnement (.venv) avec les versions exactes de uv.lock
uv sync

# 4. Activer le nettoyage automatique à chaque commit
uv run pre-commit install

# 5. Vérifier que tout fonctionne
uv run pre-commit run --all-files
uv run pytest

# 6. Installer les extensions VS Code du projet
code --install-extension ms-python.python
code --install-extension charliermarsh.ruff
```

Ces commandes fonctionnent dans PowerShell comme dans un terminal Linux. Alternative dans VS Code : onglet Extensions, taper `@recommended`, puis « Install Workspace Recommended Extensions ».

La liste des extensions est celle de `.vscode/extensions.json` : si on en ajoute une, mettre à jour les deux.

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
| `gitleaks` | aucun secret (clé API W&B, token Hugging Face, mot de passe) dans le code |

Si un hook a modifié des fichiers, le commit est interrompu pour que tu puisses voir les changements : refaire `git add` puis `git commit`. Si une erreur ne peut pas être corrigée automatiquement (une variable non définie par exemple), ruff indique le fichier et la ligne à corriger à la main.

Pour lancer les vérifications sans committer : `uv run pre-commit run --all-files`.

## Au quotidien

Après chaque `git pull` qui modifie `uv.lock`, relancer `uv sync` pour mettre l'environnement à jour.

Pour ajouter une dépendance : `uv add <paquet>` (ou `uv add --dev <paquet>` pour un outil de développement), puis commiter `pyproject.toml` et `uv.lock`.

PyTorch : `uv sync` installe la version CPU sous Windows et la version CUDA 13 sous Linux (DCE). En local,
un smoke run (`uv run python tools/pretrain.py experiment=smoke`) vérifie l'installation sans GPU.

Données : rien n'est téléchargé à l'installation. `uv run python tools/prepare_data.py` les place dans `data/`
(voir [docs/DATASETS.md](docs/DATASETS.md)).

Pour le cluster DCE, voir [docs/CLUSTER.md](docs/CLUSTER.md), et [slurm/COMMANDS.md](slurm/COMMANDS.md) pour
toutes les commandes du projet.
