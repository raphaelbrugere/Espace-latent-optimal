"""Vide les dossiers generes par le projet (donnees, resultats, caches), apres confirmation.

    lo clear --all                          # toutes les cibles
    lo clear --target features mlflow       # une partie
    lo clear --help

Cibles (emplacements resolus comme le reste du code, variables d'environnement comprises) :
    results   runs (OUTPUT_ROOT, defaut <racine>/results) : poids perdus s'ils ne sont pas
              publies (`lo hub push`)
    datasets  DATA_ROOT/datasets : re-telecharges par `lo data prepare`
    features  features en cache (FEATURES_ROOT, defaut DATA_ROOT/features) : recalculees
    models    poids externes (MODELS_ROOT, defaut DATA_ROOT/models) : `lo data prepare`
    mlflow    DATA_ROOT/mlflow.db, mlflow_hub/, mlflow_artifacts/ : `lo mlflow sync`
    cache     caches Python et outils, sorties Hydra/Lightning/MLflow a la racine du projet

Seuls les fichiers non suivis par git sont supprimes : tout chemin suivi, ou qui contient un
fichier suivi (ex. les .gitkeep de data/), est ecarte ; les dossiers de data/ sont vides, jamais
supprimes.
La liste de ce qui sera supprime et sa taille sont affichees, puis une confirmation y/n est
toujours demandee : sans reponse `y`, rien n'est supprime.
"""

import argparse
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from latentopt.data.paths import PROJECT_ROOT, data_root, datasets_root, features_root, models_root
from latentopt.hub import results_root

KEEP = {".gitkeep"}
# a la racine du projet ; __pycache__ est cherche partout sauf dans SKIP
ROOT_CACHES = [".pytest_cache", ".ruff_cache", ".mypy_cache", "outputs", "multirun",
               "lightning_logs", "mlruns", "mlflow.db"]  # fmt: skip
SKIP = {".venv", ".git"}


def contents(folder: Path) -> list[Path]:
    """Contenu de `folder`, sans les .gitkeep (le dossier lui-meme reste)."""
    return sorted(p for p in folder.iterdir() if p.name not in KEEP) if folder.is_dir() else []


def mlflow_paths() -> list[Path]:
    paths = (data_root() / n for n in ("mlflow.db", "mlflow_hub", "mlflow_artifacts"))
    return [p for p in paths if p.exists()]


def cache_paths() -> list[Path]:
    found = [p for p in (PROJECT_ROOT / n for n in ROOT_CACHES) if p.exists()]
    for top in PROJECT_ROOT.iterdir():
        if top.is_dir() and top.name not in SKIP:
            found += top.rglob("__pycache__") if top.name != "__pycache__" else [top]
    return sorted(set(found))


TARGETS: dict[str, Callable[[], list[Path]]] = {
    "results": lambda: contents(results_root()),
    "datasets": lambda: contents(datasets_root()),
    "features": lambda: contents(features_root()),
    "models": lambda: contents(models_root()),
    "mlflow": mlflow_paths,
    "cache": cache_paths,
}


def tracked() -> set[Path]:
    """Fichiers suivis par git et leurs dossiers parents ; vide hors d'un depot git."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=PROJECT_ROOT, capture_output=True,
                             check=True, text=True).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return set()
    files = [(PROJECT_ROOT / f).resolve() for f in out.split("\0") if f]
    return {*files, *(parent for f in files for parent in f.parents)}


def size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(f.lstat().st_size for f in path.rglob("*") if f.is_file() or f.is_symlink())


def human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def confirm(question: str) -> bool:
    """Vrai seulement sur une reponse `y` ; une entree fermee (EOF) vaut non."""
    try:
        return input(f"{question} [y/n] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("-a", "--all", action="store_true", help="toutes les cibles")
    which.add_argument("-t", "--target", nargs="+", choices=list(TARGETS), metavar="CIBLE",
                       help=f"parmi {', '.join(TARGETS)}")  # fmt: skip
    args = p.parse_args(argv)

    names = list(TARGETS) if args.all else list(dict.fromkeys(args.target))
    protected = tracked()
    plan = {name: [p for p in TARGETS[name]() if p.resolve() not in protected] for name in names}
    total = 0
    for name, paths in plan.items():
        sizes = [size(path) for path in paths]
        total += sum(sizes)
        print(f"[{name}] {len(paths)} element(s), {human(sum(sizes))}")
        for path, s in zip(paths, sizes, strict=True):
            print(f"  {path}  ({human(s)})")
    if not any(plan.values()):
        print("Rien a supprimer.")
        return 0
    if not confirm(f"Supprimer definitivement {human(total)} ({', '.join(names)}) ?"):
        print("Annule, rien n'a ete supprime.")
        return 1
    for paths in plan.values():
        for path in paths:
            remove(path)
    print("Supprime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
