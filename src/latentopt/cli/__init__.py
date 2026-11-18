"""Commande `lo` (alias `latentopt`) : point d'entree unique du projet.

    lo <commande> [arguments]       ex. lo pretrain experiment=smoke
    lo --help                       liste des commandes
    python -m latentopt.cli ...     equivalent, sans le script installe

Les commandes sont declarees ici, independamment de l'emplacement de leur code : un module peut
etre deplace ou renomme sans changer la commande. Chaque module n'est importe qu'a l'appel de
sa commande (`lo --help` reste instantane). Ajouter une commande : une ligne dans COMMANDS.
"""

import importlib
import sys

# "mots de la commande" -> (module sous latentopt.cli, fonction, famille, description)
COMMANDS = {
    tuple(words.split()): (f"latentopt.cli.{module}", func, family, desc)
    for words, module, func, family, desc in [
        ("pretrain", "run.pretrain", "main", "run", "pre-entraine un backbone (Hydra)"),
        ("train", "run.train", "main", "run", "entraine une tete sur un backbone gele (Hydra)"),
        ("eval", "run.eval", "main", "run", "evalue un backbone et ses tetes (Hydra)"),
        ("data prepare", "data.prepare", "main", "data", "telecharge datasets et poids externes"),
        ("hub push", "store.hub", "push", "store", "publie un run sur Hugging Face"),
        ("hub pull", "store.hub", "pull", "store", "telecharge un run depuis Hugging Face"),
        ("mlflow sync", "store.mlflow_sync", "main", "store", "regroupe les runs MLflow"),
    ]
}

FAMILIES = {
    "run": "Lancer des runs",
    "data": "Preparer les donnees",
    "store": "Tracabilite et stockage",
}


def usage() -> str:
    """Texte d'aide de `lo --help`, genere depuis COMMANDS."""
    lines = ["usage : lo <commande> [arguments]", ""]
    for family, title in FAMILIES.items():
        lines.append(f"{title} :")
        for words, (_, _, fam, desc) in COMMANDS.items():
            if fam == family:
                lines.append(f"  lo {' '.join(words):<22} {desc}")
        lines.append("")
    lines.append("Aide d'une commande : lo <commande> --help")
    return "\n".join(lines)


def resolve(argv: list[str]) -> tuple[tuple[str, ...], list[str]]:
    """Commande la plus longue qui prefixe argv, et le reste des arguments."""
    for n in (2, 1):
        words = tuple(argv[:n])
        if len(words) == n and words in COMMANDS:
            return words, argv[n:]
    raise KeyError(" ".join(argv[:2]))


def main(argv: list[str] | None = None) -> int:
    """Execute une commande ; renvoie 2 si elle est inconnue, sinon son code (0 par defaut)."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    try:
        words, rest = resolve(argv)
    except KeyError:
        group = [w for w in COMMANDS if w[0] == argv[0]]
        hint = ", ".join("lo " + " ".join(w) for w in group) if group else "lo --help"
        print(f"commande inconnue : lo {' '.join(argv[:2])}\nvoir : {hint}", file=sys.stderr)
        return 2
    module, func, family, _ = COMMANDS[words]
    # ici, sur la machine qui lance (PC, passerelle du DCE) : un job Slurm ne peut pas repondre
    if family == "run" and "--help" not in rest and "-h" not in rest:
        from latentopt.run import confirm_pushed

        if not confirm_pushed():
            print("Run annule : git push, puis relancer.", file=sys.stderr)
            return 1
    fn = getattr(importlib.import_module(module), func)
    # la commande se presente comme un programme a part entiere (" usage: lo hub push ... ")
    saved, sys.argv = sys.argv, [f"lo {' '.join(words)}", *rest]
    try:
        result = fn() if family == "run" else fn(rest)  # @hydra.main lit sys.argv
    finally:
        sys.argv = saved
    return result if isinstance(result, int) else 0
