"""Publie et telecharge des backbones sur le Hub Hugging Face, dans l'organisation du projet
(docs/HUB.md). Un depot par backbone, copie de son dossier results/<run>/ : backbone/ et
downstream/<tache>-<neck>-<tete>-<date>/ (avec ses evaluations eval/<split>-<date>/).

    lo hub push <run>                        # parties terminees du dossier (depot prive)
    lo hub push <run> --dry-run              # affiche sans publier
    lo hub pull <run>                        # -> results/<run>/ (tout le depot)
    lo hub pull <run> --only "downstream/pets_seg-*"        # une partie seulement

Travail a plusieurs : `pull`, entrainer ou evaluer, puis `push`. Un dossier de run n'est
jamais modifie une fois ecrit : le push n'envoie que les fichiers absents du depot, donc
n'ecrase jamais ce que d'autres ont publie. Si quelqu'un publie pendant le push, le Hub
refuse le commit : relancer le push.

Prerequis : `hf auth login` (token Write pour publier). `pull` est rarement necessaire : un
nom passe a la place d'un chemin (backbone.run=<run>, downstream=[<run>/downstream/...]) est
telecharge automatiquement s'il manque en local.
"""

import argparse
import json
from pathlib import Path

from latentopt import hub


def push(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lo hub push", description="Publie un backbone sur le Hub.")
    p.add_argument("run", type=Path, help="nom ou dossier d'un backbone (ou d'une de ses parties)")
    p.add_argument("--public", action="store_true", help="depot public (defaut : prive)")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="publie des runs lances avec un arbre git non commite",
    )
    p.add_argument("--dry-run", action="store_true", help="affiche ce qui serait publie")
    args = p.parse_args(argv)

    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
    run = hub.locate(args.run, pull=False)  # chemin, ou nom du backbone dans results/
    try:
        repo = hub.repo_id(hub.run_root(run).name)
        # commit et fichiers du depot ; --dry-run : sans reseau, tout parait nouveau
        parent, published = hub.remote(repo, api) if api else (None, set())
        upload = hub.plan(run, published)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return 1

    dirty = [
        f
        for f in upload.files
        if f.endswith("results.json")
        and json.loads((upload.folder / f).read_text(encoding="utf-8")).get("git_dirty")
    ]
    if dirty and not args.allow_dirty:
        print("runs lances avec un arbre git non commite (--allow-dirty pour publier quand meme) :")
        print("\n".join(f"  {upload.folder / d}" for d in dirty))
        return 1

    print(f"{upload.repo_id}\n  {upload.folder}")
    print(f"  {len(upload.present)} fichiers deja publies, {len(upload.files)} nouveaux :")
    print("\n".join(f"    {f}" for f in upload.files))
    if args.dry_run or not upload.files:
        return 0
    hub.push(upload, parent=parent, private=not args.public, api=api)
    print(f"publie : https://huggingface.co/{upload.repo_id}")
    return 0


def pull(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lo hub pull", description="Telecharge des backbones dans leur dossier local."
    )
    p.add_argument("names", nargs="+", help="noms de backbones (depots)")
    p.add_argument("--only", nargs="+", default=None,
                   help='motifs des fichiers a prendre, ex. "downstream/pets_seg-*"')  # fmt: skip
    args = p.parse_args(argv)
    for name in args.names:
        print(f"{hub.repo_id(name)} -> {hub.fetch(name, only=args.only)}")
    return 0
