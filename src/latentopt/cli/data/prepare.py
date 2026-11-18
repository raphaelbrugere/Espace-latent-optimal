"""Telecharge les datasets dans DATA_ROOT/datasets et les poids externes dans
DATA_ROOT/models (defaut : <racine>/data/...).
Seul point du code qui accede au reseau : datasets et backbones, eux, ne font que lire.
Sur le DCE, a lancer depuis la passerelle, pour ne pas occuper un GPU pendant le telechargement.

    lo data prepare                          # tout
    lo data prepare imagenette oxford_pets   # une partie
    lo data prepare dino_vits8               # poids DINO du ViT-S/8
    lo data prepare oxford_pets --force-download  # supprime et re-telecharge ce dataset
    lo data prepare --force-splits           # reecrit aussi les splits figes qui existent

Chaque dataset prepare cree ses splits figes manquants (src/latentopt/data/splits/*.json), sans
toucher a ceux qui existent : ils sont ainsi ecrits ici, une fois, et non par le premier job qui
charge le dataset. Sur une installation a jour, le diff git doit rester vide.
`--force-splits` les reecrit : a ne faire que pour changer volontairement les ensembles, les
scores ne sont plus comparables aux anciens.
`--force-download` supprime d'abord le dossier du dataset (et son archive) : sans lui, un
telechargement deja present n'est pas refait, meme interrompu ou corrompu. Il ne concerne
que les datasets, pas les poids.
Chaque dataset annonce s'il est deja present (utilise tel quel) ou telecharge, et ses splits
s'ils sont crees, conserves ou reecrits.
`--verbose` ajoute les barres de progression et la duree des telechargements ; sans lui, les
barres sont masquees (torchvision et Hugging Face les affichent sinon toujours, et remplissent
les logs Slurm).
"""

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

from latentopt.data.datasets import flowers102, galaxy10, imagenette, oxford_pets
from latentopt.data.paths import datasets_root
from latentopt.models.backbones.external import DOWNLOADS as WEIGHTS

DATASETS = {
    "imagenette": imagenette,
    "flowers102": flowers102,
    "galaxy10": galaxy10,
    "oxford_pets": oxford_pets,
}


@contextmanager
def progress_bars(enabled: bool) -> Iterator[None]:
    """Masque les barres de telechargement hors `--verbose`, puis retablit l'etat d'origine."""
    if enabled:
        yield
        return
    import datasets.utils.logging as datasets_logging
    import huggingface_hub.utils as hub_utils
    from torchvision.datasets import utils as tv_utils

    tqdm, datasets_on = tv_utils.tqdm, datasets_logging.is_progress_bar_enabled()
    hub_off = hub_utils.are_progress_bars_disabled()
    tv_utils.tqdm = partial(tqdm, disable=True)  # `_urlretrieve` lit ce nom a chaque appel
    datasets_logging.disable_progress_bar()
    hub_utils.disable_progress_bars()
    try:
        yield
    finally:
        tv_utils.tqdm = tqdm
        if datasets_on:
            datasets_logging.enable_progress_bar()
        if not hub_off:
            hub_utils.enable_progress_bars()


@contextmanager
def console_log() -> Iterator[None]:
    """Affiche les messages `info` du paquet (ex. creation des splits) pendant le bloc."""
    logger, handler = logging.getLogger("latentopt"), logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("  %(message)s"))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    names = [*DATASETS, *WEIGHTS]
    p.add_argument("names", nargs="*", help=f"parmi {', '.join(names)} (defaut : tous)")
    p.add_argument("--force-splits", action="store_true", help="reecrit les JSON de splits")
    p.add_argument(
        "--force-download", action="store_true", help="supprime et re-telecharge les datasets"
    )
    p.add_argument("--verbose", action="store_true", help="affiche les barres de progression")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(names)
    if unknown:
        p.error(f"inconnus : {sorted(unknown)}")
    root = datasets_root()
    root.mkdir(parents=True, exist_ok=True)

    with progress_bars(args.verbose), console_log():
        for name in args.names or names:
            if name in WEIGHTS:
                print(f"[{name}] poids -> {WEIGHTS[name]()}", flush=True)
                continue
            module = DATASETS[name]
            if not args.force_download and module.present(root):
                print(f"[{name}] deja present dans {root}, utilise tel quel", flush=True)
            else:
                print(f"[{name}] telechargement dans {root}", flush=True)
                start = time.perf_counter()
                module.download(root, force=args.force_download)
                if args.verbose:
                    print(f"[{name}] telecharge en {time.perf_counter() - start:.0f} s", flush=True)
            if hasattr(module, "freeze_splits"):
                module.freeze_splits(root, force=args.force_splits)  # dit lui-meme ce qu'il fait


if __name__ == "__main__":
    main()
