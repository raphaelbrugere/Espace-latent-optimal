"""Télécharge les datasets dans DATA_ROOT/datasets et les poids externes dans
DATA_ROOT/models (défaut : <racine>/data/...).
Seul point du code qui accède au réseau : datasets et backbones, eux, ne font que lire.
Sur le DCE, à lancer depuis la passerelle, pour ne pas occuper un GPU pendant le téléchargement.

    python tools/prepare_data.py                          # tout
    python tools/prepare_data.py imagenette oxford_pets   # une partie
    python tools/prepare_data.py dino_vits8               # poids DINO du ViT-S/8
    python tools/prepare_data.py --freeze-splits          # + recalcule les splits figés

`--freeze-splits` réécrit src/latentopt/data/splits/*.json : à ne faire qu'une fois,
le diff git doit être vide sur une installation à jour.
"""

import argparse

from latentopt.data.datasets import flowers102, galaxy10, imagenette, oxford_pets
from latentopt.data.paths import datasets_root
from latentopt.models.backbones.external import DOWNLOADS as WEIGHTS

DATASETS = {
    "imagenette": imagenette,
    "flowers102": flowers102,
    "galaxy10": galaxy10,
    "oxford_pets": oxford_pets,
}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    names = [*DATASETS, *WEIGHTS]
    p.add_argument("names", nargs="*", help=f"parmi {', '.join(names)} (défaut : tous)")
    p.add_argument("--freeze-splits", action="store_true")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(names)
    if unknown:
        p.error(f"inconnus : {sorted(unknown)}")
    root = datasets_root()
    root.mkdir(parents=True, exist_ok=True)

    for name in args.names or names:
        if name in WEIGHTS:
            print(f"[{name}] poids -> {WEIGHTS[name]()}", flush=True)
            continue
        module = DATASETS[name]
        print(f"[{name}] téléchargement dans {root}", flush=True)
        module.download(root)
        if args.freeze_splits and hasattr(module, "freeze_splits"):
            print(f"[{name}] splits figés -> {module.freeze_splits(root)}", flush=True)


if __name__ == "__main__":
    main()
