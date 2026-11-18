"""Galaxy10 DECaLS : 17 736 images 256px, 10 classes morphologiques déséquilibrées.

Source : Hugging Face, révision épinglée, train et test HF concaténés dans cet ordre.
Pas de split officiel exploitable : stratifié 80/10/10 figé, indexé
par ligne de cette concaténation.

Le téléchargement passe par le cache HF puis `save_to_disk` ; le chargement lit
uniquement ce dossier (`load_from_disk`), sans aucun accès réseau.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from latentopt.data import splits
from latentopt.data.datasets.base import Subset, raw, require, reset
from latentopt.data.paths import datasets_root

NAME = "galaxy10"
NUM_CLASSES = 10
REPO = "matthieulel/galaxy10_decals"
# CRITIQUE : commit HF des données téléchargées. NE PAS LE MODIFIER à la légère : les splits
# figés désignent les images par numéro de ligne, valable pour cette révision seulement. Le
# changer impose `lo data prepare galaxy10 --force-download --force-splits` (les données sur
# disque ne gardent pas trace de leur révision : rien ne détecte un oubli), et rend les
# scores incomparables aux anciens.
REVISION = "d1cfafdcce69ca7cd25aecce1a0b578cf07e6a30"
FOLDER = "galaxy10_decals"
# CRITIQUE : nom court du dataset dans les noms de runs et de dépôts Hub. NE JAMAIS LE
# MODIFIER : il renommerait tous les runs existants. Il doit aussi être DIFFÉRENT de celui
# de tous les autres datasets (vérifié par tests/data/test_abbreviations.py).
# Le « d » distingue DECaLS (256px) de Galaxy10 SDSS (69px), un autre dataset.
ABBREV = "gal10d"


class Source:
    """Train + test HF concaténés -> (image PIL RGB, label)."""

    def __init__(self, root: Path | None = None) -> None:
        from datasets import load_from_disk

        root = root or datasets_root()
        require(root / FOLDER, NAME)
        self.ds = load_from_disk(str(root / FOLDER))

    def ids(self) -> list[str]:
        return [str(i) for i in range(len(self.ds))]

    def labels(self) -> list[int]:
        return [int(y) for y in self.ds["label"]]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        item = self.ds[i]
        return item["image"].convert("RGB"), int(item["label"])


def present(root: Path | None = None) -> bool:
    """Le dataset est-il déjà téléchargé dans `root` ?"""
    return ((root or datasets_root()) / FOLDER).exists()


def download(root: Path | None = None, force: bool = False) -> None:
    from datasets import concatenate_datasets, load_dataset

    root = root or datasets_root()
    cache = root / ".hf_cache"
    if force:
        reset(root / FOLDER, cache)
    if (root / FOLDER).exists():
        return  # temporaire : seule la copie save_to_disk est gardée
    parts = [
        load_dataset(REPO, split=s, revision=REVISION, cache_dir=str(cache))
        for s in ("train", "test")
    ]
    concatenate_datasets(parts).save_to_disk(str(root / FOLDER))
    shutil.rmtree(cache, ignore_errors=True)


def freeze_splits(root: Path | None = None, force: bool = False) -> Path:
    """Fige le stratifié 80/10/10 de la révision épinglée, sauf si déjà fait."""

    def compute() -> splits.Computed:
        source = Source(root)
        meta = {"repo": REPO, "revision": REVISION, "seed": 0}
        return splits.stratified(source.labels()), source.ids(), meta

    return splits.freeze(NAME, compute, force)


def build(split: str, transform: Callable, root: Path | None = None) -> Subset:
    freeze_splits(root)  # crée le JSON s'il manque, sinon ne fait rien
    frozen = splits.load(NAME)
    if frozen.get("revision") != REVISION:
        raise ValueError(f"{NAME} : splits figés sur {frozen.get('revision')}, code sur {REVISION}")
    source = Source(root)
    return Subset(source, splits.indices(NAME, split, source.ids()), transform)


def pretraining_images(root: Path | None = None) -> Subset:
    """Pool de pré-entraînement (configs/data/galaxy10.yaml) : split train figé seul (80 %).
    Val (sonde en ligne) et test n'y entrent pas. Hors protocole, dont le pool est
    Imagenette."""
    freeze_splits(root)  # crée le JSON s'il manque, sinon ne fait rien
    source = Source(root)
    return Subset(source, splits.indices(NAME, "train", source.ids()), raw)
