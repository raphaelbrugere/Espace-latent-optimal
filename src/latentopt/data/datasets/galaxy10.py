"""Galaxy10 DECaLS : 17 736 images 256px, 10 classes morphologiques déséquilibrées.

Source : Hugging Face, révision épinglée, train et test HF concaténés dans cet ordre
(comme l'arène). Pas de split officiel exploitable : stratifié 80/10/10 figé, indexé
par ligne de cette concaténation.

Le téléchargement passe par le cache HF puis `save_to_disk` ; le chargement lit
uniquement ce dossier (`load_from_disk`), sans aucun accès réseau.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from latentopt.data import splits
from latentopt.data.datasets.base import Subset, require
from latentopt.data.paths import datasets_root

NAME = "galaxy10"
NUM_CLASSES = 10
REPO = "matthieulel/galaxy10_decals"
REVISION = "d1cfafdcce69ca7cd25aecce1a0b578cf07e6a30"
FOLDER = "galaxy10_decals"


class Source:
    """Train + test HF concaténés -> (image PIL RGB, label)."""

    def __init__(self, root: Path | None = None):
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

    def __getitem__(self, i: int):
        item = self.ds[i]
        return item["image"].convert("RGB"), int(item["label"])


def download(root: Path | None = None) -> None:
    from datasets import concatenate_datasets, load_dataset

    root = root or datasets_root()
    if (root / FOLDER).exists():
        return
    cache = root / ".hf_cache"  # temporaire : seule la copie save_to_disk est gardée
    parts = [
        load_dataset(REPO, split=s, revision=REVISION, cache_dir=str(cache))
        for s in ("train", "test")
    ]
    concatenate_datasets(parts).save_to_disk(str(root / FOLDER))
    shutil.rmtree(cache, ignore_errors=True)


def freeze_splits(root: Path | None = None) -> Path:
    source = Source(root)
    return splits.save(
        NAME, splits.stratified(source.labels()), source.ids(), repo=REPO, revision=REVISION, seed=0
    )


def build(split: str, transform: Callable, root: Path | None = None) -> Subset:
    frozen = splits.load(NAME)
    if frozen.get("revision") != REVISION:
        raise ValueError(f"{NAME} : splits figés sur {frozen.get('revision')}, code sur {REVISION}")
    source = Source(root)
    return Subset(source, splits.indices(NAME, split, source.ids()), transform)
