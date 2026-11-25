"""Galaxy10 DECaLS : 17 736 images 256px, 10 classes morphologiques desequilibrees.

Source : Hugging Face, revision epinglee, train et test HF concatenes dans cet ordre.
Pas de split officiel exploitable : stratifie 80/10/10 fige, indexe
par ligne de cette concatenation.

Le telechargement passe par le cache HF puis `save_to_disk` ; le chargement lit
uniquement ce dossier (`load_from_disk`), sans aucun acces reseau.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from latentopt.data import splits
from latentopt.data.datasets.base import ClsSource, DownloadableDataset, Subset, require, reset
from latentopt.data.paths import datasets_root

REPO = "matthieulel/galaxy10_decals"
# CRITIQUE : commit HF des donnees telechargees. NE PAS LE MODIFIER a la legere : les splits
# figes designent les images par numero de ligne, valable pour cette revision seulement. Le
# changer impose `lo data prepare galaxy10 --force-download --force-splits` (les donnees sur
# disque ne gardent pas trace de leur revision : rien ne detecte un oubli), et rend les
# scores incomparables aux anciens.
REVISION = "d1cfafdcce69ca7cd25aecce1a0b578cf07e6a30"
FOLDER = "galaxy10_decals"


class Galaxy10Source(ClsSource):
    """Train + test HF concatenes -> (image PIL RGB, label)."""

    def __init__(self, root: Path | None = None) -> None:
        from datasets import load_from_disk

        root = root or datasets_root()
        require(root / FOLDER, Galaxy10.NAME)
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


class Galaxy10Pool(ClsSource):
    """Split train fige seul (80 %) -> (image PIL RGB, label) : val et test n'y entrent pas."""

    def __init__(self, root: Path | None = None) -> None:
        DATASET.check_splits(root)
        self.source, self.split = Galaxy10Source(root), "train"
        self.rows = splits.indices(Galaxy10.NAME, "train", self.source.ids())

    def ids(self) -> list[str]:
        return [str(row) for row in self.rows]  # meme convention que Galaxy10Source

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        return self.source[self.rows[i]]


class Galaxy10(DownloadableDataset):
    """Classification 10 classes desequilibrees (`cls`) ; pool : train fige."""

    NAME = "galaxy10"
    # CRITIQUE : fige, voir DatasetSpec.ABBREV. Le " d " distingue DECaLS (256px) de
    # Galaxy10 SDSS (69px), un autre dataset.
    ABBREV = "gal10d"

    def present(self, root: Path | None = None) -> bool:
        return ((root or datasets_root()) / FOLDER).exists()

    def download(self, root: Path | None = None, force: bool = False) -> None:
        from datasets import concatenate_datasets, load_dataset

        root = root or datasets_root()
        cache = root / ".hf_cache"
        if force:
            reset(root / FOLDER, cache)
        if (root / FOLDER).exists():
            return  # temporaire : seule la copie save_to_disk est gardee
        parts = [
            load_dataset(REPO, split=s, revision=REVISION, cache_dir=str(cache))
            for s in ("train", "test")
        ]
        concatenate_datasets(parts).save_to_disk(str(root / FOLDER))
        shutil.rmtree(cache, ignore_errors=True)

    def freeze_splits(self, root: Path | None = None, force: bool = False) -> list[Path]:
        """Fige le stratifie 80/10/10 de la revision epinglee, sauf si deja fait."""

        def compute() -> splits.Computed:
            source = Galaxy10Source(root)
            meta = {"repo": REPO, "revision": REVISION, "seed": 0}
            indices = splits.stratified(source.labels())
            return splits.Computed(indices=indices, ids=source.ids(), meta=meta)

        return [splits.freeze(self.NAME, compute, force)]

    def check_splits(self, root: Path | None = None) -> None:
        """Fige les splits s'il le faut, puis verifie qu'ils viennent de la revision du code.
        Sinon leurs numeros de ligne designent d'autres images : val ou test pourraient
        entrer dans l'entrainement sans que rien ne le signale. Appelee avant toute lecture,
        par la tache comme par le pool de pre-entrainement."""
        self.freeze_splits(root)  # cree le JSON s'il manque, sinon ne fait rien
        frozen = splits.load(self.NAME).get("revision")
        if frozen != REVISION:
            raise ValueError(
                f"{self.NAME} : splits figes sur la revision {frozen}, code sur {REVISION} ; "
                f"lancer `lo data prepare {self.NAME} --force-download --force-splits`"
            )

    def tasks(self) -> dict[str, Callable]:
        return {"cls": self.build}

    def build(self, split: str, transform: Callable, root: Path | None = None) -> Subset:
        self.check_splits(root)
        source = Galaxy10Source(root)
        return Subset(source, splits.indices(self.NAME, split, source.ids()), transform)

    def pretraining_images(self, root: Path | None = None) -> Galaxy10Pool:
        """Pool de pre-entrainement (configs/data/galaxy10.yaml). Hors protocole, dont le
        pool est Imagenette."""
        return Galaxy10Pool(root)


DATASET = Galaxy10()
