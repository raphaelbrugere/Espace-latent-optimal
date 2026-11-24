"""Liste des datasets du projet. Ajouter un dataset : ecrire sa classe (base.DatasetSpec ou
base.DownloadableDataset) et son instance `DATASET` dans son module, puis l'ajouter a `ALL`.

Separe de `__init__.py` : importer `datasets.base` (resolvers de config, a chaque commande `lo`)
ne charge ainsi ni torchvision ni les modules des datasets.
"""

from latentopt.data.datasets import flowers102, galaxy10, imagenette, oxford_pets, synthetic
from latentopt.data.datasets.base import DatasetSpec, DownloadableDataset

ALL: tuple[DatasetSpec, ...] = (
    imagenette.DATASET,
    flowers102.DATASET,
    galaxy10.DATASET,
    oxford_pets.DATASET,
    synthetic.DATASET,
)
BY_NAME = {d.NAME: d for d in ALL}
# ce que `lo data prepare` propose : synthetic, genere a la volee, n'y est pas
DOWNLOADABLE = {d.NAME: d for d in ALL if isinstance(d, DownloadableDataset)}
