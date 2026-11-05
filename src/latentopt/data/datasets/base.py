from collections.abc import Callable, Sequence
from pathlib import Path

from torch.utils.data import Dataset


class Subset(Dataset):
    """Vue `indices` d'une source (image PIL, cible brute), avec transform conjoint."""

    def __init__(self, source, indices: Sequence[int], transform: Callable):
        self.source, self.indices, self.transform = source, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, target = self.source[self.indices[i]]
        return self.transform(img, target)


def require(path: Path, name: str) -> None:
    """Échoue tout de suite, avec la commande à lancer, si les données manquent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{name} absent de {path} : lance `python tools/prepare_data.py {name}`"
        )
