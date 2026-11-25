"""Cache des features d'un backbone gele, pour entrainer et evaluer des tetes sans
recalculer le backbone a chaque epoch.

    data/features/<date de creation>_<sha backbone[:16]>/
    |-- info.json                                    backbone (config, origine), date
    `-- <tache>/<split>[_<version>]__<empreinte>__<precision>__<cles>.pt

- Dossier : la date (AAAA-MM-JJ_HH-MM-SS) le rend lisible et triable ; l'empreinte des
  poids permet de le retrouver (`*_<sha>`) : un backbone relit toujours ses propres
  features, jamais celles d'un autre.
- Jobs simultanes (les 5 taches d'un backbone en parallele, `-m` sur le DCE) : tous
  retiennent le plus ancien dossier `*_<sha>`, et chaque fichier est ecrit sous un nom
  temporaire puis renomme. Un job ne lit donc jamais un fichier a moitie ecrit ; au pire,
  deux jobs extraient la meme chose.

- Cles : seules les entrees lues par le neck (`neck.in_keys`, reprises par
  `Downstream.in_keys`) sont stockees : `embed` pese ~2 Ko par image, les 4 cartes d'un
  ViT-S/8 en 128px ~0,8 Mo (fp16).
- Augmentation : elle ne commute pas avec le backbone. Chaque version d'une augmentation
  `Enumerate` (ex. inchangee / retournee) est donc extraite une fois (`Pipeline.variants`),
  et `CachedFeatures` en tire une au hasard, uniformement, a chaque passage : meme
  distribution d'augmentation que sans cache, tirages differents.
- Precision : features stockees dans le dtype produit par le backbone (autocast), donc
  sans perte par rapport a un calcul a la volee. Elle fait partie du nom de fichier : des
  features fp32 ne sont jamais relues pour un run fp16.
- Empreinte : celle du pipeline de la version (`Pipeline.fingerprint` : preparation,
  augmentation, resolution) est dans le nom du fichier. Modifier le traitement des images
  ne relit donc jamais d'anciennes features : elles sont recalculees. FORMAT ne change
  qu'avec le format du fichier lui-meme.
"""

import json
import logging
import os
import random
import shutil
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from latentopt.data.paths import features_root
from latentopt.data.transforms import Pipeline
from latentopt.evaluation.tasks import Task
from latentopt.models.contract import weights_hash

log = logging.getLogger(__name__)
FORMAT = 3  # 2 : precision dans le nom, dossier date ; 3 : empreinte du pipeline dans le nom
AUTOCAST = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def _write_atomic(path: Path, write: Callable[[Path], Any]) -> None:
    """Ecrit via un fichier temporaire du meme dossier, puis le renomme : un lecteur voit le
    fichier complet ou rien (os.replace est atomique, y compris sur le disque reseau)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def backbone_dir(sha: str, info: dict | None = None) -> Path:
    """Dossier du backbone : le plus ancien `*_<sha16>`, sinon un nouveau, date.

    Deux jobs qui creent chacun le leur au meme moment retiennent ensuite tous deux le plus
    ancien, et le perdant supprime le sien (encore vide : seulement info.json)."""
    root = features_root()
    pattern = f"*_{sha[:16]}"
    existing = sorted(root.glob(pattern))
    if existing:
        return existing[0]
    path = root / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{sha[:16]}"
    path.mkdir(parents=True, exist_ok=True)  # meme seconde : meme dossier
    created = time.strftime("%Y-%m-%d %H:%M:%S")
    text = json.dumps({"backbone_sha256": sha, "created": created, **(info or {})}, indent=2)
    _write_atomic(path / "info.json", lambda f: f.write_text(text + "\n", encoding="utf-8"))
    oldest = sorted(root.glob(pattern))[0]
    if oldest != path:
        shutil.rmtree(path, ignore_errors=True)
    return oldest


def cache_path(
    base: Path,
    task: str,
    split: str,
    variant: str,
    fingerprint: str,
    precision: str,
    keys: list[str],
) -> Path:
    head = f"{split}_{variant}" if variant else split
    return base / task / f"{head}__{fingerprint}__{precision}__{'+'.join(keys)}.pt"


def _images_and_targets(batch: list[tuple[Tensor, Any]]) -> tuple[Tensor, list[Any]]:
    return torch.stack([x for x, _ in batch]), [y for _, y in batch]


@torch.no_grad()
def extract(
    backbone: nn.Module,
    dataset: Dataset,
    keys: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """Features `keys` de tout le dataset (ordre conserve) et cibles, sous autocast."""
    backbone = backbone.to(device).eval()
    dtype = AUTOCAST[precision]
    feats, targets = {k: [] for k in keys}, []
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_images_and_targets
    )
    for x, y in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            out = backbone(x.to(device))
        for k in keys:
            feats[k].append(out[k].cpu())
        targets.extend(y)
    return {"feats": {k: torch.cat(v) for k, v in feats.items()}, "targets": targets}


class CachedFeatures(Dataset):
    """(features d'une image, cible) depuis le cache ; une version tiree uniformement par
    passage quand il y en a plusieurs (augmentation), aucun tirage sinon."""

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self.keys = list(entries[0]["feats"])

    def __len__(self) -> int:
        return len(self.entries[0]["targets"])

    def __repr__(self) -> str:
        meta = self.entries[0].get("meta", {})
        where = f"{meta['task']}/{meta['split']}, " if meta else ""
        versions = [e.get("meta", {}).get("variant", "?") or "base" for e in self.entries]
        precision = f", {meta['precision']}" if meta else ""
        return (
            f"CachedFeatures({where}{len(self)} images, versions={versions}, "
            f"cles={self.keys}{precision})"
        )

    def __getitem__(self, i: int) -> tuple[dict[str, Tensor], Any]:
        entry = random.choice(self.entries) if len(self.entries) > 1 else self.entries[0]
        return {k: entry["feats"][k][i] for k in self.keys}, entry["targets"][i]


class FeatureStore:
    """Features d'un backbone gele, extraites une fois puis relues : dossier du backbone
    (`backbone_dir`), device, precision et DataLoader d'extraction sont fixes une fois ;
    `dataset(tache, split, pipeline, cles)` renvoie un `CachedFeatures`."""

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        precision: str = "fp32",
        num_workers: int = 0,
        batch_size: int = 256,
        info: dict | None = None,
    ) -> None:
        """`info` (config et origine du backbone) va dans info.json a la creation du dossier."""
        self.backbone, self.device, self.precision = backbone, device, precision
        self.num_workers, self.batch_size = num_workers, batch_size
        self.sha = weights_hash(backbone)
        self.dir = backbone_dir(self.sha, info)

    def __repr__(self) -> str:
        return f"FeatureStore({self.dir.name}, {self.precision}, {self.device})"

    def dataset(
        self, task: Task, split: str, pipeline: Pipeline, keys: list[str]
    ) -> "CachedFeatures":
        return CachedFeatures(self.entries(task, split, pipeline, keys))

    def entries(
        self, task: Task, split: str, pipeline: Pipeline, keys: list[str]
    ) -> list[dict[str, Any]]:
        """Une entree par version du pipeline ; extraite une fois, relue ensuite."""
        keys = list(keys)
        return [
            self._load_or_extract(task, split, variant, t, keys)
            for variant, t in pipeline.variants().items()
        ]

    def _load_or_extract(
        self, task: Task, split: str, variant: str, pipeline: Pipeline, keys: list[str]
    ) -> dict[str, Any]:
        fp = pipeline.fingerprint()
        path = cache_path(self.dir, task.name, split, variant, fp, self.precision, keys)
        if path.exists():
            entry = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            coherent = entry["backbone_sha256"] == self.sha and entry["format"] == FORMAT
            if not coherent or entry["meta"]["precision"] != self.precision:
                raise ValueError(f"{path} : cache incoherent, le supprimer pour le recalculer")
            return entry
        label = f"{task.name}/{split}" + (f" ({variant})" if variant else "")
        log.info("extraction des features %s -> %s", label, path)
        entry = extract(self.backbone, task.build(split, pipeline), keys, self.batch_size,
                        self.num_workers, self.device, self.precision)  # fmt: skip
        entry["meta"] = {
            "task": task.name,
            "split": split,
            "variant": variant,
            "pipeline": repr(pipeline),
            "keys": keys,
            "precision": self.precision,
        }
        entry["format"], entry["backbone_sha256"] = FORMAT, self.sha
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, partial(torch.save, entry))
        return entry


def collate(native_targets: bool) -> Callable[[list], tuple[dict[str, Tensor], Any]]:
    """Features empilees ; cibles empilees, ou laissees en liste (masques a resolution native)."""

    def fn(batch: list) -> tuple[dict[str, Tensor], Any]:
        feats = default_collate([f for f, _ in batch])
        targets = [t for _, t in batch]
        return feats, (targets if native_targets else default_collate(targets))

    return fn
