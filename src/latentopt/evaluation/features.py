"""Cache des features d'un backbone gelé, pour entraîner et évaluer des têtes sans
recalculer le backbone à chaque epoch.

    data/features/<date de création>_<sha backbone[:16]>/
    ├── info.json                                    backbone (config, origine), date
    └── <tâche>/<split>[_flip]__<précision>__<clés>.pt

- Dossier : la date (AAAA-MM-JJ_HH-MM-SS) le rend lisible et triable ; l'empreinte des
  poids permet de le retrouver (`*_<sha>`) : un backbone relit toujours ses propres
  features, jamais celles d'un autre.

- Clés : seules les entrées lues par la tête (`head.in_keys`) sont stockées : `embed`
  pèse ~2 Ko par image, les 4 cartes d'un ViT-S/8 en 128px ~0,8 Mo (fp16).
- Augmentation : le retournement horizontal de la baseline (seg, det) ne commute pas avec
  le backbone. Les deux variantes (jamais / toujours retournée) sont donc extraites, et
  `CachedFeatures` tire à pile ou face à chaque passage, avec la probabilité du transform.
  Même distribution d'augmentation que sans cache ; tirages différents (graine).
- Précision : features stockées dans le dtype produit par le backbone (autocast), donc
  sans perte par rapport à un calcul à la volée. Elle fait partie du nom de fichier : des
  features fp32 ne sont jamais relues pour un run fp16.
- Les transforms d'éval et la résolution font partie du protocole et ne sont pas dans la
  clé : les modifier impose d'incrémenter FORMAT.
"""

import copy
import json
import logging
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, default_collate

from latentopt.data.paths import features_root

log = logging.getLogger(__name__)
FORMAT = 2  # 2 : précision dans le nom, dossier daté
AUTOCAST = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def variants(transform) -> dict[str, object]:
    """Transform à retournement aléatoire -> ses deux variantes déterministes."""
    if not getattr(transform, "hflip", 0):
        return {"": transform}
    keep, flip = copy.copy(transform), copy.copy(transform)
    keep.hflip, flip.hflip = 0.0, 1.0
    return {"": keep, "flip": flip}


def backbone_dir(sha: str, info: dict | None = None) -> Path:
    """Dossier du backbone : le dossier existant `*_<sha16>`, sinon un nouveau, daté."""
    root = features_root()
    existing = sorted(root.glob(f"*_{sha[:16]}"))
    if existing:
        return existing[-1]
    path = root / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{sha[:16]}"
    path.mkdir(parents=True)
    created = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {"backbone_sha256": sha, "created": created, **(info or {})}
    (path / "info.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def cache_path(
    base: Path, task: str, split: str, variant: str, precision: str, keys: list[str]
) -> Path:
    name = f"{split}{'_' + variant if variant else ''}__{precision}__{'+'.join(keys)}.pt"
    return base / task / name


def _images_and_targets(batch):
    return torch.stack([x for x, _ in batch]), [y for _, y in batch]


@torch.no_grad()
def extract(backbone: nn.Module, dataset, keys, batch_size, num_workers, device, precision) -> dict:
    """Features `keys` de tout le dataset (ordre conservé) et cibles, sous autocast."""
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


def load_or_extract(
    backbone,
    sha,
    task,
    split,
    transform,
    keys,
    batch_size,
    num_workers,
    device,
    precision,
    info: dict | None = None,
) -> list[dict]:
    """Une entrée par variante du transform ; extraite une fois, relue ensuite.
    `info` (config et origine du backbone) va dans info.json à la création du dossier."""
    base = backbone_dir(sha, info)
    entries = []
    for variant, t in variants(transform).items():
        path = cache_path(base, task.name, split, variant, precision, keys)
        if path.exists():
            entry = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            coherent = entry["backbone_sha256"] == sha and entry["format"] == FORMAT
            if not coherent or entry["meta"]["precision"] != precision:
                raise ValueError(f"{path} : cache incohérent, le supprimer pour le recalculer")
        else:
            label = f"{task.name}/{split}" + (f" ({variant})" if variant else "")
            log.info("extraction des features %s -> %s", label, path)
            dataset = task.build(split, t)
            entry = extract(backbone, dataset, keys, batch_size, num_workers, device, precision)
            entry["meta"] = {
                "task": task.name,
                "split": split,
                "variant": variant,
                "keys": list(keys),
                "precision": precision,
            }
            entry["format"], entry["backbone_sha256"] = FORMAT, sha
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(entry, path)
        entries.append(entry)
    return entries


class CachedFeatures(Dataset):
    """(features d'une image, cible) depuis le cache ; variante retournée avec la
    probabilité `hflip` du transform d'origine (aucun tirage s'il n'y a qu'une variante)."""

    def __init__(self, entries: list[dict], hflip: float = 0.0):
        self.entries, self.hflip = entries, hflip
        self.keys = list(entries[0]["feats"])

    def __len__(self) -> int:
        return len(self.entries[0]["targets"])

    def __getitem__(self, i: int):
        flip = len(self.entries) > 1 and random.random() < self.hflip
        entry = self.entries[1 if flip else 0]
        return {k: entry["feats"][k][i] for k in self.keys}, entry["targets"][i]


def collate(native_targets: bool):
    """Features empilées ; cibles empilées, ou laissées en liste (masques à résolution native)."""

    def fn(batch):
        feats = default_collate([f for f, _ in batch])
        targets = [t for _, t in batch]
        return feats, (targets if native_targets else default_collate(targets))

    return fn
