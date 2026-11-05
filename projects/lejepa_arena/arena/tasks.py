"""Small-league tasks: data, splits, official loss and metric.

    task = get_task("pets_seg")
    train = task.dataset("train", transform=my_augmentation)
    val   = task.eval_dataset("val")
    loader = DataLoader(val, collate_fn=task.collate_fn, ...)

Splits: "train" and "val" never overlap with "test". "test" is the leaderboard
split: never train or tune on it.

Task             | data              | input | metric                  | splits
imagenette_cls   | Imagenette 160px  | 128px | top-1                   | 95/5 of train, test = official val
flowers102_cls   | Oxford Flowers-102| 128px | top-1                   | official train / val / test
galaxy10_cls     | Galaxy10 DECaLS   | 128px | balanced accuracy       | fixed stratified 80/10/10
pets_seg         | Oxford-IIIT Pet   | 128px | mIoU at NATIVE res      | 95/5 of trainval, test = official test
pets_det         | Pet head boxes    | 128px | localization accuracy   | fixed stratified 80/10/10 of trainval

Set ARENA_SYNTHETIC=1 to replace every dataset with a tiny generated one
(no download, runs on CPU).
"""
import os
import pathlib
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import box_iou, generalized_box_iou_loss
from torchvision.transforms.v2 import functional as TF

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SYNTHETIC = os.environ.get("ARENA_SYNTHETIC", "0") == "1"
IMG_SIZE = 64 if SYNTHETIC else 128
DATA_ROOT = os.environ.get("ARENA_DATA", "./data")


def normalize(img):
    return TF.normalize(TF.to_dtype(TF.to_image(img), torch.float32, scale=True), MEAN, STD)


def to_device(y, device):
    """Move targets to the device, whatever their structure (tensor, list, dict)."""
    if torch.is_tensor(y):
        return y.to(device, non_blocking=True)
    if isinstance(y, dict):
        return {k: to_device(v, device) for k, v in y.items()}
    if isinstance(y, (list, tuple)):
        return [to_device(v, device) for v in y]
    return y


def to_float(out):
    """Cast head outputs to fp32 (after autocast), whatever their structure."""
    if torch.is_tensor(out):
        return out.float()
    if isinstance(out, dict):
        return {k: v.float() for k, v in out.items()}
    return out


def holdout(n, split, frac=0.05, seed=0):
    perm = np.random.RandomState(seed).permutation(n)
    k = max(1, int(n * frac))
    return perm[k:].tolist() if split == "train" else perm[:k].tolist()


def stratified(labels, split, fracs=(0.8, 0.1, 0.1), seed=0):
    """Fixed per-class split into train / val / test."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        a, b = int(len(idx) * fracs[0]), int(len(idx) * (fracs[0] + fracs[1]))
        out["train"] += idx[:a].tolist()
        out["val"] += idx[a:b].tolist()
        out["test"] += idx[b:].tolist()
    return sorted(out[split])


class Split(torch.utils.data.Dataset):
    def __init__(self, raw, indices, transform):
        self.raw, self.indices, self.transform = raw, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, target = self.raw[self.indices[i]]
        return self.transform(img, target)


# Raw sources: each returns (PIL RGB image, target) ------------------------------------
def pick_columns(features, repo=""):
    """Find the image column and the label column of an HF dataset. The label can be a
    ClassLabel or a plain integer column with a usual name (Galaxy10 uses uint8 'label')."""
    kind = {k: type(v).__name__ for k, v in features.items()}
    images = [k for k, t in kind.items() if t == "Image"]
    named = [k for k in ("label", "labels", "class") if k in features]
    class_labels = [k for k, t in kind.items() if t == "ClassLabel"]
    labels = named or class_labels
    if not images or not labels:
        raise KeyError(f"{repo}: no image or label column found in {kind}")
    return images[0], labels[0]


class HFImages:
    """Any HF image-classification dataset; splits are concatenated if `splits` is a list."""

    def __init__(self, repo, splits, config=None):
        from datasets import concatenate_datasets, load_dataset
        parts = [load_dataset(repo, config, split=s) for s in splits]
        self.ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        self.img_key, self.label_key = pick_columns(self.ds.features, repo)

    def labels(self):
        return self.ds[self.label_key]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return item[self.img_key].convert("RGB"), int(item[self.label_key])


class TorchvisionImages:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, target = self.ds[i]
        return img.convert("RGB"), target


def imagenette(split):
    """Imagenette 160px (official fast.ai archive) through torchvision, downloaded on first use.
    torchvision refuses download=True once the folder exists, so only download when missing."""
    from torchvision.datasets import Imagenette
    try:
        ds = Imagenette(DATA_ROOT, split=split, size="160px", download=False)
    except RuntimeError:
        ds = Imagenette(DATA_ROOT, split=split, size="160px", download=True)
    return TorchvisionImages(ds)


def synthetic_images(split, k):
    n = {"train": 1000, "val": 150, "test": 300, "all": 1500}[split]
    return SyntheticShapes(n, k=k, seed={"train": 0, "val": 2, "test": 1, "all": 3}[split])


# Classification ---------------------------------------------------------------------
class Top1:
    def __init__(self):
        self.correct = self.total = 0

    def update(self, logits, y):
        self.correct += (logits.argmax(1) == y).sum().item()
        self.total += y.numel()

    def compute(self):
        return {"top1": self.correct / max(1, self.total)}


class BalancedAccuracy:
    """Mean per-class recall: robust to class imbalance."""

    def __init__(self, k):
        self.hit, self.count = torch.zeros(k), torch.zeros(k)

    def update(self, logits, y):
        pred = logits.argmax(1).cpu()
        y = y.cpu()
        self.count += torch.bincount(y, minlength=len(self.count)).float()
        self.hit += torch.bincount(y[pred == y], minlength=len(self.hit)).float()

    def compute(self):
        seen = self.count > 0
        acc = (self.hit[seen] / self.count[seen]).mean().item()
        return {"balanced_acc": acc, "top1": (self.hit.sum() / self.count.sum()).item()}


class ClassificationTask:
    """Output: logits (B, num_classes)."""
    img_size = IMG_SIZE
    head_max_params = 2_000_000
    collate_fn = None

    def eval_transform(self, img, label):
        s = self.img_size
        img = TF.center_crop(TF.resize(img, s), s)   # as in the LeJEPA minimal example
        return normalize(img), label

    def raw(self, split):
        raise NotImplementedError

    def dataset(self, split, transform=None):
        raw, idx = self.raw(split)
        return Split(raw, idx, transform or self.eval_transform)

    def eval_dataset(self, split):
        return self.dataset(split)

    def loss(self, logits, y):
        return F.cross_entropy(logits, y)

    def metric(self):
        return Top1()


class ImagenetteCls(ClassificationTask):
    name, num_classes, primary_metric = "imagenette_cls", 10, "top1"

    def raw(self, split):
        if SYNTHETIC:
            raw = synthetic_images("test" if split == "test" else "train", 10)
        else:
            raw = imagenette("val" if split == "test" else "train")
        idx = list(range(len(raw))) if split == "test" else holdout(len(raw), split)
        return raw, idx


class Flowers102Cls(ClassificationTask):
    name, num_classes, primary_metric = "flowers102_cls", 102, "top1"

    def raw(self, split):
        if SYNTHETIC:
            raw = synthetic_images(split, 102)
        else:
            from torchvision.datasets import Flowers102
            raw = TorchvisionImages(Flowers102(DATA_ROOT, split=split, download=True))
        return raw, list(range(len(raw)))


class Galaxy10Cls(ClassificationTask):
    """Galaxy10 DECaLS (17,736 images, 256px, 10 imbalanced classes), downscaled to
    the league resolution. No official split: a fixed stratified 80/10/10 is used."""
    name, num_classes, primary_metric = "galaxy10_cls", 10, "balanced_acc"

    def eval_transform(self, img, label):
        return normalize(TF.resize(img, [self.img_size, self.img_size])), label  # centered, square

    def raw(self, split):
        if SYNTHETIC:
            raw = synthetic_images("all", 10)
            labels = [i % 10 for i in range(len(raw))]
        else:
            raw = HFImages("matthieulel/galaxy10_decals", ["train", "test"])
            labels = raw.labels()
        return raw, stratified(labels, split)

    def metric(self):
        return BalancedAccuracy(self.num_classes)


# Segmentation ---------------------------------------------------------------------
class PetsRaw:
    def __init__(self, official_split):
        if SYNTHETIC:
            self.ds = SyntheticSeg(600 if official_split == "trainval" else 200, seed=official_split == "trainval")
            self.offset = 0
        else:
            from torchvision.datasets import OxfordIIITPet
            self.ds = OxfordIIITPet(DATA_ROOT, split=official_split, target_types="segmentation", download=True)
            self.offset = 1  # raw trimap {1, 2, 3} -> {0, 1, 2}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, mask = self.ds[i]
        return img.convert("RGB"), mask


def collate_native(batch):
    """Images share the input size; masks keep their native size, so they stay a list."""
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class PetsSeg:
    """Output: logits (B, 3, h, w) at any resolution. The input is resized to a fixed
    square (reversible anisotropic scaling), and the metric is computed at the NATIVE
    mask resolution after upsampling the logits, so resolution loss is penalized."""
    name, num_classes, primary_metric = "pets_seg", 3, "miou"
    img_size = IMG_SIZE
    head_max_params = 3_000_000
    collate_fn = staticmethod(collate_native)

    def __init__(self):
        self.offset = 0 if SYNTHETIC else 1

    def _mask(self, mask):
        return torch.from_numpy(np.array(mask, dtype=np.int64)) - self.offset

    def train_transform(self, img, mask, hflip=False):
        s = [self.img_size, self.img_size]
        img, mask = TF.resize(img, s), TF.resize(mask, s, interpolation=TF.InterpolationMode.NEAREST)
        if hflip:
            img, mask = TF.hflip(img), TF.hflip(mask)
        return normalize(img), self._mask(mask)

    def flip_transform(self, img, mask):
        """A simple training augmentation (used by the baseline)."""
        return self.train_transform(img, mask, hflip=random.random() < 0.5)

    def eval_transform(self, img, mask):
        return normalize(TF.resize(img, [self.img_size, self.img_size])), self._mask(mask)

    def _raw(self, split):
        raw = PetsRaw("test" if split == "test" else "trainval")
        return raw, list(range(len(raw))) if split == "test" else holdout(len(raw), split)

    def dataset(self, split, transform=None):
        """Training data: masks resized with the image (fixed size, default collate)."""
        raw, idx = self._raw(split)
        return Split(raw, idx, transform or self.train_transform)

    def eval_dataset(self, split):
        """Scoring data: masks at native resolution (use collate_fn)."""
        raw, idx = self._raw(split)
        return Split(raw, idx, self.eval_transform)

    def loss(self, logits, y):
        return F.cross_entropy(upsample(logits, y.shape[-2:]), y)

    def metric(self):
        return MeanIoU(self.num_classes)


def upsample(logits, size):
    if logits.shape[-2:] != tuple(size):
        logits = F.interpolate(logits.float(), size=tuple(size), mode="bilinear", align_corners=False)
    return logits


class MeanIoU:
    def __init__(self, k):
        self.k, self.inter, self.union = k, torch.zeros(k), torch.zeros(k)

    def update(self, logits, targets):
        """targets: a (B, H, W) tensor, or a list of (H_i, W_i) native-size masks."""
        if torch.is_tensor(targets):
            targets = list(targets)
        for lg, y in zip(logits, targets):
            y = y.to(lg.device)
            self.update_pred(upsample(lg[None], y.shape)[0].argmax(0), y)

    def update_pred(self, pred, y):
        for c in range(self.k):
            p, t = pred == c, y == c
            self.inter[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()

    def compute(self):
        return {"miou": (self.inter / self.union.clamp(min=1)).mean().item()}


# Detection: Oxford-IIIT Pet head boxes ---------------------------------------------------
class PetsDetRaw:
    """(image, {"box": xyxy normalized to [0, 1], "label": 0 cat / 1 dog}), one head per image.
    Head boxes exist for trainval images only (the xmls/ annotations)."""

    def __init__(self):
        if SYNTHETIC:
            self.items, self.synthetic = None, SyntheticDet(900)
            return
        from torchvision.datasets import OxfordIIITPet
        OxfordIIITPet(DATA_ROOT, split="trainval", download=True)  # images + annotations (incl. xmls/)
        base = pathlib.Path(DATA_ROOT) / "oxford-iiit-pet"
        self.images, xmls = base / "images", base / "annotations" / "xmls"
        self.items = []
        for line in (base / "annotations" / "trainval.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, species, _ = line.split()
            if (xmls / f"{name}.xml").exists():
                self.items.append((name, int(species) - 1, self._box(xmls / f"{name}.xml")))

    @staticmethod
    def _box(path):
        root = ET.parse(path).getroot()
        w, h = float(root.find("size/width").text), float(root.find("size/height").text)
        b = root.find("object/bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        return [x1 / w, y1 / h, x2 / w, y2 / h]

    def labels(self):
        return self.synthetic.labels() if self.items is None else [lbl for _, lbl, _ in self.items]

    def __len__(self):
        return len(self.synthetic) if self.items is None else len(self.items)

    def __getitem__(self, i):
        if self.items is None:
            return self.synthetic[i]
        name, label, box = self.items[i]
        img = Image.open(self.images / f"{name}.jpg").convert("RGB")
        return img, {"box": torch.tensor(box).clamp(0, 1), "label": label}


class LocalizationAccuracy:
    """A prediction is correct if the class is right AND IoU >= threshold.
    Primary: accuracy averaged over IoU thresholds 0.5, 0.55, ..., 0.95 (COCO style),
    which rewards precise boxes. IoU between axis-aligned boxes is unchanged by
    stretching the image along its axes, so normalized coordinates give native-res IoU."""
    thresholds = torch.arange(0.5, 0.96, 0.05)

    def __init__(self):
        self.ious, self.correct_cls = [], []

    def update(self, out, y):
        iou = box_iou(out["box"].float(), y["box"].float()).diagonal()
        self.ious.append(iou.cpu())
        self.correct_cls.append((out["logits"].argmax(1) == y["label"]).cpu())

    def compute(self):
        iou, cls = torch.cat(self.ious), torch.cat(self.correct_cls)
        hits = torch.stack([(iou >= t) & cls for t in self.thresholds]).float().mean(1)
        return {"loc_acc": hits.mean().item(), "acc_iou50": hits[0].item(), "acc_iou75": hits[5].item(),
                "mean_iou": iou.mean().item(), "cls_acc": cls.float().mean().item()}


class PetsDet:
    """Output: {"box": (B, 4) xyxy normalized to [0, 1], "logits": (B, 2) cat/dog}.
    The input is resized to a fixed square; boxes are normalized, so they follow."""
    name, num_classes, primary_metric = "pets_det", 2, "loc_acc"
    img_size = IMG_SIZE
    head_max_params = 3_000_000
    collate_fn = None

    def eval_transform(self, img, target, hflip=False):
        img = normalize(TF.resize(img, [self.img_size, self.img_size]))
        box = target["box"].clone()
        if hflip:
            img, box = TF.hflip(img), torch.stack([1 - box[2], box[1], 1 - box[0], box[3]])
        return img, {"box": box, "label": torch.tensor(target["label"])}

    def flip_transform(self, img, target):
        """A simple training augmentation (used by the baseline)."""
        return self.eval_transform(img, target, hflip=random.random() < 0.5)

    def dataset(self, split, transform=None):
        raw = PetsDetRaw()
        return Split(raw, stratified(raw.labels(), split), transform or self.eval_transform)

    def eval_dataset(self, split):
        return self.dataset(split)

    def loss(self, out, y):
        """Baseline loss: cross-entropy + L1 + GIoU. Students may use any loss they like."""
        box = out["box"].float()
        return (F.cross_entropy(out["logits"].float(), y["label"]) + 5 * F.l1_loss(box, y["box"])
                + 2 * generalized_box_iou_loss(box, y["box"], reduction="mean"))

    def metric(self):
        return LocalizationAccuracy()


TASKS = {t.name: t for t in (ImagenetteCls, Flowers102Cls, Galaxy10Cls, PetsSeg, PetsDet)}


def get_task(name):
    if name not in TASKS:
        raise KeyError(f"unknown task '{name}', available: {sorted(TASKS)}")
    return TASKS[name]()


def pretraining_images():
    """The official small-league pretraining pool: Imagenette train images (labels are
    returned only for an online probe on DETACHED embeddings)."""
    return synthetic_images("train", 10) if SYNTHETIC else imagenette("train")


# Synthetic stand-ins (ARENA_SYNTHETIC=1) ----------------------------------------------
class SyntheticShapes:
    def __init__(self, n, k=10, size=80, seed=0):
        self.n, self.k, self.size, self.seed = n, k, size, int(seed)
        self.palette = np.random.RandomState(1234).randint(30, 225, size=(k, 3))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 1_000_003 + i)
        label = i % self.k
        color = tuple(int(c) for c in np.clip(self.palette[label] + rng.randn(3) * 15, 0, 255))
        img = Image.new("RGB", (self.size, self.size), color)
        draw = ImageDraw.Draw(img)
        for _ in range(1 + label % 4):
            x, y = rng.randint(0, self.size - 12, size=2)
            w, h = rng.randint(8, self.size // 2, size=2)
            draw.ellipse([x, y, x + w, y + h], fill=tuple(int(c) for c in rng.randint(0, 255, 3)))
        return img, label


class SyntheticSeg:
    """Non-square on purpose, to exercise the native-resolution metric."""
    palette = [(40, 40, 40), (220, 60, 60), (60, 200, 90)]

    def __init__(self, n, size=(96, 72), seed=0):
        self.n, self.size, self.seed = n, size, int(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 1_000_003 + i)
        w0, h0 = self.size
        arr = np.clip(rng.randn(h0, w0, 3) * 20 + 128, 0, 255).astype(np.uint8)
        img, mask = Image.fromarray(arr), Image.new("L", (w0, h0), 0)
        for _ in range(3):
            c = int(rng.randint(1, 3))
            x, y = rng.randint(0, w0 - 16), rng.randint(0, h0 - 16)
            w, h = rng.randint(12, w0 // 2), rng.randint(12, h0 // 2)
            ImageDraw.Draw(img).ellipse([x, y, x + w, y + h], fill=self.palette[c])
            ImageDraw.Draw(mask).ellipse([x, y, x + w, y + h], fill=c)
        return img, mask


class SyntheticDet:
    """One colored ellipse ("head") per non-square image; class = color family."""
    palette = [(220, 80, 60), (60, 120, 220)]

    def __init__(self, n, size=(96, 72)):
        self.n, self.size = n, size

    def labels(self):
        return [i % 2 for i in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(7_000_003 + i)
        w0, h0 = self.size
        arr = np.clip(rng.randn(h0, w0, 3) * 20 + 128, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        label = i % 2
        w, h = rng.randint(14, w0 // 2), rng.randint(14, h0 // 2)
        x, y = rng.randint(0, w0 - w), rng.randint(0, h0 - h)
        color = tuple(int(c) for c in np.clip(np.array(self.palette[label]) + rng.randn(3) * 20, 0, 255))
        ImageDraw.Draw(img).ellipse([x, y, x + w, y + h], fill=color)
        return img, {"box": torch.tensor([x / w0, y / h0, (x + w) / w0, (y + h) / h0]), "label": label}
