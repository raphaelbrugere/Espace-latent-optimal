"""Cache de features : identique au calcul a la volee, relu sans recalcul, jamais
partage entre backbones, precisions ou traitements d'image, augmentation preservee."""

import random

import pytest
import torch

from latentopt.data import transforms as T
from latentopt.data.paths import features_root
from latentopt.evaluation import features as Fe
from latentopt.evaluation.tasks import get_task
from latentopt.models.contract import weights_hash

CPU = torch.device("cpu")


def test_cache_path_encodes_split_version_fingerprint_precision_and_keys(tmp_path):
    p = Fe.cache_path(tmp_path, "pets_seg", "train", "flip", "abc", "fp16", ["block3", "block6"])
    assert p == tmp_path / "pets_seg" / "train_flip__abc__fp16__block3+block6.pt"
    name = Fe.cache_path(tmp_path, "t", "val", "", "abc", "fp32", ["embed"]).name
    assert name == "val__abc__fp32__embed.pt"


def test_backbone_dir_is_created_once_and_reused(isolated_data):
    sha = "ab" * 32
    first = Fe.backbone_dir(sha, {"backbone": {"x": 1}})
    assert first.parent == features_root() and first.name.endswith("_" + sha[:16])
    assert (first / "info.json").exists()
    assert Fe.backbone_dir(sha) == first


def test_concurrent_jobs_agree_on_the_oldest_backbone_dir(isolated_data, monkeypatch):
    """Un autre job cree son dossier pendant que celui-ci cree le sien : tous deux retiennent
    le plus ancien, et le perdant supprime le sien."""
    sha = "cd" * 32
    other = features_root() / f"2000-01-01_00-00-00_{sha[:16]}"
    write = Fe._write_atomic

    def racing(path, fn):
        write(path, fn)
        other.mkdir(parents=True)  # l'autre job, juste apres la creation du notre

    monkeypatch.setattr(Fe, "_write_atomic", racing)
    assert Fe.backbone_dir(sha) == other
    assert [p.name for p in features_root().iterdir()] == [other.name]


def test_write_atomic_leaves_only_the_complete_file(tmp_path):
    target = tmp_path / "x.pt"
    Fe._write_atomic(target, lambda f: f.write_text("ok", encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == "ok"
    assert [p.name for p in tmp_path.iterdir()] == ["x.pt"]
    with pytest.raises(OSError):
        Fe._write_atomic(tmp_path / "y.pt", lambda f: (_ for _ in ()).throw(OSError("disque")))
    assert [p.name for p in tmp_path.iterdir()] == ["x.pt"]  # rien de partiel


def test_extract_matches_backbone_outputs_in_order(pyramid):
    task = get_task("synthetic_cls")
    ds = torch.utils.data.Subset(task.dataset("val"), range(10))
    entry = Fe.extract(pyramid, ds, ["embed", "c8"], 4, 0, CPU, "fp32")
    assert entry["feats"]["embed"].shape == (10, pyramid.embed_dim)
    assert entry["targets"] == [ds[i][1] for i in range(10)]
    x = torch.stack([ds[i][0] for i in range(10)])
    with torch.no_grad():
        ref = pyramid(x)
    assert torch.allclose(entry["feats"]["embed"], ref["embed"], atol=1e-6)
    assert torch.allclose(entry["feats"]["c8"], ref["c8"], atol=1e-6)


@pytest.fixture
def small_task(monkeypatch):
    """synthetic_det reduit a 12 images par split (rapide)."""
    from latentopt.data.datasets import synthetic

    monkeypatch.setattr(synthetic, "SIZES", {"train": 12, "val": 12, "test": 12})
    return get_task("synthetic_det")


def call(backbone, task, split="train", transform=None, precision="fp32", keys=("c8",)):
    store = Fe.FeatureStore(backbone, CPU, precision, batch_size=8)
    return store.entries(task, split, transform or task.train_pipeline, list(keys))


def test_feature_store_dataset_wraps_the_entries(pyramid, small_task):
    store = Fe.FeatureStore(pyramid, CPU, batch_size=8)
    assert store.sha == weights_hash(pyramid) and store.dir.name.endswith(store.sha[:16])
    ds = store.dataset(small_task, "val", small_task.eval_pipeline, ["c8"])
    assert isinstance(ds, Fe.CachedFeatures) and len(ds) == 12
    assert "fp32" in repr(store)


def test_entries_are_extracted_once_then_read(pyramid, small_task, monkeypatch):
    calls = []
    real = Fe.extract
    monkeypatch.setattr(Fe, "extract", lambda *a, **k: calls.append(1) or real(*a, **k))

    first = call(pyramid, small_task)
    assert len(calls) == 2  # train : variantes normale et retournee
    second = call(pyramid, small_task)
    assert len(calls) == 2  # relu, pas recalcule
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a["feats"]["c8"], b["feats"]["c8"])
        assert a["meta"] == b["meta"]


def test_flip_variant_holds_features_of_flipped_images(pyramid, small_task):
    normal, flipped = call(pyramid, small_task)
    assert normal["meta"]["variant"] == "" and flipped["meta"]["variant"] == "flip"
    assert not torch.equal(normal["feats"]["c8"], flipped["feats"]["c8"])
    box_n, box_f = normal["targets"][0]["box"], flipped["targets"][0]["box"]
    assert torch.allclose(box_f, torch.stack([1 - box_n[2], box_n[1], 1 - box_n[0], box_n[3]]))


def test_cache_is_separated_by_backbone_and_precision(pyramid, small_task):
    call(pyramid, small_task, split="val", transform=small_task.eval_pipeline)
    call(pyramid, small_task, split="val", transform=small_task.eval_pipeline, precision="bf16")
    other = type(pyramid)()
    call(other, small_task, split="val", transform=small_task.eval_pipeline)
    dirs = sorted(p.name for p in features_root().iterdir())
    assert len(dirs) == 2  # un dossier par backbone
    fp = small_task.eval_pipeline.fingerprint()
    files = sorted(p.name for p in features_root().glob("*/synthetic_det/*.pt"))
    assert files == [
        f"val__{fp}__bf16__c8.pt",
        f"val__{fp}__fp32__c8.pt",
        f"val__{fp}__fp32__c8.pt",
    ]


def test_changing_the_image_pipeline_never_reuses_old_features(pyramid, small_task, monkeypatch):
    """Un autre traitement des images (ici une autre taille) a une autre empreinte : ses
    features sont extraites a part, jamais relues depuis celles de l'ancien."""
    calls = []
    real = Fe.extract
    monkeypatch.setattr(Fe, "extract", lambda *a, **k: calls.append(1) or real(*a, **k))
    call(pyramid, small_task, split="val", transform=small_task.eval_pipeline)
    other = T.Pipeline(T.square(32))
    assert other.fingerprint() != small_task.eval_pipeline.fingerprint()
    call(pyramid, small_task, split="val", transform=other)
    assert len(calls) == 2
    assert len(list(features_root().glob("*/synthetic_det/val__*.pt"))) == 2


def test_incoherent_cache_is_refused(pyramid, small_task):
    call(pyramid, small_task, split="val", transform=small_task.eval_pipeline)
    path = next(features_root().glob("*/synthetic_det/val__*__fp32__c8.pt"))
    entry = torch.load(path, weights_only=False)
    entry["backbone_sha256"] = "0" * 64
    torch.save(entry, path)
    with pytest.raises(ValueError, match="incoherent"):
        call(pyramid, small_task, split="val", transform=small_task.eval_pipeline)


def test_cached_features_draw_one_version_uniformly(pyramid, small_task):
    entries = call(pyramid, small_task)
    single = Fe.CachedFeatures(entries[:1])
    assert len(single) == 12
    assert torch.equal(single[3][0]["c8"], entries[0]["feats"]["c8"][3])  # aucun tirage
    random.seed(0)
    both = Fe.CachedFeatures(entries)
    flips = sum(torch.equal(both[0][0]["c8"], entries[1]["feats"]["c8"][0]) for _ in range(400))
    assert 140 < flips < 260


def test_collate_stacks_features_and_optionally_targets():
    batch = [({"embed": torch.zeros(4)}, 1), ({"embed": torch.ones(4)}, 0)]
    feats, y = Fe.collate(native_targets=False)(batch)
    assert feats["embed"].shape == (2, 4) and torch.equal(y, torch.tensor([1, 0]))
    masks = [({"embed": torch.zeros(4)}, torch.zeros(3, 5)), ({"embed": torch.ones(4)},
             torch.zeros(6, 2))]  # fmt: skip
    _, ys = Fe.collate(native_targets=True)(masks)
    assert isinstance(ys, list) and [m.shape for m in ys] == [(3, 5), (6, 2)]
