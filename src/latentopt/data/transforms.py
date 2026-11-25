"""Traitement des images des taches aval (evaluation et entrainement des tetes).

Un `Pipeline` applique a (image, cible), dans cet ordre :
    augment   aleatoire, entrainement seulement (ex. `hflip()`) ; None a l'evaluation
    prepare   deterministe, toujours : taille et conversion (`center_crop`, `square`)
puis convertit la cible au format du modele. Les etapes sont des transforms torchvision v2 :
la cible (masque `tv_tensors.Mask`, boites `tv_tensors.BoundingBoxes`) suit l'image toute
seule, quelles que soient les augmentations combinees.

`prepare` fait partie du protocole : la modifier change les scores. Le cache de features
calcule toutes les versions d'une augmentation `Enumerate` (voir `Pipeline.variants`), et
l'empreinte d'un pipeline (`fingerprint`) entre dans sa cle.

Les images sortent en float32 dans [0, 1] : c'est le backbone qui normalise
(models/contract.py). Les vues du pre-entrainement sont ailleurs (methods/lejepa.py).
"""

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF


class Resize(v2.Resize):
    """`v2.Resize`, sauf pour les masques : plus proche voisin "exact", celui de PIL. Le plus
    proche voisin par defaut de torchvision choisit d'autres pixels (une centaine de pixels
    de classe differents sur un masque 128x128)."""

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, tv_tensors.Mask):
            return TF.resize(inpt, self.size, interpolation=TF.InterpolationMode.NEAREST_EXACT)
        return super().transform(inpt, params)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, masks=nearest_exact"


class ToFloatImage(v2.Transform):
    """Image -> (3, H, W) float32 dans [0, 1] ; la cible n'est pas touchee. Remplace
    `ToImage` + `ToDtype`, dont le repr tait le type : il est ici dans le repr, donc dans
    l'empreinte du pipeline."""

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, (Image.Image, tv_tensors.Image)):
            return TF.to_dtype(TF.to_image(inpt), torch.float32, scale=True)
        return inpt

    def extra_repr(self) -> str:
        return "float32, [0, 1]"


def center_crop(size: int) -> v2.Compose:
    """Petit cote a `size`, puis crop central carre (classification)."""
    return v2.Compose([Resize(size), v2.CenterCrop(size), ToFloatImage()])


def square(size: int) -> v2.Compose:
    """Resize anisotrope en carre, sans crop : rien n'est coupe (objets centres, masques,
    boites). L'IoU entre boites alignees est invariante par ce scaling des axes."""
    return v2.Compose([Resize([size, size]), ToFloatImage()])


def steps(t: v2.Transform | None) -> str:
    """Etapes d'un transform sur une ligne, `Compose` aplatis : "A > B > C"."""
    if t is None:
        return "aucune"
    if isinstance(t, v2.Compose):
        return " > ".join(steps(c) for c in t.transforms)
    return repr(t)


class Enumerate(v2.Transform):
    """Une augmentation a nombre fini de versions (nom -> transform deterministe) : une
    version tiree au hasard par appel, et le cache de features les calcule toutes."""

    def __init__(self, choices: dict[str, v2.Transform]) -> None:
        super().__init__()
        self.choices = choices

    def extra_repr(self) -> str:
        # les versions sont dans le repr, donc dans l'empreinte du pipeline
        return ", ".join(f"{name!r}: {t!r}" for name, t in self.choices.items())

    def forward(self, *inputs: Any) -> Any:
        names = list(self.choices)
        name = names[int(torch.randint(len(names), ()))]
        return self.choices[name](*inputs)


def hflip() -> Enumerate:
    """Retournement horizontal une fois sur deux : versions "" (inchangee) et "flip"."""
    return Enumerate({"": v2.Identity(), "flip": v2.RandomHorizontalFlip(p=1.0)})


def to_model_target(target: Any, size: tuple[int, int]) -> Any:
    """Cible torchvision -> format du modele : label int inchange ; masque -> (H, W) int64 ;
    {"box": BoundingBoxes (1, 4) en pixels, "label"} -> {"box": (4,) xyxy dans [0, 1],
    "label": tenseur int64}. `size` : (H, W) de l'image au moment de la conversion."""
    if isinstance(target, tv_tensors.Mask):
        return target.as_subclass(Tensor).to(torch.int64)
    if isinstance(target, dict):
        if not target["box"].is_floating_point():
            raise TypeError(
                "boites entieres : torchvision les arrondit au pixel a chaque transform, "
                "les creer en float32 dans la source"
            )
        h, w = size
        box = target["box"].as_subclass(Tensor)[0] / torch.tensor([w, h, w, h])
        return {"box": box.clamp(0, 1), "label": torch.tensor(target["label"])}
    return target


@dataclass(frozen=True)
class Pipeline:
    """(image PIL, cible torchvision) -> (image (3, size, size) float32 [0, 1], cible du
    modele). `native_target` : la cible garde sa resolution d'origine (evaluation de la
    segmentation : la metrique y remonte les logits, la perte de resolution de la tete est
    donc penalisee)."""

    prepare: v2.Transform
    augment: v2.Transform | None = None
    native_target: bool = False

    def __call__(self, img: Image.Image, target: Any) -> tuple[Tensor, Any]:
        if self.augment is not None:
            img, target = self.augment(img, target)
        if self.native_target:
            img = self.prepare(img)
        else:
            img, target = self.prepare(img, target)
        return img, to_model_target(target, tuple(img.shape[-2:]))

    def variants(self) -> dict[str, "Pipeline"]:
        """Versions deterministes a extraire dans le cache : une par choix d'une
        augmentation `Enumerate`, une seule sans augmentation."""
        if self.augment is None:
            return {"": self}
        if not isinstance(self.augment, Enumerate):
            raise NotImplementedError(
                f"augmentation non enumerable ({type(self.augment).__name__}) : le cache de "
                "features ne sait calculer que les versions d'un `Enumerate`"
            )
        return {name: replace(self, augment=t) for name, t in self.augment.choices.items()}

    def __repr__(self) -> str:
        native = ", cible a sa resolution d'origine" if self.native_target else ""
        return f"Pipeline(augment: {steps(self.augment)} | prepare: {steps(self.prepare)}{native})"

    def fingerprint(self) -> str:
        """Empreinte du repr (chaque etape et ses parametres) : change des que le traitement
        change, donc des que les features ou les scores peuvent changer."""
        return hashlib.sha256(repr(self).encode()).hexdigest()[:12]


def collate_native(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, list[Tensor]]:
    """Images de taille commune, masques a leur resolution native : liste, pas de stack."""
    return torch.stack([x for x, _ in batch]), [y for _, y in batch]
