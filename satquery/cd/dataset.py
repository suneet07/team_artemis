"""SpaceNet 7 date pairs as a change-detection dataset.

The VLM answers change questions by inference and gets 1.2 points of its
counting accuracy from actually looking at the images. A segmentation model
answers them by *measurement*: emit a per-pixel change mask, then presence,
direction, magnitude and location are arithmetic on it rather than guesses.

**Masks are rasterised on the fly, not staged.** A tile holds tens of
footprints, so building the mask costs less than reading a cached one would,
and it keeps the label definition in one place. Staging masks would mean a
second copy of the truth that can drift from the GeoJSON it came from.

**The label is the ID difference, not a pixel difference.** SpaceNet 7's
footprints carry persistent IDs, so "changed" means a building present at one
date and absent at the other -- exactly the definition ``gen_change.py`` uses
for its questions. Differencing pixels instead would mark every shadow, every
seasonal roof, and every registration error as change.

Licence: SpaceNet 7 / MUDS, CC BY-SA 4.0, AWS Open Data Program.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

__all__ = ["BuildingDataset", "ChangePairDataset", "build_pairs"]

#: Tiles are 1024 px at 4 m. Training on the full frame would fit few samples
#: per batch and spend most of its capacity on unchanged background, so crops
#: are the standard practice for this task and 256 is what the CD literature
#: uses on LEVIR-CD.
CROP_PX = 256


def build_pairs(root: Path, max_gap_months: int = 0) -> list[dict[str, Any]]:
    """Every usable (t0, t1) pair in a staged SpaceNet 7 plan.

    ``max_gap_months`` of 0 takes every combination. A cap is worth having
    because adjacent months mostly show nothing: the mask is empty, the loss is
    already near zero on it, and the batch is spent on background.
    """
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for aoi, entry in plan["aois"].items():
        months = entry["months"]
        for a, b in combinations(months, 2):
            if max_gap_months:
                ya, ma = (int(v) for v in a.split("-"))
                yb, mb = (int(v) for v in b.split("-"))
                if abs((yb * 12 + mb) - (ya * 12 + ma)) > max_gap_months:
                    continue
            images = {
                m: root / aoi / "images" / Path(entry["images"][m]).name
                for m in (a, b)
            }
            labels = {
                m: root / aoi / "labels" / Path(entry["labels"][m]).name
                for m in (a, b)
            }
            if not all(p.exists() for p in (*images.values(), *labels.values())):
                continue
            pairs.append(
                {
                    "aoi": aoi,
                    "months": [a, b],
                    "images": [images[a], images[b]],
                    "labels": [labels[a], labels[b]],
                }
            )
    return pairs


class BuildingDataset:
    """One crop per item: a single date, and every building in it.

    The change-pair framing asks a network to find the ~0.5% of pixels that
    differ between two dates. This asks it to find buildings -- 5-15% of a
    crop, dense supervision, and the task all three top SpaceNet 7 solutions
    actually trained. Change is then derived by comparing per-date detections,
    which is arithmetic rather than learning.

    Motokimura, 4th place, listed multi-frame stacking under "things I have
    tried but did not help". Our own pair model converged at F1 0.29 and
    plateaued, which agrees with him.
    """

    def __init__(
        self,
        pairs: list[dict[str, Any]],
        *,
        crop: int = 160,
        upscale: int = 3,
        crops_per_image: int = 2,
        train: bool = True,
        seed: int = 0,
    ):
        # Each pair contributes both its dates as independent samples: the
        # detector never sees two dates at once, so a "pair" is just a
        # convenient list of images we already know how to find.
        seen: dict[Any, Any] = {}
        for pair in pairs:
            for image, label in zip(pair["images"], pair["labels"], strict=True):
                seen[image] = {"aoi": pair["aoi"], "image": image, "label": label}
        self.items = list(seen.values())
        if not self.items:
            raise ValueError("BuildingDataset needs at least one image")
        self.crop = crop
        # 3-4x enlargement before the network was one of motokimura's four
        # "especially important" items: at 4 m a building is a handful of
        # pixels, and upscaling is what makes small ones separable at all.
        self.upscale = upscale
        self.crops_per_image = crops_per_image
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items) * self.crops_per_image

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import rasterio
        import torch
        import torch.nn.functional as F
        from rasterio.features import rasterize

        item = self.items[index // self.crops_per_image]
        rng = np.random.default_rng(self.seed + index)

        with rasterio.open(item["image"]) as src:
            array = src.read([1, 2, 3]).astype("float32")
            transform = src.transform
            shape = (src.height, src.width)

        document = json.loads(Path(item["label"]).read_text(encoding="utf-8"))
        geometries = [
            f["geometry"] for f in document.get("features", []) if f.get("geometry")
        ]
        mask = (
            rasterize(
                [(g, 1) for g in geometries],
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype="uint8",
            )
            if geometries
            else np.zeros(shape, dtype="uint8")
        )

        limit_y = max(1, shape[0] - self.crop)
        limit_x = max(1, shape[1] - self.crop)
        if self.train and mask.any() and rng.random() < 0.8:
            ys, xs = np.nonzero(mask)
            pick = rng.integers(len(ys))
            top = int(np.clip(ys[pick] - self.crop // 2, 0, limit_y - 1))
            left = int(np.clip(xs[pick] - self.crop // 2, 0, limit_x - 1))
        else:
            top, left = int(rng.integers(limit_y)), int(rng.integers(limit_x))

        sl = (slice(top, top + self.crop), slice(left, left + self.crop))
        pixels = array[:, sl[0], sl[1]]
        target = mask[sl].astype("float32")
        valid = (pixels.sum(axis=0) > 0).astype("float32")

        # Enlarge with torch rather than skimage: torch is already a dependency
        # here and scikit-image is not in the training image. Bilinear for the
        # imagery, nearest for mask and validity -- interpolating a label would
        # invent fractional buildings along every edge, and the loss reads the
        # mask as a probability, so those fractions become real supervision.
        size = self.crop * self.upscale
        pixels_t = torch.from_numpy(np.ascontiguousarray(pixels / 255.0)).float()
        pixels_t = F.interpolate(
            pixels_t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
        target_t = torch.from_numpy(np.ascontiguousarray(target)).float()
        valid_t = torch.from_numpy(np.ascontiguousarray(valid)).float()
        target_t = F.interpolate(
            target_t[None, None], size=(size, size), mode="nearest"
        )[0, 0]
        valid_t = F.interpolate(
            valid_t[None, None], size=(size, size), mode="nearest"
        )[0, 0]

        return {
            "pixels": pixels_t,
            "mask": target_t,
            "valid": valid_t,
            "aoi": item["aoi"],
        }


class ChangePairDataset:
    """One crop per item: two dates stacked, plus the binary change mask."""

    def __init__(
        self,
        pairs: list[dict[str, Any]],
        *,
        crop: int = CROP_PX,
        crops_per_pair: int = 4,
        train: bool = True,
        seed: int = 0,

    ):
        if not pairs:
            raise ValueError("ChangePairDataset needs at least one pair")
        self.pairs = pairs
        self.crop = crop
        self.crops_per_pair = crops_per_pair
        self.train = train
        self.seed = seed


    def __len__(self) -> int:
        return len(self.pairs) * self.crops_per_pair

    def _ids(self, path: Path) -> dict[int, Any]:
        document = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for feature in document.get("features", []):
            properties = feature.get("properties") or {}
            identifier = properties.get("Id", properties.get("id"))
            geometry = feature.get("geometry")
            if identifier is None or not geometry:
                continue
            out[identifier] = geometry
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import rasterio
        import torch
        from rasterio.features import rasterize

        pair = self.pairs[index // self.crops_per_pair]
        rng = np.random.default_rng(self.seed + index)

        arrays, transforms = [], []
        for path in pair["images"]:
            with rasterio.open(path) as src:
                # First three bands: SpaceNet 7 is RGB(A) Planet, and the alpha
                # channel is a validity mask rather than imagery.
                data = src.read([1, 2, 3]).astype("float32")
                arrays.append(data)
                transforms.append(src.transform)
                shape = (src.height, src.width)

        before, after = (self._ids(p) for p in pair["labels"])
        # Changed = appeared or demolished. Symmetric difference of the ID sets,
        # which is the same definition the question generator answers with.
        changed = [
            geometry
            for identifier, geometry in {**before, **after}.items()
            if (identifier in before) != (identifier in after)
        ]
        mask = (
            rasterize(
                [(g, 1) for g in changed],
                out_shape=shape,
                transform=transforms[0],
                fill=0,
                dtype="uint8",
            )
            if changed
            else np.zeros(shape, dtype="uint8")
        )

        # Crop selection is biased toward change when training: an unbiased
        # crop of a 1024 px tile is almost always empty, and a model fed mostly
        # empty masks learns to predict "no change" everywhere -- which scores
        # well on pixel accuracy and zero on F1.
        top, left = self._pick_crop(mask, rng, shape)
        sl = (slice(top, top + self.crop), slice(left, left + self.crop))
        crops = [a[:, sl[0], sl[1]] for a in arrays]
        mask_crop = mask[sl]

        # Validity: a pixel counts only where BOTH dates carry imagery.
        # SpaceNet 7 writes cloud-covered regions as pure black and keeps the
        # footprints underneath, so a masked region asks the model to segment
        # change it cannot see -- and unlike the VQA corpus, where that merely
        # made a question unanswerable, here the per-pixel loss actively
        # rewards predicting change on black. Measured across the mosaics:
        # 131 of 1,423 are more than 5% masked.
        valid = np.ones_like(mask_crop, dtype="float32")
        for crop in crops:
            valid *= (crop.sum(axis=0) > 0).astype("float32")

        stacked = np.concatenate(crops, axis=0) / 255.0
        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(stacked)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_crop)).float(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).float(),
            "aoi": pair["aoi"],
            "months": pair["months"],
        }

    def _pick_crop(self, mask, rng, shape) -> tuple[int, int]:
        import numpy as np

        height, width = shape
        limit_y, limit_x = max(1, height - self.crop), max(1, width - self.crop)
        if self.train and mask.any() and rng.random() < 0.7:
            ys, xs = np.nonzero(mask)
            pick = rng.integers(len(ys))
            top = int(np.clip(ys[pick] - self.crop // 2, 0, limit_y - 1))
            left = int(np.clip(xs[pick] - self.crop // 2, 0, limit_x - 1))
            return top, left
        return int(rng.integers(limit_y)), int(rng.integers(limit_x))
