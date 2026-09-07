"""P8 — geo-referenced mask export (section 4.8, and the C18 mask rule).

**C18 is the rule this file exists to make unbreakable.** The problem statement
lists masks among the hidden set's reference annotation types, so every mask a
tool produces is a potentially scoreable artifact. Therefore:

* every mask is written as a GeoTIFF,
* in the **source CRS**,
* at **full scene resolution** — mosaicked back from tiles where tiling was
  used, never left at tile resolution,
* and never only as a web overlay.

:func:`write_mask` refuses to write a mask that would violate any of that, rather
than writing something that looks like a deliverable and scores zero. The CI mask
conformance check (Phase 1 item 19) asserts against this function.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

__all__ = ["MaskConformanceError", "MaskRecord", "check_mask_conformance", "write_mask"]


class MaskConformanceError(ValueError):
    """A mask that would not be scoreable as delivered."""


@dataclass(frozen=True)
class MaskRecord:
    """What the trace records for one written mask."""

    uri: str
    crs: str
    resolution_m: float | None
    shape: tuple[int, int]
    positive_px: int
    area_km2: float | None

    def as_dict(self) -> dict:
        return {
            "mask_uri": self.uri,
            "mask_crs": self.crs,
            "mask_res_m": self.resolution_m,
            "mask_shape": list(self.shape),
            "positive_px": self.positive_px,
            "area_km2": None if self.area_km2 is None else round(self.area_km2, 6),
        }


def check_mask_conformance(
    mask: np.ndarray,
    scene_shape: tuple[int, int],
    crs: str | None,
    transform: Affine | None,
) -> None:
    """Raise unless this mask is scoreable as delivered (C18)."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise MaskConformanceError(f"a mask must be 2-D, got shape {mask.shape}")
    if tuple(mask.shape) != tuple(scene_shape):
        raise MaskConformanceError(
            f"mask is {mask.shape} but the scene is {tuple(scene_shape)}: masks are graded "
            f"outputs and must be at full scene resolution, mosaicked back from tiles "
            f"where tiling was used (C18)"
        )
    if not crs:
        raise MaskConformanceError(
            "mask has no CRS: a mask that cannot be loaded in QGIS at the right place is "
            "not a scoreable artifact (C18)"
        )
    if transform is None:
        raise MaskConformanceError("mask has no affine transform, so it cannot be geo-referenced")


def write_mask(
    path: Path | str,
    mask: np.ndarray,
    *,
    crs: str | None,
    transform: Affine | None,
    scene_shape: tuple[int, int] | None = None,
    nodata: int | None = None,
    description: str | None = None,
) -> MaskRecord:
    """Write a boolean mask as a geo-referenced single-band GeoTIFF."""
    mask = np.asarray(mask, dtype=bool)
    shape = tuple(scene_shape) if scene_shape is not None else tuple(mask.shape)
    check_mask_conformance(mask, shape, crs, transform)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        # Tiled and internally overviewed so a full-scene mask opens in QGIS
        # without loading the whole raster.
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
        if description:
            dst.set_band_description(1, description)
        dst.update_tags(SATQUERY_MASK="1", SATQUERY_SCHEMA="c18-full-scene-source-crs")

    resolution = abs(transform.a) if transform is not None else None
    area = (
        float(np.count_nonzero(mask)) * (resolution**2) / 1_000_000.0
        if resolution
        else None
    )
    return MaskRecord(
        uri=str(path),
        crs=str(crs),
        resolution_m=resolution,
        shape=(mask.shape[0], mask.shape[1]),
        positive_px=int(np.count_nonzero(mask)),
        area_km2=area,
    )
