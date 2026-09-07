"""Turn a file on disk into the :class:`~satquery.tools.base.Scene` tools consume.

This is the join between P1 (ingest and validation) and P6 (the tools), and it
is where two of the plan's non-negotiables are enforced once rather than in
every tool:

* **Indices are computed on raw band values.** The percentile stretch exists for
  model input. Running it before a spectral index changes the ratio between the
  two terms and therefore changes the index, which silently invalidates every
  physical threshold in ``preprocessing.yaml``. So the Scene carries raw bands
  and the stretch travels separately as provenance.
* **``tiling.max_pixels`` is a cap on what reaches the *model*, not on what the
  deterministic tools may read.** A threshold, a connected-component analysis or
  an area in km² is more accurate at full resolution and costs nothing in vision
  tokens, so this reads the raster as it is and leaves
  :meth:`TilingConfig.check_within_cap` to guard the model path. Downsampling
  here would coarsen the ground sample distance of a graded mask for no reason.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.ingest.ingest import IngestResult, ingest_raster
from satquery.ingest.reader import open_raster
from satquery.tools.base import Scene

__all__ = ["LoadedScene", "load_scene"]


@dataclass
class LoadedScene:
    scene: Scene
    ingest: IngestResult
    warnings: list[str]

    def as_trace_input(self) -> dict:
        """The ``inputs[]`` record for this scene (trace schema v2)."""
        meta = self.ingest.meta
        inventory = self.ingest.inventory
        record: dict = {
            "file": Path(meta.path).name,
            "modality": self.ingest.modality,
            "modality_source": self.ingest.modality_source,
            "bands": sorted(inventory.bands),
            "computable_indices": list(inventory.computable_indices),
            "swir_available": inventory.has_swir,
            "bit_depth": meta.bit_depth,
            "bit_depth_source": meta.bit_depth_source,
            "nodata_frac": round(self.ingest.compatibility.nodata_frac, 6),
        }
        if meta.crs:
            record["crs"] = meta.crs
        if meta.pixel_size_m:
            record["pixel_size_m"] = meta.pixel_size_m
        if meta.native_gsd_m:
            record["native_gsd_m"] = meta.native_gsd_m
        if inventory.polarisations:
            record["polarisations"] = list(inventory.polarisations)
        if inventory.sar_band:
            record["sar_band"] = inventory.sar_band
        return record


def _band_names(result: IngestResult, count: int) -> list[str]:
    """Name each band slot; unmapped slots keep a neutral, honest name."""
    by_index = {index: name for name, index in result.inventory.bands.items()}
    return [by_index.get(slot, f"b{slot}") for slot in range(1, count + 1)]


def load_scene(
    path: str | Path,
    *,
    modality_override: str | None = None,
    date: str | None = None,
    config: PreprocessingConfig | None = None,
) -> LoadedScene:
    """Read a raster into a Scene the deterministic tools can operate on."""
    cfg = config if config is not None else preprocessing_config()
    result = ingest_raster(path, cfg, modality_override=modality_override)
    meta = result.meta
    warnings = list(result.compatibility.warnings)

    with open_raster(path) as src:
        data = src.read(masked=True)
        transform = src.transform
        designations = tuple(src.descriptions or ())
        if cfg.tiling.max_pixels and src.width * src.height > cfg.tiling.max_pixels:
            warnings.append(
                f"scene is {src.width}x{src.height} px, above the frozen "
                f"tiling.max_pixels cap of {cfg.tiling.max_pixels:,}. Deterministic tools "
                f"run at full resolution, which is correct; anything routed to the model "
                f"must go through P4 tiling first, or the processor will silently "
                f"downsample it."
            )

    planes = np.ma.filled(data.astype(np.float64), np.nan)
    nodata_mask = np.ma.getmaskarray(data).any(axis=0)
    names = _band_names(result, data.shape[0])

    scene = Scene(
        name=Path(meta.path).name,
        modality=result.modality,
        bands={name: planes[index] for index, name in enumerate(names)},
        inventory=result.inventory,
        transform=transform,
        crs=meta.crs,
        pixel_size_m=meta.pixel_size_m,
        nodata_mask=nodata_mask,
        date=date,
        designations=designations,
    )
    return LoadedScene(scene=scene, ingest=result, warnings=warnings)
