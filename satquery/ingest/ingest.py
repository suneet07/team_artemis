from dataclasses import dataclass
from pathlib import Path

from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.ingest.band_inventory import BandInventory
from satquery.ingest.bands import build_band_inventory
from satquery.ingest.compatibility import CompatibilityReport
from satquery.ingest.modality import detect_modality
from satquery.ingest.radiometry import nodata_fraction
from satquery.ingest.reader import RasterMeta, build_meta, open_raster, read_overview

MODALITY_OVERRIDES = ("optical", "sar")


@dataclass
class IngestResult:
    meta: RasterMeta
    modality: str
    modality_source: str
    inventory: BandInventory
    compatibility: CompatibilityReport


def ingest_raster(
    path: str | Path,
    config: PreprocessingConfig | None = None,
    *,
    modality_override: str | None = None,
) -> IngestResult:
    if modality_override is not None and modality_override not in MODALITY_OVERRIDES:
        raise ValueError(
            f"modality_override must be one of {MODALITY_OVERRIDES}, got {modality_override!r}"
        )
    cfg = config if config is not None else preprocessing_config()

    with open_raster(path) as src:
        overview = read_overview(src, max_side=cfg.ingest.overview_max_side)
        meta, notes = build_meta(path, src, overview)

    if modality_override is not None:
        modality = modality_override
        modality_notes = [f"modality declared by caller: '{modality_override}'"]
        modality_source = "declared"
    else:
        modality, modality_notes, modality_source = detect_modality(meta, overview, cfg.modality)

    inventory, band_notes = build_band_inventory(meta, modality, cfg.bands)

    warnings = notes + modality_notes + band_notes
    if modality == "unknown":
        warnings.append("modality unresolved; ask the user and record the answer in the trace")
    if meta.crs is None:
        warnings.append("no CRS: georeferencing unavailable until user supplies it")
    if not any(meta.descriptions):
        warnings.append("band descriptions absent; band semantics inferred")
    frac = nodata_fraction(overview)
    if frac > cfg.ingest.nodata_warn_fraction:
        warnings.append(f"nodata fraction {frac:.2f} exceeds {cfg.ingest.nodata_warn_fraction}")

    compatibility = CompatibilityReport(
        format_ok=True,
        crs_valid=meta.crs is not None,
        modality=modality,
        modality_source=modality_source,
        bands_present=sorted(inventory.bands),
        computable_indices=list(inventory.computable_indices),
        nodata_frac=frac,
        bit_depth=meta.bit_depth,
        bit_depth_source=meta.bit_depth_source,
        pixel_size_m=meta.pixel_size_m,
        native_gsd_m=meta.native_gsd_m,
        warnings=warnings,
    )
    return IngestResult(
        meta=meta,
        modality=modality,
        modality_source=modality_source,
        inventory=inventory,
        compatibility=compatibility,
    )
