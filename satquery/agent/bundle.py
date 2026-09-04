from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from satquery.ingest.band_inventory import BandInventory


@dataclass
class ImageRef:
    scene_id: str
    path: Path | str
    modality: str  # "optical" | "sar" | "unknown"
    crs: str | None = None
    pixel_size_m: float | None = None
    native_gsd_m: float | None = None
    bands: list[str] = field(default_factory=list)
    swir_available: bool = False
    bit_depth: int | None = None
    bit_depth_source: str | None = None
    nodata_frac: float = 0.0
    role: str | None = None
    polarisations: list[str] = field(default_factory=list)
    sar_band: str | None = None
    computable_indices: list[str] = field(default_factory=list)
    modality_source: str | None = None


@dataclass
class CoregReport:
    coregistered: bool = True
    rmse_px: float | None = None
    correction_applied: bool = False
    method: str | None = None
    common_crs: str | None = None
    checks_passed: list[str] = field(default_factory=list)


@dataclass
class TileIndex:
    tile_count: int
    tile_size_px: int
    overlap_frac: float = 0.0
    tiles: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProvenanceStep:
    stage: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    at: str = ""
    scene_id: str | None = None


@dataclass
class ImageBundle:
    bundle_id: str
    images: list[ImageRef]
    band_inventory: BandInventory
    pair_type: Literal["single", "crossmodal", "bitemporal"] = "single"
    coreg: CoregReport | None = None
    tiles: TileIndex | None = None
    provenance: list[ProvenanceStep] = field(default_factory=list)
    status: str = "ready"  # "queued" | "preparing" | "ready" | "failed"
    supported_tasks: list[str] = field(default_factory=list)
    blocked_tasks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
