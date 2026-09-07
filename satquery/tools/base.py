"""The contract every tool in the registry implements (master plan section 4.6).

A tool is a manifest plus a callable. The manifest is the enforceable half — it
is what the section 4.5.4 parameter gate validates against, and a parameter not
declared there cannot be set. This module is the executable half: what a tool
receives, what it must hand back, and how its output becomes a trace step.

Two rules from the plan are enforced structurally here rather than left to each
tool author's memory:

* **C18 — every mask is a graded artifact.** A tool that produces a mask hands
  back a full-scene boolean array plus the CRS and transform to write it with.
  There is no path for returning a tile-resolution mask or a web overlay alone.
* **Confidence has a stated basis.** Every result names where its number came
  from, and a deterministic proposer is capped so it can never outrank learned
  grounding (4.6.9).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from rasterio.transform import Affine

from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.ingest.band_inventory import BandInventory

__all__ = ["Scene", "Tool", "ToolContext", "ToolResult", "area_km2"]


def area_km2(mask: np.ndarray, pixel_size_m: float | None) -> float | None:
    """Area of a boolean mask, or None when the grid has no metric scale.

    Returning None rather than a pixel count is deliberate: an area in "pixels"
    that reaches a graded output reads as square kilometres to anyone skimming it.
    """
    if pixel_size_m is None or pixel_size_m <= 0:
        return None
    return float(np.count_nonzero(mask)) * (pixel_size_m**2) / 1_000_000.0


@dataclass
class Scene:
    """One prepared input the tools operate on.

    ``bands`` maps a canonical band or polarisation name to a 2-D array at full
    scene resolution. For SAR these are dB sigma-nought planes; for optical they
    are the raw (unstretched) reflectance-proxy planes, because a spectral index
    must be computed before any per-band stretch — stretching first would change
    the ratio and therefore the index value.
    """

    name: str
    modality: str
    bands: dict[str, np.ndarray]
    inventory: BandInventory
    transform: Affine | None = None
    crs: str | None = None
    pixel_size_m: float | None = None
    nodata_mask: np.ndarray | None = None
    date: str | None = None
    #: Sensor band designation per slot, verbatim from the file -- ``("B02",
    #: "B03", ...)``. The canonical ``bands`` keys cannot stand in for this:
    #: they collapse B11 and B12 to one ``swir``, so the second SWIR band ends
    #: up named ``b6`` and the short-wave composite becomes unbuildable from
    #: the inventory alone. Empty when the file names nothing.
    designations: tuple[str | None, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        first = next(iter(self.bands.values()))
        return tuple(np.asarray(first).shape[-2:])  # type: ignore[return-value]

    def band(self, name: str) -> np.ndarray:
        key = name.lower()
        for candidate, values in self.bands.items():
            if candidate.lower() == key:
                return np.asarray(values, dtype=np.float64)
        raise KeyError(f"scene '{self.name}' has no band '{name}'")

    def has(self, name: str) -> bool:
        return any(candidate.lower() == name.lower() for candidate in self.bands)

    def valid_mask(self) -> np.ndarray:
        stack = np.stack([np.asarray(v, dtype=np.float64) for v in self.bands.values()])
        valid = np.isfinite(stack).all(axis=0)
        if self.nodata_mask is not None:
            valid &= ~np.asarray(self.nodata_mask, dtype=bool)
        return valid


@dataclass
class ToolContext:
    """Everything a tool is allowed to look at."""

    query_text: str
    scenes: list[Scene]
    params: dict[str, Any]
    config: PreprocessingConfig = field(default_factory=preprocessing_config)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Scene:
        return self.scenes[0]

    def scene_of(self, modality: str) -> Scene | None:
        for scene in self.scenes:
            if scene.modality == modality:
                return scene
        return None


@dataclass
class ToolResult:
    """What a tool hands back. Becomes one ``steps[]`` entry in the trace."""

    outputs: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    confidence_basis: str = "heuristic"
    warnings: list[str] = field(default_factory=list)
    #: Full-scene boolean mask. C18: never tile-resolution, never overlay-only.
    mask: np.ndarray | None = None
    mask_crs: str | None = None
    mask_transform: Affine | None = None
    #: Continuous response the mask was thresholded from, for downstream tools.
    response: np.ndarray | None = None
    #: Extra fields merged into the recorded step params (threshold provenance).
    param_provenance: dict[str, Any] = field(default_factory=dict)

    def capped(self, ceiling: float) -> "ToolResult":
        """Apply a confidence ceiling — used for deterministic proposers."""
        if self.confidence is not None:
            self.confidence = min(self.confidence, ceiling)
        return self


class Tool(Protocol):
    """A registered tool: ``name`` matching its manifest, plus ``run``."""

    name: str

    def run(self, context: ToolContext) -> ToolResult: ...
