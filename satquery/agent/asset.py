from dataclasses import dataclass
from typing import Any, Literal

AssetKind = Literal[
    "mask_geotiff",
    "overlay_png",
    "chart_png",
    "report_pdf",
    "scene_preview",
    "bbox_geojson",
]


@dataclass
class AssetRef:
    asset_id: str
    kind: AssetKind
    label: str
    produced_by: str  # tool name — REQUIRED
    media_type: str
    bytes: int = 0
    crs: str | None = None  # null => not georeferenced
    bounds_wgs84: list[float] | None = None
    bbox_px: list[list[float]] | None = None  # grounding boxes in scene pixel space
    tile_url_template: str | None = None
    overlay_url: str | None = None
    download_url: str = ""
    colour: str | None = None
    stats: dict[str, Any] | None = None
