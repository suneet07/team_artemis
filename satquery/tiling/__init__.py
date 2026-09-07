"""P4 — Tiling and tile relevance (master plan section 4.4).

``GeoTiler`` is the streaming, rasterio-backed tiler; ``plan_tiles`` is the
config-driven planner that also produces the global overview and the tile cap
the query SLA depends on. Both write geo-indexed tiles, because evidence has to
map back to scene coordinates (step 3).
"""

from satquery.tiling.clip_scorer import ClipTileScorer, clip_available
from satquery.tiling.scorer import ScoredTile, TileScorer, content_scorer, score_tiles
from satquery.tiling.tiler import GeoTiler
from satquery.tiling.tiling import Tile, TilePlan, mosaic_masks, plan_tiles

__all__ = [
    "ClipTileScorer",
    "GeoTiler",
    "ScoredTile",
    "Tile",
    "TilePlan",
    "TileScorer",
    "clip_available",
    "content_scorer",
    "mosaic_masks",
    "plan_tiles",
    "score_tiles",
]
