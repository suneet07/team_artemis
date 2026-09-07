"""P4 — Tiling (master plan section 4.4), GeoPixel-style partitioning.

A Cartosat scene can be 10,000x10,000 or larger. Downsampling that to 512 px
destroys everything the question is about. The plan's answer, taken from
GeoPixel's partitioning method:

1. adaptive partition into local tiles plus one global overview;
2. overlap tiles by ~10% so a feature on a boundary is whole in some tile;
3. geo-index every tile — store its transform so evidence maps back to scene
   coordinates;
4. relevance-score tiles against the query text (``tile_scorer``);
5. process only the top-k, with a hard cap so latency stays bounded;
6. mosaic outputs back into the scene CRS.

Step 4 is deliberately **not** a CLIP call here. The plan names CLIP-style
similarity, but the headless evaluation path has to run CPU-only with no network
and no model download (section 4.11), and the cut-list fallback for this whole
pipeline is "global downsample plus one query-relevant detail crop". So the
scorer is a pluggable interface with a deterministic content-statistics default,
and a CLIP scorer can be registered on top when a GPU is present without
changing anything downstream.
"""

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from rasterio.windows import Window

from satquery.config import PreprocessingConfig, preprocessing_config

__all__ = ["Tile", "TilePlan", "mosaic_masks", "plan_tiles"]


@dataclass(frozen=True)
class Tile:
    """One geo-indexed tile of a scene."""

    index: int
    window: Window
    transform: Affine
    is_overview: bool = False
    score: float = 0.0

    @property
    def bounds_px(self) -> tuple[int, int, int, int]:
        """(row_off, col_off, height, width) in *scene* pixel coordinates."""
        return (
            int(self.window.row_off),
            int(self.window.col_off),
            int(self.window.height),
            int(self.window.width),
        )

    def as_trace_entry(self) -> dict:
        row, col, height, width = self.bounds_px
        return {
            "index": self.index,
            "row_off": row,
            "col_off": col,
            "height": height,
            "width": width,
            "is_overview": self.is_overview,
            "score": round(self.score, 4),
        }


@dataclass
class TilePlan:
    scene_shape: tuple[int, int]
    tiles: list[Tile]
    overview: Tile
    tile_side_px: int
    overlap_px: int
    truncated: bool = False
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def top_k(self, k: int) -> list[Tile]:
        """Highest-scoring ``k`` tiles, always with the overview first.

        The overview is not optional: without it a model answering "what
        dominates this scene" only ever sees crops.
        """
        ranked = sorted(self.tiles, key=lambda t: t.score, reverse=True)[: max(0, k)]
        return [self.overview, *ranked]


def _offsets(extent: int, side: int, stride: int) -> list[int]:
    if extent <= side:
        return [0]
    offsets = list(range(0, extent - side + 1, stride))
    if offsets[-1] + side < extent:
        offsets.append(extent - side)  # flush the last tile to the edge
    return offsets


def plan_tiles(
    scene_shape: tuple[int, int],
    scene_transform: Affine | None = None,
    config: PreprocessingConfig | None = None,
) -> TilePlan:
    """Partition a scene into overlapping geo-indexed tiles plus an overview.

    ``scene_transform`` is the rasterio affine of the full scene; every tile
    carries its own derived transform so a mask produced on a tile can be written
    back at scene coordinates (section 4.4 step 3 and the C18 mask rule).
    """
    cfg = config if config is not None else preprocessing_config()
    height, width = int(scene_shape[0]), int(scene_shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"scene shape {scene_shape} is empty")
    transform = scene_transform if scene_transform is not None else Affine.identity()

    side = min(cfg.tiling.tile_side_px, max(height, width))
    stride = max(1, int(round(side * (1.0 - cfg.tiling.tile_overlap_fraction))))
    overlap_px = side - stride

    overview = Tile(
        index=0,
        window=Window(0, 0, width, height),
        transform=transform,
        is_overview=True,
        score=float("inf"),
    )

    warnings: list[str] = []
    tiles: list[Tile] = []
    index = 1
    truncated = False
    for row in _offsets(height, min(side, height), stride):
        for col in _offsets(width, min(side, width), stride):
            if len(tiles) >= cfg.tiling.max_tiles:
                truncated = True
                break
            window = Window(col, row, min(side, width), min(side, height))
            tiles.append(
                Tile(
                    index=index,
                    window=window,
                    transform=transform * Affine.translation(col, row),
                    is_overview=False,
                )
            )
            index += 1
        if truncated:
            break

    if truncated:
        warnings.append(
            f"scene partitions into more than {cfg.tiling.max_tiles} tiles; capped so the "
            f"query SLA stays bounded (section 4.4 step 5). Coverage is partial — the "
            f"overview still covers the whole scene."
        )
    return TilePlan(
        scene_shape=(height, width),
        tiles=tiles,
        overview=overview,
        tile_side_px=side,
        overlap_px=overlap_px,
        truncated=truncated,
        warnings=warnings,
    )


def mosaic_masks(
    scene_shape: tuple[int, int],
    pieces: list[tuple[Tile, np.ndarray]],
) -> np.ndarray:
    """Mosaic per-tile boolean masks back to full scene resolution (step 6).

    Overlapping tiles are combined by logical OR: a feature detected in either
    view of the overlap is a detection. The result is full-scene, which is what
    the C18 mask rule requires of every graded mask — never left at tile
    resolution.
    """
    height, width = int(scene_shape[0]), int(scene_shape[1])
    out = np.zeros((height, width), dtype=bool)
    for tile, mask in pieces:
        row, col, tile_h, tile_w = tile.bounds_px
        piece = np.asarray(mask, dtype=bool)
        if piece.shape != (tile_h, tile_w):
            raise ValueError(
                f"tile {tile.index} mask is {piece.shape}, expected {(tile_h, tile_w)}; "
                f"a mask must be mosaicked at the resolution it was produced at"
            )
        out[row : row + tile_h, col : col + tile_w] |= piece
    return out
