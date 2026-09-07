"""Full-scene queries through P4 tiling (master plan section 4.4).

``answer_query`` reads a scene whole, which is right for a benchmark chip and
wrong for a Cartosat scene: at 10,000 x 10,000 the interesting object is a few
hundred pixels and every deterministic threshold is computed against a histogram
dominated by everything else.

This is the plan's step 5, "process top-k tiles only", wired into the query path:

1. partition into overlapping geo-indexed tiles plus a global overview;
2. score the tiles against the query (``tile_scorer``);
3. answer on the top-k only;
4. **mosaic the masks back to full scene resolution** and write one geo-referenced
   GeoTIFF per target.

Step 4 is not optional and is the reason this is not a loop over crops. C18 says
a mask is a graded artifact at full scene resolution in the source CRS; a
per-tile mask, or k separate masks, is not the deliverable no matter how good the
answer text is.

The Week-6 cut list degrades this to "global downsample plus one query-relevant
detail crop". :func:`tiled_answer_query` with ``top_k=1`` is that fallback, so
taking the cut is a parameter change rather than a rewrite.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from satquery.agent.pipeline import QueryOutcome, answer_query
from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.ingest.scene import load_scene
from satquery.report.masks import write_mask
from satquery.tiling import Tile, mosaic_masks, plan_tiles, score_tiles
from satquery.tiling.scorer import TileScorer

__all__ = ["TiledOutcome", "tiled_answer_query"]


@dataclass
class TiledOutcome:
    """One answer per selected tile, plus the mosaicked full-scene masks."""

    scene_shape: tuple[int, int]
    selected: list[dict[str, Any]]
    outcomes: list[QueryOutcome]
    mask_uris: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def best(self) -> QueryOutcome | None:
        """The highest-confidence tile answer — the one to show first."""
        answered = [o for o in self.outcomes if not o.refused]
        return max(answered, key=lambda o: o.confidence, default=None)

    def as_trace_block(self) -> dict[str, Any]:
        return {
            "scene_shape": list(self.scene_shape),
            "tiles_selected": self.selected,
            "mask_uris": list(self.mask_uris),
        }


def _write_tile(
    source: rasterio.DatasetReader, tile: Tile, directory: Path, index: int
) -> Path:
    """Write one tile as a GeoTIFF carrying its own geo-referencing.

    Written to disk rather than passed as an array because the whole query path
    — ingest, modality detection, band inventory, the compatibility report —
    keys off a raster. A tile that skips ingest would also skip the checks, and
    those checks are a named deliverable.
    """
    row, col, height, width = tile.bounds_px
    window = rasterio.windows.Window(col, row, width, height)
    data = source.read(window=window, masked=True)

    profile = source.profile.copy()
    profile.update(
        height=height,
        width=width,
        transform=source.window_transform(window),
        driver="GTiff",
        compress="deflate",
    )
    path = directory / f"tile_{index:04d}.tif"
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.ma.filled(data, source.nodata or 0))
        for band in range(1, source.count + 1):
            description = source.descriptions[band - 1]
            if description:
                destination.set_band_description(band, description)
        destination.update_tags(**source.tags())
    return path


def tiled_answer_query(
    question: str,
    image_path: str | Path,
    *,
    top_k: int | None = None,
    scorer: TileScorer | None = None,
    output_dir: Path | str | None = None,
    config: PreprocessingConfig | None = None,
    modality: str | None = None,
) -> TiledOutcome:
    """Answer ``question`` over a full scene by way of the top-k relevant tiles.

    ``scorer`` defaults to the deterministic content scorer;
    :class:`~satquery.tiling.ClipTileScorer` is the drop-in upgrade when weights
    are present.
    """
    cfg = config if config is not None else preprocessing_config()
    top_k = cfg.tiling.top_k_tiles if top_k is None else top_k
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else None
    warnings: list[str] = []

    loaded = load_scene(image_path, modality_override=modality, config=cfg)
    scene = loaded.scene
    plan = plan_tiles(scene.shape, scene.transform, config=cfg)
    warnings.extend(plan.warnings)

    if not plan.tiles:
        # Scene fits in one tile: tiling would add cost and change nothing.
        outcome = answer_query(
            question,
            [image_path],
            modalities=[modality] if modality else None,
            output_dir=output_dir,
            config=cfg,
            write_evidence=output_dir is not None,
        )
        return TiledOutcome(
            scene_shape=scene.shape,
            selected=[plan.overview.as_trace_entry()],
            outcomes=[outcome],
            mask_uris=list(outcome.trace["graded"]["outputs"].get("masks") or []),
            warnings=warnings,
        )

    # --- score the tiles against the query --------------------------------
    plane = np.stack([np.asarray(v, dtype=np.float64) for v in scene.bands.values()]).mean(
        axis=0
    )
    crops = []
    for tile in plan.tiles:
        row, col, height, width = tile.bounds_px
        crops.append((tile.index, plane[row : row + height, col : col + width]))
    ranked = score_tiles(question, crops, scorer=scorer)
    scores = {entry.index: entry.score for entry in ranked}
    basis = ranked[0].basis if ranked else "content_statistics"

    ordered = sorted(plan.tiles, key=lambda t: scores.get(t.index, 0.0), reverse=True)
    selected = ordered[: max(1, top_k)]

    # --- answer on each selected tile -------------------------------------
    outcomes: list[QueryOutcome] = []
    per_tile_masks: list[tuple[Tile, np.ndarray]] = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        with rasterio.open(image_path) as source:
            for tile in selected:
                tile_path = _write_tile(source, tile, directory, tile.index)
                outcome = answer_query(
                    question,
                    [tile_path],
                    modalities=[modality] if modality else None,
                    output_dir=None,
                    config=cfg,
                    write_evidence=False,
                )
                outcomes.append(outcome)
                # Prefer the fused mask when both modalities contributed, else
                # whichever mask-producing tool ran. Mixing masks of different
                # targets into one mosaic would produce a raster that means
                # nothing, so only the first is taken per tile.
                for name in ("fused", "spectral_index", "sar_backscatter", "texture_seg"):
                    mask = outcome.masks.get(name)
                    if mask is not None:
                        per_tile_masks.append((tile, np.asarray(mask, dtype=bool)))
                        break

    # --- mosaic back to full scene resolution (C18) -----------------------
    mask_uris: list[str] = []
    if per_tile_masks:
        mosaic = mosaic_masks(scene.shape, per_tile_masks)
        if output_dir is not None:
            record = write_mask(
                output_dir / "scene_mask.tif",
                mosaic,
                crs=scene.crs,
                transform=scene.transform,
                scene_shape=scene.shape,
                description="mosaicked from tiles",
            )
            mask_uris.append(record.uri)
        else:
            warnings.append(
                "tiles produced masks but no output_dir was given, so the mosaicked "
                "full-scene mask was not written; a mask that is not exported is not a "
                "graded artifact (C18)"
            )
    elif selected:
        warnings.append(
            "no tile produced a mask, so there is nothing to mosaic; the answer rests "
            "on the per-tile text only"
        )

    return TiledOutcome(
        scene_shape=scene.shape,
        selected=[
            {**tile.as_trace_entry(), "score": round(scores.get(tile.index, 0.0), 4),
             "scoring_basis": basis}
            for tile in selected
        ],
        outcomes=outcomes,
        mask_uris=mask_uris,
        warnings=warnings,
    )
