"""Tiling must not change what a scene measures (§12, Rule 7).

The regression these guard against: the executor looped a tool once per tile,
every iteration recomputed the identical whole scene, and the results were
summed -- so a 63-tile scene reported 63x its true area in graded.outputs.
"""

import numpy as np
import pytest

from satquery.agent.bundle import ImageBundle, ImageRef, TileIndex
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.tools import spectral_index, tiling_support

SCENE_PX = 100


def _bundle(tile_count: int) -> ImageBundle:
    """Same scene every time; only how it is carved into tiles varies."""
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDVI", "NDWI"],
    )
    img = ImageRef(
        scene_id="s0",
        path="scene.tif",
        modality="optical",
        crs="EPSG:32644",
        pixel_size_m=10.0,
        computable_indices=["NDVI", "NDWI"],
    )
    tiles = None
    if tile_count > 1:
        # Contiguous row strips covering the scene exactly once.
        step = SCENE_PX / tile_count
        tiles = [
            {
                "tile_id": f"t{i}",
                "bbox_px": [0, round(i * step), SCENE_PX, round((i + 1) * step)],
                "nodata_frac": 0.0,
            }
            for i in range(tile_count)
        ]
    index = (
        TileIndex(tile_count=tile_count, tile_size_px=SCENE_PX, overlap_frac=0.0, tiles=tiles)
        if tiles
        else None
    )
    return ImageBundle(
        bundle_id=f"b_{tile_count}",
        images=[img],
        band_inventory=inv,
        pair_type="single",
        tiles=index,
    )


def _area_of(tile_count: int) -> float:
    res = run_query(_bundle(tile_count), "What is the water extent in this scene?")
    steps = [s for s in res.trace["steps"] if s["tool"] == "spectral_index"]
    assert steps, f"spectral_index did not run for tile_count={tile_count}"
    return float(steps[0]["outputs"]["area_km2"])


@pytest.mark.parametrize("tile_count", [4, 10, 63])
def test_area_is_invariant_to_tile_count(tile_count: int):
    """The same scene measures the same area however it is tiled."""
    baseline = _area_of(1)
    assert baseline > 0.0
    assert _area_of(tile_count) == pytest.approx(baseline, rel=0.02), (
        "area_km2 changed with tile_count -- per-tile results are being summed "
        "instead of mosaicked"
    )


def test_overlapping_tiles_do_not_double_count():
    """Seam pixels belong to two tiles and must still be measured once (§12.4)."""
    bundle = _bundle(4)
    # Grow every tile by 10 px so neighbours overlap.
    for tile in bundle.tiles.tiles:
        x0, y0, x1, y1 = tile["bbox_px"]
        tile["bbox_px"] = [x0, max(0, y0 - 10), x1, min(SCENE_PX, y1 + 10)]
    bundle.tiles.overlap_frac = 0.1

    res = run_query(bundle, "What is the water extent in this scene?")
    step = next(s for s in res.trace["steps"] if s["tool"] == "spectral_index")
    assert float(step["outputs"]["area_km2"]) == pytest.approx(_area_of(1), rel=0.02)


def test_tiled_mask_is_full_scene_not_tile_resolution():
    """Rule 7: the recorded mask is the whole scene, mosaicked back."""
    rasterio = pytest.importorskip("rasterio")
    res = run_query(_bundle(10), "What is the water extent in this scene?")
    masks = [a for a in res.evidence if a.kind == "mask_geotiff"]
    assert masks, "no mask asset produced"
    # download_url is an API endpoint (§18); the file itself is named in stats.
    assert masks[0].download_url.startswith("/assets/")
    with rasterio.open(masks[0].stats["source_path"]) as src:
        assert (src.height, src.width) == (SCENE_PX, SCENE_PX)


def test_tool_without_tiling_support_runs_whole_scene_and_warns():
    """A tool that ignores current_tile must not be looped, and must say so."""
    original = spectral_index.SUPPORTS_TILING
    spectral_index.SUPPORTS_TILING = False
    try:
        res = run_query(_bundle(10), "What is the water extent in this scene?")
        step = next(s for s in res.trace["steps"] if s["tool"] == "spectral_index")
        assert float(step["outputs"]["area_km2"]) == pytest.approx(_area_of(1), rel=0.02)
        assert any("does not support per-tile execution" in w for w in res.warnings)
    finally:
        spectral_index.SUPPORTS_TILING = original


def test_resolve_window_rejects_tiles_that_do_not_locate_a_window():
    """A tile carrying only an id and a score is not a window."""
    assert tiling_support.resolve_window({"tile_id": "t0", "score": 0.9}, (100, 100)) is None
    assert tiling_support.resolve_window({"bbox_px": [0, 0, 0, 0]}, (100, 100)) is None
    assert tiling_support.resolve_window({"bbox_px": [200, 200, 300, 300]}, (100, 100)) is None
    assert tiling_support.resolve_window({"bbox_px": [0, 10, 100, 40]}, (100, 100)) == (
        10,
        0,
        40,
        100,
    )


def test_crop_is_identity_without_a_window():
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert tiling_support.crop(arr, None) is arr
    assert tiling_support.crop(arr, (1, 2, 3, 5)).shape == (2, 3)
