"""P4 tiling (master plan section 4.4).

The windows arithmetic was already right. What these tests add is the part the
plan cares about and the earlier version quietly violated: a tile has to stay
under the frozen ``tiling.max_pixels`` cap, or the processor downsamples it and
the ground sample distance of everything downstream is wrong with no record of
it. The old test constructed 1024x1024 tiles, which are four times the cap.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from rasterio.transform import from_origin
from rasterio.windows import Window

from satquery.config import CapExceededError, preprocessing_config
from satquery.tiling import mosaic_masks, plan_tiles, score_tiles
from satquery.tiling.tiler import GeoTiler


def mock_dataset(width: int, height: int) -> MagicMock:
    dataset = MagicMock()
    dataset.width = width
    dataset.height = height
    return dataset


def test_tiler_yields_correct_number_of_windows():
    side = preprocessing_config().tiling.tile_side_px
    dataset = mock_dataset(side * 2, side * 2)

    tiler = GeoTiler(tile_size=side, overlap=0)
    tiles = list(tiler.get_tiles(dataset))
    assert len(tiles) == 4
    assert tiles[0]["window"] == Window(col_off=0, row_off=0, width=side, height=side)
    assert tiles[3]["window"] == Window(col_off=side, row_off=side, width=side, height=side)

    tiler2 = GeoTiler(tile_size=side, overlap=side // 2)
    tiles2 = list(tiler2.get_tiles(dataset))
    assert len(tiles2) == 16
    assert tiles2[0]["window"] == Window(col_off=0, row_off=0, width=side, height=side)
    # The last window clamps to the scene edge.
    last = tiles2[-1]["window"]
    assert last.width == side // 2 and last.height == side // 2


def test_tiler_defaults_to_the_frozen_contract():
    cfg = preprocessing_config().tiling
    tiler = GeoTiler()
    assert tiler.tile_size == cfg.tile_side_px
    assert tiler.overlap == round(cfg.tile_side_px * cfg.tile_overlap_fraction)


def test_tile_larger_than_the_cap_is_refused():
    """A 1024 px tile is 4x the frozen cap; the processor would resize it."""
    with pytest.raises(CapExceededError):
        GeoTiler(tile_size=1024, overlap=0)


def test_overlap_must_be_smaller_than_the_tile():
    with pytest.raises(ValueError):
        GeoTiler(tile_size=256, overlap=256)


def test_tile_count_is_capped_so_the_query_sla_holds():
    cfg = preprocessing_config().tiling
    side = cfg.tile_side_px
    # A scene big enough to produce far more than max_tiles.
    huge = mock_dataset(side * (cfg.max_tiles + 40), side * 4)
    tiles = list(GeoTiler(tile_size=side, overlap=0).get_tiles(huge))
    assert len(tiles) == cfg.max_tiles


def test_plan_tiles_geo_indexes_every_tile_and_adds_an_overview():
    transform = from_origin(0.0, 12000.0, 10.0, 10.0)
    plan = plan_tiles((1200, 1500), transform)
    assert plan.overview.is_overview and plan.overview.window.width == 1500
    assert plan.tiles, "a scene larger than one tile must partition"
    # Every tile carries its own transform, so evidence maps back to the scene.
    first = plan.tiles[0]
    assert first.transform.c == transform.c
    offset = next(tile for tile in plan.tiles if tile.bounds_px[1] > 0)
    assert offset.transform.c > transform.c


def test_mosaic_returns_masks_at_full_scene_resolution():
    """C18: a mask left at tile resolution is not a scoreable artifact."""
    plan = plan_tiles((600, 800))
    pieces = [
        (tile, np.ones((tile.bounds_px[2], tile.bounds_px[3]), dtype=bool))
        for tile in plan.tiles[:2]
    ]
    mosaic = mosaic_masks((600, 800), pieces)
    assert mosaic.shape == (600, 800)
    assert mosaic.any()


def test_mosaic_rejects_a_mask_of_the_wrong_size():
    plan = plan_tiles((600, 800))
    with pytest.raises(ValueError):
        mosaic_masks((600, 800), [(plan.tiles[0], np.ones((10, 10), dtype=bool))])


def test_tile_scorer_ranks_by_the_query():
    rng = np.random.default_rng(0)
    textured = rng.random((64, 64))
    flat = np.full((64, 64), 0.5)
    ranked = score_tiles("locate the buildings", [(0, textured), (1, flat)])
    assert ranked[0].index == 0
    ranked = score_tiles("where is the lake", [(0, textured), (1, flat)])
    assert ranked[0].index == 1
