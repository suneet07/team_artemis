from satquery.agent.bundle import ImageBundle, ImageRef, TileIndex
from satquery.agent.tiling_policy import decide_tile_plan
from satquery.ingest.band_inventory import BandInventory


def _make_tiled_bundle(tile_count: int) -> ImageBundle:
    tiles = [{"tile_id": f"tile_{i}", "score": 1.0 - (i * 0.01)} for i in range(tile_count)]
    tile_index = (
        TileIndex(tile_count=tile_count, tile_size_px=512, overlap_frac=0.1, tiles=tiles)
        if tile_count > 1
        else None
    )
    img = ImageRef(scene_id="s0", path="p0.tif", modality="optical")
    inv = BandInventory(
        bands={"b": 1},
        has_swir=False,
        has_nir=False,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=[],
    )
    return ImageBundle(
        bundle_id="tile_bundle",
        images=[img],
        band_inventory=inv,
        pair_type="single",
        tiles=tile_index,
    )


def test_tiling_policy_63_tiles_learned_budget():
    bundle = _make_tiled_bundle(63)
    plan = decide_tile_plan(bundle)

    assert plan.is_tiled is True
    assert plan.total_tile_count == 63
    assert plan.deterministic_tiles is not None
    assert len(plan.deterministic_tiles) == 63
    assert plan.learned_tiles is not None
    assert len(plan.learned_tiles) == 4  # default budget
    assert "tiles: 4 of 63 selected" in plan.note


def test_tiling_policy_single_tile_whole_scene():
    bundle = _make_tiled_bundle(1)
    plan = decide_tile_plan(bundle)

    assert plan.is_tiled is False
    assert plan.deterministic_tiles is None
    assert plan.learned_tiles is None
    assert plan.selection_method == "whole_scene"
