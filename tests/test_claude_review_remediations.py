from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.formatter import format_change_answer
from satquery.agent.graph import run_query
from satquery.agent.task_enum import Task
from satquery.agent.validator import validate_query_compatibility
from satquery.ingest.band_inventory import BandInventory
from satquery.tools import change_map, change_stats, sar_backscatter, spectral_index, tile_scorer


def _make_dummy_bundle(modalities: list[str], crs: str = "EPSG:32644") -> ImageBundle:
    images = [
        ImageRef(
            scene_id=f"scene_{i}",
            path=f"img_{i}.tif",
            modality=m,
            crs=crs,
            nodata_frac=0.0,
            polarisations=["VV"] if m == "sar" else [],
            swir_available=True,
        )
        for i, m in enumerate(modalities)
    ]
    inv = BandInventory(
        bands={"green": 1, "nir": 2, "swir": 3},
        has_swir=True,
        has_nir=True,
        is_pan_only=False,
        polarisations=["VV"] if "sar" in modalities else [],
        sar_band="C" if "sar" in modalities else None,
        sensor_hint=None,
        computable_indices=["NDWI", "NDVI"],
    )
    return ImageBundle(
        bundle_id="b_test",
        images=images,
        band_inventory=inv,
        pair_type="bitemporal" if len(images) > 1 else "single",
    )


def test_fix_1_validator_exposes_all_failures():
    """Verify validator exposes all collected failures, not just the first one."""
    bundle = _make_dummy_bundle(["sar"], crs="")
    res = validate_query_compatibility(bundle, "Produce a green color change map", Task.CHANGE_MAP)
    assert res.passed is False
    assert res.refusal is not None
    assert len(res.failures) >= 3
    # Check that each distinct failure category is preserved in the list
    categories = {f["category"] for f in res.failures}
    assert "missing_input" in categories
    assert "modality_limitation" in categories
    assert "validator" in categories


def test_fix_2_real_geotiff_written_to_disk_and_readable():
    """Verify spectral_index, sar_backscatter, and change_map write real GeoTIFF files."""
    ctx: dict[str, Any] = {"crs": "EPSG:32644"}

    # 1. spectral_index
    res_opt = spectral_index.execute({"index": "NDWI"}, context=ctx)
    p_opt = Path(res_opt["mask_uri"])
    assert p_opt.exists() and p_opt.is_file()
    assert p_opt.stat().st_size > 0
    with rasterio.open(p_opt) as src:
        assert src.crs.to_string() == "EPSG:32644"
        data = src.read(1)
        assert data.shape == (100, 100)

    # 2. sar_backscatter
    res_sar = sar_backscatter.execute({"pol": "VV"}, context=ctx)
    p_sar = Path(res_sar["mask_uri"])
    assert p_sar.exists() and p_sar.is_file()
    assert p_sar.stat().st_size > 0
    with rasterio.open(p_sar) as src:
        assert src.crs.to_string() == "EPSG:32644"
        data = src.read(1)
        assert data.shape == (100, 100)

    # 3. change_map
    res_chg = change_map.execute({"mode": "semantic"}, context=ctx)
    p_chg = Path(res_chg["mask_uri"])
    assert p_chg.exists() and p_chg.is_file()
    assert p_chg.stat().st_size > 0
    with rasterio.open(p_chg) as src:
        assert src.crs.to_string() == "EPSG:32644"
        data = src.read(1)
        assert data.shape == (100, 100)


def test_fix_3_emit_node_results_preserved():
    """Verify emit_node does not overwrite execution results."""
    bundle = _make_dummy_bundle(["optical"])
    res = run_query(bundle, "Where are the water bodies in this scene?")
    assert res.state == "succeeded"
    assert res.evidence is not None
    assert len(res.evidence) > 0
    # Asset bytes measured from real GeoTIFF file
    assert res.evidence[0].bytes > 0


def test_fix_4_tile_scorer_data_driven_and_uniform():
    """Verify tile_scorer uses data-driven mask partitioning or uniform fallback."""
    # When a mask exists in mask_cache, scores reflect mask presence
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[:50, :50] = 1  # top-left quadrant is foreground
    ctx_with_mask = {"mask_cache": {"NDWI": test_mask}}
    res_mask = tile_scorer.execute({"top_k": 4}, context=ctx_with_mask)
    scores = [s["score"] for s in res_mask["scores"]]
    assert len(scores) == 4
    # The top-left quadrant tile should have a higher score than the others
    assert max(scores) > min(scores)

    # When no tiles and no mask exist, assign uniform scores without arbitrary formulas
    ctx_empty: dict[str, Any] = {}
    res_uniform = tile_scorer.execute({"top_k": 4}, context=ctx_empty)
    scores_uniform = [s["score"] for s in res_uniform["scores"]]
    assert len(scores_uniform) == 4
    assert all(s == 0.80 for s in scores_uniform)


def test_fix_5_change_stats_baseline_unknown_when_no_t1():
    """Verify change_stats returns None for area_before_km2 when no T1 mask is available."""
    # 1. No baseline mask in mask_cache
    ctx_no_t1: dict[str, Any] = {"mask_cache": {}}
    res = change_stats.execute({"target_class": "urban"}, context=ctx_no_t1)
    stats = res["stats"]
    assert stats["area_before_km2"] is None
    assert stats["area_after_km2"] is None
    ans = format_change_answer(stats)
    assert "baseline area before change: unknown" in ans

    # 2. When T1 baseline mask is present in mask_cache
    t1_mask = np.ones((100, 100), dtype=np.uint8) * 1  # 10000 px = 1.0 km2 at 10m
    ctx_with_t1 = {"mask_cache": {"spectral_index": t1_mask}}
    res_t1 = change_stats.execute({"target_class": "urban"}, context=ctx_with_t1)
    stats_t1 = res_t1["stats"]
    assert stats_t1["area_before_km2"] == 1.0
    ans_t1 = format_change_answer(stats_t1)
    assert "area changed from 1.00 km²" in ans_t1
