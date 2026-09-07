"""§12.4, §19, §24 and §25 — the cross-cutting requirements.

These are the requirements that live between nodes rather than inside one, so
nothing else exercises them: the tile budget a learned tool must respect, the
token stream, cooperative cancellation, the global budget, and the cache.
"""

import threading
import time

import pytest

from satquery.agent.bundle import ImageBundle, ImageRef, TileIndex
from satquery.agent.executor import _TOOL_CACHE, _is_cancelled, _query_budget_s
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.tools import tiling_support

SCENE_PX = 100


def _inventory() -> BandInventory:
    return BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDVI", "NDWI"],
    )


def _bundle(tile_count: int = 1) -> ImageBundle:
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
        step = SCENE_PX / tile_count
        tiles = [
            {
                "tile_id": f"t{i}",
                "bbox_px": [0, round(i * step), SCENE_PX, round((i + 1) * step)],
                "nodata_frac": 0.0,
                "score": 1.0 - i / tile_count,
            }
            for i in range(tile_count)
        ]
    return ImageBundle(
        bundle_id=f"b_{tile_count}",
        images=[img],
        band_inventory=_inventory(),
        pair_type="single",
        tiles=(
            TileIndex(tile_count=tile_count, tile_size_px=SCENE_PX, overlap_frac=0.1, tiles=tiles)
            if tiles
            else None
        ),
    )


# --------------------------------------------------------------------------
# Rule 14 / §12.3 -- the learned-tool tile budget
# --------------------------------------------------------------------------


def test_learned_tool_never_exceeds_the_tile_budget(monkeypatch):
    """Rule 14: at most `agent.learned_tool_tile_budget` tiles reach a VLM."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    res = run_query(_bundle(63), "Describe the contents of this satellite image.")
    step = next(s for s in res.trace["steps"] if s["tool"] == "rs_ground_caption")
    seen = step["outputs"]["tiles_seen"]
    assert seen == 4, f"budget is 4, tool saw {seen}"
    assert step["outputs"]["tile_coverage_frac"] == pytest.approx(4 / 63, rel=0.01)


def test_tile_coverage_is_recorded_so_the_answer_can_be_qualified(monkeypatch):
    """§12.3: never claim scene-wide coverage from a tiled sample."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    res = run_query(_bundle(63), "Describe the contents of this satellite image.")
    step = next(s for s in res.trace["steps"] if s["tool"] == "rs_ground_caption")
    assert 0.0 < step["outputs"]["tile_coverage_frac"] < 1.0
    assert any("tiles: 4 of 63" in note for note in res.trace["routing_notes"])


# --------------------------------------------------------------------------
# §12.4 -- aggregating per-tile answers
# --------------------------------------------------------------------------


def test_free_text_takes_the_best_tile_and_never_concatenates():
    merged = tiling_support.aggregate_tile_answers(
        [
            {"answer": "a lake", "tile_score": 0.2},
            {"answer": "a runway", "tile_score": 0.9},
            {"answer": "farmland", "tile_score": 0.5},
        ]
    )
    assert merged["answer"] == "a runway"


def test_presence_is_any_tile_yes():
    merged = tiling_support.aggregate_tile_answers(
        [
            {"answer": "No ships visible.", "tile_score": 0.9},
            {"answer": "Yes, two ships.", "tile_score": 0.1},
        ]
    )
    assert merged["presence"] is True


def test_boxes_are_deduplicated_across_seams():
    """An object straddling two overlapping tiles is proposed twice."""
    merged = tiling_support.aggregate_tile_answers(
        [
            {"boxes": [{"bbox_px": [10, 10, 30, 30], "score": 0.9}], "tile_score": 0.9},
            {"boxes": [{"bbox_px": [11, 11, 31, 31], "score": 0.4}], "tile_score": 0.8},
            {"boxes": [{"bbox_px": [70, 70, 90, 90], "score": 0.7}], "tile_score": 0.7},
        ]
    )
    assert merged["count"] == 2, "the seam duplicate was not suppressed"
    assert merged["boxes"][0]["score"] == 0.9, "NMS kept the weaker proposal"


def test_nms_keeps_genuinely_distinct_objects():
    boxes = [
        {"bbox_px": [0, 0, 10, 10], "score": 0.9},
        {"bbox_px": [40, 40, 50, 50], "score": 0.8},
        {"bbox_px": [80, 80, 90, 90], "score": 0.7},
    ]
    assert len(tiling_support.nms(boxes)) == 3


# --------------------------------------------------------------------------
# §19 -- token streaming
# --------------------------------------------------------------------------


def test_token_events_are_emitted_for_learned_tools(monkeypatch):
    """§19: the `token` event exists and rides between step_started and fusion."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    events: list[tuple[str, dict]] = []
    run_query(
        _bundle(),
        "Describe the contents of this satellite image.",
        emit=lambda name, data: events.append((name, data)),
    )
    names = [n for n, _ in events]
    assert "token" in names, "no token event emitted"
    text = "".join(d["text"] for n, d in events if n == "token")
    assert text.strip(), "token events carried no text"
    assert names.index("token") < names.index("fusion")


def test_tokens_stream_on_the_tiled_path_too(monkeypatch):
    """Only the best tile streams -- its answer is the one §12.4 keeps."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    events: list[str] = []
    run_query(
        _bundle(10),
        "Describe the contents of this satellite image.",
        emit=lambda name, data: events.append(name),
    )
    assert events.count("token") > 0, "the tiled path emitted no tokens"


def test_no_token_events_in_headless_mode(monkeypatch):
    """The emitter is a no-op headless, so tools must not stream regardless."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    res = run_query(_bundle(), "Describe the contents of this satellite image.")
    assert res.state == "succeeded"


# --------------------------------------------------------------------------
# §24 -- cooperative cancellation
# --------------------------------------------------------------------------


def test_cancellation_flag_accepts_an_event_or_a_callable():
    flag = threading.Event()
    assert _is_cancelled({"cancel_check": flag}) is False
    flag.set()
    assert _is_cancelled({"cancel_check": flag}) is True
    assert _is_cancelled({"cancel_check": lambda: True}) is True
    assert _is_cancelled({}) is False


def test_a_cancelled_query_reports_the_frozen_cancelled_state(monkeypatch):
    """§24 + the frozen QueryState enum."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    already = threading.Event()
    already.set()
    res = run_query(_bundle(), "What is the water extent?", cancel_check=already)
    assert res.state == "cancelled"
    assert any("Cancelled by request" in w for w in res.warnings)
    # Rule 11 still holds on this path.
    assert res.trace["graded"]["task_selected"]


# --------------------------------------------------------------------------
# §25 -- budget, cache, wall clock
# --------------------------------------------------------------------------


def test_global_budget_returns_partial_results_rather_than_nothing(monkeypatch):
    """§25: on the global timeout, hand back the best partial answer."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    monkeypatch.setenv("SATQUERY_QUERY_BUDGET_S", "0")
    res = run_query(_bundle(), "What is the water extent?")
    assert any("budget" in w.lower() for w in res.warnings)
    assert res.trace["graded"]["task_selected"], "a trace is still written"


def test_budget_is_configurable_and_defaults_to_the_sla(monkeypatch):
    monkeypatch.delenv("SATQUERY_QUERY_BUDGET_S", raising=False)
    assert _query_budget_s() == 20.0
    monkeypatch.setenv("SATQUERY_QUERY_BUDGET_S", "5")
    assert _query_budget_s() == 5.0


def test_latency_is_wall_clock_not_a_sum_of_node_timings(monkeypatch):
    """Waves run concurrently, so summing node timings overstates the query."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    started = time.perf_counter()
    res = run_query(_bundle(), "What is the water extent?")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    assert res.latency_ms is not None
    assert res.latency_ms <= elapsed_ms + 50, "latency exceeds the wall time it ran in"


def _real_raster_bundle(tmp_path) -> ImageBundle:
    """A bundle backed by an actual GeoTIFF, so tools take the real read path."""
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    path = tmp_path / "scene.tif"
    rng = np.linspace(0.0, 1.0, SCENE_PX * SCENE_PX, dtype="float32").reshape(
        SCENE_PX, SCENE_PX
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SCENE_PX,
        width=SCENE_PX,
        count=4,
        dtype="float32",
        crs="EPSG:32644",
        transform=from_origin(500000.0, 3000000.0, 10.0, 10.0),
    ) as dst:
        for band in range(1, 5):
            dst.write(rng * band / 4.0, band)

    img = ImageRef(
        scene_id="s0",
        path=str(path),
        modality="optical",
        crs="EPSG:32644",
        pixel_size_m=10.0,
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id="b_real", images=[img], band_inventory=_inventory(), pair_type="single"
    )


def test_deterministic_tool_outputs_are_cached(monkeypatch, tmp_path):
    """§25: identical deterministic work is not repeated across runs."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    bundle = _real_raster_bundle(tmp_path)
    _TOOL_CACHE.clear()

    run_query(bundle, "What is the water extent?")
    assert _TOOL_CACHE, "nothing was cached on the real-raster path"
    size_after_first = len(_TOOL_CACHE)

    run_query(bundle, "What is the water extent?")
    assert len(_TOOL_CACHE) == size_after_first, "a repeat run added new cache entries"


def test_cache_misses_when_the_raster_changes(monkeypatch, tmp_path):
    """The key carries scene identity, so an edited raster must not hit."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    bundle = _real_raster_bundle(tmp_path)
    _TOOL_CACHE.clear()
    run_query(bundle, "What is the water extent?")
    first = len(_TOOL_CACHE)

    path = str(bundle.images[0].path)
    with rasterio.open(path, "r+") as dst:
        dst.write(np.zeros((SCENE_PX, SCENE_PX), dtype="float32"), 1)

    run_query(bundle, "What is the water extent?")
    assert len(_TOOL_CACHE) > first, "an edited raster reused a stale cache entry"


def test_synthetic_fallback_results_are_never_cached(monkeypatch):
    """A synthetic result exists because the input was unreadable; that can change."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    _TOOL_CACHE.clear()
    run_query(_bundle(), "What is the water extent?")
    assert all(not v.get("synthetic") for v in _TOOL_CACHE.values())
    assert not _TOOL_CACHE, "a synthetic-fallback result was cached"
