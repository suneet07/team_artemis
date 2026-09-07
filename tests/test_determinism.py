import difflib
import json

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.executor import asset_id_for
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.tools import (
    centroid_prior,
    object_box_fallback,
    sar_backscatter,
    spectral_index,
    texture_seg,
)


def _make_test_bundle() -> ImageBundle:
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint="optical_test",
        computable_indices=["NDVI", "NDWI"],
    )
    img = ImageRef(
        scene_id="scene_det",
        path="det_test.tif",
        modality="optical",
        crs="EPSG:32644",
        pixel_size_m=10.0,
        bands=["blue", "green", "red", "nir"],
        swir_available=False,
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id="bundle_det",
        images=[img],
        band_inventory=inv,
        pair_type="single",
        status="ready",
    )


def test_spectral_index_pure_determinism():
    params = {"index": "NDWI", "threshold_method": "otsu"}
    out1 = spectral_index.execute(params, context={})
    out2 = spectral_index.execute(params, context={})
    assert out1 == out2
    assert out1["threshold_value"] == out2["threshold_value"]
    assert out1["area_km2"] == out2["area_km2"]
    assert out1["threshold_method"] == out2["threshold_method"]


def test_centroid_prior_pure_determinism():
    ctx: dict = {}
    spectral_index.execute({"index": "NDWI"}, context=ctx)

    out1 = centroid_prior.execute({"min_area_px": 10}, context=ctx)
    out2 = centroid_prior.execute({"min_area_px": 10}, context=ctx)
    assert out1 == out2
    assert out1["centroid"] == out2["centroid"]
    assert out1["area_px"] == out2["area_px"]


def test_sar_backscatter_pure_determinism():
    params = {"pol": "VV", "threshold_method": "otsu"}
    out1 = sar_backscatter.execute(params, context={})
    out2 = sar_backscatter.execute(params, context={})
    assert out1 == out2
    assert out1["threshold_db"] == out2["threshold_db"]
    assert out1["area_km2"] == out2["area_km2"]


def test_texture_seg_and_box_fallback_determinism():
    ctx: dict = {}
    tex1 = texture_seg.execute({"window_size": 5}, context=ctx)
    tex2 = texture_seg.execute({"window_size": 5}, context=ctx)
    assert tex1 == tex2

    box1 = object_box_fallback.execute({"target_class": "bridge"}, context=ctx)
    box2 = object_box_fallback.execute({"target_class": "bridge"}, context=ctx)
    assert box1 == box2
    assert box1["boxes"] == box2["boxes"]


def test_e2e_query_determinism():
    bundle = _make_test_bundle()
    query = "What is the vegetation and water extent?"

    res1 = run_query(bundle, query, query_id="fixed_q1")
    res2 = run_query(bundle, query, query_id="fixed_q2")

    # Graded outputs and planned steps must match exactly
    assert res1.state == res2.state == "succeeded"
    assert res1.answer == res2.answer
    assert res1.confidence == res2.confidence

    trace1_steps = res1.trace["graded"]["permitted_parameters"]
    trace2_steps = res2.trace["graded"]["permitted_parameters"]
    assert trace1_steps == trace2_steps


def _masked(trace: dict) -> str:
    """The trace with the three fields §25 permits to vary removed."""
    t = json.loads(json.dumps(trace))
    t.pop("query_id", None)
    t.pop("timestamp", None)
    for step in t.get("steps", []):
        step.pop("latency_ms", None)
    return json.dumps(t, indent=1, sort_keys=True)


def test_whole_trace_is_byte_identical_across_runs():
    """§25: same bundle + same question => byte-identical trace, modulo id/ts/latency.

    The criterion is about the *whole* trace, not the answer. This previously
    passed while `evidence` carried a fresh uuid4 asset id on every run, because
    nothing compared the two documents.
    """
    bundle = _make_test_bundle()
    query = "What is the vegetation and water extent?"

    first = _masked(run_query(bundle, query, query_id="fixed_q1").trace)
    second = _masked(run_query(bundle, query, query_id="fixed_q2").trace)

    if first != second:
        diff = chr(10).join(
            difflib.unified_diff(
                first.splitlines(), second.splitlines(), "run1", "run2", lineterm="", n=1
            )
        )
        raise AssertionError("trace differs between identical runs:" + chr(10) + diff)


def test_asset_ids_are_derived_from_content_not_random():
    """Asset ids reach the trace, so they must be reproducible."""
    bundle = _make_test_bundle()
    query = "What is the vegetation and water extent?"

    ids1 = [a.asset_id for a in run_query(bundle, query).evidence]
    ids2 = [a.asset_id for a in run_query(bundle, query).evidence]
    assert ids1 and ids1 == ids2

    # Distinct identities still get distinct ids.
    assert asset_id_for("spectral_index", "mask_geotiff", "a.tif") != asset_id_for(
        "spectral_index", "mask_geotiff", "b.tif"
    )
    assert asset_id_for("spectral_index", "mask_geotiff", "a.tif") != asset_id_for(
        "sar_backscatter", "mask_geotiff", "a.tif"
    )
