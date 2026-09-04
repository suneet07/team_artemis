from typing import Any

from satquery.agent.bundle import CoregReport, ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.agent.trace import validate_trace
from satquery.ingest.band_inventory import BandInventory


def _make_dummy_bundle(status: str = "ready") -> ImageBundle:
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
        scene_id="scene_001",
        path="dummy_optical.tif",
        modality="optical",
        crs="EPSG:32644",
        pixel_size_m=10.0,
        native_gsd_m=10.0,
        bands=["blue", "green", "red", "nir"],
        swir_available=False,
        bit_depth=12,
        nodata_frac=0.01,
        computable_indices=["NDVI", "NDWI"],
    )
    coreg = CoregReport(coregistered=True, checks_passed=["crs_match"])
    return ImageBundle(
        bundle_id="bundle_test_01",
        images=[img],
        band_inventory=inv,
        pair_type="single",
        coreg=coreg,
        status=status,
    )


def test_agent_foundation_happy_path():
    bundle = _make_dummy_bundle()
    events: list[tuple[str, dict[str, Any]]] = []

    def sink(event: str, data: dict[str, Any]) -> None:
        events.append((event, data))

    res = run_query(bundle, "What dominates this scene?", emit=sink)

    assert res.state == "succeeded"
    assert res.bundle_id == "bundle_test_01"
    assert res.query_id is not None
    assert res.refusal is None
    assert res.trace != {}

    # Validate schema version 2
    validate_trace(res.trace)
    assert res.trace["schema_version"] == 2
    assert res.trace["graded"]["parameter_check"]["passed"] is True
    assert res.trace["graded"]["outputs"]["answer"] is not None

    # Check SSE events
    event_names = [e[0] for e in events]
    assert event_names == [
        "accepted",
        "router",
        "validator",
        "plan",
        "step_started",
        "step_completed",
        "evidence",
        "fusion",
        "done",
    ]
    assert events[0][1]["query_id"] == res.query_id
    assert events[-1][1]["state"] == "succeeded"


def test_agent_foundation_unready_bundle():
    bundle = _make_dummy_bundle(status="preparing")
    events: list[tuple[str, dict[str, Any]]] = []

    def sink(event: str, data: dict[str, Any]) -> None:
        events.append((event, data))

    res = run_query(bundle, "What changed?", emit=sink)

    assert res.state == "refused"
    assert res.refusal is not None
    assert res.refusal["category"] == "missing_input"
    assert "not ready" in res.refusal["reason"]

    # Refusal trace MUST also be valid schema
    validate_trace(res.trace)
    assert res.trace["graded"]["parameter_check"]["passed"] is False

    # Check SSE events for refusal
    event_names = [e[0] for e in events]
    assert event_names == ["accepted", "done"]
    assert events[1][1]["state"] == "refused"


def test_defaults_applied_recorded_in_trace():
    # Bundle without computable indices forces dummy_tool fallback in planner
    inv = BandInventory(
        bands={"blue": 1},
        has_swir=False,
        has_nir=False,
        is_pan_only=True,
        polarisations=[],
        sar_band=None,
        sensor_hint="pan",
        computable_indices=[],
    )
    img = ImageRef(scene_id="s1", path="pan.tif", modality="optical")
    bundle = ImageBundle(
        bundle_id="b_defaults",
        images=[img],
        band_inventory=inv,
        pair_type="single",
        status="ready",
    )
    res = run_query(bundle, "Describe this scene")
    assert res.state == "succeeded"
    planned_steps = res.trace["graded"]["permitted_parameters"]
    assert len(planned_steps) > 0
    # dummy_tool only had {"index": "ALPHA"} from planner, default scale was applied
    dummy_step = next((s for s in planned_steps if s["tool"] == "dummy_tool"), None)
    if dummy_step:
        assert dummy_step.get("defaults_applied") == ["scale"]
