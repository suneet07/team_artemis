from typing import Any

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory


def _make_optical_bundle() -> ImageBundle:
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
        scene_id="scene_001",
        path="opt.tif",
        modality="optical",
        crs="EPSG:32644",
        native_gsd_m=10.0,
        pixel_size_m=10.0,
        bands=["blue", "green", "red", "nir"],
        swir_available=False,
        bit_depth=12,
        nodata_frac=0.01,
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id="bundle_sse_test",
        images=[img],
        band_inventory=inv,
        pair_type="single",
    )


def test_sse_sequence_success_path():
    bundle = _make_optical_bundle()
    events: list[str] = []

    def sink(event: str, data: dict[str, Any]) -> None:
        events.append(event)

    res = run_query(bundle, "What dominates this scene?", emit=sink)
    assert res.state == "succeeded"

    # Frozen sequence: accepted -> router -> validator -> plan ->
    # (step_started -> step_completed -> evidence)* -> fusion -> done
    assert events == [
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


def test_sse_sequence_refusal_at_validator():
    bundle = _make_optical_bundle()
    events: list[str] = []

    def sink(event: str, data: dict[str, Any]) -> None:
        events.append(event)

    # Change query on 1-image bundle triggers V1 refusal at validator
    res = run_query(bundle, "How has the land cover changed?", emit=sink)
    assert res.state == "refused"

    # Refusal ends after validator, followed by done with state "refused"
    assert events == ["accepted", "router", "validator", "done"]


def test_sse_sequence_refusal_unready_bundle():
    bundle = _make_optical_bundle()
    bundle.status = "preparing"
    events: list[str] = []

    def sink(event: str, data: dict[str, Any]) -> None:
        events.append(event)

    res = run_query(bundle, "What is here?", emit=sink)
    assert res.state == "refused"
    assert events == ["accepted", "done"]
