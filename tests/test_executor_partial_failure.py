from unittest.mock import patch

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.agent.trace import validate_trace
from satquery.ingest.band_inventory import BandInventory


def _make_bundle() -> ImageBundle:
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDVI"],
    )
    img = ImageRef(
        scene_id="scene_001",
        path="opt.tif",
        modality="optical",
        crs="EPSG:32644",
        computable_indices=["NDVI"],
    )
    return ImageBundle(
        bundle_id="bundle_partial_fail",
        images=[img],
        band_inventory=inv,
        pair_type="single",
    )


def test_executor_partial_failure_recovers():
    bundle = _make_bundle()

    # Simulate tool raising an exception during execution
    with patch(
        "satquery.tools.spectral_index.execute",
        side_effect=RuntimeError("Simulated memory allocation failure"),
    ):
        res = run_query(bundle, "What is the vegetation extent?")

    # Query should not crash; state should complete
    assert res.state == "succeeded"
    assert any("Simulated memory allocation failure" in w for w in res.warnings)

    # Trace is valid schema v2
    validate_trace(res.trace)
    assert any("failed with exception" in w for w in res.trace["warnings"])
