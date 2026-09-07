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


def test_executor_all_tools_failure_fails_with_internal():
    bundle = _make_bundle()

    # When all tools in the plan fail, query state is 'failed' per §10 N6 and N-4
    with patch(
        "satquery.tools.spectral_index.execute",
        side_effect=RuntimeError("Simulated memory allocation failure"),
    ):
        res = run_query(bundle, "What is the vegetation extent?")

    # Query should not crash; state should be failed (not succeeded)
    assert res.state == "failed"
    assert any("Simulated memory allocation failure" in w for w in res.warnings)

    # Trace is valid schema v2
    validate_trace(res.trace)
    assert any("failed with exception" in w for w in res.trace["warnings"])


def test_executor_partial_failure_recovers_with_surviving_tool():
    # Crossmodal pair with optical and SAR
    img_opt = ImageRef(
        scene_id="s_opt",
        path="opt.tif",
        modality="optical",
        crs="EPSG:32644",
        computable_indices=["NDWI"],
    )
    img_sar = ImageRef(
        scene_id="s_sar",
        path="sar.tif",
        modality="sar",
        crs="EPSG:32644",
        polarisations=["VV"],
    )
    inv = BandInventory(
        bands={"red": 1, "green": 2, "blue": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=["VV"],
        sar_band="C",
        computable_indices=["NDWI"],
    )
    bundle = ImageBundle(
        bundle_id="b_crossmodal_fail",
        images=[img_opt, img_sar],
        band_inventory=inv,
        pair_type="crossmodal",
    )

    # Deliberately fail spectral_index while sar_backscatter survives
    with patch(
        "satquery.tools.spectral_index.execute",
        side_effect=RuntimeError("Simulated optical failure"),
    ):
        res = run_query(bundle, "Extract water footprint combining optical and sar")

    # Partial failure is survivable: sar_backscatter succeeded
    assert res.state == "succeeded"
    assert any("Simulated optical failure" in w for w in res.warnings)
    validate_trace(res.trace)
