from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.planner import plan_query
from satquery.agent.task_enum import Task
from satquery.ingest.band_inventory import BandInventory


def test_replan_substitutes_rejected_index():
    inv = BandInventory(
        bands={"green": 2, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDWI"],
    )
    bundle = ImageBundle(
        bundle_id="b_no_swir",
        images=[ImageRef(scene_id="opt", path="opt.tif", modality="optical")],
        band_inventory=inv,
        pair_type="single",
    )

    # Initial plan with NDBI rejected because SWIR is absent
    initial_plan = [
        {"tool": "spectral_index", "params": {"index": "NDBI"}, "depends_on": []}
    ]
    rejection = ["parameter 'index'='NDBI' requires band 'swir' which the source does not provide"]

    replanned, notes = plan_query(
        Task.SINGLE_VQA,
        bundle,
        "Extract built-up",
        replan_count=1,
        gate_rejected=rejection,
    )

    # Second plan differs from first
    assert replanned != initial_plan
    assert replanned[0]["params"]["index"] == "NDWI"
    assert any("Substituted" in n for n in notes)


def test_replan_empty_or_double_failure_refuses_parameter_gate():
    inv = BandInventory(
        bands={"green": 2},
        has_swir=False,
        has_nir=False,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=[],  # No computable indices at all
    )
    bundle = ImageBundle(
        bundle_id="b_empty",
        images=[ImageRef(scene_id="opt", path="opt.tif", modality="optical")],
        band_inventory=inv,
        pair_type="single",
    )

    # Question with invalid parameter that fails
    rejection = ["parameter 'index'='NDBI' requires band 'swir' which the source does not provide"]
    replanned, notes = plan_query(
        Task.SINGLE_VQA,
        bundle,
        "Extract index",
        replan_count=1,
        gate_rejected=rejection,
    )
    assert any("Dropped spectral_index" in n for n in notes)


def test_replan_drops_unknown_parameter_materially_differing():
    inv = BandInventory(
        bands={"green": 2, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDWI"],
    )
    bundle = ImageBundle(
        bundle_id="b_test",
        images=[ImageRef(scene_id="opt", path="opt.tif", modality="optical")],
        band_inventory=inv,
        pair_type="single",
    )

    first_plan = [
        {"tool": "spectral_index", "params": {"index": "NDWI", "thresh": 0.5}, "depends_on": []}
    ]
    rejection = ["unknown parameter 'thresh' not in manifest"]

    replanned, notes = plan_query(
        Task.SINGLE_VQA,
        bundle,
        "Extract water",
        replan_count=1,
        gate_rejected=rejection,
        previous_plan=first_plan,
    )

    assert replanned != first_plan
    assert "thresh" not in replanned[0]["params"]
    assert any("Dropping unknown parameter 'thresh'" in n for n in notes)


def test_replan_drops_or_resets_out_of_range_parameter():
    inv = BandInventory(
        bands={"green": 2, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDWI"],
    )
    bundle = ImageBundle(
        bundle_id="b_test",
        images=[ImageRef(scene_id="opt", path="opt.tif", modality="optical")],
        band_inventory=inv,
        pair_type="single",
    )

    first_plan = [
        {
            "tool": "spectral_index",
            "params": {"index": "NDWI", "threshold_value": 47},
            "depends_on": [],
        }
    ]
    rejection = ["parameter 'threshold_value'=47 outside permitted range [-1.0, 1.0]"]

    replanned, notes = plan_query(
        Task.SINGLE_VQA,
        bundle,
        "Extract water",
        replan_count=1,
        gate_rejected=rejection,
        previous_plan=first_plan,
    )

    assert replanned != first_plan
    assert "threshold_value" not in replanned[0]["params"]
    assert any("threshold_value" in n for n in notes)

