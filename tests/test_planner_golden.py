from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.planner import plan_query
from satquery.agent.task_enum import Task
from satquery.ingest.band_inventory import BandInventory


def _make_optical_bundle(has_swir: bool = True) -> ImageBundle:
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5}
        if has_swir
        else {"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=has_swir,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDVI", "NDWI", "MNDWI", "NDBI"] if has_swir else ["NDVI", "NDWI"],
    )
    img = ImageRef(scene_id="s0", path="opt.tif", modality="optical", swir_available=has_swir)
    return ImageBundle(
        bundle_id="b_opt",
        images=[img],
        band_inventory=inv,
        pair_type="single",
    )


def test_planner_single_grounding_in_vocab():
    bundle = _make_optical_bundle()
    plan, notes = plan_query(Task.SINGLE_GROUNDING, bundle, "Where is the aircraft in this image?")

    tools = [s["tool"] for s in plan]
    assert "spectral_index" in tools
    assert "centroid_prior" in tools

    # centroid_prior must depend on spectral_index
    centroid_step = next(s for s in plan if s["tool"] == "centroid_prior")
    assert centroid_step["depends_on"] == ["spectral_index"]


def test_planner_single_grounding_out_of_vocab():
    bundle = _make_optical_bundle()
    plan, notes = plan_query(Task.SINGLE_GROUNDING, bundle, "Locate the bridge across the river")

    tools = [s["tool"] for s in plan]
    assert "texture_seg" in tools
    assert "object_box_fallback" in tools
    assert any("Out-of-vocabulary" in n for n in notes)

    fallback_step = next(s for s in plan if s["tool"] == "object_box_fallback")
    assert fallback_step["params"]["target_class"] == "bridge"


def test_planner_crossmodal_extraction():
    opt_img = ImageRef(scene_id="opt", path="opt.tif", modality="optical")
    sar_img = ImageRef(scene_id="sar", path="sar.tif", modality="sar", polarisations=["VV"])
    inv = BandInventory(
        bands={"green": 2, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=["VV"],
        sar_band="C",
        sensor_hint=None,
        computable_indices=["NDWI"],
    )
    bundle = ImageBundle(
        bundle_id="b_cross",
        images=[opt_img, sar_img],
        band_inventory=inv,
        pair_type="crossmodal",
    )

    plan, notes = plan_query(
        Task.CROSSMODAL_EXTRACTION, bundle, "Identify urban features across modalities."
    )
    tools = [s["tool"] for s in plan]
    assert "spectral_index" in tools
    assert "sar_backscatter" in tools


def test_planner_single_vqa_reaches_rs_vqa():
    bundle = _make_optical_bundle()
    plan, notes = plan_query(
        Task.SINGLE_VQA, bundle, "How many ships are docked in the harbour?"
    )
    tools = [s["tool"] for s in plan]
    assert "rs_vqa" in tools
    assert "spectral_index" not in tools


def test_planner_single_vqa_spectral_query():
    bundle = _make_optical_bundle()
    plan, notes = plan_query(
        Task.SINGLE_VQA, bundle, "What is the water extent in this scene?"
    )
    tools = [s["tool"] for s in plan]
    assert "spectral_index" in tools
    assert "rs_vqa" not in tools

