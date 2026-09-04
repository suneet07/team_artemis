from satquery.agent.bundle import CoregReport, ImageBundle, ImageRef
from satquery.agent.refusals import ACTION_ADD_2ND_IMAGE
from satquery.agent.task_enum import Task
from satquery.agent.validator import validate_query_compatibility
from satquery.ingest.band_inventory import BandInventory


def _make_bundle(
    modalities: list[str],
    coreg: CoregReport | None = None,
    has_swir: bool = False,
    is_pan_only: bool = False,
    pols: list[str] | None = None,
    nodata_frac: float = 0.0,
    crs: str | None = "EPSG:32644",
) -> ImageBundle:
    images = []
    for idx, mod in enumerate(modalities):
        images.append(
            ImageRef(
                scene_id=f"scene_{idx}",
                path=f"img_{idx}.tif",
                modality=mod,
                crs=crs,
                nodata_frac=nodata_frac,
                polarisations=pols or ([] if mod == "optical" else ["VV"]),
                swir_available=has_swir,
            )
        )
    inv = BandInventory(
        bands={"b": 1, "g": 2, "r": 3},
        has_swir=has_swir,
        has_nir=True,
        is_pan_only=is_pan_only,
        polarisations=pols or [],
        sar_band="C" if "sar" in modalities else None,
        sensor_hint=None,
        computable_indices=["NDVI"] if not is_pan_only else [],
    )
    return ImageBundle(
        bundle_id="test_bundle",
        images=images,
        band_inventory=inv,
        pair_type="single" if len(images) == 1 else "bitemporal",
        coreg=coreg,
    )


def test_v1_change_query_with_one_image_refuses():
    bundle = _make_bundle(["optical"])
    res = validate_query_compatibility(
        bundle, "How has the land cover changed?", Task.CHANGE_DESCRIPTION
    )
    assert res.passed is False
    assert res.refusal is not None
    assert res.refusal["category"] == "missing_input"
    assert res.refusal["remedy"]["action"] == ACTION_ADD_2ND_IMAGE
    assert "remedy" in res.refusal


def test_v2_coregistration_failure_refuses():
    coreg = CoregReport(coregistered=False, rmse_px=2.35)
    bundle = _make_bundle(["optical", "optical"], coreg=coreg)
    res = validate_query_compatibility(bundle, "What changed?", Task.CHANGE_MAP)
    assert res.passed is False
    assert res.refusal is not None
    assert res.refusal["category"] == "validator"
    assert "2.35 px" in res.refusal["reason"]
    assert res.refusal["remedy"]["action"] == "reupload"


def test_v3_sar_only_colour_question_refuses():
    bundle = _make_bundle(["sar"])
    res = validate_query_compatibility(
        bundle, "What is the green vegetation color?", Task.SINGLE_VQA
    )
    assert res.passed is False
    assert res.refusal is not None
    assert res.refusal["category"] == "modality_limitation"
    assert res.refusal["remedy"]["action"] == "add_optical"
    assert "suggested_questions" in res.refusal["remedy"]


def test_v4_absent_swir_band_notes_reroute():
    bundle = _make_bundle(["optical"], has_swir=False)
    res = validate_query_compatibility(bundle, "What is the water extent?", Task.SINGLE_VQA)
    assert res.passed is True
    assert any("SWIR" in n for n in res.routing_notes)


def test_v5_pan_only_spectral_question_notes_texture_reroute():
    bundle = _make_bundle(["optical"], is_pan_only=True)
    res = validate_query_compatibility(
        bundle, "Assess the vegetation health with NDVI", Task.SINGLE_VQA
    )
    assert res.passed is True
    assert any("texture" in n for n in res.routing_notes)


def test_v6_single_pol_polarimetric_degrades():
    bundle = _make_bundle(["sar"], pols=["VV"])
    res = validate_query_compatibility(
        bundle, "Perform polarimetric decomposition", Task.SINGLE_VQA
    )
    assert res.passed is True
    assert any("Single-polarisation" in w for w in res.warnings)


def test_v7_high_nodata_warns():
    bundle = _make_bundle(["optical"], nodata_frac=0.35)
    res = validate_query_compatibility(bundle, "What is here?", Task.SINGLE_VQA)
    assert res.passed is True
    assert any("nodata" in w.lower() for w in res.warnings)


def test_v8_missing_crs_on_change_map_refuses():
    bundle = _make_bundle(["optical", "optical"], crs="")
    res = validate_query_compatibility(bundle, "Produce a change map", Task.CHANGE_MAP)
    assert res.passed is False
    assert res.refusal is not None
    assert res.refusal["category"] == "validator"
    assert res.refusal["remedy"]["action"] == "reupload"
    # Verify that failures for both images were collected without early break
    assert len(res.failures) == 2
    assert "scene_0" in res.failures[0]["reason"]
    assert "scene_1" in res.failures[1]["reason"]


def test_v9_out_of_vocab_grounding_routes_to_fallback():
    bundle = _make_bundle(["optical"])
    res = validate_query_compatibility(
        bundle, "Locate the bridge across the river", Task.SINGLE_GROUNDING
    )
    assert res.passed is True
    assert any("object_box_fallback" in n for n in res.routing_notes)


def test_validator_collects_all_simultaneous_failures():
    # Bundle has 1 SAR image with no CRS, querying change map with color question
    # Triggers V1 (single image for change), V3 (SAR + color), and V8 (missing CRS for change_map)
    bundle = _make_bundle(["sar"], crs="")
    res = validate_query_compatibility(
        bundle, "Produce green change map", Task.CHANGE_MAP
    )
    assert res.passed is False
    assert res.refusal is not None
    # Must have collected multiple distinct failures (V1, V3, V8)
    assert len(res.failures) >= 3
    categories = [f["category"] for f in res.failures]
    assert "missing_input" in categories  # V1
    assert "modality_limitation" in categories  # V3
    assert "validator" in categories  # V8

