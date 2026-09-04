import json

import jsonschema

from satquery.agent.bundle import CoregReport, ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.paths import TRACE_SCHEMA_PATH


def _get_trace_schema():
    return json.loads(TRACE_SCHEMA_PATH.read_text(encoding="utf-8"))


def _make_bundle(modalities, coreg=None, crs="EPSG:32644", has_swir=True):
    images = []
    for idx, mod in enumerate(modalities):
        images.append(
            ImageRef(
                scene_id=f"scene_{idx}",
                path=f"img_{idx}.tif",
                modality=mod,
                crs=crs,
                nodata_frac=0.01,
                polarisations=["VV"] if mod == "sar" else [],
                swir_available=has_swir,
            )
        )
    bands = (
        {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5}
        if has_swir
        else {"blue": 1, "green": 2, "red": 3, "nir": 4}
    )
    indices = (
        ["NDVI", "NDWI", "MNDWI", "NDBI"] if has_swir else ["NDVI", "NDWI"]
    )
    inv = BandInventory(
        bands=bands,
        has_swir=has_swir,
        has_nir=True,
        is_pan_only=False,
        polarisations=["VV"] if "sar" in modalities else [],
        sar_band="C" if "sar" in modalities else None,
        sensor_hint=None,
        computable_indices=indices,
    )
    pair_type = (
        "single"
        if len(images) == 1
        else ("crossmodal" if len(set(modalities)) > 1 else "bitemporal")
    )
    return ImageBundle(
        bundle_id="test_bundle",
        images=images,
        band_inventory=inv,
        pair_type=pair_type,
        coreg=coreg,
    )


def test_graph_path_success_single_optical():
    schema = _get_trace_schema()
    bundle = _make_bundle(["optical"])
    res = run_query(bundle, "What is the water extent in this scene?")
    assert res.state == "succeeded"
    assert res.answer is not None
    assert res.confidence is not None
    assert res.trace is not None
    jsonschema.validate(instance=res.trace, schema=schema)


def test_graph_path_success_crossmodal_fusion():
    schema = _get_trace_schema()
    bundle = _make_bundle(["optical", "sar"])
    res = run_query(bundle, "Extract water bodies using optical and SAR")
    assert res.state == "succeeded"
    assert res.trace is not None
    assert "agreement" in res.trace
    jsonschema.validate(instance=res.trace, schema=schema)


def test_graph_path_success_change_detection():
    schema = _get_trace_schema()
    bundle = _make_bundle(["optical", "optical"])
    res = run_query(bundle, "How much water area changed between before and after?")
    assert res.state == "succeeded"
    assert "change" in res.answer.lower()
    jsonschema.validate(instance=res.trace, schema=schema)


def test_graph_path_refusal_v1_missing_image():
    schema = _get_trace_schema()
    bundle = _make_bundle(["optical"])
    res = run_query(bundle, "How much did the forest change between the dates?")
    assert res.state == "refused"
    assert res.refusal is not None
    assert res.refusal["category"] == "missing_input"
    jsonschema.validate(instance=res.trace, schema=schema)


def test_graph_path_refusal_v2_coreg_failure():
    schema = _get_trace_schema()
    coreg = CoregReport(coregistered=False, rmse_px=3.2)
    bundle = _make_bundle(["optical", "optical"], coreg=coreg)
    res = run_query(bundle, "Calculate change area between acquisitions")
    assert res.state == "refused"
    assert res.refusal["category"] == "validator"
    jsonschema.validate(instance=res.trace, schema=schema)


def test_graph_path_refusal_v3_sar_spectral_question():
    schema = _get_trace_schema()
    bundle = _make_bundle(["sar"])
    res = run_query(bundle, "What is the green chlorophyll spectral signature?")
    assert res.state == "refused"
    assert res.refusal["category"] == "modality_limitation"
    jsonschema.validate(instance=res.trace, schema=schema)


def test_graph_path_refusal_v8_change_map_no_crs():
    schema = _get_trace_schema()
    bundle = _make_bundle(["optical", "optical"], crs="")
    res = run_query(bundle, "Produce a georeferenced change map")
    assert res.state == "refused"
    assert res.refusal["category"] == "validator"
    jsonschema.validate(instance=res.trace, schema=schema)
