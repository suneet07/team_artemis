import json
from pathlib import Path

import jsonschema

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.paths import TRACE_SCHEMA_PATH

FIXTURES_DIR = Path(__file__).parent.parent / "frontend" / "mocks" / "fixtures"


def _get_schema():
    return json.loads(TRACE_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_mock_fixtures_validate_against_schema():
    schema = _get_schema()
    fixtures = [
        "trace_crossmodal_success.json",
        "trace_param_rejected.json",
        "trace_refusal_missing_input.json",
    ]
    for fname in fixtures:
        fpath = FIXTURES_DIR / fname
        data = json.loads(fpath.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)


def test_refusal_trace_matches_fixture_shape():
    schema = _get_schema()
    fix_path = FIXTURES_DIR / "trace_refusal_missing_input.json"
    fix_data = json.loads(fix_path.read_text(encoding="utf-8"))

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
        scene_id="scene_0",
        path="cartosat_mx.tif",
        modality="optical",
        crs="EPSG:32644",
        nodata_frac=0.02,
        swir_available=False,
    )
    bundle = ImageBundle(
        bundle_id="b_diff",
        images=[img],
        band_inventory=inv,
        pair_type="single",
    )

    res = run_query(bundle, "What changed between the two dates?")
    assert res.state == "refused"
    trace = res.trace

    # Check top-level keys match fixture
    top_keys = [
        "schema_version",
        "graded",
        "router_path",
        "inputs",
        "steps",
        "evidence",
        "warnings",
    ]
    for k in top_keys:
        assert k in trace
        assert k in fix_data

    # Check graded keys
    graded_keys = [
        "task_selected",
        "tools_invoked",
        "permitted_parameters",
        "parameter_check",
        "outputs",
    ]
    for gk in graded_keys:
        assert gk in trace["graded"]
        assert gk in fix_data["graded"]

    assert trace["graded"]["outputs"]["refusal"]["category"] == "missing_input"
    jsonschema.validate(instance=trace, schema=schema)
