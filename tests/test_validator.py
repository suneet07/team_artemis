"""Validator and parameter-gate catch rate (sections 4.5.3, 4.5.4, and 6.2).

Section 6.2 reports an **invalid-config catch rate**: deliberately malformed
inputs go in, and the question is whether the validator refuses. These are those
inputs.

The earlier version of this file built manifests by hand out of keys the manifest
schema does not have — ``required: true``, ``type: boolean_array`` — and asserted
against the second, weaker validator that read them. It therefore passed while
the real gate was never exercised, and it never once checked a value against a
declared range, which is the specific failure section 4.5.4 exists to prevent.
Manifests here are schema-valid, and the gate under test is the shipped one.
"""

import numpy as np
import pytest

from satquery.agent.router import D3BandRouter, QueryContext, route
from satquery.agent.task_enum import Task
from satquery.agent.trace import TraceBuilder
from satquery.agent.validator import ToolValidator, validate
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.manifest import ToolManifest
from satquery.tools.registry import ToolRegistry


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.default()


def cartosat_mx() -> BandInventory:
    return BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_nir=True,
        computable_indices=["NDVI", "NDWI"],
    )


# --------------------------------------------------------------------------
# 4.5.4 — the parameter gate
# --------------------------------------------------------------------------


def test_missing_required_param(registry):
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters("spectral_index", {}, manifest)
    assert res["passed"] is False
    assert any("index" in reason for reason in res["rejected"])


def test_unknown_param_is_rejected_not_ignored(registry):
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters(
        "spectral_index", {"index": "NDWI", "hallucinated_param": True}, manifest
    )
    assert res["passed"] is False
    assert any("hallucinated_param" in reason for reason in res["rejected"])


def test_out_of_range_value_is_rejected_never_clamped(registry):
    """The plan's worked example: a threshold of 47 must not run.

    Under v3 the system would have executed it, produced a meaningless mask, and
    faithfully written ``"threshold_value": 47`` into the graded artifact --
    documenting our own error instead of preventing it.
    """
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters(
        "spectral_index", {"index": "NDWI", "threshold_value": 47.0}, manifest
    )
    assert res["passed"] is False
    assert any("47" in reason for reason in res["rejected"])


def test_off_enum_value_is_rejected(registry):
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters(
        "spectral_index", {"index": "NDXX"}, manifest
    )
    assert res["passed"] is False


def test_wrong_type_is_rejected(registry):
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters(
        "spectral_index", {"index": "NDWI", "threshold_value": "not a number"}, manifest
    )
    assert res["passed"] is False


def test_band_prerequisite_is_enforced(registry):
    """D3, belt and braces: NDBI needs SWIR, and Cartosat MX has none."""
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters(
        "spectral_index",
        {"index": "NDBI", "threshold_method": "otsu"},
        manifest,
        band_inventory=cartosat_mx(),
    )
    assert res["passed"] is False
    assert any("swir" in reason.lower() for reason in res["rejected"])


def test_valid_params_pass(registry):
    manifest = registry.get("spectral_index")
    res = ToolValidator.validate_parameters(
        "spectral_index",
        {"index": "NDWI", "threshold_method": "otsu"},
        manifest,
        band_inventory=cartosat_mx(),
    )
    assert res == {"passed": True, "rejected": []}


def test_raw_manifest_mapping_still_accepted():
    """Callers held the parsed mapping as well as the dataclass."""
    raw = {
        "name": "probe",
        "description": "schema-valid probe manifest",
        "required_modalities": ["optical"],
        "permitted_parameters": {"scale": {"type": "float", "range": [0.0, 1.0]}},
        "outputs": {"answer": {"type": "text"}},
    }
    assert ToolValidator.validate_parameters("probe", {"scale": 0.5}, raw)["passed"] is True
    assert ToolValidator.validate_parameters("probe", {"scale": 9.0}, raw)["passed"] is False
    assert isinstance(ToolManifest.from_dict(raw), ToolManifest)


def test_manifest_name_mismatch_is_caught(registry):
    res = ToolValidator.validate_parameters(
        "sar_backscatter", {"index": "NDWI"}, registry.get("spectral_index")
    )
    assert res["passed"] is False


# --------------------------------------------------------------------------
# 4.5.3 — the scene validator rule table
# --------------------------------------------------------------------------


def _decide(question: str, modalities: list[str], inventories, image_count: int = 1):
    context = QueryContext(question, modalities, inventories, image_count=image_count)
    return context, route(context)


def test_change_query_with_one_image_refuses_and_asks_for_the_second():
    context, decision = _decide("How has the built-up area changed?", ["optical"], [cartosat_mx()])
    result = validate(context, decision)
    assert result.passed is False
    assert result.refusal.category == "missing_input"
    assert "second" in result.refusal.reason.lower()


def test_sar_only_colour_question_explains_the_modality_limit():
    sar = BandInventory(polarisations=["VV", "VH"], sar_band="C")
    context, decision = _decide("What colour is the water here?", ["sar"], [sar])
    result = validate(context, decision)
    assert result.passed is False
    assert result.refusal.category == "modality_limitation"
    assert "backscatter" in result.refusal.reason.lower()


def test_pan_only_spectral_question_refuses_with_an_explanation():
    pan = BandInventory(bands={"pan": 1}, is_pan_only=True, computable_indices=[])
    context, decision = _decide("Is the vegetation healthy?", ["optical"], [pan])
    result = validate(context, decision)
    assert result.passed is False
    assert result.refusal.category == "modality_limitation"


def test_single_pol_polarimetric_question_warns_rather_than_refusing():
    """Answer what one channel supports, and say what it cannot."""
    single = BandInventory(bands={"hh": 1}, polarisations=["HH"], sar_band="X")
    context, decision = _decide(
        "What is the polarimetric signature here?", ["sar"], [single]
    )
    result = validate(context, decision)
    assert result.passed is True
    assert any("polarisation" in note.lower() for note in result.routing_notes)


def test_absent_band_reroutes_rather_than_running_a_doomed_step():
    """D3: a step needing a band the source lacks never reaches the gate."""
    context = QueryContext("How much built-up area is there?", ["optical"], [cartosat_mx()])
    decision = route(context)
    decision.plan = [{"tool": "spectral_index", "params": {"index": "NDBI"}}]
    result = validate(context, decision)
    assert all(step["params"].get("index") != "NDBI" for step in result.plan)
    assert result.dropped_steps


def test_high_nodata_warns_and_proceeds():
    context, decision = _decide("Is there water here?", ["optical"], [cartosat_mx()])
    result = validate(context, decision, nodata_fractions=[0.8], nodata_warn_fraction=0.5)
    assert result.passed is True
    assert any("nodata" in warning for warning in result.warnings)


# --------------------------------------------------------------------------
# D3 band router
# --------------------------------------------------------------------------


def test_d3_band_router_graceful_refusal():
    inventory = BandInventory(
        bands={"PAN": 1}, computable_indices=[], sar_band=None, is_pan_only=True
    )
    res = D3BandRouter(inventory).route("vegetation")
    assert res["tool"] == "refusal"
    assert "Input is Pan-only" in res["reason"]


def test_d3_band_router_prefers_mndwi_then_ndwi():
    with_swir = BandInventory(computable_indices=["NDWI", "MNDWI"], has_swir=True)
    assert D3BandRouter(with_swir).route("water")["params"]["index"] == "MNDWI"
    assert D3BandRouter(cartosat_mx()).route("water")["params"]["index"] == "NDWI"


# --------------------------------------------------------------------------
# The rejection reaches the graded artifact
# --------------------------------------------------------------------------


def test_trace_builder_records_rejection():
    builder = TraceBuilder(query_text="Test query")
    builder.set_routing(Task.SINGLE_VQA, "rules")
    builder.set_parameter_check(passed=False, rejected=["Invalid type"])
    trace = builder.build()
    assert trace["graded"]["parameter_check"]["passed"] is False
    assert trace["graded"]["parameter_check"]["rejected"] == ["Invalid type"]


def test_masks_are_not_smuggled_through_the_parameter_channel(registry):
    """Pixels must not travel as parameters (section 4.5.4).

    A tool whose raster arrives in ``params`` cannot pass its own gate, which is
    what forced the controller to execute stubs instead of tools.
    """
    manifest = registry.get("change_stats")
    res = ToolValidator.validate_parameters(
        "change_stats", {"mask_t1": np.zeros((4, 4), dtype=bool)}, manifest
    )
    assert res["passed"] is False
    assert any("mask_t1" in reason for reason in res["rejected"])
