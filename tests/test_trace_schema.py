import json

import pytest
from jsonschema import ValidationError

from satquery.agent.task_enum import Task
from satquery.agent.trace import TraceBuilder, validate_trace


def minimal_builder() -> TraceBuilder:
    return (
        TraceBuilder("what is here?")
        .set_routing(Task.SINGLE_VQA, "rules")
        .set_parameter_check(True, [])
        .set_outputs(answer="water", confidence=0.9)
    )


def test_minimal_trace_is_valid_and_graded_first():
    trace = minimal_builder().build()
    validate_trace(trace)
    assert list(trace)[:5] == ["schema_version", "query_id", "timestamp", "query_text", "graded"]
    assert trace["schema_version"] == 2
    graded = trace["graded"]
    assert graded["task_selected"] == "single_vqa"
    assert graded["parameter_check"] == {"passed": True, "rejected": []}
    assert graded["outputs"]["confidence"] == 0.9


def test_refusal_output_is_schema_valid():
    trace = (
        minimal_builder()
        .set_outputs(
            answer="Refused: this sensor cannot answer spectral questions.",
            refusal={"reason": "no NIR band", "category": "modality_limitation"},
        )
        .build()
    )
    validate_trace(trace)
    assert trace["graded"]["outputs"]["refusal"]["category"] == "modality_limitation"


def test_invalid_refusal_category_rejected_by_schema():
    trace = minimal_builder().build()
    trace["graded"]["outputs"]["refusal"] = {"reason": "x", "category": "vibes"}
    with pytest.raises(ValidationError):
        validate_trace(trace)


def test_defaults_applied_field_is_schema_valid():
    trace = (
        minimal_builder()
        .add_planned_step("dummy_tool", {"scale": 0.5}, True, defaults_applied=["scale"])
        .build()
    )
    assert trace["graded"]["permitted_parameters"][0]["defaults_applied"] == ["scale"]
    validate_trace(trace)


def test_compatibility_ingest_block_is_schema_valid():
    trace = (
        minimal_builder()
        .set_compatibility(
            {
                "ingest": {
                    "format_ok": True,
                    "crs_valid": False,
                    "modality": "unknown",
                    "modality_source": "intensity_signature",
                    "bands_present": [],
                    "computable_indices": [],
                    "nodata_frac": 0.1,
                    "bit_depth": 16,
                    "bit_depth_source": "container_dtype",
                    "pixel_size_m": None,
                    "native_gsd_m": None,
                    "warnings": ["no CRS"],
                }
            }
        )
        .set_inputs(
            [
                {
                    "file": "chip.png",
                    "modality": "optical",
                    "modality_source": "declared",
                    "computable_indices": [],
                    "bit_depth": 8,
                    "bit_depth_source": "nbits_metadata",
                    "stretch_bounds": [0.02, 0.98],
                }
            ]
        )
        .build()
    )
    validate_trace(trace)


def test_invalid_task_rejected_by_schema():
    trace = minimal_builder().build()
    trace["graded"]["task_selected"] = "teleport"
    with pytest.raises(ValidationError):
        validate_trace(trace)


def test_unknown_top_level_field_rejected():
    trace = minimal_builder().build()
    trace["surprise"] = 1
    with pytest.raises(ValidationError):
        validate_trace(trace)


def test_unknown_graded_field_rejected():
    trace = minimal_builder().build()
    trace["graded"]["secret_note"] = "x"
    with pytest.raises(ValidationError):
        validate_trace(trace)


def test_confidence_bounds_enforced():
    trace = minimal_builder().build()
    trace["graded"]["outputs"]["confidence"] = 1.5
    with pytest.raises(ValidationError):
        validate_trace(trace)


def test_missing_parameter_check_raises_in_builder():
    builder = TraceBuilder("q").set_routing(Task.SINGLE_VQA, "rules")
    with pytest.raises(ValueError):
        builder.build()


def test_missing_routing_raises_in_builder():
    builder = TraceBuilder("q").set_parameter_check(True, [])
    with pytest.raises(ValueError):
        builder.build()


def test_tools_invoked_defaults_to_executed_steps():
    trace = (
        minimal_builder()
        .add_step("spectral_index", {"index": "NDWI"}, {"area_km2": 1.0})
        .add_step("sar_backscatter", {"pol": "VV"}, {})
        .build()
    )
    assert trace["graded"]["tools_invoked"] == ["spectral_index", "sar_backscatter"]
    assert trace["steps"][0]["param_source"] == "manifest_validated"


def test_rich_trace_with_all_sections_validates():
    trace = (
        minimal_builder()
        .set_inputs(
            [
                {
                    "file": "cartosat_mx.tif",
                    "modality": "optical",
                    "native_gsd_m": 2.0,
                    "pixel_size_m": 2.0,
                    "crs": "EPSG:32644",
                    "bands": ["B", "G", "R", "NIR"],
                    "swir_available": False,
                    "bit_depth": 12,
                    "nodata_frac": 0.02,
                }
            ]
        )
        .set_compatibility({"coregistered": True, "rmse_px": 0.8, "checks_passed": ["crs_match"]})
        .add_routing_note("SWIR unavailable; NDBI skipped")
        .add_planned_step("spectral_index", {"index": "NDWI"}, within_manifest=True)
        .add_step(
            "spectral_index",
            {"index": "NDWI", "threshold_method": "otsu"},
            {"area_km2": 3.4},
            confidence=0.79,
            latency_ms=180,
        )
        .set_agreement(0.86, "consistent", None)
        .set_fusion("qwen3vl-4b-instruct+rs_vqa@v1", "water body present", 0.81)
        .add_evidence("overlay.png")
        .add_warning("NDBI unavailable: source lacks SWIR band")
        .build()
    )
    validate_trace(trace)
    assert trace["inputs"][0]["swir_available"] is False
    assert trace["agreement"]["iou"] == 0.86


def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "out" / "trace.json"
    trace = minimal_builder().write_json(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == trace
