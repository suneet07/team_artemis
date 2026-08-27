from satquery.agent.trace import validate_trace
from satquery.evalcli.smoke import run_dummy_query


def test_happy_path_emits_scoreable_trace():
    trace = run_dummy_query(
        "What dominates this scene?",
        {"index": "ALPHA", "scale": 0.3},
        modality="optical",
    )
    validate_trace(trace)
    graded = trace["graded"]
    assert trace["schema_version"] == 2
    assert graded["parameter_check"] == {"passed": True, "rejected": []}
    plan = graded["permitted_parameters"][0]
    assert plan["within_manifest"] is True
    assert plan["params"] == {"index": "ALPHA", "scale": 0.3}
    assert "defaults_applied" not in plan
    assert graded["tools_invoked"] == ["dummy_tool"]
    assert graded["outputs"]["answer"].startswith("dummy_tool[ALPHA]")
    assert graded["outputs"]["area_km2"] == 3.0
    assert len(trace["steps"]) == 1
    assert trace["steps"][0]["params"] == plan["params"]


def test_gate_failure_emits_answer_and_structured_refusal():
    trace = run_dummy_query("What dominates this scene?", {"index": "OMEGA", "scale": 0.3})
    validate_trace(trace)
    graded = trace["graded"]
    outputs = graded["outputs"]
    assert graded["parameter_check"]["passed"] is False
    assert any("OMEGA" in r for r in graded["parameter_check"]["rejected"])
    assert outputs["answer"].startswith("Refused: parameter validation failed")
    assert "GAMMA" not in outputs["answer"]
    assert "OMEGA" in outputs["answer"]
    assert outputs["refusal"]["category"] == "parameter_gate"
    assert "OMEGA" in outputs["refusal"]["reason"]
    assert outputs["confidence"] == 0.0
    assert graded["permitted_parameters"][0]["within_manifest"] is False
    assert any("Refused" in w for w in trace["warnings"])
    assert graded["tools_invoked"] == []


def test_defaults_applied_recorded_and_graded_params_match_run():
    trace = run_dummy_query("q", {"index": "BETA"}, modality="optical")
    graded = trace["graded"]
    plan = graded["permitted_parameters"][0]
    assert plan["params"] == {"index": "BETA", "scale": 0.5}
    assert plan["defaults_applied"] == ["scale"]
    assert trace["steps"][0]["params"] == plan["params"]


def test_modality_mismatch_refuses(dummy_like_context=None):
    trace = run_dummy_query(
        "q", {"index": "ALPHA", "scale": 0.5}, modality="sar"
    )
    validate_trace(trace)
    outputs = trace["graded"]["outputs"]
    assert outputs["refusal"]["category"] == "parameter_gate"
    assert "requires modality 'optical'" in outputs["refusal"]["reason"]
