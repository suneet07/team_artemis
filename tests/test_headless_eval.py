"""Headless evaluation mode (master plan section 4.11).

This mode is three things at once, which is why it is tested rather than
demonstrated: the artifact we submit if the organisers run our code, the
venue-demo parachute for a room with no internet, and the CI smoke test.

The property that matters most here is not the answer text — it is that the
trace is schema-valid, that the parameter gate really ran, and that the answer
is grounded in a tool that really executed. The previous batch runner wrote
``"Mocked answer for spectral_index"`` with ``parameter_check.passed = True``,
which is indistinguishable from a real run in the submitted file.
"""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.agent.trace import validate_trace
from satquery.evalcli.__main__ import main, run_single
from satquery.evalcli.batch import process_batch


@pytest.fixture
def optical_scene(tmp_path):
    path = tmp_path / "scene.tif"
    rng = np.random.default_rng(3)
    gradient = np.linspace(400, 3600, 4 * 64 * 64).reshape(4, 64, 64)
    data = np.clip(gradient + rng.integers(-150, 150, (4, 64, 64)), 0, 4095).astype("uint16")
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32644",
        "transform": from_origin(0.0, 640.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(("Blue", "Green", "Red", "NIR"), start=1):
            dst.set_band_description(index, name)
    return path


def test_single_query_emits_a_schema_valid_trace(optical_scene, tmp_path):
    payload, trace = run_single(
        "Where is the water body?",
        [str(optical_scene)],
        output_dir=tmp_path / "evidence",
        write_evidence=True,
    )
    validate_trace(trace)
    assert trace["graded"]["parameter_check"]["passed"] is True
    assert trace["graded"]["tools_invoked"], "a real tool must have run"
    assert payload["answer"] and "mock" not in payload["answer"].lower()


def test_every_executed_step_carries_manifest_validated_parameters(optical_scene):
    """Section 6.2's parameter-gate conformance metric, as a test."""
    _, trace = run_single("Is there water here?", [str(optical_scene)])
    for step in trace["steps"]:
        assert step["param_source"] == "manifest_validated"
    planned = {entry["tool"] for entry in trace["graded"]["permitted_parameters"]}
    for step in trace["steps"]:
        assert step["tool"] in planned, "an executed step that was never gated"


def test_refusal_is_schema_valid_and_explained(optical_scene):
    """A graceful, explained refusal scores better than a hallucination."""
    payload, trace = run_single("How has this area changed?", [str(optical_scene)])
    validate_trace(trace)
    assert payload["refused"] is True
    refusal = trace["graded"]["outputs"]["refusal"]
    assert refusal["category"] == "missing_input"
    # The UI's `remedy` field must not reach the trace: the schema forbids it.
    assert set(refusal) == {"reason", "category"}


def test_benchmark_formatting_is_applied_when_asked(optical_scene):
    payload, _ = run_single(
        "Is there water here?", [str(optical_scene)], benchmark="rsvqa_lr_presence"
    )
    assert payload["formatted_answer"] in ("yes", "no")


def test_batch_writes_answers_and_traces(optical_scene, tmp_path):
    batch = tmp_path / "queries.jsonl"
    batch.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "a", "images": [str(optical_scene)], "question": "Is there water here?"},
                {"id": "b", "images": [str(optical_scene)], "question": "Describe this scene."},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary = process_batch(str(batch), str(tmp_path / "out"))
    assert summary == {"rows": 2, "failed": 0}

    answers = [
        json.loads(line)
        for line in (tmp_path / "out" / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    traces = [
        json.loads(line)
        for line in (tmp_path / "out" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in answers] == ["a", "b"]
    for trace in traces:
        validate_trace(trace)


def test_one_bad_row_does_not_lose_the_batch(optical_scene, tmp_path):
    batch = tmp_path / "queries.jsonl"
    batch.write_text(
        json.dumps({"id": "good", "images": [str(optical_scene)], "question": "Is there water?"})
        + "\n"
        + json.dumps({"id": "bad", "images": ["/no/such/file.tif"], "question": "Is there water?"})
        + "\n",
        encoding="utf-8",
    )
    summary = process_batch(str(batch), str(tmp_path / "out"))
    assert summary["rows"] == 2 and summary["failed"] == 1
    answers = [
        json.loads(line)
        for line in (tmp_path / "out" / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert answers[0]["id"] == "good" and "error" not in answers[0]
    assert "error" in answers[1]


def test_selftest_exits_zero():
    """The CI smoke test: no data, no network, no GPU."""
    assert main(["--selftest"]) == 0
