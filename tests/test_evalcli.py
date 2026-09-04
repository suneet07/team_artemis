import json
import subprocess
import sys

from satquery.agent.trace import validate_trace


def test_evalcli_success_exit_code_and_output(tmp_path):
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "-m",
        "satquery.evalcli",
        "--query",
        "What is the vegetation extent?",
        "--out",
        str(out_dir),
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0

    trace_file = out_dir / "trace.json"
    result_file = out_dir / "result.json"

    assert trace_file.exists()
    assert result_file.exists()

    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    validate_trace(trace_data)

    result_data = json.loads(result_file.read_text(encoding="utf-8"))
    assert result_data["state"] == "succeeded"


def test_evalcli_refusal_exit_code_1():
    # A change query without a 2nd image refuses with code 1
    cmd = [
        sys.executable,
        "-m",
        "satquery.evalcli",
        "--query",
        "How has the land cover changed between 2020 and 2024?",
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1


def test_evalcli_single_file_output(tmp_path):
    out_file = tmp_path / "combined.json"
    cmd = [
        sys.executable,
        "-m",
        "satquery.evalcli",
        "--question",
        "What is here?",
        "--out",
        str(out_file),
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "result" in data
    assert "trace" in data
    assert "metadata" in data
    assert data["result"]["state"] == "succeeded"


def test_evalcli_images_and_batch_mode(tmp_path):
    batch_file = tmp_path / "batch.jsonl"
    batch_file.write_text(
        json.dumps({"id": "q1", "images": ["fake1.tif"], "question": "What is here?"}) + "\n"
        + json.dumps({"id": "q2", "images": ["fake2.tif"], "question": "What changed?"}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch_out"
    cmd = [
        sys.executable,
        "-m",
        "satquery.evalcli",
        "--batch-file",
        str(batch_file),
        "--out",
        str(out_dir),
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    summary_file = out_dir / "summary.jsonl"
    raw_lines = summary_file.read_text(encoding="utf-8").splitlines()
    lines = [json.loads(ln) for ln in raw_lines if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["id"] == "q1"
    assert lines[0]["state"] == "succeeded"
    assert lines[1]["id"] == "q2"
    assert lines[1]["state"] == "refused"
