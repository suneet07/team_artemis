"""The 300-query routing evaluation set (Phase 0 item 1, section 6.2).

The plan builds this set in the same sitting as the task-enum freeze and measures
routing accuracy in Week 1, because the risk register's trigger is concrete:
*hybrid routing below 90% on unambiguous cases means the rules get fixed now.*
A set that cannot detect that is worse than no set at all.

The previous generator could not. It drew the *expected task* and the *expected
router path* at random, so the ground truth was noise; it wrote to a hard-coded
path on one machine; it was unseeded, so the file changed on every run and any
accuracy number moved with it; and every "expected_tools" entry was
``["dummy_tool"]``. These tests pin the properties that make the number mean
something.
"""

import json
import sys
from pathlib import Path

import pytest

from satquery.agent.task_enum import RouterPath, Task
from satquery.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).parent.parent
DATASET = REPO_ROOT / "training" / "eval" / "routing_300.jsonl"

sys.path.insert(0, str(REPO_ROOT / "training" / "eval"))


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    assert DATASET.exists(), f"routing set not found at {DATASET}"
    return [json.loads(line) for line in DATASET.open(encoding="utf-8") if line.strip()]


def test_dataset_shape(rows):
    assert len(rows) == 300
    assert len({row["id"] for row in rows}) == 300
    for row in rows:
        truth = row["ground_truth"]
        assert truth["expected_task"] in {task.value for task in Task}
        assert truth["expected_router_path"] in {path.value for path in RouterPath}
        assert isinstance(truth["unambiguous"], bool)
        context = row["input_context"]
        assert context["modalities"] and context["bands"] and context["crs"]


def test_every_task_is_covered(rows):
    covered = {row["ground_truth"]["expected_task"] for row in rows}
    assert covered == {task.value for task in Task}


def test_expected_tools_are_real_registered_tools(rows):
    known = set(ToolRegistry.default().names())
    for row in rows:
        for tool in row["ground_truth"]["expected_tools"]:
            assert tool in known, f"{row['id']} expects unregistered tool '{tool}'"


def test_hard_cases_are_present_in_useful_numbers(rows):
    """Refusals and reroutes are what section 6.2's catch rate is measured on."""
    categories = [row["ground_truth"]["expected_category"] for row in rows]
    assert categories.count("refusal") >= 30
    assert categories.count("reroute_d3") >= 15
    assert categories.count("deterministic_fallback") >= 10
    assert categories.count("ambiguous") >= 10
    assert categories.count("stress_single_pol") >= 10


def test_query_text_is_varied(rows):
    """A rule that matches one phrasing must not be able to score 100%."""
    assert len({row["query_text"] for row in rows}) >= 100


def test_refusal_categories_are_schema_valid(rows):
    valid = {
        "parameter_gate",
        "validator",
        "modality_limitation",
        "missing_input",
        "unsupported_class",
    }
    for row in rows:
        category = row["ground_truth"]["expected_refusal_category"]
        if category is not None:
            assert category in valid


def test_generation_is_deterministic(tmp_path):
    """Byte-identical on every run, or an accuracy delta means nothing."""
    from generate_routing_300 import write_dataset

    first = write_dataset(tmp_path / "a.jsonl")
    second = write_dataset(tmp_path / "b.jsonl")
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == DATASET.read_bytes(), (
        "the committed routing set is out of date; regenerate it with "
        "`python training/eval/generate_routing_300.py`"
    )


def test_rules_router_clears_the_risk_register_threshold():
    """Part 10: hybrid routing below 90% on unambiguous cases is a trigger."""
    from score_routing import evaluate

    report = evaluate()
    assert report["unambiguous_accuracy"] >= 0.9, (
        f"routing accuracy {report['unambiguous_accuracy']:.1%} on unambiguous cases is "
        f"below the 90% trigger. Misroutes: {report['misroutes'][:5]}"
    )
    assert report["invalid_config_catch_rate"] >= 0.9
    # A mostly-rules trace is more defensible than a mostly-LLM one (4.5.2).
    assert report["rules_share"] >= 0.8


def test_score_routing_reports_and_exits_zero(capsys):
    """CI runs this script as a gate, so its exit code has to mean something."""
    from score_routing import main

    assert main() == 0
    assert "unambiguous" in capsys.readouterr().out
