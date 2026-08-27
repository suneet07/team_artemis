import json

from satquery.agent.task_enum import RouterPath, Task
from satquery.paths import TRACE_SCHEMA_PATH

EXPECTED_TASKS = {
    "single_vqa",
    "single_caption",
    "single_grounding",
    "change_description",
    "change_vqa",
    "change_map",
    "crossmodal_extraction",
    "crossmodal_vqa",
}


def test_task_enum_is_frozen():
    assert {t.value for t in Task} == EXPECTED_TASKS


def test_router_path_enum_is_frozen():
    assert {r.value for r in RouterPath} == {"rules", "llm"}


def test_trace_schema_and_task_enum_stay_in_sync():
    schema = json.loads(TRACE_SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = schema["properties"]["graded"]["properties"]["task_selected"]["enum"]
    assert set(enum) == {t.value for t in Task}
