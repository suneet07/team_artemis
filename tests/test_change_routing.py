"""Every task with a trained adapter must actually plan it.

`change_vqa` is the mandatory G4 gate. Its adapter is trained and measured --
AA 68.0% on official CDVQA Val against a 45.0% blind ceiling -- and it was
missing from the plan entirely. Bi-temporal questions ran `coreg_check`, found
`change_map` absent (a closed decision), handed `change_stats` no class maps,
and fell through to "no learned adapter is loaded".

Nothing errored. The gate was simply answered by nothing, and every status
surface said the adapter was loaded, because it *was* loaded -- just never
called.
"""

from __future__ import annotations

import pytest

from satquery.agent.router import QueryContext, plan_for, route
from satquery.agent.task_enum import Task

#: The learned tool each task is meant to reach. `change_map` is excluded on
#: purpose: it asks for a raster, and the text adapter is not one.
LEARNED_FOR_TASK = {
    Task.SINGLE_VQA: "rs_vqa",
    Task.SINGLE_CAPTION: "rs_ground_caption",
    Task.SINGLE_GROUNDING: "rs_ground_caption",
    Task.CHANGE_VQA: "change_vqa",
    Task.CHANGE_DESCRIPTION: "change_vqa",
}


def _context(text: str, modalities: list[str]) -> QueryContext:
    return QueryContext(
        query_text=text,
        modalities=modalities,
        inventories=[],
        image_count=len(modalities),
        dates=["2018-01-01", "2020-01-01"][: len(modalities)],
    )


def _tools(plan: list[dict]) -> list[str]:
    return [step["tool"] for step in plan]


@pytest.mark.parametrize(
    "task,expected", sorted(LEARNED_FOR_TASK.items(), key=lambda kv: kv[0].value)
)
def test_each_task_plans_its_learned_tool(task, expected):
    modalities = ["optical", "optical"] if task.value.startswith("change") else ["optical"]
    plan = plan_for(task, _context("what changed here?", modalities), [])
    assert expected in _tools(plan), (
        f"{task.value} plans {_tools(plan)} and never reaches {expected}; "
        "the trained weights would be loaded and never called"
    )


def test_change_vqa_reaches_its_adapter_from_a_real_question():
    """The routed path, not just plan_for called directly."""
    decision = route(
        _context("What changed between the two dates?", ["optical", "optical"])
    )
    assert decision.task is Task.CHANGE_VQA
    assert "change_vqa" in _tools(decision.plan)


def test_change_map_still_plans_the_mask_tool_it_cannot_run():
    """A request for a raster should say what was attempted, not silently
    substitute prose. `change_map` is absent by decision; the executor records
    that and the trace carries it."""
    plan = plan_for(
        Task.CHANGE_MAP, _context("Produce a change mask.", ["optical", "optical"]), []
    )
    assert "change_map" in _tools(plan)
    assert "change_vqa" not in _tools(plan), (
        "a text answer is not a raster; CHANGE_MAP must not quietly answer in prose"
    )


def test_the_graded_change_paths_plan_no_step_that_cannot_run():
    """`change_stats` needs `class_map_before`/`class_map_after`, which come
    from `change_map` or from two thresholded masks -- and nothing currently
    produces either. Planning them added two steps to every graded bi-temporal
    query that could only emit a warning.

    The arithmetic itself is sound: 100% AA on 2,012 rows when handed
    ground-truth footprints. It is perception that is missing, so this asserts
    the *current* contract rather than a permanent one -- restore both steps the
    day a mask source lands, and change this test with them.
    """
    for task in (Task.CHANGE_VQA, Task.CHANGE_DESCRIPTION):
        tools = _tools(
            plan_for(task, _context("What changed?", ["optical", "optical"]), [])
        )
        assert "change_stats" not in tools, (
            f"{task.value} plans change_stats with nothing to feed it"
        )
        assert "change_map" not in tools, (
            f"{task.value} plans change_map, which is absent by decision"
        )
        assert tools == ["coreg_check", "change_vqa"], tools
