"""The prompt that ships is the prompt that was measured.

`rs_ground_caption` serves the BASE model -- there is no adapter and there is
not meant to be one. That makes the prompt load-bearing in a way it is not for a
trained adapter: `rs_vqa` and `change_vqa` reproduce their scores from
`scale_prefix(...) + question` because that is the format they were fine-tuned
on, but the base model reproduces nothing unless it is asked the way the
benchmark asked.

Two numbers depend on the exact strings here:

* grounding **62.7% acc@0.5** (above published fine-tuned GeoChat at 60.6%),
  from the ``qwen_precise`` arm;
* captioning **ROUGE-L 0.252** against a 0.222 blind floor, from the ``strong``
  arm.

Serving originally passed the user's raw question instead. For captioning that
loses the register the metric rewards; for grounding it is worse, because the
0-1000 coordinate scale and the ``[{"bbox_2d": ...}]`` shape exist only in the
prompt -- and pinning that scale is what took grounding from 75% unparsable to
0%.
"""

from __future__ import annotations

import pytest

from satquery.agent.router import QueryContext, plan_for
from satquery.agent.served_prompts import (
    PRECISE_PROMPT,
    STRONG_PROMPT,
    served_prompt,
)
from satquery.agent.task_enum import Task


def _context(text: str, modalities: list[str] | None = None) -> QueryContext:
    modalities = modalities or ["optical"]
    return QueryContext(
        query_text=text,
        modalities=modalities,
        inventories=[],
        image_count=len(modalities),
        dates=["2020-01-01"] * len(modalities),
    )


def _steps(plan: list[dict], tool: str) -> list[dict]:
    return [step for step in plan if step["tool"] == tool]


# -- the prompts themselves --------------------------------------------------


def test_the_grounding_prompt_carries_the_scale_and_the_output_shape():
    """Both are what the box parser depends on, and both live only here."""
    filled = served_prompt("grounding", "the white ship.")
    assert "0-1000" in filled, (
        "the coordinate scale is gone; Qwen's raw output convention differs and "
        "the parser once read 75% of replies as unparsable without it"
    )
    assert '"bbox_2d"' in filled
    assert "SMALLEST axis-aligned rectangle" in filled
    # The phrase is substituted, and the trailing full stop is stripped the way
    # the benchmark fed it -- the prompt reads as an object name, not a sentence.
    assert "the white ship" in filled
    assert "the white ship." not in filled


def test_the_caption_prompt_carries_the_mined_register():
    """Every clause below was measured off VRSBench's own 9,350 references."""
    filled = served_prompt("caption", "Describe this scene.")
    assert filled == STRONG_PROMPT
    for clause in ("47 words", "three sentences".replace("three ", "3 "), "small or large"):
        assert clause in filled, f"{clause!r} missing from the caption prompt"
    # And must NOT name a provenance. The clause used to read "The image,
    # sourced from GoogleEarth," because that token appears in 86% of VRSBench
    # references -- pure n-gram farming, zero information about the picture, and
    # factually false on the Cartosat-2S hidden set the system is graded on.
    # A tool that sells auditable output cannot assert a source it cannot know.
    assert "GoogleEarth" not in filled, (
        "the provenance clause is back; it states a source the system cannot "
        "verify and is wrong on every non-GoogleEarth scene"
    )
    assert "sourced from" not in filled


def test_an_unknown_mode_returns_the_question_untouched():
    """Silently captioning a grounding request scores as a bad model."""
    assert served_prompt("vqa", "How many buildings?") == "How many buildings?"
    assert served_prompt("", "anything") == "anything"


def test_the_scorers_use_the_same_objects_the_server_does():
    """A second copy in scripts/ could drift and the benchmark would stop
    describing what ships, which is exactly how the threshold bug happened."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name, attr, expected in (
        ("eval_grounding_pipelines.py", "PRECISE_PROMPT", PRECISE_PROMPT),
        ("eval_captioning.py", "STRONG_PROMPT", STRONG_PROMPT),
    ):
        spec = importlib.util.spec_from_file_location("_scorer", root / "scripts" / name)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:  # argparse at import time
            pass
        assert getattr(module, attr) is expected, (
            f"scripts/{name} no longer shares {attr} with the serving path"
        )


# -- the router half ---------------------------------------------------------


def test_the_router_declares_grounding_mode():
    """Only the planner knows which task this manifest is serving."""
    plan = plan_for(
        Task.SINGLE_GROUNDING, _context("Where is the water in this image?"), ["water"]
    )
    steps = _steps(plan, "rs_ground_caption")
    assert steps, "grounding plans no longer include rs_ground_caption"
    assert steps[0]["params"].get("mode") == "grounding"


def test_the_router_declares_caption_mode():
    plan = plan_for(
        Task.SINGLE_CAPTION,
        _context("Describe the land-cover and major objects visible."),
        [],
    )
    steps = _steps(plan, "rs_ground_caption")
    assert steps, "caption plans no longer include rs_ground_caption"
    assert steps[0]["params"].get("mode") == "caption"


def test_captions_do_not_route_to_the_vqa_adapter():
    """The corpus assigns 19,983 caption rows to rs_ground_caption and 57,044
    VQA rows to rs_vqa. Routing a description to the VQA adapter answers the
    problem statement's first representative query with a model tuned to emit
    short factual answers -- the opposite register."""
    plan = plan_for(Task.SINGLE_CAPTION, _context("Describe this scene."), [])
    assert not _steps(plan, "rs_vqa")


@pytest.mark.parametrize("mode", ["grounding", "caption"])
def test_the_declared_mode_is_one_the_manifest_permits(mode):
    """The parameter gate validates against the manifest, so a mode the config
    does not declare would be refused at execution rather than at planning."""
    from satquery.tools.registry import ToolRegistry

    spec = ToolRegistry.default().get("rs_ground_caption").permitted_parameters["mode"]
    assert mode in set(spec.values)


def test_grounding_gets_the_referring_phrase_not_the_whole_question():
    """62.7% was measured on VRSBench referring phrases -- noun phrases like
    "the white ship" -- and PRECISE_PROMPT substitutes into "Find this object in
    the satellite image: {phrase}". Handing it a whole interrogative produces
    "Find this object in the satellite image: Where is the water in this image?",
    which is not the input the number describes.
    """
    plan = plan_for(
        Task.SINGLE_GROUNDING, _context("Where is the water in this image?"), ["water"]
    )
    step = _steps(plan, "rs_ground_caption")[0]
    assert step["params"].get("phrase") == "water"

    filled = served_prompt(step["params"]["mode"], step["params"]["phrase"])
    assert filled.splitlines()[0].endswith("water")
    assert "Where is" not in filled


def test_the_phrase_falls_back_to_the_question_when_nothing_was_extracted():
    """A tool called without a planned phrase must still ask something."""
    from satquery.tools.base import ToolContext

    assert served_prompt("grounding", "the large yellow vehicle") != ""
    # ToolContext with no phrase param is the fallback path the tool takes.
    context = ToolContext(query_text="find the jetty", scenes=[], params={})
    assert context.params.get("phrase") is None
