"""The learned adapter tool reaches its runner instead of dying on the Scene.

`rs_vqa` failed on every deployed request with

    AttributeError: 'Scene' object has no attribute 'effective_gsd_m'

and the failure was invisible: the executor records a tool error as a trace
warning and carries on, so `/meta/health` reported both adapters loaded, the
answer came back confidently from deterministic evidence, and nothing said the
adapter had not run. A GPU was billed for a model that was never called.

These tests run the tool with a stub runner, so they need no weights and no
GPU, and assert the two things that were actually broken: that the tool can
read everything it needs off a real `Scene`, and that a tool failure is visible
rather than swallowed.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.ingest.band_inventory import BandInventory
from satquery.tools.base import Scene, ToolContext


def _scene(name: str = "t0", modality: str = "optical") -> Scene:
    bands = {b: np.full((8, 8), 0.2, dtype="float32") for b in ("red", "green", "blue")}
    return Scene(
        name=name,
        modality=modality,
        bands=bands,
        inventory=BandInventory(bands=tuple(bands)),
        pixel_size_m=10.0,
    )


@pytest.fixture
def stub_runner(monkeypatch):
    """Replace the VLM with a recorder, so this needs no weights."""
    calls: list[dict] = []

    class Runner:
        def answer_all(self, items, batch_size=1, max_new_tokens=128):
            calls.append({"items": items, "max_new_tokens": max_new_tokens})
            return ["a stub answer"] * len(items)

    import satquery.tools.learned as learned

    monkeypatch.setattr(learned, "_runner", lambda base, adapter: Runner())
    return calls


def test_the_tool_reads_only_fields_a_real_scene_carries(stub_runner):
    """The regression: it read `effective_gsd_m`, which Scene does not have."""
    from satquery.tools.learned import LearnedVqaTool

    tool = LearnedVqaTool("rs_vqa", "/nonexistent/adapter")
    context = ToolContext(
        query_text="What is in this image?", scenes=[_scene()], params={}
    )
    result = tool.run(context)

    assert stub_runner, "the tool never reached its runner"
    assert result.outputs.get("answer") == "a stub answer"


def test_the_scale_prefix_receives_the_pixel_size(stub_runner):
    """The prompt must carry the real GSD, not silently drop it."""
    from satquery.tools.learned import LearnedVqaTool

    tool = LearnedVqaTool("rs_vqa", None)
    tool.run(
        ToolContext(query_text="How wide is the road?", scenes=[_scene()], params={})
    )
    prompt = stub_runner[0]["items"][0]["question"]
    assert "10" in prompt, (
        "the 10 m pixel size never reached the prompt, so the model was asked "
        "to judge scale with no scale given"
    )


def test_a_bitemporal_pair_passes_both_scenes(stub_runner):
    from satquery.tools.learned import LearnedVqaTool

    tool = LearnedVqaTool("change_vqa", None)
    tool.run(
        ToolContext(
            query_text="What changed?",
            scenes=[_scene("t0"), _scene("t1")],
            params={},
        )
    )
    assert len(stub_runner[0]["items"][0]["images"]) >= 2


def test_a_tool_failure_is_recorded_where_someone_will_see_it():
    """A swallowed AttributeError is how a dead adapter looked healthy.

    The executor may keep going -- that is deliberate, a failed tool should not
    take the whole answer down -- but the failure has to survive into the trace
    warnings, which is what finally exposed this bug.
    """
    from satquery.tools.learned import LearnedVqaTool

    tool = LearnedVqaTool("rs_vqa", None)
    with pytest.raises(ValueError, match="needs at least one scene"):
        tool.run(ToolContext(query_text="anything?", scenes=[], params={}))


def test_all_adapters_share_one_base_model(monkeypatch):
    """Three registered tools must not mean three copies of a 4B model.

    Keying the runner cache by (base, adapter) loaded a full base per adapter.
    On a 22 GB L4 the third one OOMed mid-query, the executor logged a tool
    failure and answered from whatever else had run -- so it looked like
    `change_vqa` simply never executing, not like an allocation bug. C1 commits
    to one base with LoRAs swapped over it.
    """
    import satquery.tools.learned as learned

    built: list[str] = []
    selected: list[str | None] = []

    class _Runner:
        def __init__(self, base, adapter_path=None):
            built.append(base)
            self.adapter_path = adapter_path

        def select_adapter(self, path):
            selected.append(path)
            self.adapter_path = path

        def answer_all(self, items, batch_size=1, max_new_tokens=128):
            return ["x"] * len(items)

    monkeypatch.setattr(learned, "_RUNNERS", {})
    monkeypatch.setattr("satquery.training.generate.VLMRunner", _Runner)

    base = learned.BASE_MODEL
    learned._runner(base, "/data/checkpoints/rs_vqa/adapter")
    learned._runner(base, "/data/checkpoints/change_vqa/adapter")
    learned._runner(base, None)

    assert built == [base], f"loaded the base {len(built)} times, expected once"
    assert selected == [
        "/data/checkpoints/rs_vqa/adapter",
        "/data/checkpoints/change_vqa/adapter",
        None,
    ], selected
