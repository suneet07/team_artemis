"""``lulc_classifier`` is registered, routed, and gated by its own contract.

The tool was measured before it was wired: 74.95% on 6,000 held-out cross-modal
rows against a 50% majority floor, radar alone, no training. None of that
reaches a user until a plan asks for it, and a registered-but-unrouted tool is
indistinguishable from an absent one -- so these tests pin the wiring rather
than the model.
"""

import pytest

from satquery.agent.router import QueryContext, plan_for
from satquery.agent.task_enum import Task
from satquery.tools.base import ToolContext
from satquery.tools.catalog import DETERMINISTIC_TOOLS, available_tools, implementation
from satquery.tools.lulc import CLASSES, LulcClassifierTool


class _Inventory:
    """Minimal stand-in for BandInventory; plan_for only reads these."""

    def __init__(self, indices):
        self.bands = {"blue": 1, "green": 2, "red": 3, "nir": 4}
        self.has_nir = True
        self.computable_indices = list(indices)


def _context(question, modalities, indices=("NDVI", "NDWI"), image_count=1):
    return QueryContext(
        question,
        list(modalities),
        [_Inventory(indices) for _ in modalities],
        image_count=image_count,
    )


def _tools(plan):
    return [step["tool"] for step in plan]


def test_registered_and_reachable():
    assert "lulc_classifier" in available_tools()
    assert implementation("lulc_classifier") is not None


def test_excluded_from_the_deterministic_set():
    """It needs torch and downloaded weights.

    The headless CPU path (section 4.11) is built from ``DETERMINISTIC_TOOLS``.
    A learned tool listed there would plan cleanly and fail at execution, which
    is the failure the split exists to prevent.
    """
    assert "lulc_classifier" not in DETERMINISTIC_TOOLS


def test_caption_plan_grounds_the_description_in_named_classes():
    plan = plan_for(
        Task.SINGLE_CAPTION,
        _context("Describe the land-cover and major objects visible", ["sar"]),
        [],
    )
    tools = _tools(plan)
    # rs_ground_caption, not rs_vqa: the corpus assigns 19,983 caption rows and
    # 39,996 grounding rows to rs_ground_caption, and 57,044 VQA rows to
    # rs_vqa. Captioning ships the base model behind that manifest name.
    assert tools == ["lulc_classifier", "rs_ground_caption"]
    # Order is declared, not incidental: the caption step consumes the class
    # inventory, so it must not be scheduled before it exists.
    caption = next(step for step in plan if step["tool"] == "rs_ground_caption")
    assert caption["depends_on"] == ["lulc_classifier"]


def test_crossmodal_plan_keeps_both_deterministic_decisions():
    """The classifier is added beside D1, never instead of it.

    Section 4.7 fuses at the decision level: an optical mask and a SAR mask,
    then reconciliation. Replacing either with a single learned answer would
    remove the disagreement the fusion table exists to explain.
    """
    plan = plan_for(
        Task.CROSSMODAL_EXTRACTION,
        _context("Identify built-up and water regions", ["optical", "sar"]),
        ["water"],
    )
    tools = _tools(plan)
    assert "lulc_classifier" in tools
    assert "sar_backscatter" in tools
    assert tools[0] == "coreg_check"
    assert any(t in tools for t in ("spectral_index", "texture_seg"))


def test_not_planned_for_optical_only_scenes():
    """Optical land cover belongs to ``spectral_index`` and the VLM.

    Both are measured and both work; the radar classifier scored 0.4968 on
    optical -- chance -- against 0.7525 on radar, over the same questions. So
    an optical-only caption keeps the single learned step it always had.
    """
    plan = plan_for(
        Task.SINGLE_CAPTION, _context("Describe this scene", ["optical"]), []
    )
    assert _tools(plan) == ["rs_ground_caption"]


def test_refuses_a_scene_with_no_radar():
    """Radar only, and the message says so.

    This asserted "optical or SAR" while the tool still fell back to the optical
    checkpoint when no radar view was present. That fallback is gone -- the
    optical arm scores 0.4968, chance -- so the refusal is now unconditional on
    a bundle without radar, and the message names the number that decided it.
    See `tests/test_lulc_is_radar_only.py` for the three guards around it.
    """
    tool = LulcClassifierTool()
    with pytest.raises(ValueError, match="radar only"):
        tool.run(ToolContext(query_text="what is here", scenes=[], params={}))


def test_class_list_matches_the_checkpoint_head():
    """19 outputs, and the order is the only thing mapping a logit to a class."""
    assert len(CLASSES) == 19
    assert CLASSES[0] == "Agro-forestry areas"
    assert CLASSES[-1] == "Urban fabric"
