"""A crossmodal query must execute, not merely plan.

The fusion branch read `context.scene_of("optical")` from the router's
`QueryContext`, which has no such method. Every crossmodal query failed with an
AttributeError -- and 411 tests, plus all 44 route checks, passed while it did.

Nothing caught it because nothing executed the branch. It is reached only when
an optical mask and a SAR mask both materialise, and the offline route checker
stops at planning: it asserts every planned tool exists, never that the plan
runs. So the fix is not another assertion about the source, it is actually
running a co-registered pair through `answer_query` and requiring an answer.

The pair is synthesised rather than fixtured -- no data dependency -- but built
so both thresholds clear their floors. A pair where either mask comes back empty
never reaches fusion, and a test that silently skips the branch it exists to
cover is worse than no test.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

rasterio = pytest.importorskip("rasterio")


def _pair(tmp_path: Path):
    from verify_routes import _synthetic_pair

    return _synthetic_pair(tmp_path)


def test_a_crossmodal_query_returns_an_answer(tmp_path):
    from satquery.agent.pipeline import answer_query

    optical, sar = _pair(tmp_path)
    outcome = answer_query(
        "Use the optical and SAR images together to identify built-up and "
        "water-covered regions.",
        [str(optical), str(sar)],
        modalities=["optical", "sar"],
        output_dir=tmp_path,
        write_evidence=False,
    )
    assert not outcome.refused, outcome.answer
    assert outcome.answer


def test_the_fusion_branch_is_actually_reached(tmp_path):
    """The point of the test. Without this the pair could route crossmodal,
    execute cleanly, and still never touch the line that was broken."""
    from satquery.agent.pipeline import answer_query

    optical, sar = _pair(tmp_path)
    outcome = answer_query(
        "Use the optical and SAR images together to identify built-up and "
        "water-covered regions.",
        [str(optical), str(sar)],
        modalities=["optical", "sar"],
        output_dir=tmp_path,
        write_evidence=False,
    )
    trace = outcome.trace if isinstance(outcome.trace, dict) else {}
    assert trace.get("agreement") is not None, (
        "no agreement in the trace: both masks did not materialise, so "
        "fuse_masks and disagreement_hints never ran and this test proves nothing"
    )


def test_the_router_context_still_has_no_scenes():
    """The premise behind the bug. If `QueryContext` ever grows `scene_of`, the
    original confusion becomes harmless and this file can be reconsidered."""
    from satquery.agent.router import QueryContext
    from satquery.tools.base import ToolContext

    assert not hasattr(QueryContext, "scene_of")
    assert hasattr(ToolContext, "scene_of")
