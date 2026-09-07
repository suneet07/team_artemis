"""P5 — plan construction (master plan section 4.5.2).

The plan is the smallest sufficient tool DAG for a task, and it is built by
:func:`satquery.agent.router.plan_for`, which is where the band inventory lives.
That co-location is deliberate: **D3 is a planning decision**. Whether
``spectral_index`` can enter a plan at all depends on which bands the source
has, so a planner that cannot see the inventory has to plan optimistically and
let the gate reject the step afterwards — which shows up in the graded trace as
a rejection we caused ourselves.

What the earlier version of this module did wrong, all of it visible in a trace:

* ``SINGLE_CAPTION`` planned ``rs_ground_caption``. Captioning is ``rs_vqa``
  (section 4.6.1); the grounding adapter emits boxes.
* ``SINGLE_GROUNDING`` planned ``centroid_prior`` **first**, with the adapter
  depending on it — but a centroid needs a mask, and nothing in the plan produced
  one, so D2's point prior was always ``None``.
* There was no ``object_box_fallback`` branch at all ("Simplified: always
  assuming in-vocab"), so a query for a class outside the trained vocabulary —
  the exact case that tool exists for, and one the plan puts on the never-cut
  list — scored zero.
* ``CROSSMODAL_EXTRACTION`` planned ``spectral_index`` with no ``index``, which
  the parameter gate must reject, so the centrepiece demo could not run.
* ``coreg_check`` and ``tile_scorer`` never appeared in any plan, so P3 and P4
  never appeared in the trace.
* ``except Exception: pass`` around manifest lookup hid every one of these.
"""

from satquery.agent.router import GROUNDING_VOCABULARY, QueryContext, plan_for
from satquery.agent.state import PlannedStep
from satquery.agent.task_enum import Task
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.registry import ToolRegistry, effective_params

__all__ = ["plan_for_task"]


def plan_for_task(
    task: Task,
    band_inventory: BandInventory,
    modalities: list[str],
    registry: ToolRegistry,
    question_noun: str = "",
    image_count: int | None = None,
) -> list[PlannedStep]:
    """Build the plan for ``task`` under the bands and modalities available.

    ``question_noun`` is the class a grounding query names. It decides between
    the learned adapter and the deterministic proposer, which is the C39 split:
    covered classes (regions, buildings, aircraft, ships) go to
    ``rs_ground_caption``; everything else goes to ``object_box_fallback`` and is
    labelled a deterministic proposal in the trace.
    """
    if image_count is None:
        image_count = 2 if task.value.startswith(("change", "crossmodal")) else 1

    targets = [question_noun] if question_noun else []
    context = QueryContext(
        query_text=question_noun,
        modalities=list(modalities),
        inventories=[band_inventory],
        image_count=image_count,
    )
    plan = plan_for(task, context, targets)

    resolved: list[PlannedStep] = []
    for step in plan:
        params = dict(step.get("params", {}))
        # Fail loudly on an unknown tool. The previous `except Exception: pass`
        # meant a typo in a tool name produced a plan that looked fine here and
        # was rejected at the gate, with the cause three layers away.
        manifest = registry.get(step["tool"])
        resolved.append(
            {
                "tool": step["tool"],
                "params": effective_params(manifest, params),
                "within_manifest": True,
                "depends_on": list(step.get("depends_on") or []),
            }
        )
    return resolved


def grounding_is_in_vocabulary(noun: str) -> bool:
    """Whether the learned adapter covers this class (C36/C37/C39/C53)."""
    return noun.strip().lower() in GROUNDING_VOCABULARY
