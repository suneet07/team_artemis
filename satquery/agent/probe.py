"""Bundle probe: for each task, decide whether this bundle can support it (§21).

Reuses validate_query_compatibility rules and applies the §15 permitted-tool
∩ BandInventory filter so the supported/blocked classification reflects actual
tool reachability, not just validator rules.
"""

from satquery.agent.bundle import ImageBundle
from satquery.agent.planner import PERMITTED_TOOLS
from satquery.agent.task_enum import Task
from satquery.agent.validator import validate_query_compatibility


def probe_bundle(bundle: ImageBundle) -> tuple[list[str], list[dict[str, str]]]:
    """For each of the 8 tasks, decide whether this bundle can support it (§21).

    Applies:
    1. validate_query_compatibility rules (V1–V9) per bundle + task.
    2. §15 permitted-tool set filtered by BandInventory.computable_indices —
       if the permitted tools for a task are ALL blocked by missing bands,
       that task is blocked with reason "No permitted tools have computable bands."

    Returns (supported_tasks, blocked_tasks).
    """
    supported: list[str] = []
    blocked: list[dict[str, str]] = []

    modalities = {img.modality for img in bundle.images}
    is_sar_only = bool(modalities == {"sar"})
    has_optical = "optical" in modalities
    has_sar = "sar" in modalities
    computable = set(bundle.band_inventory.computable_indices if bundle.band_inventory else [])

    for task in Task:
        # Cross-modal bundle requirement check
        if task in (Task.CROSSMODAL_EXTRACTION, Task.CROSSMODAL_VQA):
            if not (has_optical and has_sar) and bundle.pair_type != "crossmodal":
                blocked.append({
                    "task": task.value,
                    "reason": "Cross-modal tasks require both optical and SAR acquisitions.",
                })
                continue

        # Single captioning requirement check (optical only)
        if task == Task.SINGLE_CAPTION and is_sar_only:
            blocked.append({
                "task": task.value,
                "reason": "Captioning models require optical imagery.",
            })
            continue

        # Use a representative question for validator invocation
        # (question-dependent rules like V3 SAR + colour are exercised via modality checks above)
        representative_question = _representative_question(task)

        # Run validator on the bundle and task
        val = validate_query_compatibility(bundle, question=representative_question, task=task)
        if not val.passed and val.refusal:
            blocked.append({
                "task": task.value,
                "reason": val.refusal["reason"],
            })
            continue

        # §15 + BandInventory filter: check if any permitted tool can actually run
        permitted = set(PERMITTED_TOOLS.get(task, []))
        if permitted and computable:
            # Tools with requires_bands constraints may be dropped; check if at least
            # one tool in the permitted set does not require missing bands.
            # We conservatively consider the task reachable if at least one permitted
            # tool does not have a band requirement (i.e. it's not purely index-based).
            # Full check is performed at plan time; this is a lightweight probe.
            from satquery.tools.registry import ToolRegistry  # lazy import

            registry = ToolRegistry.default()
            all_blocked_by_bands = True
            for tool_name in permitted:
                if tool_name not in registry.names():
                    continue
                manifest = registry.get(tool_name)
                # Check if any param requires a band not in computable_indices
                requires_any_band = any(
                    spec.requires_bands
                    for spec in manifest.permitted_parameters.values()
                    if spec.requires_bands
                )
                if not requires_any_band:
                    # This tool has no band requirement — task is reachable
                    all_blocked_by_bands = False
                    break
                # requires_bands is a dict of {band_name: required}; check if any
                # required band is in the bundle's computable_indices
                tool_required_bands = [
                    band
                    for spec in manifest.permitted_parameters.values()
                    if spec.requires_bands
                    for band in (spec.requires_bands if isinstance(spec.requires_bands, list)
                                 else list(spec.requires_bands.keys())
                                 if isinstance(spec.requires_bands, dict)
                                 else [str(spec.requires_bands)])
                ]
                if any(b in computable for b in tool_required_bands):
                    all_blocked_by_bands = False
                    break

            if all_blocked_by_bands and permitted:
                blocked.append({
                    "task": task.value,
                    "reason": (
                        f"No permitted tools for {task.value!r} have computable bands "
                        f"(available: {sorted(computable)!r})."
                    ),
                })
                continue

        supported.append(task.value)

    return supported, blocked


def _representative_question(task: Task) -> str:
    """Returns a short representative question for each task used for validator invocation."""
    questions = {
        Task.SINGLE_VQA: "What is in this image?",
        Task.SINGLE_CAPTION: "Describe this satellite scene.",
        Task.SINGLE_GROUNDING: "Where is the building?",
        Task.CHANGE_DESCRIPTION: "Describe what changed.",
        Task.CHANGE_VQA: "Did the area change?",
        Task.CHANGE_MAP: "How much area changed?",
        Task.CROSSMODAL_EXTRACTION: "Extract features across optical and SAR.",
        Task.CROSSMODAL_VQA: "Compare optical and SAR observations.",
    }
    return questions.get(task, "describe scene")
