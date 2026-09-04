from satquery.agent.bundle import ImageBundle
from satquery.agent.task_enum import Task
from satquery.agent.validator import validate_query_compatibility


def probe_bundle(bundle: ImageBundle) -> tuple[list[str], list[dict[str, str]]]:
    """For each of the 8 tasks, decide whether this bundle can support it (§21).

    Reuses validate_query_compatibility rules that depend on bundle properties.
    Returns (supported_tasks, blocked_tasks).
    """
    supported: list[str] = []
    blocked: list[dict[str, str]] = []

    modalities = {img.modality for img in bundle.images}
    is_sar_only = bool(modalities == {"sar"})
    has_optical = "optical" in modalities
    has_sar = "sar" in modalities

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

        # Run validator on the bundle and task
        val = validate_query_compatibility(bundle, question="describe scene", task=task)
        if not val.passed and val.refusal:
            blocked.append({
                "task": task.value,
                "reason": val.refusal["reason"],
            })
            continue

        supported.append(task.value)

    return supported, blocked

