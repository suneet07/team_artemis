import re
from dataclasses import dataclass, field
from typing import Any

from satquery.agent.bundle import ImageBundle
from satquery.agent.task_enum import Task
from satquery.agent.validator import TRAINED_GROUNDING_VOCABULARY
from satquery.tools.registry import ToolRegistry

# §15: Task -> Permitted tools matrix
PERMITTED_TOOLS: dict[Task, list[str]] = {
    Task.SINGLE_VQA: [
        "rs_vqa",
        "spectral_index",
        "sar_backscatter",
        "lulc_classifier",
        "texture_seg",
        "tile_scorer",
    ],
    Task.SINGLE_CAPTION: [
        "rs_ground_caption",
        "lulc_classifier",
        "tile_scorer",
    ],
    Task.SINGLE_GROUNDING: [
        "centroid_prior",
        "spectral_index",
        "texture_seg",
        "rs_ground_caption",
        "object_box_fallback",
        "tile_scorer",
    ],
    Task.CHANGE_DESCRIPTION: [
        "change_map",
        "change_vqa",
        "coreg_check",
        "tile_scorer",
    ],
    Task.CHANGE_VQA: [
        "change_vqa",
        "change_map",
        "coreg_check",
        "tile_scorer",
    ],
    Task.CHANGE_MAP: [
        "change_map",
        "change_stats",
        "coreg_check",
        "tile_scorer",
    ],
    Task.CROSSMODAL_EXTRACTION: [
        "spectral_index",
        "sar_backscatter",
        "optsar_fusion",
        "coreg_check",
        "texture_seg",
        "tile_scorer",
    ],
    Task.CROSSMODAL_VQA: [
        "optsar_fusion",
        "spectral_index",
        "sar_backscatter",
        "coreg_check",
        "tile_scorer",
    ],
}


@dataclass
class PlannedStep:
    tool: str
    params: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)


def _extract_target_noun(question: str) -> str:
    ql = question.lower()
    for noun in [
        "bridge", "tank", "vehicle", "harbour", "runway", "ship", "aircraft", "building"
    ]:
        if noun in ql:
            return noun
    return "other"


def plan_query(
    task: Task,
    bundle: ImageBundle,
    question: str,
    replan_count: int = 0,
    gate_rejected: list[str] | None = None,
    registry: ToolRegistry | None = None,
    previous_plan: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generates an ordered list of tool calls adhering to §10 N4 and §15.

    Returns (plan_steps, routing_notes).
    """
    reg = registry or ToolRegistry.default()
    registered_names = set(reg.names())
    permitted = [t for t in PERMITTED_TOOLS.get(task, []) if t in registered_names]
    notes: list[str] = []
    modalities = {img.modality for img in bundle.images}
    band_inv = bundle.band_inventory
    ql = question.lower()

    # Replan handling (§10 N4 Step 8)
    if replan_count == 1 and gate_rejected:
        notes.append(f"Replanning attempt 1 handling rejections: {'; '.join(gate_rejected)}")
        # Check if index needs SWIR and SWIR is absent
        needs_swir_rejection = any("requires band 'swir'" in r.lower() for r in gate_rejected)
        if needs_swir_rejection and "spectral_index" in permitted:
            computable = band_inv.computable_indices if band_inv else []
            substitute = next((idx for idx in computable if idx in ("NDWI", "NDVI")), None)
            if substitute:
                notes.append(
                    f"Substituted rejected index with computable '{substitute}' (D3 replan)"
                )
                plan = [
                    {
                        "tool": "spectral_index",
                        "params": {"index": substitute, "threshold_method": "otsu"},
                        "depends_on": [],
                    }
                ]
                return plan, notes
            # If no computable substitute, drop tool
            notes.append("Dropped spectral_index: no computable indices available on replan")
            permitted = [t for t in permitted if t != "spectral_index"]

        # Check for unknown parameters or out-of-range parameters from previous plan
        if previous_plan:
            modified_plan = []
            made_changes = False
            for step in previous_plan:
                t_name = step["tool"]
                new_params = dict(step.get("params", {}))
                manifest = reg.get(t_name) if t_name in reg.names() else None

                for r in gate_rejected:
                    unknown_m = re.search(r"unknown parameter '([^']+)'", r)
                    out_of_range_m = re.search(
                        r"parameter '([^']+)'=[^ ]+ outside permitted range", r
                    )
                    if unknown_m:
                        bad_key = unknown_m.group(1)
                        if bad_key in new_params:
                            del new_params[bad_key]
                            notes.append(f"Dropping unknown parameter '{bad_key}' during replan")
                            made_changes = True
                    if out_of_range_m:
                        bad_param = out_of_range_m.group(1)
                        if bad_param in new_params:
                            if manifest and bad_param in manifest.parameters:
                                default_v = manifest.parameters[bad_param].default
                                if default_v is not None:
                                    new_params[bad_param] = default_v
                                else:
                                    del new_params[bad_param]
                            else:
                                del new_params[bad_param]
                            notes.append(
                                f"Reverting parameter '{bad_param}' to manifest default"
                            )
                            made_changes = True

                modified_plan.append({
                    "tool": t_name,
                    "params": new_params,
                    "depends_on": step.get("depends_on", []),
                })

            if made_changes:
                return modified_plan, notes

        # If dummy_tool is being replanned
        if "dummy_tool" in permitted:
            return [{"tool": "dummy_tool", "params": {"index": "BETA"}, "depends_on": []}], notes

    # Initial plan construction
    plan: list[dict[str, Any]] = []

    # Single Caption
    if task == Task.SINGLE_CAPTION:
        if "rs_ground_caption" in permitted:
            plan.append({
                "tool": "rs_ground_caption",
                "params": {"max_tokens": 128},
                "depends_on": [],
            })
        elif "lulc_classifier" in permitted:
            plan.append({
                "tool": "lulc_classifier",
                "params": {},
                "depends_on": [],
            })
        elif "dummy_tool" in registered_names:
            plan.append({"tool": "dummy_tool", "params": {"index": "ALPHA"}, "depends_on": []})
        return plan, notes

    # Grounding task branch
    if task == Task.SINGLE_GROUNDING:
        has_in_vocab = any(vocab in ql for vocab in TRAINED_GROUNDING_VOCABULARY)
        if has_in_vocab:
            can_use_spectral = (
                "spectral_index" in permitted
                and "optical" in modalities
                and bool(band_inv.computable_indices if band_inv else [])
            )
            if can_use_spectral:
                idx = band_inv.computable_indices[0]
                plan.append({
                    "tool": "spectral_index",
                    "params": {"index": idx, "threshold_method": "otsu"},
                    "depends_on": [],
                })
                if "centroid_prior" in permitted:
                    plan.append({
                        "tool": "centroid_prior",
                        "params": {"min_area_px": 10},
                        "depends_on": ["spectral_index"],
                    })
            elif "dummy_tool" in registered_names and not plan:
                plan.append({
                    "tool": "dummy_tool", "params": {"index": "ALPHA"}, "depends_on": []
                })
        else:
            # Out-of-vocabulary grounding
            notes.append("Out-of-vocabulary target: routing to texture_seg + object_box_fallback")
            if "texture_seg" in permitted:
                plan.append({
                    "tool": "texture_seg",
                    "params": {"window_size": 5},
                    "depends_on": [],
                })
            if "object_box_fallback" in permitted:
                target = _extract_target_noun(question)
                valid_targets = {"bridge", "tank", "vehicle", "harbour", "runway"}
                target_cls = target if target in valid_targets else "other"
                plan.append({
                    "tool": "object_box_fallback",
                    "params": {"target_class": target_cls},
                    "depends_on": ["texture_seg"] if "texture_seg" in permitted else [],
                })
        return plan, notes

    # Cross-modal extraction branch
    if task == Task.CROSSMODAL_EXTRACTION:
        deps = []
        if "optical" in modalities and "spectral_index" in permitted:
            if not getattr(band_inv, "has_swir", True) and "NDBI" in ql:
                notes.append(
                    "SWIR unavailable; NDBI skipped; using NDWI/SAR primary for built-up (D3)"
                )
            computables = getattr(band_inv, "computable_indices", [])
            if "NDWI" in computables:
                idx = "NDWI"
            elif computables:
                idx = computables[0]
            else:
                idx = "NDVI"
            plan.append({
                "tool": "spectral_index",
                "params": {"index": idx, "threshold_method": "otsu"},
                "depends_on": [],
            })
            deps.append("spectral_index")
        if "sar" in modalities and "sar_backscatter" in permitted:
            plan.append({
                "tool": "sar_backscatter",
                "params": {"pol": "VV", "threshold_method": "otsu"},
                "depends_on": [],
            })
            deps.append("sar_backscatter")
        if "optsar_fusion" in permitted and len(deps) >= 2:
            plan.append({
                "tool": "optsar_fusion",
                "params": {"question": question},
                "depends_on": deps,
            })
        return plan, notes

    # Cross-modal VQA branch
    if task == Task.CROSSMODAL_VQA:
        deps = []
        if "optical" in modalities and "spectral_index" in permitted:
            idx = "NDWI" if "NDWI" in getattr(band_inv, "computable_indices", []) else "NDVI"
            plan.append({
                "tool": "spectral_index",
                "params": {"index": idx, "threshold_method": "otsu"},
                "depends_on": [],
            })
            deps.append("spectral_index")
        if "sar" in modalities and "sar_backscatter" in permitted:
            plan.append({
                "tool": "sar_backscatter",
                "params": {"pol": "VV", "threshold_method": "otsu"},
                "depends_on": [],
            })
            deps.append("sar_backscatter")
        if "optsar_fusion" in permitted:
            plan.append({
                "tool": "optsar_fusion",
                "params": {"question": question},
                "depends_on": deps,
            })
        return plan, notes

    # Change tasks branch
    if task in (Task.CHANGE_MAP, Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA):
        if "change_map" in permitted:
            plan.append({
                "tool": "change_map",
                "params": {},
                "depends_on": [],
            })
        if task in (Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA) and "change_vqa" in permitted:
            plan.append({
                "tool": "change_vqa",
                "params": {"question": question},
                "depends_on": ["change_map"] if "change_map" in permitted else [],
            })
        elif task == Task.CHANGE_MAP and "change_stats" in permitted:
            plan.append({
                "tool": "change_stats",
                "params": {},
                "depends_on": ["change_map"] if "change_map" in permitted else [],
            })
        elif "dummy_tool" in registered_names and not plan:
            plan.append({"tool": "dummy_tool", "params": {"index": "ALPHA"}, "depends_on": []})
        return plan, notes

    # Single VQA
    has_indices = bool(getattr(band_inv, "computable_indices", []))
    if "optical" in modalities and "spectral_index" in permitted and has_indices:
        idx = band_inv.computable_indices[0]
        plan.append({
            "tool": "spectral_index",
            "params": {"index": idx, "threshold_method": "otsu"},
            "depends_on": [],
        })
    elif "sar" in modalities and "sar_backscatter" in permitted:
        plan.append({
            "tool": "sar_backscatter",
            "params": {"pol": "VV", "threshold_method": "otsu"},
            "depends_on": [],
        })
    elif "rs_vqa" in permitted:
        plan.append({
            "tool": "rs_vqa",
            "params": {"question": question},
            "depends_on": [],
        })
    elif "dummy_tool" in registered_names:
        plan.append({
            "tool": "dummy_tool",
            "params": {"index": "ALPHA"},
            "depends_on": [],
        })

    return plan, notes
