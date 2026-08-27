from typing import Any

from satquery.agent.task_enum import Task
from satquery.agent.trace import TraceBuilder
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.dummy_tool import execute
from satquery.tools.manifest import ToolManifest
from satquery.tools.registry import ToolRegistry, check_parameters, effective_params


def _permitted_summary(manifest: ToolManifest) -> str:
    return ", ".join(
        f"{name}:{spec.type}" for name, spec in manifest.permitted_parameters.items()
    )


def _refusal_answer(manifest: ToolManifest, reasons: list[str]) -> str:
    return (
        f"Refused: parameter validation failed ({'; '.join(reasons)}). "
        f"{manifest.name} accepts: {_permitted_summary(manifest)}."
    )


def run_dummy_query(
    query_text: str,
    params: dict[str, Any],
    band_inventory: BandInventory | None = None,
    modality: str | list[str] | None = None,
) -> dict[str, Any]:
    manifest = ToolRegistry.default().get("dummy_tool")
    merged = effective_params(manifest, params)
    defaults_applied = sorted(set(merged) - set(params))
    check = check_parameters(
        manifest, merged, band_inventory=band_inventory, modalities=modality
    )
    builder = (
        TraceBuilder(query_text)
        .set_routing(Task.SINGLE_VQA, "rules")
        .add_planned_step(
            manifest.name,
            merged,
            within_manifest=check.passed,
            defaults_applied=defaults_applied or None,
        )
        .set_parameter_check(check.passed, check.rejected)
    )
    if not check.passed:
        answer = _refusal_answer(manifest, check.rejected)
        builder.add_warning(answer)
        builder.set_outputs(
            answer=answer,
            confidence=0.0,
            refusal={"reason": "; ".join(check.rejected), "category": "parameter_gate"},
        )
        return builder.build()
    output = execute(merged)
    builder.add_step(manifest.name, merged, output, confidence=0.99)
    builder.set_outputs(answer=output["answer"], area_km2=output["area_km2"], confidence=0.99)
    return builder.build()
