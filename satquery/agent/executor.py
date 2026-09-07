"""P5 — the parameter gate and the DAG executor (sections 4.5.4 and 4.5.6).

The order is the whole point:

1. the router produces a plan;
2. the validator gate accepts, reroutes or refuses it (4.5.3);
3. **the parameter gate checks every step against its manifest (4.5.4)**;
4. only then does any node run, appending to ``steps[]`` (4.5.6).

Two things this module used to get wrong, both of which changed answers:

* **``depends_on`` was declared and ignored.** Every step was submitted to the
  pool at once, so ``change_stats`` could run before ``change_map`` produced the
  map it consumes, and ``centroid_prior`` before the mask it takes a centroid
  of. Dependencies are now honoured: independent steps run in parallel,
  dependent ones wait for their producers, and a cycle is an error rather than a
  hang.
* **A missing tool raised.** One learned adapter that has not landed yet would
  take the whole query down, which inverts the plan's scheduling rule that
  training is never on the demo's critical path. A tool with no implementation
  is now a recorded gap and the deterministic path answers without it.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from satquery.agent.trace import TraceBuilder
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.base import ToolContext, ToolResult
from satquery.tools.catalog import implementation
from satquery.tools.registry import ToolRegistry, check_parameters, effective_params

__all__ = [
    "DagExecutor",
    "ExecutedStep",
    "ExecutionResult",
    "GateOutcome",
    "check_plan",
    "execute_plan",
]


@dataclass
class GateOutcome:
    """Result of the section 4.5.4 gate over a whole plan."""

    passed: bool
    rejected: list[str] = field(default_factory=list)
    #: One entry per planned step, in plan order. This is what the trace's
    #: ``graded.permitted_parameters`` block is built from -- a graded field,
    #: so it records rejected steps too rather than omitting them.
    checked: list[dict[str, Any]] = field(default_factory=list)
    runnable: list[dict[str, Any]] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


@dataclass
class ExecutedStep:
    tool: str
    params: dict[str, Any]
    result: ToolResult
    latency_ms: int


@dataclass
class ExecutionResult:
    steps: list[ExecutedStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def tools_invoked(self) -> list[str]:
        return list(dict.fromkeys(step.tool for step in self.steps))

    def output_of(self, tool: str) -> dict[str, Any] | None:
        for step in self.steps:
            if step.tool == tool:
                return step.result.outputs
        return None


def check_plan(
    plan: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    band_inventory: BandInventory | None = None,
    modalities: list[str] | None = None,
) -> GateOutcome:
    """Validate every planned step against its manifest before anything runs."""
    rejected: list[str] = []
    checked: list[dict[str, Any]] = []
    runnable: list[dict[str, Any]] = []
    unavailable: list[str] = []

    for step in plan:
        name = step["tool"]
        try:
            manifest = registry.get(name)
        except KeyError:
            rejected.append(f"tool '{name}' is not in the registry")
            checked.append(
                {"tool": name, "params": dict(step.get("params", {})), "within_manifest": False}
            )
            continue

        supplied = dict(step.get("params", {}))
        merged = effective_params(manifest, supplied)
        defaults_applied = sorted(set(merged) - set(supplied))
        outcome = check_parameters(
            manifest, merged, band_inventory=band_inventory, modalities=modalities
        )
        record: dict[str, Any] = {
            "tool": name,
            "params": merged,
            "within_manifest": outcome.passed,
        }
        if defaults_applied:
            record["defaults_applied"] = defaults_applied
        checked.append(record)

        if not outcome.passed:
            rejected.extend(outcome.rejected)
            continue
        if implementation(name) is None:
            # A learned tool whose adapter has not landed. Recorded, not fatal:
            # the whole app is built against the zero-shot base and upgraded as
            # each adapter arrives (Part 8, the scheduling rule).
            unavailable.append(name)
            continue
        runnable.append(
            {"tool": name, "params": merged, "depends_on": list(step.get("depends_on") or [])}
        )

    return GateOutcome(
        passed=not rejected,
        rejected=rejected,
        checked=checked,
        runnable=runnable,
        unavailable=unavailable,
    )


def _run_one(step: dict[str, Any], context: ToolContext) -> tuple[ExecutedStep | None, str | None]:
    tool = implementation(step["tool"])
    if tool is None:
        return None, f"tool '{step['tool']}' has no implementation registered"
    # Each step sees its own gate-approved parameters over the shared scene and
    # artifact state. Sharing one params dict would let the last step planned
    # decide what every step runs with.
    step_context = replace(context, params=dict(step["params"]))
    step_context.artifacts = context.artifacts
    started = time.perf_counter()
    try:
        result = tool.run(step_context)
    except Exception as error:  # noqa: BLE001 - one tool must not lose the query
        return None, f"tool '{step['tool']}' failed: {type(error).__name__}: {error}"
    return (
        ExecutedStep(
            step["tool"],
            dict(step["params"]),
            result,
            int((time.perf_counter() - started) * 1000),
        ),
        None,
    )


def _waves(runnable: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group steps into dependency waves; each wave runs in parallel."""
    pending = {step["tool"]: step for step in runnable}
    done: set[str] = set()
    waves: list[list[dict[str, Any]]] = []
    while pending:
        ready = [
            step
            for step in pending.values()
            # A dependency that is not in this plan at all was dropped by the
            # validator or has no adapter; waiting for it forever would hang.
            if all(dep in done or dep not in pending for dep in step.get("depends_on", []))
        ]
        if not ready:
            raise ValueError(
                f"circular dependency in the plan among {sorted(pending)}; a plan the "
                f"executor cannot order is a planner bug, not something to run partially"
            )
        waves.append(ready)
        for step in ready:
            done.add(step["tool"])
            pending.pop(step["tool"])
    return waves


def execute_plan(
    runnable: list[dict[str, Any]],
    context: ToolContext,
    *,
    max_workers: int = 4,
) -> ExecutionResult:
    """Execute a gate-approved plan, parallel within each dependency wave."""
    execution = ExecutionResult(artifacts=context.artifacts)

    def absorb(step: ExecutedStep | None, failure: str | None) -> None:
        if failure:
            execution.failures.append(failure)
            execution.warnings.append(failure)
            return
        assert step is not None
        execution.steps.append(step)
        execution.warnings.extend(step.result.warnings)
        if step.result.mask is not None:
            context.artifacts.setdefault("masks", {})[step.tool] = step.result.mask
            context.artifacts["mask"] = step.result.mask
            context.artifacts["mask_crs"] = step.result.mask_crs
            context.artifacts["mask_transform"] = step.result.mask_transform
        if step.result.response is not None:
            context.artifacts.setdefault("responses", {})[step.tool] = step.result.response

    for wave in _waves(runnable):
        if len(wave) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(wave))) as pool:
                results = [
                    future.result()
                    for future in [pool.submit(_run_one, step, context) for step in wave]
                ]
            for step, failure in results:
                absorb(step, failure)
        else:
            absorb(*_run_one(wave[0], context))

    execution.artifacts = context.artifacts
    return execution


class DagExecutor:
    """Callable-registry executor kept for callers holding plain functions.

    The agent path uses :func:`execute_plan` with the real tool objects. This
    wrapper exists for the array-in/dict-out call style, and unlike its earlier
    version it honours ``depends_on`` and records real latency rather than
    reporting ``0`` for every step.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def execute_plan(
        self,
        plan: list[dict[str, Any]],
        tool_registry: dict[str, Any],
        trace_builder: TraceBuilder,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for wave in _waves([dict(step) for step in plan]):
            results: list[tuple[dict[str, Any], Any, int, str | None]] = []

            def call(step: dict[str, Any]):
                name = step["tool"]
                function = tool_registry.get(name)
                if function is None:
                    return step, None, 0, f"tool '{name}' is not in the registry"
                started = time.perf_counter()
                try:
                    value = function(step.get("params", {}))
                except Exception as error:  # noqa: BLE001
                    return step, None, 0, f"{type(error).__name__}: {error}"
                return step, value, int((time.perf_counter() - started) * 1000), None

            if len(wave) > 1:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(wave))) as pool:
                    results = [f.result() for f in [pool.submit(call, s) for s in wave]]
            else:
                results = [call(wave[0])]

            for step, value, latency_ms, failure in results:
                name = step["tool"]
                if failure:
                    trace_builder.add_warning(f"Tool {name} failed: {failure}")
                    outputs[name] = {"error": failure, "confidence": 0.0}
                    continue
                trace_builder.add_step(
                    tool=name,
                    params=step.get("params", {}),
                    outputs=value,
                    confidence=value.get("confidence"),
                    latency_ms=latency_ms,
                    param_source="manifest_validated"
                    if step.get("within_manifest")
                    else "override",
                )
                outputs[name] = value
        return outputs

    def shutdown(self) -> None:
        """No-op: pools are now scoped to a wave and closed when it finishes."""
