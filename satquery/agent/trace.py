import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

from satquery.agent.task_enum import RouterPath, Task
from satquery.paths import TRACE_SCHEMA_PATH

SCHEMA_VERSION = 2

_SCHEMA: dict[str, Any] | None = None


def _schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(TRACE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


def validate_trace(trace: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator(_schema()).validate(trace)


class TraceBuilder:
    def __init__(
        self,
        query_text: str,
        query_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        self._trace: dict[str, Any] = {
            "query_id": query_id or str(uuid.uuid4()),
            "timestamp": timestamp or datetime.now(UTC).isoformat(),
            "query_text": query_text,
        }
        self._task: Task | None = None
        self._router_path: str | None = None
        self._plan: list[dict[str, Any]] = []
        self._parameter_check: dict[str, Any] | None = None
        self._tools_invoked: list[str] | None = None
        self._steps: list[dict[str, Any]] = []
        self._outputs: dict[str, Any] = {}

    def set_routing(self, task: Task, router_path: RouterPath | str) -> "TraceBuilder":
        """Record which path chose the task: the rule router or the LLM fallback.

        Written into ``self._trace``, not only onto the builder. ``build()``
        emits ``self._trace``; an attribute set beside it is checked for
        presence and then dropped, which is what happened here -- every trace
        validated as routed and every trace shipped ``router_path`` absent.

        It matters because it is the honest account of *how* the task was
        chosen. Section 4.5 measures rule-router accuracy against the LLM
        fallback rate, and the frontend renders the field; a trace that omits
        it cannot support either.
        """
        self._task = task
        if isinstance(router_path, RouterPath):
            router_path = router_path.value
        self._router_path = router_path
        self._trace["router_path"] = router_path
        return self

    def set_inputs(self, inputs: list[dict[str, Any]]) -> "TraceBuilder":
        self._trace["inputs"] = inputs
        return self

    def set_compatibility(self, compatibility: dict[str, Any]) -> "TraceBuilder":
        self._trace["compatibility"] = compatibility
        return self

    def add_routing_note(self, note: str) -> "TraceBuilder":
        self._trace.setdefault("routing_notes", []).append(note)
        return self

    def add_planned_step(
        self,
        tool: str,
        params: dict[str, Any],
        within_manifest: bool,
        defaults_applied: list[str] | None = None,
    ) -> "TraceBuilder":
        record: dict[str, Any] = {
            "tool": tool,
            "params": params,
            "within_manifest": within_manifest,
        }
        if defaults_applied:
            record["defaults_applied"] = defaults_applied
        self._plan.append(record)
        return self

    def set_parameter_check(self, passed: bool, rejected: list[str]) -> "TraceBuilder":
        self._parameter_check = {"passed": passed, "rejected": rejected}
        return self

    def set_tools_invoked(self, tools: list[str]) -> "TraceBuilder":
        self._tools_invoked = tools
        return self

    def add_step(
        self,
        tool: str,
        params: dict[str, Any],
        outputs: dict[str, Any],
        confidence: float | None = None,
        latency_ms: int | None = None,
        param_source: str | None = "manifest_validated",
    ) -> "TraceBuilder":
        step: dict[str, Any] = {"tool": tool, "params": params}
        if param_source is not None:
            step["param_source"] = param_source
        step["outputs"] = outputs
        if confidence is not None:
            step["confidence"] = confidence
        if latency_ms is not None:
            step["latency_ms"] = latency_ms
        self._steps.append(step)
        return self

    def set_agreement(
        self,
        iou: float | None,
        verdict: str | None,
        disagreement_cause: str | None,
    ) -> "TraceBuilder":
        self._trace["agreement"] = {
            "iou": iou,
            "verdict": verdict,
            "disagreement_cause": disagreement_cause,
        }
        return self

    def set_fusion(
        self, model: str, answer: str | None, confidence: float | None
    ) -> "TraceBuilder":
        self._trace["fusion"] = {"model": model, "answer": answer, "confidence": confidence}
        return self

    def set_outputs(
        self,
        answer: str | None = None,
        masks: list[str] | None = None,
        area_km2: float | None = None,
        confidence: float | None = None,
        extra: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
    ) -> "TraceBuilder":
        outputs: dict[str, Any] = {}
        if answer is not None:
            outputs["answer"] = answer
        if masks is not None:
            outputs["masks"] = masks
        if area_km2 is not None:
            outputs["area_km2"] = area_km2
        if confidence is not None:
            outputs["confidence"] = confidence
        if refusal is not None:
            outputs["refusal"] = refusal
        if extra:
            outputs.update(extra)
        self._outputs = outputs
        return self

    def add_evidence(self, item: str) -> "TraceBuilder":
        self._trace.setdefault("evidence", []).append(item)
        return self

    def add_warning(self, warning: str) -> "TraceBuilder":
        self._trace.setdefault("warnings", []).append(warning)
        return self

    @property
    def warnings(self) -> list[str]:
        """Warnings recorded so far; the confidence layer counts them (4.7.3)."""
        return list(self._trace.get("warnings", []))

    @property
    def routing_notes(self) -> list[str]:
        return list(self._trace.get("routing_notes", []))

    def build(self) -> dict[str, Any]:
        if self._task is None or self._router_path is None:
            raise ValueError("routing not set: call set_routing() before build()")
        if self._parameter_check is None:
            raise ValueError(
                "parameter_check not recorded: call set_parameter_check() before build()"
            )
        tools = self._tools_invoked
        if tools is None:
            tools = list(dict.fromkeys(step["tool"] for step in self._steps))
        graded: dict[str, Any] = {
            "task_selected": self._task.value,
            "tools_invoked": tools,
            "permitted_parameters": self._plan,
            "parameter_check": self._parameter_check,
            "outputs": self._outputs,
        }
        ordered: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for key, value in self._trace.items():
            ordered[key] = value
            if key == "query_text":
                ordered["graded"] = graded
                ordered["steps"] = self._steps
        validate_trace(ordered)
        return ordered

    def write_json(self, path: Path) -> dict[str, Any]:
        trace = self.build()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return trace
