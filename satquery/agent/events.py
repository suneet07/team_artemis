from typing import Any


def emit_event(emit: callable, event_name: str, payload: dict[str, Any]) -> None:
    """
    Helper to emit an SSE event using the provided emit callback.
    The callback signature is expected to be `emit(event_name: str, payload: dict)`.
    """
    if emit is not None:
        emit(event_name, payload)


# Specific event helpers

def emit_accepted(emit: callable, query_id: str, queued_ms: int = 0) -> None:
    emit_event(emit, "accepted", {"query_id": query_id, "queued_ms": queued_ms})


def emit_router(
    emit: callable, router_path: str, task_selected: str | None, notes: list[str]
) -> None:
    emit_event(emit, "router", {
        "router_path": router_path,
        "task_selected": task_selected,
        "notes": notes
    })


def emit_validator(emit: callable, passed: bool, rejections: list[str]) -> None:
    emit_event(emit, "validator", {
        "passed": passed,
        "rejections": rejections
    })


def emit_plan(emit: callable, plan: list[dict[str, Any]]) -> None:
    emit_event(emit, "plan", {"plan": plan})


def emit_step_started(emit: callable, tool: str) -> None:
    emit_event(emit, "step_started", {"tool": tool})


def emit_step_completed(emit: callable, tool: str, result: dict[str, Any]) -> None:
    emit_event(emit, "step_completed", {"tool": tool, "result": result})


def emit_agreement(emit: callable, winning_modality: str, explanation: str) -> None:
    emit_event(emit, "agreement", {
        "winning_modality": winning_modality,
        "explanation": explanation
    })


def emit_fusion(emit: callable, fused_answer: str, confidence: float) -> None:
    emit_event(emit, "fusion", {
        "fused_answer": fused_answer,
        "confidence": confidence
    })


def emit_done(emit: callable, query_id: str) -> None:
    emit_event(emit, "done", {"query_id": query_id})
