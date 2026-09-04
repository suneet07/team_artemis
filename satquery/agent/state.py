from collections.abc import Callable
from typing import Any, TypedDict

from satquery.agent.asset import AssetRef
from satquery.agent.bundle import ImageBundle
from satquery.agent.task_enum import RouterPath, Task
from satquery.agent.trace import TraceBuilder
from satquery.ingest.band_inventory import BandInventory


class AgentState(TypedDict, total=False):
    query_id: str
    query_text: str
    bundle: ImageBundle
    band_inventory: BandInventory
    modalities: list[str]
    pair_type: str
    allow_llm_router: bool
    task: Task | None
    router_path: RouterPath | None
    routing_notes: list[str]
    validation_ok: bool
    refusal: dict[str, Any] | None
    failures: list[dict[str, Any]]
    tile_plan: Any | None
    plan: list[dict[str, Any]]
    gate_passed: bool
    gate_rejected: list[str]
    replan_count: int
    results: dict[str, Any]
    mask_cache: dict[str, Any]
    agreement: dict[str, Any] | None
    fused_answer: str | None
    confidence: float | None
    confidence_basis: str | None
    assets: list[AssetRef]
    warnings: list[str]
    trace: TraceBuilder
    emit: Callable[[str, dict[str, Any]], None]
    timings: dict[str, int]
    query_state: str
    trace_dict: dict[str, Any]
