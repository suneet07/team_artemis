import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from satquery.agent.asset import AssetRef
from satquery.agent.bundle import ImageBundle
from satquery.agent.events import EventSink, create_emitter
from satquery.agent.executor import execute_plan
from satquery.agent.planner import plan_query
from satquery.agent.refusals import create_refusal
from satquery.agent.router import route_query
from satquery.agent.state import AgentState
from satquery.agent.task_enum import RouterPath, Task
from satquery.agent.tiling_policy import decide_tile_plan
from satquery.agent.trace import TraceBuilder
from satquery.agent.validator import validate_query_compatibility
from satquery.confidence.calculator import calculate_confidence
from satquery.fusion.reconcile import reconcile_crossmodal
from satquery.serving.client import base_model
from satquery.tools.registry import ToolRegistry, check_parameters, effective_params


@dataclass
class QueryResult:
    query_id: str
    bundle_id: str
    question: str
    state: str  # queued | running | succeeded | refused | failed | cancelled
    answer: str | None
    confidence: float | None
    confidence_basis: str | None
    latency_ms: int | None
    refusal: dict[str, Any] | None
    failures: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[AssetRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def _fusion_model_name(results: dict[str, Any]) -> str:
    """Model string for the trace and the fusion event (§17).

    Names the base model, plus the learned adapter when one contributed, so the
    trace records what actually produced the answer rather than a bare product
    name.
    """
    base = base_model()
    for out in results.values():
        if isinstance(out, dict) and out.get("adapter"):
            # §17 writes this as base+tool@version; the adapter already carries
            # the base (Rule 9), so drop it from the suffix rather than repeat it.
            adapter = str(out["adapter"]).replace(f"@{base}-", "@", 1)
            return f"{base}+{adapter}"
    return base


def ingest_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    # N1 step 6: start the wall clock. Node timings measure work; only this
    # measures what the user waited, which is what the 20 s SLA is about.
    wall_start = state.get("wall_start") or start_t
    submitted_at = state.get("submitted_at")
    queued_ms = int((wall_start - submitted_at) * 1000) if submitted_at else 0
    query_id = state.get("query_id") or str(uuid.uuid4())
    question = state.get("query_text", "")
    bundle = state.get("bundle")

    emit = state.get("emit")
    if emit:
        emit("accepted", {"query_id": query_id, "queued_ms": max(0, queued_ms)})

    trace = state.get("trace") or TraceBuilder(query_text=question, query_id=query_id)
    warnings: list[str] = list(state.get("warnings") or [])

    if bundle is None or bundle.status != "ready":
        bundle_id = getattr(bundle, "bundle_id", "unknown") if bundle else "missing"
        status = getattr(bundle, "status", "missing") if bundle else "missing"
        refusal = create_refusal(
            category="missing_input",
            reason=f"Bundle '{bundle_id}' is not ready (status: '{status}')",
            action="none",
            label="Bundle not ready",
        )
        trace.set_routing(Task.SINGLE_VQA, RouterPath.RULES)
        trace.set_parameter_check(passed=False, rejected=["bundle_not_ready"])
        trace.add_warning(f"Bundle not ready: {refusal['reason']}")
        warnings.append(refusal["reason"])
        elapsed_ms = int((time.perf_counter() - start_t) * 1000)
        timings = dict(state.get("timings") or {})
        timings["ingest"] = elapsed_ms
        return {
            "query_id": query_id,
            "wall_start": wall_start,
            "refusal": refusal,
            "validation_ok": False,
            "warnings": warnings,
            "trace": trace,
            "timings": timings,
        }

    inputs: list[dict[str, Any]] = []
    for img in bundle.images:
        input_rec: dict[str, Any] = {
            "file": str(img.path),
            "modality": img.modality,
        }
        if img.native_gsd_m is not None:
            input_rec["native_gsd_m"] = img.native_gsd_m
        if img.pixel_size_m is not None:
            input_rec["pixel_size_m"] = img.pixel_size_m
        if img.crs is not None:
            input_rec["crs"] = img.crs
        if img.bands:
            input_rec["bands"] = img.bands
        input_rec["swir_available"] = img.swir_available
        if img.bit_depth is not None:
            input_rec["bit_depth"] = img.bit_depth
        if img.bit_depth_source is not None:
            input_rec["bit_depth_source"] = img.bit_depth_source
        input_rec["nodata_frac"] = img.nodata_frac
        if img.polarisations:
            input_rec["polarisations"] = img.polarisations
        if img.sar_band is not None:
            input_rec["sar_band"] = img.sar_band
        if img.computable_indices:
            input_rec["computable_indices"] = img.computable_indices
        if img.modality_source is not None:
            input_rec["modality_source"] = img.modality_source
        inputs.append(input_rec)

    trace.set_inputs(inputs)

    if bundle.coreg:
        compat: dict[str, Any] = {
            "coregistered": bundle.coreg.coregistered,
            "checks_passed": bundle.coreg.checks_passed,
        }
        if bundle.coreg.rmse_px is not None:
            compat["rmse_px"] = bundle.coreg.rmse_px
        if bundle.coreg.common_crs is not None:
            compat["common_crs"] = bundle.coreg.common_crs
        trace.set_compatibility(compat)

    if bundle.warnings:
        for w in bundle.warnings:
            trace.add_warning(w)
            warnings.append(w)

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["ingest"] = elapsed_ms

    return {
        "query_id": query_id,
        "wall_start": wall_start,
        "bundle": bundle,
        "band_inventory": bundle.band_inventory,
        "modalities": [img.modality for img in bundle.images],
        "pair_type": bundle.pair_type,
        "trace": trace,
        "warnings": warnings,
        "timings": timings,
    }


def route_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    trace = state["trace"]
    bundle = state.get("bundle")
    question = state["query_text"]
    modalities = state.get("modalities")
    if modalities is None:
        modalities = [img.modality for img in bundle.images] if bundle else []
    pair_type = state.get("pair_type") or (bundle.pair_type if bundle else None)
    img_count = len(bundle.images) if bundle else 1

    task, router_path, notes = route_query(
        question=question,
        modalities=modalities,
        pair_type=pair_type,
        image_count=img_count,
        allow_llm=state.get("allow_llm_router", True),
    )

    trace.set_routing(task, router_path)
    all_notes = list(state.get("routing_notes") or [])
    for n in notes:
        trace.add_routing_note(n)
        all_notes.append(n)

    emit = state.get("emit")
    if emit:
        emit("router", {
            "router_path": router_path.value,
            "task_selected": task.value,
            "notes": notes,
        })

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["router"] = elapsed_ms

    return {
        "task": task,
        "router_path": router_path,
        "routing_notes": all_notes,
        "trace": trace,
        "timings": timings,
    }


def validator_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    trace = state["trace"]
    bundle = state["bundle"]
    question = state["query_text"]
    task = state["task"] or Task.SINGLE_VQA

    res = validate_query_compatibility(bundle, question, task)

    warnings = list(state.get("warnings") or [])
    for w in res.warnings:
        trace.add_warning(w)
        warnings.append(w)

    notes = list(state.get("routing_notes") or [])
    for n in res.routing_notes:
        trace.add_routing_note(n)
        notes.append(n)

    emit = state.get("emit")
    if emit:
        emit("validator", {
            "passed": res.passed,
            "refusal": res.refusal,
            "warnings": res.warnings,
        })

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["validator"] = elapsed_ms

    return {
        "validation_ok": res.passed,
        "refusal": res.refusal,
        "failures": res.failures,
        "warnings": warnings,
        "routing_notes": notes,
        "trace": trace,
        "timings": timings,
    }


def tiling_policy_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    trace = state["trace"]
    bundle = state["bundle"]

    tile_plan = decide_tile_plan(bundle)
    notes = list(state.get("routing_notes") or [])
    if tile_plan.note:
        trace.add_routing_note(tile_plan.note)
        notes.append(tile_plan.note)

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["tiling"] = elapsed_ms

    return {
        "tile_plan": tile_plan,
        "routing_notes": notes,
        "trace": trace,
        "timings": timings,
    }


def planner_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    trace = state["trace"]
    task = state["task"] or Task.SINGLE_VQA
    bundle = state["bundle"]
    question = state["query_text"]
    replan_count = state.get("replan_count", 0)
    gate_rejected = state.get("gate_rejected")

    plan, notes = plan_query(
        task=task,
        bundle=bundle,
        question=question,
        replan_count=replan_count,
        gate_rejected=gate_rejected,
        previous_plan=state.get("plan"),
    )

    all_notes = list(state.get("routing_notes") or [])
    for n in notes:
        trace.add_routing_note(n)
        all_notes.append(n)

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["planner"] = elapsed_ms

    return {
        "plan": plan,
        "routing_notes": all_notes,
        "trace": trace,
        "timings": timings,
    }


def parameter_gate_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    registry = ToolRegistry.default()
    all_rejected: list[str] = []
    plan = list(state.get("plan") or [])
    trace = state["trace"]

    # Reset planned steps in trace if replanning
    if state.get("replan_count", 0) > 0:
        trace._plan.clear()  # noqa: SLF001 — clear() on private list; use trace.get_planned_steps() to read

    for step in plan:
        tool_name = step["tool"]
        if tool_name not in registry.names():
            all_rejected.append(f"tool '{tool_name}' is not registered in ToolRegistry")
            continue
        manifest = registry.get(tool_name)
        merged = effective_params(manifest, step["params"])
        defaults_applied = sorted(set(merged) - set(step["params"]))
        result = check_parameters(
            manifest,
            merged,
            band_inventory=state.get("band_inventory"),
            modalities=state.get("modalities"),
        )
        trace.add_planned_step(
            manifest.name,
            merged,
            within_manifest=result.passed,
            defaults_applied=defaults_applied or None,
        )
        if not result.passed:
            all_rejected.extend(result.rejected)
        step["params"] = merged

    gate_passed = not all_rejected
    trace.set_parameter_check(passed=gate_passed, rejected=all_rejected)

    emit = state.get("emit")
    if emit:
        emit("plan", {
            "steps": trace.get_planned_steps(),
            "parameter_check": {"passed": gate_passed, "rejected": all_rejected},
        })

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["gate"] = elapsed_ms

    if not gate_passed:
        replan_count = state.get("replan_count", 0)
        if replan_count == 0:
            return {
                "gate_passed": False,
                "gate_rejected": all_rejected,
                "replan_count": 1,
                "timings": timings,
                "trace": trace,
            }
        else:
            # Failed twice
            refusal = create_refusal(
                category="parameter_gate",
                reason=f"Parameter validation failed after replan: {'; '.join(all_rejected)}",
                action="ask_different_question",
                label="Ask a different question",
            )
            return {
                "gate_passed": False,
                "gate_rejected": all_rejected,
                "refusal": refusal,
                "validation_ok": False,
                "timings": timings,
                "trace": trace,
            }

    return {
        "gate_passed": True,
        "gate_rejected": [],
        "plan": plan,
        "timings": timings,
        "trace": trace,
    }


def executor_node(state: AgentState) -> dict[str, Any]:
    res = execute_plan(state)
    return res


def fusion_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    trace = state["trace"]
    results = state.get("results") or {}
    task = state.get("task")
    emit = state.get("emit")

    opt_res = results.get("spectral_index")
    sar_res = results.get("sar_backscatter")
    mask_cache = state.get("mask_cache") or {}
    bundle = state.get("bundle")

    if task in (Task.CROSSMODAL_EXTRACTION, Task.CROSSMODAL_VQA) or (opt_res and sar_res):
        opt_mask = None
        for k in ("spectral_index", "NDWI", "NDVI", "MNDWI"):
            if k in mask_cache:
                opt_mask = mask_cache[k]
                break
        sar_mask = mask_cache.get("sar_backscatter")
        context_meta = {}
        if bundle and bundle.images:
            context_meta["crs"] = bundle.images[0].crs

        fusion_res = reconcile_crossmodal(
            opt_res,
            sar_res,
            opt_mask=opt_mask,
            sar_mask=sar_mask,
            context_meta=context_meta,
        )
        fused_answer = fusion_res.fused_answer
        if "optsar_fusion" in results:
            optsar_out = results["optsar_fusion"]
            if isinstance(optsar_out, dict):
                optsar_ans = optsar_out.get("answer")
                if optsar_ans and not optsar_ans.startswith("MODEL_UNAVAILABLE"):
                    fused_answer = f"{fused_answer} {optsar_ans}".strip()
        trace.set_agreement(
            iou=fusion_res.iou,
            verdict=fusion_res.verdict,
            disagreement_cause=fusion_res.disagreement_cause,
        )
        if emit:
            emit("agreement", {
                "iou": fusion_res.iou,
                "verdict": fusion_res.verdict,
                "disagreement_cause": fusion_res.disagreement_cause,
                "winning_modality": fusion_res.winning_modality,
                "explanation": fusion_res.explanation,
            })
        agreement_dict = {
            "iou": fusion_res.iou,
            "verdict": fusion_res.verdict,
            "disagreement_cause": fusion_res.disagreement_cause,
            "winning_modality": fusion_res.winning_modality,
            "explanation": fusion_res.explanation,
        }
    else:
        agreement_dict = None
        # Single tool answer extraction
        if "dummy_tool" in results and results["dummy_tool"].get("answer"):
            fused_answer = results["dummy_tool"]["answer"]
        elif "change_stats" in results and results["change_stats"].get("answer"):
            fused_answer = results["change_stats"]["answer"]
        elif "change_map" in results and results["change_map"].get("change_ratio") is not None:
            ratio = results["change_map"].get("change_ratio", 0.0)
            fused_answer = (
                f"Change map computed: change ratio is {ratio:.1%} across the analyzed scene."
            )
        elif "change_vqa" in results and results["change_vqa"].get("answer"):
            fused_answer = results["change_vqa"]["answer"]
        elif "rs_vqa" in results and results["rs_vqa"].get("answer"):
            fused_answer = results["rs_vqa"]["answer"]
        elif "rs_ground_caption" in results and (
            results["rs_ground_caption"].get("answer")
            or results["rs_ground_caption"].get("caption")
        ):
            fused_answer = (
                results["rs_ground_caption"].get("answer")
                or results["rs_ground_caption"].get("caption")
            )
        elif (
            "spectral_index" in results and results["spectral_index"].get("area_km2") is not None
        ):
            idx = results["spectral_index"].get("index", "Index")
            area = results["spectral_index"].get("area_km2", 0.0)
            fused_answer = f"Computed {idx} mask: target identified across {area:.2f} km²."
        elif (
            "sar_backscatter" in results and results["sar_backscatter"].get("area_km2") is not None
        ):
            area = results["sar_backscatter"].get("area_km2", 0.0)
            fused_answer = f"Calibrated SAR backscatter thresholding identified {area:.2f} km²."
        elif (
            "object_box_fallback" in results
            and results["object_box_fallback"].get("count") is not None
        ):
            cnt = results["object_box_fallback"].get("count", 0)
            if cnt == 0:
                refusal = create_refusal(
                    category="unsupported_class",
                    reason=(
                        "Grounding target class is outside trained vocabulary and "
                        "fallback proposer found no candidate features (V9)."
                    ),
                    action="ask_different_question",
                    label="Ask a different question",
                    suggested_questions=[
                        "Where are the buildings in this scene?",
                        "Locate water bodies.",
                    ],
                )
                trace.add_warning(f"V9 refusal: {refusal['reason']}")
                warnings_list = list(state.get("warnings") or [])
                warnings_list.append(refusal["reason"])
                elapsed_ms = int((time.perf_counter() - start_t) * 1000)
                timings = dict(state.get("timings") or {})
                timings["fusion"] = elapsed_ms
                return {
                    "refusal": refusal,
                    "validation_ok": False,
                    "fused_answer": f"Refused: {refusal['reason']}",
                    "trace": trace,
                    "warnings": warnings_list,
                    "timings": timings,
                }
            fused_answer = f"Proposed {cnt} candidate bounding boxes using morphological priors."
        else:
            if any("MODEL_UNAVAILABLE" in w for w in (state.get("warnings") or [])):
                fused_answer = "Model serving offline or unreachable."
            else:
                fused_answer = "Analysis completed successfully."

    warnings_list = state.get("warnings") or []
    if any("synthetic" in w.lower() for w in warnings_list):
        if "(synthetic measurement)" not in fused_answer:
            fused_answer = f"{fused_answer.rstrip('.')} (synthetic measurement)."

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["fusion"] = elapsed_ms

    return {
        "fused_answer": fused_answer,
        "agreement": agreement_dict,
        "trace": trace,
        "timings": timings,
    }


def confidence_node(state: AgentState) -> dict[str, Any]:
    start_t = time.perf_counter()
    results = state.get("results") or {}
    agreement = state.get("agreement")
    warnings = state.get("warnings") or []
    tile_plan = state.get("tile_plan")
    used_fallback = "object_box_fallback" in results

    has_unavail = any("MODEL_UNAVAILABLE" in w for w in warnings)
    has_valid_tool = any(
        isinstance(r, dict)
        and (r.get("mask_uri") or r.get("area_km2") is not None or r.get("boxes"))
        for r in results.values()
    )
    if has_unavail and not has_valid_tool:
        conf = 0.0
        basis = "heuristic"
    else:
        conf, basis = calculate_confidence(
            tool_results=results,
            agreement=agreement,
            warnings=warnings,
            tile_plan=tile_plan,
            used_fallback=used_fallback,
        )

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["confidence"] = elapsed_ms

    return {
        "confidence": conf,
        "confidence_basis": basis,
        "timings": timings,
    }


def refuse_node(state: AgentState) -> dict[str, Any]:
    refusal = state.get("refusal")
    reason = refusal["reason"] if refusal else "Query refused"
    answer = f"Refused: {reason}"
    return {
        "fused_answer": answer,
        "confidence": 0.0,
        "confidence_basis": "heuristic",
    }


def emit_node(state: AgentState) -> dict[str, Any]:
    trace = state["trace"]
    refusal = state.get("refusal")
    warnings = state.get("warnings") or []
    assets = state.get("assets") or []

    # In current pipeline, ensure trace routing and parameter check are set
    if getattr(trace, "_task", None) is None:
        trace.set_routing(
            state.get("task") or Task.SINGLE_VQA,
            state.get("router_path") or RouterPath.RULES,
        )
    if getattr(trace, "_parameter_check", None) is None:
        trace.set_parameter_check(passed=True, rejected=[])

    for asset in assets:
        trace.add_evidence(asset.asset_id)

    timings = state.get("timings") or {}
    wall_start = state.get("wall_start")
    # Wall clock, not the sum of node timings: waves run concurrently, so the
    # sum overstates a parallel query and understates one that waited on I/O.
    total_latency_ms = (
        int((time.perf_counter() - wall_start) * 1000) if wall_start else sum(timings.values())
    )

    if total_latency_ms > 20000:
        warning_msg = f"Global latency budget exceeded: {total_latency_ms} ms > 20000 ms"
        trace.add_warning(warning_msg)
        warnings.append(warning_msg)

    emit = state.get("emit")

    if refusal:
        query_state = "refused"
        answer = state.get("fused_answer") or f"Refused: {refusal['reason']}"
        confidence = 0.0
        confidence_basis = "heuristic"
        trace_refusal = {
            "reason": refusal["reason"],
            "category": refusal["category"],
        }
        trace.set_outputs(
            answer=answer,
            confidence=confidence,
            refusal=trace_refusal,
            extra={"confidence_basis": confidence_basis},
        )
    elif state.get("cancelled"):
        query_state = "cancelled"
        answer = state.get("fused_answer") or "Query cancelled."
        confidence = state.get("confidence") or 0.0
        confidence_basis = state.get("confidence_basis") or "heuristic"
        trace.set_outputs(
            answer=answer,
            confidence=confidence,
            extra={"confidence_basis": confidence_basis},
        )
    elif state.get("all_tools_failed"):
        query_state = "failed"
        answer = state.get("fused_answer") or "All planned tools failed."
        confidence = 0.0
        confidence_basis = "heuristic"
        trace.set_outputs(
            answer=answer,
            confidence=0.0,
            extra={"confidence_basis": confidence_basis},
        )
    else:
        query_state = "succeeded"
        answer = state.get("fused_answer") or "Query processed successfully."
        confidence = state.get("confidence", 0.90)
        confidence_basis = state.get("confidence_basis", "heuristic")

        # Gather mask files and area from assets/results
        masks: list[str] = [
            str(a.download_url) for a in assets if a.kind == "mask_geotiff"
        ]
        area_km2 = None
        for a in assets:
            if a.stats and "area_km2" in a.stats:
                area_km2 = float(a.stats["area_km2"])
                break

        trace.set_outputs(
            answer=answer,
            masks=masks or None,
            area_km2=area_km2,
            confidence=confidence,
            extra={"confidence_basis": confidence_basis},
        )

    if not refusal and query_state != "failed":
        trace.set_fusion(
            model=_fusion_model_name(state.get("results") or {}),
            answer=answer,
            confidence=confidence if confidence is not None else 0.0,
        )

    trace_dir = Path(os.environ.get("SATQUERY_TRACES_DIR", "traces"))
    trace_file = trace_dir / f"{state['query_id']}.json"

    try:
        trace_dict = trace.write_json(trace_file)
    except Exception as ex:
        # Minimal trace fallback per §10 N9 and Rule 11
        if getattr(trace, "_task", None) is None:
            trace.set_routing(Task.SINGLE_VQA, RouterPath.RULES)
        if getattr(trace, "_parameter_check", None) is None:
            trace.set_parameter_check(passed=False, rejected=[f"Pipeline failure: {ex}"])
        trace.set_outputs(
            answer=f"Pipeline failure: {ex}",
            confidence=0.0,
            extra={"confidence_basis": "heuristic"},
        )
        trace.add_warning(f"Trace build failed; emitting minimal trace: {ex}")
        trace_dict = trace.write_json(trace_file)

    if emit:
        if not refusal and query_state != "failed":
            emit(
                "fusion",
                {
                    "model": _fusion_model_name(state.get("results") or {}),
                    "answer": answer,
                    "confidence": confidence if confidence is not None else 0.0,
                    "confidence_basis": confidence_basis or "heuristic",
                },
            )
        emit(
            "done",
            {
                "query_id": state["query_id"],
                "state": query_state,
                "total_latency_ms": total_latency_ms,
                "trace_url": f"/traces/{state['query_id']}.json",
            },
        )

    return {
        "trace_dict": trace_dict,
        "query_state": query_state,
        "total_latency_ms": total_latency_ms,
        "fused_answer": answer,
        "confidence": confidence,
        "confidence_basis": confidence_basis,
        "warnings": warnings,
    }


def _route_after_ingest(state: AgentState) -> str:
    if not state.get("validation_ok", True):
        return "refuse"
    return "route"


def _route_after_validator(state: AgentState) -> str:
    if not state.get("validation_ok", True):
        return "refuse"
    return "tiling"


def _route_after_gate(state: AgentState) -> str:
    if not state.get("gate_passed", True):
        if state.get("replan_count", 0) == 1 and state.get("refusal") is None:
            return "plan"
        return "refuse"
    return "executor"


def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("route", route_node)
    graph.add_node("validator", validator_node)
    graph.add_node("tiling", tiling_policy_node)
    graph.add_node("plan", planner_node)
    graph.add_node("gate", parameter_gate_node)
    graph.add_node("executor", executor_node)
    graph.add_node("fusion", fusion_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("emit", emit_node)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest", _route_after_ingest, {"refuse": "refuse", "route": "route"}
    )
    graph.add_edge("route", "validator")
    graph.add_conditional_edges(
        "validator", _route_after_validator, {"refuse": "refuse", "tiling": "tiling"}
    )
    graph.add_edge("tiling", "plan")
    graph.add_edge("plan", "gate")
    graph.add_conditional_edges(
        "gate", _route_after_gate, {"plan": "plan", "refuse": "refuse", "executor": "executor"}
    )
    graph.add_edge("executor", "fusion")
    graph.add_edge("fusion", "confidence")
    graph.add_edge("confidence", "emit")
    graph.add_edge("refuse", "emit")
    graph.add_edge("emit", END)

    return graph.compile()


_GRAPH = None


def get_compiled_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_query(
    bundle: ImageBundle,
    question: str,
    *,
    query_id: str | None = None,
    allow_llm_router: bool = True,
    emit: EventSink | None = None,
    cancel_check: Any | None = None,
    submitted_at: float | None = None,
) -> QueryResult:
    """Run one query. §9.

    `cancel_check` is a callable or an Event the executor consults at wave
    boundaries (§24); `submitted_at` is a `time.perf_counter()` stamp from when
    the job was accepted, so `queued_ms` reports the real wait.
    """
    emitter = create_emitter(emit)
    initial_state: AgentState = {
        "query_id": query_id or str(uuid.uuid4()),
        "query_text": question,
        "bundle": bundle,
        "band_inventory": bundle.band_inventory,
        "modalities": [img.modality for img in bundle.images],
        "pair_type": bundle.pair_type,
        "allow_llm_router": allow_llm_router,
        "cancel_check": cancel_check,
        "submitted_at": submitted_at,
        "wall_start": time.perf_counter(),
        "routing_notes": [],
        "validation_ok": True,
        "refusal": None,
        "failures": [],
        "plan": [],
        "gate_passed": True,
        "gate_rejected": [],
        "replan_count": 0,
        "results": {},
        "assets": [],
        "warnings": [],
        "emit": emitter,
        "timings": {},
    }

    app = get_compiled_graph()
    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        qid = initial_state["query_id"]
        trace = initial_state.get("trace") or TraceBuilder(question, query_id=qid)
        if getattr(trace, "_task", None) is None:
            trace.set_routing(Task.SINGLE_VQA, RouterPath.RULES)
        if getattr(trace, "_parameter_check", None) is None:
            trace.set_parameter_check(passed=False, rejected=[f"Unhandled error: {exc}"])
        trace.set_outputs(
            answer=f"Unhandled error: {exc}",
            confidence=0.0,
            extra={"confidence_basis": "heuristic"},
        )
        trace.add_warning(f"Unhandled query error: {exc}")
        trace_dir = Path(os.environ.get("SATQUERY_TRACES_DIR", "traces"))
        trace_file = trace_dir / f"{qid}.json"
        try:
            trace_dict = trace.write_json(trace_file)
        except Exception:
            trace_dict = trace.build()
        return QueryResult(
            query_id=qid,
            bundle_id=bundle.bundle_id,
            question=question,
            state="failed",
            answer=f"Unhandled error: {exc}",
            confidence=0.0,
            confidence_basis="heuristic",
            latency_ms=0,
            refusal=None,
            failures=[{"step": "pipeline", "error": str(exc)}],
            evidence=[],
            warnings=[f"Unhandled error: {exc}"],
            trace=trace_dict,
        )

    trace_dict = final_state.get("trace_dict") or {}
    query_state = final_state.get("query_state", "succeeded")
    if final_state.get("cancelled"):
        query_state = "cancelled"
    elif final_state.get("all_tools_failed") and query_state != "refused":
        query_state = "failed"

    return QueryResult(
        query_id=final_state["query_id"],
        bundle_id=bundle.bundle_id,
        question=question,
        state=query_state,
        answer=final_state.get("fused_answer"),
        confidence=final_state.get("confidence"),
        confidence_basis=final_state.get("confidence_basis"),
        latency_ms=final_state.get("total_latency_ms")
        or sum((final_state.get("timings") or {}).values()),
        refusal=final_state.get("refusal"),
        failures=final_state.get("failures", []),
        evidence=final_state.get("assets", []),
        warnings=final_state.get("warnings", []),
        trace=trace_dict,
    )
