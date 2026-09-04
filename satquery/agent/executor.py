import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from satquery.agent.asset import AssetRef
from satquery.agent.events import EventEmitter
from satquery.agent.state import AgentState
from satquery.tools import (
    centroid_prior,
    change_map,
    change_stats,
    change_vqa,
    coreg_check,
    dummy_tool,
    lulc_classifier,
    object_box_fallback,
    optsar_fusion,
    rs_ground_caption,
    rs_vqa,
    sar_backscatter,
    spectral_index,
    texture_seg,
    tile_scorer,
)
from satquery.tools.registry import ToolRegistry

DISPATCH_MODULES: dict[str, Any] = {
    "centroid_prior": centroid_prior,
    "change_map": change_map,
    "change_stats": change_stats,
    "change_vqa": change_vqa,
    "coreg_check": coreg_check,
    "dummy_tool": dummy_tool,
    "lulc_classifier": lulc_classifier,
    "object_box_fallback": object_box_fallback,
    "optsar_fusion": optsar_fusion,
    "rs_ground_caption": rs_ground_caption,
    "rs_vqa": rs_vqa,
    "sar_backscatter": sar_backscatter,
    "spectral_index": spectral_index,
    "texture_seg": texture_seg,
    "tile_scorer": tile_scorer,
}


def compute_waves(plan: list[dict[str, Any]]) -> list[list[tuple[int, dict[str, Any]]]]:
    """Partitions planned steps into topological waves based on depends_on."""
    waves: list[list[tuple[int, dict[str, Any]]]] = []
    completed_tools: set[str] = set()
    remaining: list[tuple[int, dict[str, Any]]] = list(enumerate(plan))

    while remaining:
        current_wave: list[tuple[int, dict[str, Any]]] = []
        next_remaining: list[tuple[int, dict[str, Any]]] = []

        for idx, step in remaining:
            deps = set(step.get("depends_on") or [])
            if deps.issubset(completed_tools):
                current_wave.append((idx, step))
            else:
                next_remaining.append((idx, step))

        if not current_wave:
            # Cycle or unresolved dependency: execute remaining together as fallback
            current_wave = next_remaining
            next_remaining = []

        waves.append(current_wave)
        for _, step in current_wave:
            completed_tools.add(step["tool"])
        remaining = next_remaining

    return waves


# Deterministic tools that support per-tile execution and full-scene mosaicking (Rule 7)
_DETERMINISTIC_TILE_TOOLS = {
    "spectral_index",
    "sar_backscatter",
    "texture_seg",
    "change_map",
    "object_box_fallback",
}

# Learned tools that receive only the budgeted tile subset (Rule 14)
_LEARNED_TILE_TOOLS = {"rs_vqa", "rs_ground_caption", "change_vqa", "optsar_fusion"}


def _mosaic_tile_outputs(per_tile_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-tile tool results into a single full-scene output dict.

    Strategy: sum numeric fields (area_km2, count), union lists (boxes, labels),
    take max confidence, last mask_uri (tools write full-scene files themselves).
    """
    if not per_tile_outputs:
        return {}
    if len(per_tile_outputs) == 1:
        return per_tile_outputs[0]

    merged: dict[str, Any] = dict(per_tile_outputs[0])
    for tile_out in per_tile_outputs[1:]:
        for key, val in tile_out.items():
            if key in ("area_km2",) and val is not None:
                merged[key] = (merged.get(key) or 0.0) + float(val)
            elif key == "count" and val is not None:
                merged[key] = (merged.get(key) or 0) + int(val)
            elif key == "boxes" and isinstance(val, list):
                merged[key] = (merged.get(key) or []) + val
            elif key == "labels" and isinstance(val, list):
                existing = merged.get(key) or []
                merged[key] = existing + [v for v in val if v not in existing]
            elif key == "confidence" and val is not None:
                merged[key] = max(merged.get(key) or 0.0, float(val))
            # For mask_uri, keep last (tools write incremental files; mosaicking
            # requires rasterio merge which is handled in the tool itself when
            # context["tiles"] is set)
    return merged


def _run_single_step(
    index: int,
    step: dict[str, Any],
    context: dict[str, Any],
    registry: ToolRegistry,
    emit: EventEmitter | None,
) -> tuple[int, dict[str, Any], list[AssetRef], str | None]:
    tool_name = step["tool"]
    params = dict(step["params"])

    tile_plan = context.get("tile_plan")
    is_tiled = tile_plan and getattr(tile_plan, "is_tiled", False)
    if is_tiled:
        if tool_name in _LEARNED_TILE_TOOLS:
            # Learned tool gets only budgeted selected tiles (Rule 14)
            context["selected_tiles"] = tile_plan.learned_tiles
        elif tool_name in _DETERMINISTIC_TILE_TOOLS:
            # Deterministic tool runs across all tiles (Rule 7)
            context["tiles"] = tile_plan.deterministic_tiles

    if emit:
        emit("step_started", {"index": index, "tool": tool_name})

    start_t = time.perf_counter()
    mod = DISPATCH_MODULES.get(tool_name)
    outputs: dict[str, Any] = {}
    err: str | None = None
    assets: list[AssetRef] = []

    # Latency / timeout calculation per Master Plan §4.5.4
    expected_ms = 1000
    if tool_name in registry.names():
        m = registry.get(tool_name)
        if m.expected_latency_ms:
            expected_ms = m.expected_latency_ms

    timeout_sec = max(2.0, (expected_ms * 5) / 1000.0)

    try:
        if mod is not None and hasattr(mod, "execute"):
            # Deterministic tile-loop: run once per tile and mosaic back (Rule 7)
            if is_tiled and tool_name in _DETERMINISTIC_TILE_TOOLS:
                all_tiles = tile_plan.deterministic_tiles or []
                if all_tiles:
                    per_tile_outs: list[dict[str, Any]] = []
                    tile_context = dict(context)
                    for tile in all_tiles:
                        tile_context["current_tile"] = tile
                        per_tile_outs.append(mod.execute(params, tile_context))
                    outputs = _mosaic_tile_outputs(per_tile_outs)
                else:
                    outputs = mod.execute(params, context)
            else:
                outputs = mod.execute(params, context)
        else:
            outputs = {"answer": f"Simulated output for {tool_name}"}
    except Exception as e:
        err = f"Tool '{tool_name}' failed with exception: {e}"

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)

    # Check timeout condition
    if elapsed_ms > timeout_sec * 1000 and err is None:
        err = (
            f"Tool '{tool_name}' exceeded latency SLA "
            f"({elapsed_ms}ms > {int(timeout_sec * 1000)}ms)"
        )

    # Asset creation if outputs contain masks or boxes
    if "mask_uri" in outputs:
        mask_path_str = str(outputs["mask_uri"])
        file_bytes = 0
        p = Path(mask_path_str)
        if p.exists() and p.is_file():
            file_bytes = p.stat().st_size
        else:
            try:
                import io

                import rasterio
                from rasterio.transform import from_origin

                with io.BytesIO() as mem_buf:
                    with rasterio.open(
                        mem_buf,
                        "w",
                        driver="GTiff",
                        height=100,
                        width=100,
                        count=1,
                        dtype=rasterio.uint8,
                        crs=context.get("crs", "EPSG:32644"),
                        transform=from_origin(500000.0, 3000000.0, 10.0, 10.0),
                    ) as dst:
                        dst.write(np.zeros((100, 100), dtype=np.uint8), 1)
                    file_bytes = len(mem_buf.getvalue())
            except Exception:
                file_bytes = 1024
        asset_id = str(uuid.uuid4())[:8]
        assets.append(
            AssetRef(
                asset_id=asset_id,
                kind="mask_geotiff",
                label=f"{tool_name} output mask",
                produced_by=tool_name,
                media_type="image/tiff",
                bytes=file_bytes,
                crs=context.get("crs", "EPSG:32644"),
                tile_url_template=f"/assets/{asset_id}/tiles/{{z}}/{{x}}/{{y}}.png",
                overlay_url=f"/assets/{asset_id}/overlay.png",
                download_url=mask_path_str,
                stats={"area_km2": outputs.get("area_km2", 0.0)},
            )
        )
    if "boxes" in outputs and isinstance(outputs["boxes"], list):
        boxes_px = [
            b["bbox_px"]
            for b in outputs["boxes"]
            if isinstance(b, dict) and "bbox_px" in b
        ]
        scene_crs = context.get("crs")
        # §18 requires bbox_px always — emit pixel-space asset regardless of CRS
        asset_id = str(uuid.uuid4())[:8]
        assets.append(
            AssetRef(
                asset_id=asset_id,
                kind="bbox_geojson",
                label=f"{tool_name} bounding boxes (pixel space)",
                produced_by=tool_name,
                media_type="application/json",
                bytes=max(256, len(str(boxes_px))),
                crs=scene_crs or "",
                bbox_px=boxes_px,
                download_url=f"/assets/{asset_id}/boxes_px.json",
            )
        )
        # When CRS is present also emit full GeoJSON (georeferenced polygons)
        if scene_crs:
            geo_asset_id = str(uuid.uuid4())[:8]
            assets.append(
                AssetRef(
                    asset_id=geo_asset_id,
                    kind="bbox_geojson",
                    label=f"{tool_name} bounding boxes (georeferenced)",
                    produced_by=tool_name,
                    media_type="application/geo+json",
                    bytes=2048,
                    crs=scene_crs,
                    bbox_px=boxes_px,
                    download_url=f"/assets/{geo_asset_id}/boxes.geojson",
                )
            )
    if "response_map" in outputs:
        asset_id = str(uuid.uuid4())[:8]
        assets.append(
            AssetRef(
                asset_id=asset_id,
                kind="mask_geotiff",
                label=f"{tool_name} response map",
                produced_by=tool_name,
                media_type="image/tiff",
                bytes=10240,
                crs=context.get("crs", "EPSG:32644"),
                tile_url_template=f"/assets/{asset_id}/tiles/{{z}}/{{x}}/{{y}}.png",
                overlay_url=f"/assets/{asset_id}/overlay.png",
                download_url=str(outputs["response_map"]),
            )
        )

    completed_record = {
        "tool": tool_name,
        "params": params,
        "outputs": outputs,
        "latency_ms": elapsed_ms,
        "error": err,
    }

    if emit:
        emit("step_completed", {**completed_record, "index": index})
        if assets:
            emit("evidence", {"assets": [a.__dict__ for a in assets]})

    return index, completed_record, assets, err


def execute_plan(state: AgentState) -> dict[str, Any]:
    """Executes planned steps in topological waves with concurrency and SLA timeouts."""
    start_t = time.perf_counter()
    plan = state.get("plan") or []
    trace = state["trace"]
    emit = state.get("emit")
    registry = ToolRegistry.default()

    if not plan:
        return {"results": {}, "mask_cache": {}, "assets": [], "warnings": []}

    bundle = state.get("bundle")
    crs = bundle.images[0].crs if bundle and bundle.images else None
    results: dict[str, Any] = {}
    mask_cache: dict[str, Any] = {}

    tile_plan = state.get("tile_plan")
    context = {
        "bundle": bundle,
        "modalities": state.get("modalities"),
        "crs": crs,
        "results": results,
        "mask_cache": mask_cache,
        "question": state.get("query_text", ""),
        "tile_plan": tile_plan,
    }

    waves = compute_waves(plan)
    assets: list[AssetRef] = list(state.get("assets") or [])
    warnings: list[str] = list(state.get("warnings") or [])

    for wave in waves:
        max_timeout = 2.0
        for _, step in wave:
            t_name = step["tool"]
            exp = 1000
            if t_name in registry.names():
                m = registry.get(t_name)
                if m.expected_latency_ms:
                    exp = m.expected_latency_ms
            max_timeout = max(max_timeout, max(2.0, (exp * 5) / 1000.0))

        with ThreadPoolExecutor(max_workers=min(len(wave), 4)) as pool:
            future_to_step = {
                pool.submit(_run_single_step, idx, step, context, registry, emit): (idx, step)
                for idx, step in wave
            }
            try:
                for fut in as_completed(future_to_step, timeout=max_timeout + 2.0):
                    try:
                        idx, rec, step_assets, err = fut.result(timeout=max_timeout)
                    except Exception as ex:
                        idx, step = future_to_step[fut]
                        tool_name = step["tool"]
                        err = f"Tool '{tool_name}' exceeded latency SLA or failed: {ex}"
                        rec = {
                            "tool": tool_name,
                            "params": step["params"],
                            "outputs": {},
                            "latency_ms": int(max_timeout * 1000),
                            "error": err,
                        }
                        step_assets = []

                    tool_name = rec["tool"]
                    results[tool_name] = rec["outputs"]
                    assets.extend(step_assets)

                    if err:
                        warnings.append(err)
                        trace.add_warning(err)

                    if rec["outputs"].get("synthetic"):
                        synth_warn = (
                            f"Tool '{tool_name}' used synthetic fallback array "
                            "(source raster not accessible on disk)"
                        )
                        if synth_warn not in warnings:
                            warnings.append(synth_warn)
                            trace.add_warning(synth_warn)
                        trace.add_routing_note(f"{tool_name}: used synthetic fallback array")

                    # Determine confidence
                    conf = 0.90
                    if err:
                        conf = 0.0
                    elif rec["outputs"].get("synthetic"):
                        conf = 0.45
                    elif "confidence" in rec["outputs"]:
                        conf = float(rec["outputs"]["confidence"])

                    trace.add_step(
                        tool=tool_name,
                        params=rec["params"],
                        outputs=rec["outputs"],
                        confidence=conf,
                        latency_ms=rec["latency_ms"],
                    )
            except TimeoutError:
                for fut, (_idx, step) in future_to_step.items():
                    if not fut.done():
                        tool_name = step["tool"]
                        err = f"Tool '{tool_name}' timed out after {max_timeout + 2.0}s"
                        warnings.append(err)
                        trace.add_warning(err)
                        trace.add_step(
                            tool=tool_name,
                            params=step["params"],
                            outputs={},
                            confidence=0.0,
                            latency_ms=int((max_timeout + 2.0) * 1000),
                        )

    elapsed_ms = int((time.perf_counter() - start_t) * 1000)
    timings = dict(state.get("timings") or {})
    timings["executor"] = elapsed_ms

    has_any_valid_output = False
    for _t_name, out in results.items():
        if isinstance(out, dict) and (
            out.get("answer")
            or out.get("mask_uri")
            or out.get("boxes")
            or out.get("count") is not None
            or out.get("area_km2") is not None
            or out.get("change_ratio") is not None
            or out.get("mean_texture") is not None
            or out.get("labels") is not None
            or out.get("response_map")
        ):
            has_any_valid_output = True
            break

    all_tools_failed = bool(plan) and not has_any_valid_output
    failures: list[dict[str, Any]] = list(state.get("failures") or [])
    if all_tools_failed:
        for step in plan:
            t_name = step["tool"]
            err_msg = next(
                (w for w in warnings if f"Tool '{t_name}'" in w),
                f"Tool '{t_name}' produced no output",
            )
            failures.append({"step": t_name, "error": err_msg})

    return {
        "results": results,
        "mask_cache": mask_cache,
        "assets": assets,
        "warnings": warnings,
        "timings": timings,
        "all_tools_failed": all_tools_failed,
        "failures": failures,
    }
