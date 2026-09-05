import hashlib
import json
import os
import time
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



# §25: "cache deterministic tool outputs keyed by (tool, version, params,
# image_hash, tile_id) -- demo reruns are common and this is nearly free."
# Process-local and bounded; learned tools are never cached because a model
# swap or a temperature change would make a stale answer indistinguishable.
_TOOL_CACHE: dict[str, dict[str, Any]] = {}
_TOOL_CACHE_MAX = 256
_CACHEABLE = _DETERMINISTIC_TILE_TOOLS | {"centroid_prior", "change_stats", "coreg_check"}


def _cache_key(
    tool_name: str,
    version: int | None,
    params: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    """Identity of a deterministic computation, or None when it cannot be pinned."""
    if tool_name not in _CACHEABLE:
        return None
    bundle = context.get("bundle")
    images = getattr(bundle, "images", None) or []
    # Scene identity: paths plus their mtime/size, so an edited raster misses.
    scene_parts: list[str] = []
    for img in images:
        path = Path(str(img.path))
        try:
            stat = path.stat()
            # Nanosecond mtime, not seconds: a raster rewritten in-place within
            # the same second keeps its size, and a second-resolution key would
            # serve the pre-edit result back.
            scene_parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            # Unreadable: no stable identity, so refuse to cache at all rather
            # than key on a path that may point at different bytes next run.
            return None
    tile = context.get("current_tile")
    tile_id = str(tile.get("tile_id")) if isinstance(tile, dict) else "-"
    payload = "|".join(
        (
            tool_name,
            str(version),
            json.dumps(params, sort_keys=True, default=str),
            ";".join(scene_parts),
            tile_id,
        )
    )
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()



# §18: a suggested render colour per producing tool, so the evidence panel is
# consistent between runs and between masks. Not semantic colour -- the frontend
# owns that -- just a stable hint.
_ASSET_COLOURS = {
    "spectral_index": "#2b7bba",
    "sar_backscatter": "#c46a1f",
    "change_map": "#8e44ad",
    "texture_seg": "#4f7a4f",
    "object_box_fallback": "#a33b1f",
    "rs_ground_caption": "#0f6b5c",
}


def _bounds_wgs84(mask_path: str) -> list[float] | None:
    """Scene bounds in WGS84 for map fitting, or None when not georeferenced."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(mask_path) as src:
            if not src.crs:
                return None
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            return [float(west), float(south), float(east), float(north)]
    except Exception:
        return None



def _mask_crs(mask_path: str) -> str | None:
    """The CRS actually written into the file -- None when it carries none."""
    try:
        import rasterio

        with rasterio.open(mask_path) as src:
            return str(src.crs) if src.crs else None
    except Exception:
        return None


def asset_id_for(tool_name: str, kind: str, discriminator: str) -> str:
    """A stable id for an asset, derived from what the asset *is*.

    Asset ids are written into the trace's `evidence` array, so a random id makes
    two identical runs produce different traces and breaks §25. Hashing the
    identity instead means the same tool producing the same output always yields
    the same id, while two assets from one step still differ by `kind` and
    `discriminator`.
    """
    digest = hashlib.blake2s(
        "|".join((tool_name, kind, discriminator)).encode("utf-8"), digest_size=4
    )
    return digest.hexdigest()




def _query_budget_s() -> float:
    """The global per-query budget in seconds (§25: 20 s)."""
    try:
        return float(os.environ.get("SATQUERY_QUERY_BUDGET_S", "20"))
    except ValueError:
        return 20.0


def _is_cancelled(state: AgentState) -> bool:
    """Whether this query has been asked to stop (§24).

    The flag is a plain callable or event on the state so the web path can set it
    from `POST /queries/{id}/cancel` without the graph knowing about Celery.
    """
    flag = state.get("cancel_check")
    if callable(flag):
        try:
            return bool(flag())
        except Exception:
            return False
    is_set = getattr(flag, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return bool(state.get("cancelled"))


def _execute_cached(
    mod: Any,
    tool_name: str,
    version: int | None,
    params: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool, reusing a previous identical computation when there is one."""
    key = _cache_key(tool_name, version, params, context)
    if key is None:
        return mod.execute(params, context)
    hit = _TOOL_CACHE.get(key)
    if hit is not None:
        # Copy so a caller mutating outputs cannot poison the entry.
        return dict(hit)
    out = mod.execute(params, context)
    if isinstance(out, dict) and not out.get("synthetic"):
        # Never cache a synthetic-fallback result: it exists because the real
        # input was unreadable, and that can change between runs.
        if len(_TOOL_CACHE) >= _TOOL_CACHE_MAX:
            _TOOL_CACHE.clear()
        _TOOL_CACHE[key] = dict(out)
    return out


def mod_for(tool_name: str) -> Any:
    """Dispatch-table lookup, for capability checks before dispatch."""
    return DISPATCH_MODULES.get(tool_name)


def _tool_supports_tiling(mod: Any) -> bool:
    """Whether a tool honours context["current_tile"].

    A tool that does not is called once on the whole scene. Looping such a tool
    would run the same computation N times over identical data, and any scalar
    the executor combined from those runs would be wrong by a factor of N.
    """
    return bool(getattr(mod, "SUPPORTS_TILING", False))


def _mosaic_tile_outputs(
    per_tile_outputs: list[dict[str, Any]],
    pixel_size_m: float,
) -> dict[str, Any]:
    """Composite per-tile results into one full-scene output (Rule 7).

    Scalars are *derived from the mosaic*, never summed across tiles. Tiles
    overlap by `overlap_frac`, so summing per-tile areas double-counts the seams;
    compositing the masks first and measuring once handles overlap by
    construction (§12.4).
    """
    if not per_tile_outputs:
        return {}

    scene_shape = None
    for out in per_tile_outputs:
        if out.get("_scene_shape"):
            scene_shape = tuple(int(v) for v in out["_scene_shape"])
            break

    merged: dict[str, Any] = {}
    for out in per_tile_outputs:
        for key, val in out.items():
            if key.startswith("_") or key in ("boxes", "count"):
                continue
            merged.setdefault(key, val)

    # Boxes are unioned; their coordinates are already scene-space (§12.4).
    boxes: list[Any] = []
    for out in per_tile_outputs:
        tile_boxes = out.get("boxes")
        if isinstance(tile_boxes, list):
            boxes.extend(tile_boxes)
    if boxes:
        merged["boxes"] = boxes
        merged["count"] = len(boxes)

    # Composite the tile arrays into the scene canvas.
    arrays = [
        (out["_mask_array"], out.get("_tile_offset") or [0, 0])
        for out in per_tile_outputs
        if isinstance(out.get("_mask_array"), np.ndarray)
    ]
    if not arrays or scene_shape is None:
        return merged

    is_binary = all(np.asarray(a).dtype == np.uint8 for a, _ in arrays)
    canvas = np.zeros(scene_shape, dtype=np.uint8 if is_binary else np.float32)
    for arr, (row0, col0) in arrays:
        sub = np.asarray(arr)
        row1 = min(row0 + sub.shape[0], scene_shape[0])
        col1 = min(col0 + sub.shape[1], scene_shape[1])
        if row1 <= row0 or col1 <= col0:
            continue
        view = sub[: row1 - row0, : col1 - col0]
        if is_binary:
            # Union, so a pixel written by two overlapping tiles counts once.
            canvas[row0:row1, col0:col1] |= view.astype(np.uint8)
        else:
            np.maximum(canvas[row0:row1, col0:col1], view, out=canvas[row0:row1, col0:col1])

    merged["_mosaic"] = canvas
    if is_binary:
        area_px = int(np.count_nonzero(canvas))
        merged["area_km2"] = round(float(area_px * (pixel_size_m**2) / 1e6), 4)
        merged["change_ratio"] = round(float(area_px / canvas.size), 4)
    else:
        merged["mean_texture"] = round(float(np.mean(canvas)), 4)
        merged["high_texture_frac"] = round(float(np.mean(canvas > 0.5)), 4)
    return merged


def _write_mosaic_geotiff(
    canvas: "np.ndarray",
    tool_name: str,
    crs_str: str | None,
    pixel_size_m: float,
    src_transform: Any | None = None,
) -> str:
    """Write the mosaicked scene as one GeoTIFF (Rule 7).

    When the source raster was readable its own transform is carried through, so
    the mask lands where the scene does. When it was not, no CRS is written at
    all: a real EPSG code over a fabricated origin is a mask that claims to be
    somewhere it is not, which is worse than an ungeoreferenced one.
    """
    import tempfile

    mask_dir = Path(tempfile.gettempdir()) / "satquery_assets" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_file = mask_dir / f"{tool_name}_scene_mask.tif"
    try:
        import rasterio
        from rasterio.transform import from_origin

        out = canvas if canvas.dtype == np.uint8 else (canvas * 255).astype(np.uint8)
        if src_transform is not None:
            transform = src_transform
            crs_out = crs_str
        else:
            top_y = float(canvas.shape[0] * pixel_size_m)
            transform = from_origin(0.0, top_y, pixel_size_m, pixel_size_m)
            crs_out = None
        with rasterio.open(
            mask_file,
            "w",
            driver="GTiff",
            height=out.shape[0],
            width=out.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs_out,
            transform=transform,
        ) as dst:
            dst.write(out, 1)
    except Exception:
        return ""
    return str(mask_file)


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
            context["total_tile_count"] = tile_plan.total_tile_count
            if not _tool_supports_tiling(mod_for(tool_name)):
                # The budget is offered but not consumed. Say so rather than let
                # the tile_plan's coverage note imply the model saw a subset.
                context.setdefault("tiling_skipped", []).append(tool_name)
        elif tool_name in _DETERMINISTIC_TILE_TOOLS:
            # Deterministic tool runs across all tiles (Rule 7)
            context["tiles"] = tile_plan.deterministic_tiles

    if emit:
        emit("step_started", {"index": index, "tool": tool_name})
        # §19: the serving client streams the answer through here as it arrives.
        # Only nodes emit events, so the callback is the executor's, not the tool's.
        context["on_token"] = lambda text: emit("token", {"text": text})
    else:
        context.pop("on_token", None)

    start_t = time.perf_counter()
    mod = DISPATCH_MODULES.get(tool_name)
    outputs: dict[str, Any] = {}
    err: str | None = None
    assets: list[AssetRef] = []

    # Latency / timeout calculation per Master Plan §4.5.4
    expected_ms = 1000
    manifest_version: int | None = None
    if tool_name in registry.names():
        m = registry.get(tool_name)
        manifest_version = m.version
        if m.expected_latency_ms:
            expected_ms = m.expected_latency_ms

    timeout_sec = max(2.0, (expected_ms * 5) / 1000.0)

    try:
        if mod is not None and hasattr(mod, "execute"):
            # Deterministic tile-loop: run once per tile and mosaic back (Rule 7).
            # Only tools that actually window on context["current_tile"] are looped;
            # anything else runs whole-scene, because N identical runs of a
            # scene-wide computation cannot be combined into a scene-wide answer.
            all_tiles = (tile_plan.deterministic_tiles or []) if is_tiled else []
            if (
                is_tiled
                and tool_name in _DETERMINISTIC_TILE_TOOLS
                and _tool_supports_tiling(mod)
                and all_tiles
            ):
                per_tile_outs: list[dict[str, Any]] = []
                tile_context = dict(context)
                for tile in all_tiles:
                    tile_context["current_tile"] = tile
                    per_tile_outs.append(
                        _execute_cached(mod, tool_name, manifest_version, params, tile_context)
                    )
                pixel_size_m = 10.0
                for out in per_tile_outs:
                    if out.get("_pixel_size_m"):
                        pixel_size_m = float(out["_pixel_size_m"])
                        break
                outputs = _mosaic_tile_outputs(per_tile_outs, pixel_size_m)
                mosaic = outputs.pop("_mosaic", None)
                if isinstance(mosaic, np.ndarray):
                    # One full-scene, geo-referenced GeoTIFF -- never tile-resolution.
                    uri = _write_mosaic_geotiff(
                        mosaic,
                        tool_name,
                        outputs.get("_crs") or context.get("crs"),
                        pixel_size_m,
                        outputs.get("_src_transform"),
                    )
                    if uri:
                        outputs["mask_uri" if mosaic.dtype == np.uint8 else "response_map"] = uri
                    # Publish the scene-level array so later waves (centroid_prior,
                    # object_box_fallback) see the mosaic, not the last tile.
                    cache = context.setdefault("mask_cache", {})
                    if mosaic.dtype == np.uint8:
                        cache[tool_name] = mosaic
                        if outputs.get("index"):
                            cache[outputs["index"]] = mosaic
                    else:
                        context["texture_map"] = mosaic
            else:
                if is_tiled and tool_name in _DETERMINISTIC_TILE_TOOLS and all_tiles:
                    context.setdefault("tiling_skipped", []).append(tool_name)
                outputs = _execute_cached(mod, tool_name, manifest_version, params, context)
        else:
            outputs = {"answer": f"Simulated output for {tool_name}"}
    except Exception as e:
        err = f"Tool '{tool_name}' failed with exception: {e}"

    # Private mosaic keys never reach the trace: it is JSON-serialised on write.
    outputs = {k: v for k, v in outputs.items() if not k.startswith("_")}

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
        asset_id = asset_id_for(tool_name, "mask_geotiff", mask_path_str)
        assets.append(
            AssetRef(
                asset_id=asset_id,
                kind="mask_geotiff",
                label=f"{tool_name} output mask",
                produced_by=tool_name,
                media_type="image/tiff",
                bytes=file_bytes,
                crs=_mask_crs(mask_path_str) or context.get("crs"),
                bounds_wgs84=_bounds_wgs84(mask_path_str),
                tile_url_template=f"/assets/{asset_id}/tiles/{{z}}/{{x}}/{{y}}.png",
                overlay_url=f"/assets/{asset_id}/overlay.png",
                download_url=f"/assets/{asset_id}",
                colour=_ASSET_COLOURS.get(tool_name),
                stats={
                    "area_km2": outputs.get("area_km2", 0.0),
                    "source_path": mask_path_str,
                },
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
        asset_id = asset_id_for(tool_name, "bbox_px", repr(boxes_px))
        assets.append(
            AssetRef(
                asset_id=asset_id,
                kind="bbox_geojson",
                label=f"{tool_name} bounding boxes (pixel space)",
                produced_by=tool_name,
                media_type="application/json",
                bytes=max(256, len(str(boxes_px))),
                crs=None,
                bbox_px=boxes_px,
                download_url=f"/assets/{asset_id}/boxes_px.json",
            )
        )
        # When CRS is present also emit full GeoJSON (georeferenced polygons)
        if scene_crs:
            geo_asset_id = asset_id_for(tool_name, "bbox_geojson", repr(boxes_px))
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
        asset_id = asset_id_for(tool_name, "response_map", str(outputs["response_map"]))
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

    cancelled = False
    budget_exhausted = False
    for wave in waves:
        # §24: cancellation is cooperative and checked at wave boundaries, never
        # mid-tool -- a half-written GeoTIFF is worse than a late one.
        if _is_cancelled(state):
            cancelled = True
            note = "Cancelled by request; stopped at a wave boundary"
            warnings.append(note)
            trace.add_warning(note)
            break
        # §25: on the global budget, return the best partial answer rather than
        # nothing. Waves already completed stay in `results`.
        if time.perf_counter() - start_t > _query_budget_s():
            budget_exhausted = True
            note = (
                f"Global query budget of {_query_budget_s():.0f}s exhausted; "
                "returning partial results"
            )
            warnings.append(note)
            trace.add_warning(note)
            break

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

    # Rule 9 / §17: graded.tools_invoked names the adapter for learned tools and
    # the bare tool for deterministic ones. TraceBuilder auto-derives this from
    # step names, which would lose the adapter, so set it explicitly.
    invoked: list[str] = []
    for step in plan:
        name = step["tool"]
        out = results.get(name)
        if isinstance(out, dict) and out.get("adapter"):
            name = str(out["adapter"])
        if name not in invoked:
            invoked.append(name)
    if invoked:
        trace.set_tools_invoked(invoked)

    for skipped in dict.fromkeys(context.get("tiling_skipped") or []):
        note = (
            f"Tool '{skipped}' does not support per-tile execution; "
            "ran whole-scene instead of the tiled path"
        )
        if note not in warnings:
            warnings.append(note)
            trace.add_warning(note)

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
        "cancelled": cancelled,
        "budget_exhausted": budget_exhausted,
    }
