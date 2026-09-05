from typing import Any

from satquery.serving.client import (
    ADAPTER_VERSIONS,
    get_serving_client,
    qualified_adapter,
)
from satquery.tools import tiling_support


def _raise_if_unavailable(res: Any) -> None:
    """A missing model is a failed step, not an answer string (§20, F-3)."""
    if res.serving_mode == "unavailable" or res.text.startswith("MODEL_UNAVAILABLE"):
        raise RuntimeError("MODEL_UNAVAILABLE: Model serving offline or unreachable.")


def _tile_origin(tile: dict[str, Any]) -> tuple[float, float]:
    """(row, col) offset of a tile in the scene grid, or (0, 0) if unlocatable."""
    bbox = tile.get("bbox_px") if isinstance(tile, dict) else None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return float(bbox[1]), float(bbox[0])
    if isinstance(tile, dict) and "row_off" in tile and "col_off" in tile:
        return float(tile["row_off"]), float(tile["col_off"])
    return 0.0, 0.0


def _run_over_tiles(context, call):
    """Run `call(tile)` once per budgeted tile and aggregate per §12.4.

    Rule 14: a learned tool never sees more than `agent.learned_tool_tile_budget`
    tiles. The executor puts the already-budgeted selection in
    context["selected_tiles"]; this honours it rather than quietly using the
    whole scene, and reports the coverage so §10 N8 can price the sample and the
    answer can be qualified (§12.3).
    """
    tiles = (context or {}).get("selected_tiles") or []
    if not tiles:
        return None

    # §12.4 free text: the answer comes from the single highest-scoring tile, so
    # that is the only one worth streaming -- tokens from tiles whose answers are
    # discarded would show the user text that never becomes the answer.
    ranked = sorted(
        tiles,
        key=lambda t: float(t.get("score") or 0.0) if isinstance(t, dict) else 0.0,
        reverse=True,
    )
    per_tile = []
    for position, tile in enumerate(ranked):
        out = call(tile, position == 0)
        out["tile_score"] = float(tile.get("score") or 0.0) if isinstance(tile, dict) else 0.0
        per_tile.append(out)

    merged = tiling_support.aggregate_tile_answers(per_tile)
    merged.pop("tile_score", None)
    total = (context or {}).get("total_tile_count") or len(tiles)
    merged["tiles_seen"] = len(tiles)
    merged["tile_coverage_frac"] = round(len(tiles) / total, 4) if total else 1.0
    return merged


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = params.get("prompt", "")
    point_prior = params.get("point_prior")
    if not prompt and context and "bundle" in context:
        prompt = context.get("question", "")

    if context and "bundle" in context:
        bundle = context["bundle"]
        images = getattr(bundle, "images", [])
        if images:
            modalities = [img.modality for img in images]
            gsds = [float(getattr(img, "pixel_size_m", 10.0) or 10.0) for img in images]
            roles = ["single"] * len(images)
            from satquery.agent.prompt import assemble_prompt
            prompt = assemble_prompt(
                images=images,
                roles=roles,
                modalities=modalities,
                effective_gsd_m=gsds,
                question=prompt,
                point_prior=point_prior,
            )
    elif point_prior:
        prompt = f"{prompt} [point: {point_prior}]"

    max_tokens = int(params.get("max_tokens", 128))
    adapter = qualified_adapter("rs_ground_caption", ADAPTER_VERSIONS["rs_ground_caption"])
    client = get_serving_client()
    on_token = (context or {}).get("on_token")

    def _call(tile: dict[str, Any], stream: bool) -> dict[str, Any]:
        r = client.infer(
            adapter=adapter,
            prompt=prompt + f" [tile: {tile.get('tile_id', '?')}]",
            max_tokens=max_tokens,
            stream=stream and on_token is not None,
            on_token=on_token,
        )
        _raise_if_unavailable(r)
        # Boxes come back tile-local; §18 requires whole-scene pixel space, so
        # translate here at the tool boundary rather than letting two conventions
        # travel downstream. NMS then removes the seam duplicates (§12.4).
        row_off, col_off = _tile_origin(tile)
        moved = []
        for b in r.boxes or []:
            x0, y0, x1, y1 = b["bbox_px"]
            moved.append({
                **b,
                "bbox_px": [x0 + col_off, y0 + row_off, x1 + col_off, y1 + row_off],
            })
        return {"answer": r.text, "boxes": moved, "confidence": r.confidence}

    tiled = _run_over_tiles(context, _call)
    if tiled is not None:
        tiled["adapter"] = adapter
        tiled["caption"] = tiled.get("answer", "")
        return tiled

    res = client.infer(
        adapter=adapter,
        prompt=prompt,
        max_tokens=max_tokens,
        stream=on_token is not None,
        on_token=on_token,
    )

    if res.serving_mode == "unavailable" or res.text.startswith("MODEL_UNAVAILABLE"):
        raise RuntimeError("MODEL_UNAVAILABLE: Model serving offline or unreachable.")

    boxes = res.boxes or []

    return {
        "adapter": adapter,
        "boxes": boxes,
        "caption": res.text,
        "answer": res.text,
        "confidence": res.confidence,
    }

