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
    question = params.get("question", "")
    if not question and context and "bundle" in context:
        question = context.get("question", "")

    max_tokens = int(params.get("max_tokens", 64))
    temperature = float(params.get("temperature", 0.0))

    prompt = question
    if context and "bundle" in context:
        bundle = context["bundle"]
        images = getattr(bundle, "images", [])
        if images:
            modalities = [img.modality for img in images]
            gsds = [float(getattr(img, "pixel_size_m", 10.0) or 10.0) for img in images]
            pair_type = getattr(bundle, "pair_type", "single")
            if pair_type == "bitemporal":
                roles = [f"t{i}" for i in range(len(images))]
            elif pair_type == "crossmodal":
                roles = [img.modality for img in images]
            else:
                roles = ["single"] * len(images)
            from satquery.agent.prompt import assemble_prompt
            prompt = assemble_prompt(
                images=images,
                roles=roles,
                modalities=modalities,
                effective_gsd_m=gsds,
                question=question,
            )

    adapter = qualified_adapter("rs_vqa", ADAPTER_VERSIONS["rs_vqa"])
    client = get_serving_client()
    on_token = (context or {}).get("on_token")

    def _call(tile: dict[str, Any], stream: bool) -> dict[str, Any]:
        r = client.infer(
            adapter=adapter,
            prompt=prompt + f" [tile: {tile.get('tile_id', '?')}]",
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream and on_token is not None,
            on_token=on_token,
        )
        _raise_if_unavailable(r)
        return {"answer": r.text, "confidence": r.confidence}

    tiled = _run_over_tiles(context, _call)
    if tiled is not None:
        tiled["adapter"] = adapter
        return tiled

    res = client.infer(
        adapter=adapter,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=on_token is not None,
        on_token=on_token,
    )

    if res.serving_mode == "unavailable" or res.text.startswith("MODEL_UNAVAILABLE"):
        raise RuntimeError("MODEL_UNAVAILABLE: Model serving offline or unreachable.")

    return {
        "adapter": adapter,
        "answer": res.text,
        "confidence": res.confidence,
    }

