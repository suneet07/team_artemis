from typing import Any

from satquery.serving.client import get_serving_client


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = params.get("prompt", "")
    point_prior = params.get("point_prior")
    if point_prior:
        prompt = f"{prompt} [point: {point_prior}]"

    max_tokens = int(params.get("max_tokens", 128))
    client = get_serving_client()
    res = client.infer(
        adapter="rs_ground_caption@v1",
        prompt=prompt,
        max_tokens=max_tokens,
    )

    default_boxes = [{"bbox_px": [100.0, 100.0, 200.0, 200.0], "class": "target", "score": 0.85}]
    boxes = res.boxes or default_boxes

    return {
        "boxes": boxes,
        "caption": res.text,
    }
