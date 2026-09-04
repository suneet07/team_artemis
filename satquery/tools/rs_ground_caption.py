from typing import Any

from satquery.serving.client import get_serving_client


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
    client = get_serving_client()
    res = client.infer(
        adapter="rs_ground_caption@v1",
        prompt=prompt,
        max_tokens=max_tokens,
    )

    if res.serving_mode == "unavailable" or res.text.startswith("MODEL_UNAVAILABLE"):
        raise RuntimeError("MODEL_UNAVAILABLE: Model serving offline or unreachable.")

    boxes = res.boxes or []

    return {
        "boxes": boxes,
        "caption": res.text,
        "answer": res.text,
        "confidence": res.confidence,
    }

