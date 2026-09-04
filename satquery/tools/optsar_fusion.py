from typing import Any

from satquery.serving.client import get_serving_client


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    question = params.get("question", "")
    max_tokens = int(params.get("max_tokens", 64))

    prompt = question
    if context and "bundle" in context:
        bundle = context["bundle"]
        images = getattr(bundle, "images", [])
        if images:
            modalities = [img.modality for img in images]
            gsds = [float(getattr(img, "pixel_size_m", 10.0) or 10.0) for img in images]
            roles = [img.modality for img in images]
            from satquery.agent.prompt import assemble_prompt
            prompt = assemble_prompt(
                images=images,
                roles=roles,
                modalities=modalities,
                effective_gsd_m=gsds,
                question=question,
            )

    client = get_serving_client()
    res = client.infer(
        adapter="optsar_fusion@v1",
        prompt=prompt,
        max_tokens=max_tokens,
    )

    if res.serving_mode == "unavailable" or res.text.startswith("MODEL_UNAVAILABLE"):
        raise RuntimeError("MODEL_UNAVAILABLE: Model serving offline or unreachable.")

    return {
        "answer": res.text,
        "confidence": res.confidence,
    }

