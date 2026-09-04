from typing import Any

from satquery.serving.client import get_serving_client


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

    client = get_serving_client()
    res = client.infer(
        adapter="rs_vqa@v2",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return {
        "answer": res.text,
        "confidence": res.confidence,
    }
