from typing import Any

from satquery.serving.client import get_serving_client


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    question = params.get("question", "")
    if not question and context and "bundle" in context:
        question = context.get("question", "")

    max_tokens = int(params.get("max_tokens", 64))
    temperature = float(params.get("temperature", 0.0))

    client = get_serving_client()
    res = client.infer(
        adapter="rs_vqa@v2",
        prompt=question,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return {
        "answer": res.text,
        "confidence": res.confidence,
    }
