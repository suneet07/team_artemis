from typing import Any

from satquery.serving.client import get_serving_client


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    question = params.get("question", "")
    max_tokens = int(params.get("max_tokens", 64))

    client = get_serving_client()
    res = client.infer(
        adapter="change_vqa@v1",
        prompt=question,
        max_tokens=max_tokens,
    )

    return {
        "answer": res.text,
        "confidence": res.confidence,
    }
