import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Rule 9: an adapter name always carries the base model it was trained against,
# so a trace says which weights produced the answer -- not merely which LoRA
# version. §29 Q7 (Qwen3-VL-4B vs Qwen3.5-2B) is still open, so the base is a
# config value, not a constant: when the bake-off resolves, this is the one
# place that changes and every trace follows.
DEFAULT_BASE_MODEL = "qwen3vl-4b"


ADAPTER_VERSIONS: dict[str, str] = {
    "rs_vqa": "v2",
    "rs_ground_caption": "v1",
    "change_vqa": "v1",
    "optsar_fusion": "v1",
    "lulc_classifier": "v1",
}


def base_model() -> str:
    """The base model adapters are trained against (env: SATQUERY_BASE_MODEL)."""
    return os.environ.get("SATQUERY_BASE_MODEL", DEFAULT_BASE_MODEL)


def qualified_adapter(tool_name: str, version: str) -> str:
    """Build a Rule 9 adapter name, e.g. change_vqa@qwen3vl-4b-v2.

    `version` is the adapter's own revision ("v2"); the base model is inserted
    between it and the tool so the pair is always legible together.
    """
    version = version.lstrip("@")
    return f"{tool_name}@{base_model()}-{version}"




# §22: "Two concurrent queries needing different adapters will contend on swap.
# Serialise learned-tool calls behind a single queue per adapter." One lock per
# adapter, so different adapters still run concurrently but a single adapter is
# never swapped out from under an in-flight request.
_ADAPTER_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _adapter_lock(adapter: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _ADAPTER_LOCKS.setdefault(adapter, threading.Lock())


def _emit_tokens(text: str, on_token: Callable[[str], None] | None) -> None:
    """Replay a completed answer as tokens so the UI streams either way.

    A real vLLM stream would call `on_token` as chunks arrive. The stub has the
    whole string at once; splitting it keeps the §19 event sequence identical so
    the frontend is not written against a shape only one backend produces.
    """
    if on_token is None or not text:
        return
    for word in text.split(" "):
        on_token(word + " ")


@dataclass
class ModelInferenceResult:
    text: str
    confidence: float
    logprob: float | None = None
    serving_mode: str = "local_fallback"
    boxes: list[dict[str, Any]] | None = None
    latency_ms: int = 0


class ServingClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or os.environ.get(
            "SATQUERY_VLLM_URL", "http://localhost:8000/v1"
        )
        self._offline_mode = os.environ.get("SATQUERY_OFFLINE_MODE", "false").lower() in (
            "true",
            "1",
        )
        self._health_cache_time: float = 0.0
        self._health_cache_val: bool | None = None

    def is_healthy(self, ttl: float = 10.0) -> bool:
        if self._offline_mode:
            return False
        if os.environ.get("SATQUERY_STUB_SERVING") == "1":
            # Stub mode is a deliberate choice, not a fallback. Probing a port
            # nobody is listening on costs ~1s per call on Windows and every
            # learned step pays it against the 20 s budget (§25).
            return False
        import time

        now = time.monotonic()
        if self._health_cache_val is not None and (now - self._health_cache_time) < ttl:
            return self._health_cache_val

        try:
            import urllib.request

            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                healthy = bool(resp.status == 200)
        except Exception:
            healthy = False

        self._health_cache_time = now
        self._health_cache_val = healthy
        return healthy

    def get_health(self) -> dict[str, Any]:
        healthy = self.is_healthy()
        return {
            "status": "ok",
            "version": "0.4.1",
            "trace_schema_version": 2,
            "serving": "vllm" if healthy else "local_fallback",
            "gpu": False,
            "offline_mode": self._offline_mode or not healthy,
            "base_model": base_model(),
            # The adapters this build would request. Whether they are trained and
            # resident is a separate question -- `serving` above says whether any
            # model is reachable at all.
            "adapters_loaded": [] if not healthy else [
                qualified_adapter(name, ver)
                for name, ver in ADAPTER_VERSIONS.items()
            ],
            "demo_bundles_warm": 3,
        }

    def infer(
        self,
        adapter: str,
        prompt: str,
        images: list[Any] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> ModelInferenceResult:
        """Run one adapter. §22.

        `on_token` is called with each chunk as it arrives so the executor can
        forward §19's `token` event; it is a no-op in headless mode because the
        emitter is. Adapter swaps are serialised per §22 -- see `_ADAPTER_LOCKS`.
        """
        started = time.monotonic()
        with _adapter_lock(adapter):
            result = self._infer_locked(adapter, prompt, images, max_tokens, temperature)
        result.latency_ms = int((time.monotonic() - started) * 1000)
        if stream and result.serving_mode != "unavailable":
            _emit_tokens(result.text, on_token)
        return result

    def _infer_locked(
        self,
        adapter: str,
        prompt: str,
        images: list[Any] | None,
        max_tokens: int,
        temperature: float,
    ) -> ModelInferenceResult:
        # If real vLLM server is running, attempt query
        if not self._offline_mode and self.is_healthy():
            try:
                import json
                import urllib.request

                payload = {
                    "model": adapter,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    text = resp_data["choices"][0]["text"].strip()
                    return ModelInferenceResult(
                        text=text,
                        confidence=0.88,
                        logprob=-0.12,
                        serving_mode="vllm",
                    )
            except Exception:
                pass

        # Gated stub serving (§22 / C-3)
        if os.environ.get("SATQUERY_STUB_SERVING", "0") != "1":
            return ModelInferenceResult(
                text="MODEL_UNAVAILABLE: Model serving offline or unreachable.",
                confidence=0.0,
                logprob=None,
                serving_mode="unavailable",
            )

        # Deterministic local stub fallback based on adapter and prompt
        p_lower = prompt.lower()
        if "ground" in adapter or "caption" in adapter:
            if "aircraft" in p_lower:
                text = "Aircraft observed on the apron."
                boxes = [
                    {"bbox_px": [120.0, 210.0, 180.0, 310.0], "class": "aircraft", "score": 0.89}
                ]
            elif "building" in p_lower:
                text = "Building structure detected in the central cluster."
                boxes = [
                    {"bbox_px": [80.0, 140.0, 160.0, 240.0], "class": "building", "score": 0.91}
                ]
            else:
                text = f"Region of interest identified for prompt: {prompt}"
                boxes = [
                    {"bbox_px": [100.0, 100.0, 200.0, 200.0], "class": "target", "score": 0.85}
                ]
            return ModelInferenceResult(
                text=text,
                confidence=0.89,
                logprob=-0.11,
                serving_mode="local_fallback",
                boxes=boxes,
            )

        if "change" in adapter:
            if "vegetation" in p_lower or "forest" in p_lower:
                text = "Vegetation reduction and minor clearing detected across observation period."
            elif "urban" in p_lower or "building" in p_lower:
                text = "New construction and expansion of built-up surfaces observed in sector."
            else:
                text = "Noticeable spectral change detected between earlier and later observation."
            return ModelInferenceResult(
                text=text,
                confidence=0.86,
                logprob=-0.15,
                serving_mode="local_fallback",
            )

        if "fusion" in adapter:
            text = "Optical and SAR analysis confirms surface water boundary and structure."
            return ModelInferenceResult(
                text=text,
                confidence=0.91,
                logprob=-0.09,
                serving_mode="local_fallback",
            )

        # Standard VQA fallback
        if "vegetation" in p_lower:
            text = "Dense vegetation and agricultural land dominate the surveyed area."
        elif "water" in p_lower:
            text = "A prominent water body is clearly visible in the central section."
        elif "urban" in p_lower or "built" in p_lower:
            text = "Urban structures and transit corridors are identified across the scene."
        else:
            text = "Analysis indicates mixed land cover with clear terrestrial features."

        return ModelInferenceResult(
            text=text,
            confidence=0.88,
            logprob=-0.12,
            serving_mode="local_fallback",
        )


_DEFAULT_CLIENT: ServingClient | None = None


def get_serving_client() -> ServingClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = ServingClient()
    return _DEFAULT_CLIENT
