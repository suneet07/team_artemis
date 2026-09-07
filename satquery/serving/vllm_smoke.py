"""Phase 0 item 7 — does vLLM actually serve this base with two hot LoRAs?

    python -m satquery.serving.vllm_smoke \
        --adapter rs_vqa=checkpoints/rs_vqa/adapter \
        --adapter change_vqa=checkpoints/change_vqa/adapter
    python -m satquery.serving.vllm_smoke --base-only     # before any adapter exists

Section 4.9 serves one base with four LoRAs swapped per request, and C1 keeps
the architecture unmodified precisely so that this is possible. The whole
serving story -- one 8 GB base resident, four adapters of ~50 MB each, swapped
per query -- rests on a capability nobody in this repo has yet observed working.
If vLLM cannot hot-swap LoRAs on Qwen3-VL, the fallback is four full models and
the VRAM budget changes by roughly 24 GB. That is a Phase 0 finding, not a
Phase 3 surprise.

What this checks, in order, stopping at the first failure:

1. vLLM imports and reports its version.
2. The base loads with ``enable_lora=True`` at ``max_lora_rank`` matching the
   trained rank. A rank mismatch is the most common silent failure -- vLLM
   raises at *request* time, not load time, so a smoke that only loads the
   engine proves nothing.
3. The base answers a multimodal prompt.
4. Each adapter answers the same prompt through ``LoRARequest``.
5. Adapters are swapped back and forth and the outputs stay stable, which is
   what catches state leaking between requests.

**Two adapters answering identically is a failure, not a pass.** It is what the
system looks like when the LoRA is loaded and then ignored, and it is the
outcome this script exists to catch. That check only has teeth once the adapters
are actually trained; with untrained or identical adapters it is expected to
trip, and the report says so rather than reporting a false alarm as a defect.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "SMOKE_PROMPT",
    "SmokeReport",
    "build_engine",
    "run_smoke",
]

#: One prompt, asked of every arm. Comparing arms requires the prompt to be the
#: only thing that is not varying.
SMOKE_PROMPT = (
    "[ground sample distance: 10 m] Is there any water body visible in this image? "
    "Answer yes or no."
)


@dataclass
class ArmResult:
    name: str
    adapter_path: str | None
    output: str
    load_seconds: float
    generate_seconds: float


@dataclass
class SmokeReport:
    model_id: str
    vllm_version: str
    max_lora_rank: int
    arms: list[ArmResult] = field(default_factory=list)
    swap_seconds: list[float] = field(default_factory=list)
    stable_under_swap: bool | None = None
    distinct_outputs: bool | None = None
    gpu_memory_gb: float = 0.0
    failure: str | None = None
    date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return self.failure is None and bool(self.arms)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def _synthetic_image(size: int = 120):
    """A deterministic tile, so the smoke needs no staged imagery.

    Serving is being tested here, not accuracy. A fixed pattern keeps the run
    reproducible and keeps this script runnable before any corpus exists.
    """
    import numpy as np
    from PIL import Image

    grid = np.indices((size, size)).sum(axis=0) % 256
    return Image.fromarray(np.stack([grid] * 3, axis=-1).astype("uint8"))


def build_engine(model_id: str, *, max_lora_rank: int, max_loras: int, max_pixels: int):
    """Construct a vLLM engine with LoRA enabled.

    ``max_lora_rank`` must be at least the rank the adapters were trained at
    (section 5.2 freezes r=16). vLLM will accept a smaller value at load and
    then reject every LoRA request, so it is passed explicitly rather than left
    at the default of 16 by luck.
    """
    from vllm import LLM

    return LLM(
        model=model_id,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 3},
        mm_processor_kwargs={"max_pixels": max_pixels},
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )


def _generate(engine, prompt: str, image, lora_request=None) -> tuple[str, float]:
    from vllm import SamplingParams

    started = time.time()
    outputs = engine.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(temperature=0.0, max_tokens=32),
        lora_request=lora_request,
    )
    return outputs[0].outputs[0].text.strip(), time.time() - started


def run_smoke(
    model_id: str,
    adapters: dict[str, str],
    *,
    max_lora_rank: int = 16,
    max_pixels: int | None = None,
    prompt: str = SMOKE_PROMPT,
    swap_rounds: int = 2,
) -> SmokeReport:
    if max_pixels is None:
        from satquery.config import preprocessing_config

        max_pixels = preprocessing_config().tiling.max_pixels

    try:
        import vllm
    except ImportError as error:
        return SmokeReport(
            model_id=model_id,
            vllm_version="not installed",
            max_lora_rank=max_lora_rank,
            failure=(
                f"vLLM is not installed ({error}). Section 4.9 depends on it; install "
                "the serving extra before treating item 7 as blocked on anything else."
            ),
        )

    report = SmokeReport(
        model_id=model_id,
        vllm_version=getattr(vllm, "__version__", "unknown"),
        max_lora_rank=max_lora_rank,
    )
    image = _synthetic_image()

    started = time.time()
    try:
        engine = build_engine(
            model_id,
            max_lora_rank=max_lora_rank,
            # One slot per adapter plus one, so a swap never has to evict mid-run
            # and the swap timing measures a swap rather than an eviction.
            max_loras=max(1, len(adapters)),
            max_pixels=max_pixels,
        )
    except Exception as error:  # noqa: BLE001 - the failure IS the finding
        report.failure = f"engine failed to load with enable_lora=True: {error!r}"
        return report
    load_seconds = time.time() - started
    print(f"engine up in {load_seconds:.1f}s (vLLM {report.vllm_version})")

    try:
        text, seconds = _generate(engine, prompt, image)
    except Exception as error:  # noqa: BLE001
        report.failure = f"base model could not answer a multimodal prompt: {error!r}"
        return report
    report.arms.append(
        ArmResult("base", None, text, round(load_seconds, 3), round(seconds, 3))
    )
    print(f"base -> {text!r}")

    from vllm.lora.request import LoRARequest

    requests = {}
    for index, (name, path) in enumerate(adapters.items(), start=1):
        resolved = Path(path)
        if not resolved.exists():
            report.failure = (
                f"adapter {name!r} does not exist at {resolved}. Train it first; a "
                "serving smoke against a missing adapter tests nothing."
            )
            return report
        requests[name] = LoRARequest(name, index, str(resolved))

    for name, request in requests.items():
        started = time.time()
        try:
            text, seconds = _generate(engine, prompt, image, lora_request=request)
        except Exception as error:  # noqa: BLE001
            report.failure = (
                f"adapter {name!r} failed at request time: {error!r}. A rank mismatch "
                f"surfaces here rather than at load -- adapters are trained at r=16, "
                f"the engine was built with max_lora_rank={max_lora_rank}."
            )
            return report
        report.arms.append(
            ArmResult(
                name,
                str(adapters[name]),
                text,
                round(time.time() - started, 3),
                round(seconds, 3),
            )
        )
        print(f"{name} -> {text!r}")

    if len(requests) >= 2:
        names = list(requests)
        baseline = {arm.name: arm.output for arm in report.arms}
        stable = True
        for _ in range(swap_rounds):
            for name in names:
                started = time.time()
                text, _ = _generate(engine, prompt, image, lora_request=requests[name])
                report.swap_seconds.append(round(time.time() - started, 3))
                if text != baseline[name]:
                    stable = False
        report.stable_under_swap = stable
        adapter_outputs = [arm.output for arm in report.arms if arm.name in requests]
        report.distinct_outputs = len(set(adapter_outputs)) == len(adapter_outputs)

    try:
        import torch

        report.gpu_memory_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:  # noqa: BLE001 - memory reporting is not the test
        report.gpu_memory_gb = 0.0

    return report


def _markdown(report: SmokeReport) -> str:
    lines = [
        "# Phase 0 item 7 — vLLM multi-LoRA serving smoke",
        "",
        f"**Date:** {report.date}  ",
        f"**Base:** `{report.model_id}`  ",
        f"**vLLM:** {report.vllm_version}  ",
        f"**max_lora_rank:** {report.max_lora_rank}  ",
        f"**Peak VRAM:** {report.gpu_memory_gb} GB",
        "",
        f"**Result: {'PASS' if report.passed else 'FAIL'}**",
        "",
    ]
    if report.failure:
        lines += [
            "## Failure",
            "",
            report.failure,
            "",
            "Section 4.9 assumes one resident base with adapters swapped per "
            "request. If that cannot be made to work, the fallback is four "
            "separately-served models and roughly 24 GB more VRAM. Resolve this "
            "before the serving budget is quoted anywhere.",
            "",
        ]
    if report.arms:
        lines += ["| arm | adapter | output | generate (s) |", "|---|---|---|---|"]
        for arm in report.arms:
            # Escape the cell separator before interpolating; a pipe in a model
            # output would otherwise split the row and corrupt the table.
            cell = arm.output.replace("|", "\\|")[:80]
            lines.append(
                f"| {arm.name} | `{arm.adapter_path or '—'}` | "
                f"{cell} | {arm.generate_seconds} |"
            )
        lines.append("")
    if report.swap_seconds:
        mean = sum(report.swap_seconds) / len(report.swap_seconds)
        lines += [
            f"**Swap latency:** {len(report.swap_seconds)} swaps, "
            f"mean {mean:.3f}s, max {max(report.swap_seconds):.3f}s",
            "",
        ]
    if report.stable_under_swap is False:
        lines += [
            "**Outputs changed under repeated swapping.** State is leaking between "
            "requests; per-request adapter selection is not safe until this is "
            "understood.",
            "",
        ]
    if report.distinct_outputs is False:
        lines += [
            "**Every adapter produced the same string.** That is what it looks like "
            "when the LoRA is loaded and then ignored. Expected if the adapters are "
            "untrained or identical — confirm which before filing it as a defect.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="name=path",
        help="repeatable; section 4.9 needs at least two to test hot-swapping",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="load and answer with no adapters — useful before any exist",
    )
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--out", default="logs/phase0_item7_vllm_smoke.md")
    args = parser.parse_args()

    adapters: dict[str, str] = {}
    for entry in args.adapter:
        if "=" not in entry:
            raise SystemExit(f"--adapter wants name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        adapters[name] = path

    if not adapters and not args.base_only:
        raise SystemExit(
            "Pass at least two --adapter name=path, or --base-only. Item 7 is "
            "specifically the hot-swap test; a base-only run is a partial result and "
            "must be labelled as one rather than ticking the checklist item."
        )

    report = run_smoke(
        args.model,
        adapters,
        max_lora_rank=args.max_lora_rank,
        max_pixels=args.max_pixels or None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(report), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )
    print(f"\n{'PASS' if report.passed else 'FAIL'} — report at {out}")
    if report.failure:
        print(report.failure)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
