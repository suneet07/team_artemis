"""Find the largest micro-batch that survives the worst-case sequence length.

    python scripts/batch_saturation_ramp.py --source-px 512 --composites 3

Guard rail 4 says the §5.5 budget is re-derived from measurement. The 200-step
smoke cannot supply that number, because it ran on BEN chips: 120x120 px is
**14 vision tokens per image**, and ``max_pixels`` is a cap the vision tower
never upsamples to. A 512 px benchmark chip is 256 tokens. At three composites
that is 42 tokens per sample against 768 -- an eighteen-fold difference in the
vision sequence, on the same model and the same config.

So the smoke's 13.65 GB at micro-batch 4 says nothing about whether micro-batch
4 *fits* on real benchmark imagery. Nobody currently knows whether the first
long training run survives its first step. That is what this measures, before
twelve hours are committed rather than twenty minutes into them.

**Synthetic tiles, deliberately.** Peak memory and step time are functions of
tensor shapes; pixel values do not enter. Using synthetic inputs means the
ceiling can be measured without staging RSVQA-HR first, and it keeps this
honestly separate from anything that reads a loss. Nothing here is an accuracy
claim and the report says so.

**The real training path.** The model is built with ``build_lora_model`` and
stepped through the same forward/backward the trainer uses, so the number
measured is the number that will be paid. A lookalike loop that skips, say,
gradient checkpointing would report a ceiling the trainer cannot reach.

An OOM is a *result*, not a crash: it is recorded, the CUDA allocator is reset,
and the ramp stops there.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.config import preprocessing_config  # noqa: E402
from satquery.training.config import TrainingConfig  # noqa: E402
from satquery.training.dataset import SyntheticShapeDataset  # noqa: E402
from satquery.training.lora import audit_checkpointing, build_lora_model  # noqa: E402

DEFAULT_LADDER = (1, 2, 4, 8, 16, 32)

#: Enough steps for the allocator to reach steady state; fewer and the first
#: step's allocation churn dominates the timing.
WARMUP_STEPS = 2
MEASURED_STEPS = 5


def vision_tokens(source_px: int, composites: int, max_pixels: int) -> int:
    """Vision tokens per sample. The cap binds; the tower never upsamples."""
    return (min(source_px * source_px, max_pixels) // 1024) * composites


def _try_batch(bundle, dataset, micro_batch: int, source_px: int) -> dict:
    """One rung. Returns a result dict; an OOM is reported, not raised."""
    import torch
    from torch.utils.data import DataLoader

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, processor = bundle.model, bundle.processor
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    # The trainer's own collator, not a lookalike. Building the prompt by hand
    # produced text with no image placeholders, so the processor emitted 768
    # vision features with nowhere to put them:
    #   ValueError: Image features and image tokens do not match, tokens: 0,
    #   features: 768
    # Beyond fixing that, sharing the collator is what makes this measurement
    # transferable -- padding and label masking both change sequence length, and
    # a ceiling measured under different padding is a ceiling for a different
    # run.
    from satquery.training.collator import MaskedCollator

    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        collate_fn=MaskedCollator(processor),
        num_workers=0,
    )

    started = None
    steps_done = 0
    try:
        for index, batch in enumerate(loader):
            if index >= WARMUP_STEPS + MEASURED_STEPS:
                break
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if index == WARMUP_STEPS - 1:
                torch.cuda.synchronize()
                started = time.monotonic()
            elif index >= WARMUP_STEPS:
                steps_done += 1
    except torch.OutOfMemoryError as error:
        optimiser.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return {
            "micro_batch": micro_batch,
            "status": "OOM",
            "detail": str(error).splitlines()[0][:160],
        }

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    optimiser.zero_grad(set_to_none=True)
    del optimiser
    torch.cuda.empty_cache()

    seconds_per_step = elapsed / max(1, steps_done)
    return {
        "micro_batch": micro_batch,
        "status": "ok",
        "peak_vram_gb": round(peak, 2),
        "sec_per_step": round(seconds_per_step, 4),
        "samples_per_sec": round(micro_batch / seconds_per_step, 3),
        "source_px": source_px,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adapter", default="rs_vqa")
    parser.add_argument(
        "--source-px",
        type=int,
        default=512,
        help="tile side before the cap. 512 is the worst case in the training mix",
    )
    parser.add_argument("--composites", type=int, default=3)
    parser.add_argument("--ladder", default=",".join(str(n) for n in DEFAULT_LADDER))
    parser.add_argument(
        "--out",
        default=None,
        help="defaults to a name carrying the shape, so a second shape does not "
        "overwrite the first run's report",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This measures a memory ceiling on the GPU that will "
            "train; there is nothing to measure on CPU."
        )

    cfg = TrainingConfig.for_adapter(args.adapter)
    cap = preprocessing_config().tiling.max_pixels
    tokens = vision_tokens(args.source_px, args.composites, cap)

    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"source      : {args.source_px}px x {args.composites} composites")
    print(f"cap         : {cap} px -> {tokens} vision tokens/sample")
    print("building the model through the real training path...", flush=True)

    bundle = build_lora_model(cfg, verbose=False)
    bundle.model.to("cuda")
    audit = audit_checkpointing(bundle.model)
    print(
        f"attention   : {bundle.attention}   checkpointing {audit.active}/{audit.layers}",
        flush=True,
    )
    if audit.active != audit.layers:
        print(
            "WARNING: checkpointing is not fully active, so every memory number "
            "below is a ceiling for a configuration the trainer does not use.",
            flush=True,
        )

    dataset = SyntheticShapeDataset(
        source_size=args.source_px,
        composites=args.composites,
        length=max(int(n) for n in args.ladder.split(",")) * (WARMUP_STEPS + MEASURED_STEPS),
    )

    rungs = []
    for rung in (int(n) for n in args.ladder.split(",")):
        print(f"\nmicro-batch {rung}...", flush=True)
        result = _try_batch(bundle, dataset, rung, args.source_px)
        rungs.append(result)
        if result["status"] == "OOM":
            print(f"  OOM -- ceiling is below {rung}", flush=True)
            break
        print(
            f"  {result['peak_vram_gb']} GB  {result['sec_per_step']} s/step  "
            f"{result['samples_per_sec']} samples/s",
            flush=True,
        )

    ok = [r for r in rungs if r["status"] == "ok"]
    best = max(ok, key=lambda r: r["samples_per_sec"]) if ok else None
    largest = max(ok, key=lambda r: r["micro_batch"]) if ok else None

    report = {
        "run": "phase0_batch_saturation",
        "device": torch.cuda.get_device_name(0),
        "attention": bundle.attention,
        "checkpointing_active": f"{audit.active}/{audit.layers}",
        "source_px": args.source_px,
        "composites": args.composites,
        "vision_tokens_per_sample": tokens,
        "rungs": rungs,
        "largest_that_fits": largest["micro_batch"] if largest else None,
        "fastest": best["micro_batch"] if best else None,
        "caveat": (
            "Synthetic tiles. Memory and step time only -- no loss here is evidence "
            "of anything. Valid for this sequence length; a shorter one is cheaper "
            "and a longer one may not fit."
        ),
    }

    lines = [
        "# Phase 0 — batch saturation at the worst-case sequence length",
        "",
        f"`{report['device']}`, attention `{bundle.attention}`, "
        f"checkpointing {audit.active}/{audit.layers}.",
        "",
        f"Inputs: **{args.source_px}px x {args.composites} composites = "
        f"{tokens} vision tokens/sample**, against 42 for a 120px BEN chip.",
        "",
        "| micro-batch | peak VRAM | s/step | samples/s |",
        "|---|---|---|---|",
    ]
    for rung in rungs:
        if rung["status"] == "OOM":
            lines.append(f"| {rung['micro_batch']} | **OOM** | — | — |")
        else:
            lines.append(
                f"| {rung['micro_batch']} | {rung['peak_vram_gb']} GB | "
                f"{rung['sec_per_step']} | {rung['samples_per_sec']} |"
            )
    lines += [
        "",
        f"**Largest that fits: {report['largest_that_fits']}. "
        f"Fastest: {report['fastest']}.**",
        "",
        report["caveat"],
    ]

    out = Path(
        args.out
        or f"logs/phase0_batch_saturation_{args.source_px}px_x{args.composites}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nlargest that fits: {report['largest_that_fits']}  "
          f"fastest: {report['fastest']}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
