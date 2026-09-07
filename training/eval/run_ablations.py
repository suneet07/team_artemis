"""The four ablations, dispatched to harnesses that can actually fail.

    python training/eval/run_ablations.py --experiment optical_composites \
        --manifest data/eval/india_holdout_v0.jsonl
    python training/eval/run_ablations.py --experiment point_prior \
        --manifest data/eval/d2_200.jsonl
    python training/eval/run_ablations.py --list

This replaces a scaffold that printed a fixed script for each experiment and
ended with "Result: Pending ML Weights". It could not fail, so a green run meant
nothing; worse, ``tool_vs_vlm_math`` printed "Tool is mathematically perfect" as
a *result* when no arithmetic had been performed.

Three of the four experiments run zero-shot and therefore do not wait on
adapters:

``optical_composites`` (C22)
    Two composites against three. Section 4.2 adds a short-wave composite for
    ``rs_vqa`` and ``change_vqa``; the resolution policy calls the ablation
    sub-hour on real BEN chips while the cost model books +8.1 h against the
    A100 budget. Those two claims cannot both be true, and this harness
    measures which one is.

``multisensor_vs_rgb``
    Optical views alone against optical plus SAR, same questions. The C46 claim
    is that fusion earns its complexity; a wash here means it does not.

``point_prior`` (D2)
    Delegated to :mod:`training.eval.zero_shot`, which owns the arm construction.

``tool_vs_vlm_math``
    The one honest comparison of deterministic arithmetic against generation.
    It needs before/after masks in the manifest and it refuses to run without
    them, rather than asserting the tool is perfect.

Every experiment writes a markdown report and a JSON sidecar, and every score
carries the section 6.4 caveat: these are in-repo comparators, not official
scorers.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.training.config import TrainingConfig  # noqa: E402
from satquery.training.dataset import (  # noqa: E402
    RealChipDataset,
    load_canonical_manifest,
)
from satquery.training.metrics import (  # noqa: E402
    OFFICIAL_SCORER_CAVEAT,
    score_predictions,
)

EXPERIMENTS = (
    "optical_composites",
    "multisensor_vs_rgb",
    "point_prior",
    "tool_vs_vlm_math",
)

#: Numeric answers are scored with a relative tolerance, not exact string match.
#: "12.0 km2" and "12 km2" are the same answer and a formatter difference is not
#: an accuracy difference.
NUMERIC_TOLERANCE = 0.05

#: Below this, an arm has not answered the question set at all and no comparison
#: between arms is meaningful. Yes/no questions score ~0.5 by guessing, so a
#: number near zero means a format mismatch, not a weak model.
FLOOR_ACCURACY = 0.10

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def first_number(text: str) -> float | None:
    match = _NUMBER.search(str(text))
    return float(match.group()) if match else None


def numeric_accuracy(
    predictions: list[str], references: list[str], tolerance: float = NUMERIC_TOLERANCE
) -> dict[str, Any]:
    """Relative-tolerance accuracy over the first number in each answer.

    Unparsable predictions count as wrong, and are reported separately so a low
    score caused by formatting is distinguishable from one caused by arithmetic.
    """
    hits = unparsable = 0
    errors: list[float] = []
    for prediction, reference in zip(predictions, references, strict=True):
        got, want = first_number(prediction), first_number(reference)
        if want is None:
            raise ValueError(
                f"reference {reference!r} carries no number, so it cannot be scored "
                "numerically. Filter the manifest to quantitative rows first."
            )
        if got is None:
            unparsable += 1
            continue
        denominator = abs(want) if want else 1.0
        relative = abs(got - want) / denominator
        errors.append(relative)
        hits += relative <= tolerance
    total = len(predictions)
    return {
        "n": total,
        "accuracy": round(hits / total, 4) if total else 0.0,
        "tolerance": tolerance,
        "unparsable": unparsable,
        "median_relative_error": (
            round(sorted(errors)[len(errors) // 2], 4) if errors else None
        ),
    }


def _dataset(args, *, composites: int | None = None) -> RealChipDataset:
    samples = load_canonical_manifest(args.manifest, split=args.split)
    if args.limit:
        samples = samples[: args.limit]
    return RealChipDataset(
        samples,
        root=args.image_root or Path(args.manifest).parent,
        composites=composites,
        gsd_conditioning=True,
        # The optical-composites ablation scores two views against three on the
        # same rows, so changing the count is the measurement, not a mistake.
        resize_views=True,
    )


def _runner(args):
    """The model for this run, loaded at most once.

    Every handler used to build its own ``VLMRunner``, so ``--experiment all``
    would have paid four separate 8 GB loads off the volume for four experiments
    that share a model, a checkpoint and a ``max_pixels``. Memoised on the
    namespace rather than at module level so a caller running two different
    models in one process still gets two models.
    """
    existing = getattr(args, "_shared_runner", None)
    if existing is not None:
        return existing

    from satquery.training.generate import VLMRunner

    runner = VLMRunner(
        args.model, adapter_path=args.adapter or None, max_pixels=args.max_pixels or None
    )
    args._shared_runner = runner
    return runner


def _score_arm(runner, dataset: RealChipDataset, batch_size: int) -> dict[str, Any]:
    items = [dataset[i] for i in range(len(dataset))]
    predictions = runner.answer_all(items, batch_size=batch_size)
    references = [s.answer for s in dataset.samples]
    types = [s.raw.get("question_type", s.task) for s in dataset.samples]
    return score_predictions(predictions, references, types).as_dict()


# ---------------------------------------------------------------------------
# C22: two optical composites against three.
# ---------------------------------------------------------------------------


def run_optical_composites(args) -> dict[str, Any]:
    """C22: two optical composites against three, accuracy and token ceiling."""
    cfg = TrainingConfig.for_adapter("rs_vqa")
    runner = _runner(args)
    arms = {}
    for count in (2, 3):
        dataset = _dataset(args, composites=count)
        print(f"arm: {count} composites ({len(dataset)} samples)")
        arms[f"{count}_composites"] = {
            **_score_arm(runner, dataset, args.batch_size),
            # The cost half of the decision. Sequence length is linear in the
            # composite count, and so is step time. This is the cap-bound upper
            # limit: a 120 px BEN chip costs 16 tokens whatever max_pixels says,
            # so read it as a ceiling rather than a measurement.
            "vision_token_ceiling_per_sample": cfg.vision_tokens_per_image * count,
        }

    two, three = arms["2_composites"], arms["3_composites"]
    delta = three["average_accuracy"] - two["average_accuracy"]

    # Both arms at the floor means neither arm answered in the reference's
    # format, and the delta between two unusable numbers is not a result. The
    # first real run of this scored 0.006 and 0.004 on yes/no questions, where
    # guessing scores ~0.5 -- forty times below chance. The harness reported a
    # confident "drop to two composites" from it.
    #
    # The cause is structural, not a bug: zero-shot Qwen answers a yes/no
    # question with a sentence. This screen only means something once an adapter
    # has taught the answer format, or on a benchmark the base model can already
    # do. Refusing to rank is the correct output here.
    if max(two["average_accuracy"], three["average_accuracy"]) < FLOOR_ACCURACY:
        return {
            "experiment": "optical_composites",
            "arms": arms,
            "delta_average_accuracy": round(delta, 4),
            "verdict": (
                f"NO VERDICT. Both arms scored below {FLOOR_ACCURACY:.0%} "
                f"({two['average_accuracy']:.1%} and {three['average_accuracy']:.1%}), "
                "which is at or under chance for this question set. Neither arm "
                "answered in the reference format, so the difference between them "
                "measures nothing. Re-run with --adapter once one is trained, or "
                "against a benchmark the base model can already answer."
            ),
            "note": "Comparison void: both arms at the noise floor.",
        }

    if delta >= 0.02:
        verdict = (
            f"The short-wave composite earns its tokens: {delta:+.1%} average accuracy "
            "for a 50% longer vision sequence. Keep three composites for rs_vqa and "
            "change_vqa, and book the extra hours honestly."
        )
    elif delta <= -0.005:
        verdict = (
            f"Three composites are WORSE ({delta:+.1%}) as well as more expensive. Drop "
            "to two and reclaim the sequence length."
        )
    else:
        verdict = (
            f"The third composite moves accuracy {delta:+.1%} -- inside noise at "
            f"n={three['samples']} -- while adding 50% to the vision sequence and "
            "therefore roughly 50% to step time. On a fixed A100 budget that is a "
            "cost with no measured return: drop to two composites unless the "
            "adapter-trained numbers say otherwise."
        )
    return {
        "experiment": "optical_composites",
        "arms": arms,
        "delta_average_accuracy": round(delta, 4),
        "verdict": verdict,
        "note": (
            "Zero-shot. The C22 decision is about a trained adapter, so treat this as "
            "the cheap screen it is: a large zero-shot gap is evidence, a wash is not "
            "proof of a wash after fine-tuning."
        ),
    }


# ---------------------------------------------------------------------------
# C46: optical alone against optical plus SAR.
# ---------------------------------------------------------------------------


def _optical_only(samples):
    """Keep the optical views of each sample, dropping SAR.

    Rows whose ``modality`` list does not align with ``images`` are dropped with
    a count, not silently reinterpreted -- a misaligned row would put a SAR chip
    in the optical arm and quietly invert the experiment.
    """
    kept, misaligned = [], 0
    for sample in samples:
        modalities = sample.modality or []
        if len(modalities) != len(sample.images):
            misaligned += 1
            continue
        optical = [
            image
            for image, modality in zip(sample.images, modalities, strict=True)
            if str(modality).lower() != "sar"
        ]
        if not optical or len(optical) == len(sample.images):
            continue
        clone = replace(sample, images=optical)
        kept.append(clone)
    return kept, misaligned


def run_multisensor_vs_rgb(args) -> dict[str, Any]:
    """C46: optical views alone against optical plus SAR, identical questions."""
    samples = load_canonical_manifest(args.manifest, split=args.split)
    if args.limit:
        samples = samples[: args.limit]
    optical, misaligned = _optical_only(samples)
    if not optical:
        raise SystemExit(
            "No sample carries both an optical and a SAR view with an aligned "
            "`modality` list, so there is nothing to ablate. This experiment needs "
            "the C46 fusion corpus; run it against the paired manifest, not the "
            "single-sensor one."
        )
    paired = [s for s in samples if s.sample_id in {c.sample_id for c in optical}]
    root = args.image_root or Path(args.manifest).parent
    runner = _runner(args)

    print(f"arm: optical only ({len(optical)} samples)")
    rgb = _score_arm(runner, RealChipDataset(optical, root=root), args.batch_size)
    print(f"arm: optical + SAR ({len(paired)} samples)")
    fused = _score_arm(runner, RealChipDataset(paired, root=root), args.batch_size)

    delta = fused["average_accuracy"] - rgb["average_accuracy"]
    return {
        "experiment": "multisensor_vs_rgb",
        "arms": {"optical_only": rgb, "optical_plus_sar": fused},
        "delta_average_accuracy": round(delta, 4),
        "dropped_misaligned_rows": misaligned,
        "verdict": (
            f"Fusion adds {delta:+.1%} average accuracy on {fused['samples']} paired "
            "samples. "
            + (
                "The crossmodal adapter is carrying its own weight."
                if delta >= 0.02
                else "That is not a differentiator. Either the pairing is not "
                "co-registered well enough for the extra view to help, or the "
                "questions do not need SAR -- check the co-registration before "
                "concluding fusion does not work."
            )
        ),
        "note": (
            "Both arms use the same questions and the same model. Only the image "
            "stack differs, which is the only way the delta is attributable to the "
            "sensor rather than to the prompt."
        ),
    }


# ---------------------------------------------------------------------------
# D4: deterministic arithmetic against generation.
# ---------------------------------------------------------------------------


def run_tool_vs_vlm_math(args) -> dict[str, Any]:
    """D4: change_stats arithmetic against VLM generation on quantitative rows."""
    import numpy as np

    from satquery.tools.deterministic import execute_change_stats

    samples = load_canonical_manifest(args.manifest, split=args.split)
    if args.limit:
        samples = samples[: args.limit]
    root = Path(args.image_root or Path(args.manifest).parent)

    required = ("mask_t1", "mask_t2", "pixel_area_m2")
    usable = [s for s in samples if all(key in s.raw for key in required)]
    if not usable:
        raise SystemExit(
            "No manifest row carries mask_t1, mask_t2 and pixel_area_m2. The tool arm "
            "computes area change from masks; without them there is no arithmetic to "
            "compare against and the only honest output is this refusal. The previous "
            "scaffold printed 'Tool is mathematically perfect' here having computed "
            "nothing."
        )
    print(f"{len(usable)}/{len(samples)} rows carry masks and are scorable")

    tool_predictions = []
    for sample in usable:
        result = execute_change_stats(
            {
                "mask_t1": np.load(root / str(sample.raw["mask_t1"]).replace("\\", "/")),
                "mask_t2": np.load(root / str(sample.raw["mask_t2"]).replace("\\", "/")),
                "pixel_area_m2": float(sample.raw["pixel_area_m2"]),
                "class_names": sample.raw.get("class_names"),
            }
        )
        tool_predictions.append(result["answer"])

    runner = _runner(args)
    dataset = RealChipDataset(usable, root=root)
    vlm_predictions = runner.answer_all(
        [dataset[i] for i in range(len(dataset))], batch_size=args.batch_size
    )
    references = [s.answer for s in usable]

    tool_score = numeric_accuracy(tool_predictions, references)
    vlm_score = numeric_accuracy(vlm_predictions, references)
    return {
        "experiment": "tool_vs_vlm_math",
        "arms": {"change_stats_tool": tool_score, "vlm_generation": vlm_score},
        "delta_accuracy": round(tool_score["accuracy"] - vlm_score["accuracy"], 4),
        "verdict": (
            f"Tool {tool_score['accuracy']:.1%} against VLM {vlm_score['accuracy']:.1%} "
            f"at {NUMERIC_TOLERANCE:.0%} relative tolerance. "
            + (
                "Route quantitative change questions to change_stats, as section 4.6.4 "
                "specifies."
                if tool_score["accuracy"] > vlm_score["accuracy"]
                else "The tool is NOT ahead. Its arithmetic is exact, so a loss here "
                "means the masks it was handed are wrong -- fix the mask source before "
                "touching the routing rule."
            )
        ),
        "note": (
            "The tool's arithmetic over the arrays it is given is exact by "
            "construction; this experiment measures the end-to-end answer, masks "
            "included, which is the thing a user actually receives."
        ),
    }


# ---------------------------------------------------------------------------


def run_point_prior(args) -> dict[str, Any]:
    """Delegate to the D2 harness rather than reimplement its two arms."""
    from training.eval.zero_shot import run_d2

    namespace = argparse.Namespace(
        samples=args.manifest,
        model=args.model,
        image_root=args.image_root,
        limit=args.limit or 200,
        batch_size=args.batch_size,
        max_pixels=args.max_pixels,
        out=args.out or "logs/phase0_item10_d2.md",
        # Hand the loaded model across the module boundary; run_d2 loads its own
        # only when this is absent.
        _shared_runner=getattr(args, "_shared_runner", None),
    )
    result = run_d2(namespace)
    return {
        "experiment": "point_prior",
        "arms": {
            "no_prior": result["without_prior"],
            "centroid_prior": result["with_prior"],
        },
        "delta_average_accuracy": result["delta_average_accuracy"],
        "verdict": result["framing"],
        "note": "Full report at logs/phase0_item10_d2.md",
    }


HANDLERS = {
    "optical_composites": run_optical_composites,
    "multisensor_vs_rgb": run_multisensor_vs_rgb,
    "point_prior": run_point_prior,
    "tool_vs_vlm_math": run_tool_vs_vlm_math,
}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Ablation — {report['experiment']}",
        "",
        f"**Date:** {report['date']}  ",
        f"**Model:** `{report['model']}`  ",
        f"**Adapter:** `{report['adapter'] or 'none (zero-shot)'}`  ",
        f"**Manifest:** `{report['manifest']}`",
        "",
        f"> {OFFICIAL_SCORER_CAVEAT}",
        "",
        "| arm | headline | n |",
        "|---|---|---|",
    ]
    for name, arm in report["arms"].items():
        headline = arm.get("average_accuracy", arm.get("accuracy", 0.0))
        lines.append(f"| {name} | {headline:.4f} | {arm.get('samples', arm.get('n', 0))} |")
    lines += ["", "## Verdict", "", report["verdict"]]
    if report.get("note"):
        lines += ["", "## Caveat", "", report["note"]]
    return "\n".join(lines)


def _run_all(args) -> int:
    """Run every experiment in one process, on one model load.

    An experiment that cannot run is recorded as ``not_run`` with the reason and
    **never** as a result. The alternative -- letting the first missing manifest
    kill the process -- throws away the model load and the experiments that
    would have succeeded, which on a rented A100 is the expensive kind of tidy.

    The exit code still reflects failure. A partial sweep that exits 0 is how a
    missing experiment becomes a missing row in a table nobody re-reads.
    """
    reports: dict[str, Any] = {}
    failed: list[str] = []
    timings: dict[str, float] = {}

    # Load the model before the loop and time it separately. Charged to the
    # first experiment it would look like that experiment is slow, and the
    # §5.5 budget would be re-derived from a number that is mostly a download.
    load_started = time.monotonic()
    _runner(args)
    load_seconds = time.monotonic() - load_started
    print(f"model loaded in {load_seconds:.1f}s (once, shared)\n", flush=True)

    for name in EXPERIMENTS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        started = time.monotonic()
        try:
            reports[name] = HANDLERS[name](args)
        except (SystemExit, FileNotFoundError, ValueError, KeyError) as error:
            reason = str(error) or type(error).__name__
            print(f"NOT RUN: {reason}", flush=True)
            reports[name] = {"experiment": name, "status": "not_run", "reason": reason}
            failed.append(name)
        timings[name] = time.monotonic() - started
        print(f"[{timings[name]:.1f}s] {name}", flush=True)

    combined = {
        "run": "phase0_ablation_sweep",
        "date": datetime.now(UTC).isoformat(),
        "model": args.model,
        "adapter": args.adapter,
        "manifest": args.manifest,
        "experiments": reports,
        "not_run": failed,
        "timing": {
            "model_load_seconds": round(load_seconds, 1),
            "per_experiment_seconds": {k: round(v, 1) for k, v in timings.items()},
            "total_seconds": round(load_seconds + sum(timings.values()), 1),
            # What running these separately would have cost: the same work plus
            # one model load per experiment instead of one for the set.
            "separate_runs_would_add_seconds": round(load_seconds * (len(EXPERIMENTS) - 1), 1),
        },
        "caveat": OFFICIAL_SCORER_CAVEAT,
    }

    out = Path(args.out or "logs/ablation_all.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 0 — ablation sweep",
        "",
        f"Model `{args.model}`, adapter `{args.adapter or 'none (zero-shot)'}`.",
        f"One model load ({load_seconds:.0f}s) for {len(EXPERIMENTS)} experiments.",
        "",
        "| Experiment | Time | Verdict |",
        "|---|---|---|",
    ]
    for name, report in reports.items():
        seconds = f"{timings[name]:.0f}s"
        if report.get("status") == "not_run":
            lines.append(f"| {name} | {seconds} | **NOT RUN** — {report['reason'][:110]} |")
        else:
            verdict = str(report.get("verdict", "")).splitlines()[0][:110]
            lines.append(f"| {name} | {seconds} | {verdict} |")
    lines += [
        "",
        f"**Total {load_seconds + sum(timings.values()):.0f}s** "
        f"({load_seconds:.0f}s model load + {sum(timings.values()):.0f}s experiments). "
        f"Run separately this would have cost roughly "
        f"{load_seconds * (len(EXPERIMENTS) - 1):.0f}s more in repeated model loads.",
        "",
        "Timings are wall-clock on the GPU this ran on. Re-derive the §5.5 budget "
        "from them rather than from the estimates they replace.",
        "",
        OFFICIAL_SCORER_CAVEAT,
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(f"\n{len(EXPERIMENTS) - len(failed)}/{len(EXPERIMENTS)} ran -> {out}")
    if failed:
        print(f"not run: {', '.join(failed)}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"))
    parser.add_argument("--list", action="store_true", help="describe the experiments and exit")
    parser.add_argument("--manifest", help="canonical JSONL for this experiment")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None, help="LoRA path; omit for zero-shot")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.list:
        for name in EXPERIMENTS:
            doc = (HANDLERS[name].__doc__ or "").strip().split("\n")[0]
            print(f"{name:22} {doc or 'see module docstring'}")
        return 0
    if not args.experiment:
        parser.error("--experiment is required (or --list)")
    if not args.manifest:
        parser.error(f"--experiment {args.experiment} needs --manifest")

    if args.experiment == "all":
        return _run_all(args)

    report = HANDLERS[args.experiment](args)
    report.update(
        {
            "date": datetime.now(UTC).isoformat(),
            "model": args.model,
            "adapter": args.adapter,
            "manifest": args.manifest,
            "caveat": OFFICIAL_SCORER_CAVEAT,
        }
    )

    out = Path(args.out or f"logs/ablation_{args.experiment}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(report), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['verdict']}\n\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
