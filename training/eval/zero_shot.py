"""Phase 0 item 9 — the base-model bake-off, and item 10's D2 ablation.

    python training/eval/zero_shot.py bakeoff \
        --benchmark rsvqa=data/eval/rsvqa_lr.jsonl \
        --benchmark cdvqa=data/eval/cdvqa.jsonl \
        --benchmark grounding=data/eval/vrsbench_ground.jsonl

    python training/eval/zero_shot.py d2 --samples data/eval/d2_200.jsonl

Section 5.1 leaves the base open: Qwen3-VL-4B wins CDVQA test1 (AA 67.86 vs
66.94 for 8B) but Qwen3.5-2B scored highest on test2 (AA 69.56), and grounding
scales with size while grounding *is* gate G3. The plan's instruction is to test
both zero-shot on CDVQA **and** a grounding set, then commit to one -- never to
split bases across adapters, which doubles VRAM and destroys the multi-LoRA
argument.

This replaces a scaffold that printed "Result: Pending ML Weights" for all four
experiments. That file could not fail, so it never told anyone anything.

SCORING HONESTY
---------------
Scored with the in-repo comparator (``satquery.training.metrics``), not the
official scripts. Section 6.4 (C8) makes integrating the official scorers a
Phase 1 deliverable precisely because reimplemented metrics differ by large
margins. Every report carries the caveat; do not quote these as benchmark
results.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.evalcli.formatter import (  # noqa: E402
    contract_for,
    format_answer,
)
from satquery.training.dataset import (  # noqa: E402
    RealChipDataset,
    load_canonical_manifest,
)
from satquery.training.metrics import (  # noqa: E402
    OFFICIAL_SCORER_CAVEAT,
    ScoreReport,
    grounding_accuracy,
    score_predictions,
)

#: **Base model committed 2026-08-29: Qwen3-VL-4B-Instruct.** Decided by the
#: project rather than by the bake-off; Qwen3.5-2B is dropped and is not to be
#: re-added without reopening that decision.
#:
#: The bake-off keeps running against the committed base alone, because its
#: second job outlives the choice: a zero-shot baseline is what every
#: fine-tuned number is later reported against. Without it "the adapter scores
#: X" has nothing to be an improvement over.
#:
#: The measured §5.5 budget assumes this model. Changing it invalidates every
#: hour in that table.
COMMITTED_BASE = "Qwen/Qwen3-VL-4B-Instruct"
CANDIDATE_BASES = (COMMITTED_BASE,)


def _load(
    path: str,
    image_root: str | None,
    composites: int | None = None,
    keep_prefixes: tuple[str, ...] = (),
) -> RealChipDataset:
    """One benchmark as a dataset, built the way training built its corpus.

    ``composites`` must match what the adapter was trained with or the model is
    scored on an input shape it never saw. ``rs_vqa`` trains at three views per
    sample, so a one-image RSVQA row is repeated to three; leaving this None
    would quietly send one image and cost accuracy that has nothing to do with
    the model's ability to answer the question.
    """
    samples = load_canonical_manifest(path)
    if keep_prefixes:
        samples = [
            s
            for s in samples
            if str(s.raw.get("question_type", "")).startswith(keep_prefixes)
        ]
    return RealChipDataset(
        samples,
        root=image_root or Path(path).parent,
        composites=composites,
        gsd_conditioning=True,
    )


def _parse_box(text: str) -> tuple[float, float, float, float] | None:
    """Pull a 4-number box out of free text, or return None.

    Returns None rather than guessing when the model emitted no parsable box:
    :func:`grounding_accuracy` counts that as a miss, which is the honest
    treatment. Dropping unparsable rows would score only the easy samples.
    """
    import re

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if len(numbers) < 4:
        return None
    values = [float(v) for v in numbers[:4]]
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _format_row(benchmark: str, question_type: str, prediction: str, question: str):
    """One prediction through its benchmark's answer contract.

    A row whose question type has no contract (BEN's captioning and grounding,
    which belong to other gates) is passed through untouched and counted as
    recovered, so it is neither reformatted nor penalised here.
    """
    from satquery.evalcli.formatter import FormattedAnswer

    contract = contract_for(benchmark, question_type)
    if contract is None:
        return FormattedAnswer(str(prediction).strip().lower(), True, "no contract")
    return format_answer(prediction, contract, question=question)


def _ablate(items: list[dict], mode: str) -> list[dict]:
    """Break the question-image correspondence, to test whether it is used.

    A benchmark built from annotations can often be answered from the annotation
    statistics alone: "does arable land touch broad-leaved forest" has a skewed
    answer distribution *given the classes named in the question*, learnable
    without ever looking at a pixel. A score that survives this ablation was
    never about the imagery.

    ``shuffle`` pairs each question with another row's images, which keeps the
    input distribution exactly as the model expects and removes only the
    correspondence. ``blank`` substitutes flat grey, which is a blunter test:
    it also takes the input off-distribution, so a drop is weaker evidence --
    the model may be reacting to the strangeness rather than to the missing
    content. Prefer shuffle; blank is the sanity check on the sanity check.
    """
    if mode in ("", "none"):
        return items
    if mode == "blank":
        from PIL import Image

        return [
            {**item, "images": [Image.new("RGB", i.size, (127, 127, 127)) for i in item["images"]]}
            for item in items
        ]
    if mode == "shuffle":
        if len(items) < 2:
            return items
        # A fixed rotation rather than a random permutation: every row is
        # guaranteed a different row's images, and the run is reproducible.
        offset = max(1, len(items) // 2)
        return [
            {**item, "images": items[(index + offset) % len(items)]["images"]}
            for index, item in enumerate(items)
        ]
    raise SystemExit(f"--ablate-images wants none|shuffle|blank, got {mode!r}")


def run_bakeoff(args) -> dict:
    from satquery.training.generate import VLMRunner

    keep: dict[str, tuple[str, ...]] = {}
    for rule in args.keep_types or []:
        name, _, prefixes = rule.partition("=")
        if not prefixes:
            raise SystemExit(f"--keep-types wants name=prefix[,prefix], got {rule!r}")
        keep[name] = tuple(p.strip() for p in prefixes.split(",") if p.strip())

    benchmarks = {}
    for entry in args.benchmark:
        if "=" not in entry:
            raise SystemExit(f"--benchmark wants name=path.jsonl, got {entry!r}")
        name, path = entry.split("=", 1)
        # A benchmark file can carry rows for gates other than the ones being
        # scored. BEN.txt holds captioning and grounding alongside the VQA the
        # rs_vqa gates read, and a 200-word caption drags every batch it lands
        # in out to max_new_tokens while contributing to no gate here.
        dataset = _load(
            path,
            args.image_root,
            args.composites or None,
            keep.get(name, ()),
        )
        if args.limit:
            # A bounded run over the same code path, for checking that answers
            # come back in the shape the scorer expects before paying for the
            # full set. Sliced, not sampled: reproducible between runs.
            dataset.samples = dataset.samples[: args.limit]
        benchmarks[name] = dataset
        print(f"{name}: {len(benchmarks[name])} samples", flush=True)

    models = args.models or list(CANDIDATE_BASES)
    results: dict[str, dict] = {}

    for model_id in models:
        print(f"\n=== {model_id} ===")
        runner = VLMRunner(
            model_id,
            adapter_path=getattr(args, "adapter", None) or None,
            max_pixels=args.max_pixels or None,
        )
        per_benchmark: dict[str, dict] = {}

        for name, dataset in benchmarks.items():
            items = [dataset[i] for i in range(len(dataset))]
            items = _ablate(items, args.ablate_images)
            predictions = runner.answer_all(
                items, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens
            )
            references = [s.answer for s in dataset.samples]

            if name.startswith("ground"):
                boxes = [_parse_box(p) for p in predictions]
                truth = [_parse_box(r) for r in references]
                if any(t is None for t in truth):
                    raise SystemExit(
                        f"{name}: reference boxes are not parsable, so grounding "
                        "cannot be scored. Check the manifest answer format."
                    )
                score = grounding_accuracy(boxes, truth)
                per_benchmark[name] = {**score, "caveat": OFFICIAL_SCORER_CAVEAT}
                print(
                    f"  {name}: acc@0.5={score['acc@0.5']:.3f} "
                    f"mIoU={score['mean_iou']:.3f}",
                    flush=True,
                )
            else:
                types = [s.raw.get("question_type", s.task) for s in dataset.samples]
                # Section 6.4's formatter, and deliberately the *same* one the
                # headless mode uses: a judge scores what evalcli emits, so an
                # evaluation number computed through a second implementation
                # would be predicting the wrong thing. Applied to every model
                # alike, it changes how an answer is read and never what was
                # asked -- base and adapter see identical prompts.
                formatted = [
                    _format_row(name, kind, prediction, sample.question)
                    for prediction, kind, sample in zip(
                        predictions, types, dataset.samples, strict=True
                    )
                ]
                recovered = sum(1 for f in formatted if f.matched)

                if args.show:
                    # The raw string beside what the scorer will compare. A
                    # formatter that silently rescues a broken model, or a model
                    # emitting prose, both show up here and nowhere else.
                    print(f"  --- first {args.show} of {name} ---", flush=True)
                    for sample, raw, done in list(
                        zip(dataset.samples, predictions, formatted, strict=True)
                    )[: args.show]:
                        flag = "" if done.matched else "  <-- UNRECOVERED"
                        print(
                            f"    [{sample.raw.get('question_type')}] "
                            f"raw={raw[:70]!r} -> {done.text!r} "
                            f"ref={sample.answer!r}{flag}",
                            flush=True,
                        )
                report: ScoreReport = score_predictions(
                    [f.text for f in formatted], references, types
                )
                scored = report.as_dict()
                # The BEN.txt paper's IF column: how often an unambiguous answer
                # could be extracted at all. A low rate means the number below
                # is measuring format compliance, not vision.
                scored["instruction_following"] = round(
                    recovered / max(1, len(formatted)), 4
                )
                scored["unparsable"] = len(formatted) - recovered
                per_benchmark[name] = scored
                print(
                    f"  {name}: AA={report.average_accuracy:.3f} "
                    f"overall={report.overall_accuracy:.3f} "
                    f"IF={scored['instruction_following']:.3f}",
                    flush=True,
                )

            if args.dump_predictions:
                dump = Path(args.dump_predictions) / f"{_slug(model_id)}_{name}.jsonl"
                dump.parent.mkdir(parents=True, exist_ok=True)
                with dump.open("w", encoding="utf-8") as handle:
                    for sample, prediction in zip(
                        dataset.samples, predictions, strict=True
                    ):
                        handle.write(
                            json.dumps(
                                {
                                    "sample_id": sample.sample_id,
                                    "question_type": sample.raw.get("question_type"),
                                    "reference": sample.answer,
                                    "prediction": prediction,
                                    "formatted": _format_row(
                                        name,
                                        sample.raw.get("question_type", sample.task),
                                        prediction,
                                        sample.question,
                                    ).text,
                                }
                            )
                            + "\n"
                        )

        results[model_id] = per_benchmark
        del runner

    report = {
        "run": "phase0_item9_bakeoff",
        "date": datetime.now(UTC).isoformat(),
        "adapter": getattr(args, "adapter", None),
        "composites": args.composites or None,
        "ablate_images": args.ablate_images,
        "models": results,
        "caveat": OFFICIAL_SCORER_CAVEAT,
    }
    _write(args.out, report, _bakeoff_markdown(report))
    return report


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(".", "-")


#: Section 6.1 targets for the rs_vqa adapter, as percentages. BEN.txt carries
#: two of them in one manifest -- binary and MCQ are separate gates scored over
#: disjoint question types -- so the pass/fail line cannot be read off a
#: benchmark's headline average and has to be recomputed per prefix.
RS_VQA_GATES = (
    ("BEN.txt binary VQA", "ben", "binary_", (), 70.0),
    ("BEN.txt MCQ", "ben", "mcq_", (), 45.0),
    ("RSVQA-LR", "rsvqa_lr", "", (), 88.0),
    ("RSVQA-HR *(official, all types)*", "rsvqa_hr", "", (), 85.0),
    ("RSVQA-HR *(our restricted set)*", "rsvqa_hr", "", ("area",), 85.0),
)

#: Why RSVQA-HR is reported twice. ``area`` answers are stored as exact
#: integers, but the dataset's own protocol scores them as five classes
#: (Lobry et al. 2020, and the ConfigILM reference implementation quantises
#: identically). Bucketed and trained on, the type reached 87.6% -- so the
#: restricted average now sits *below* the official one, and reporting only the
#: restricted figure would understate the model rather than flatter it. Both
#: are published because the two answer different questions: the official row
#: is comparable to other work, the restricted row isolates the three types
#: whose references are unambiguous.
HR_AREA_CAVEAT = (
    "RSVQA-HR is reported twice. The official row scores all four question "
    "types, as published work does. The restricted row excludes `area`, whose "
    "references come from OpenStreetMap polygons and are noisy at the item "
    "level (66% of the test split is `0m2`). Area is quantised into the "
    "dataset's own five classes for both training and scoring, per the paper; "
    "scored as exact integers it would be unanswerable by construction."
)


def _gate_rows(benchmarks: dict) -> list[str]:
    """The four section 6.1 numbers, or a note saying which data was absent."""
    rows = []
    for label, benchmark, prefix, excluded, target in RS_VQA_GATES:
        score = benchmarks.get(benchmark)
        if not score or "per_type" not in score:
            rows.append(f"| {label} | — | {target:.0f} | not run |")
            continue
        types = {
            name: value
            for name, value in score["per_type"].items()
            if name.startswith(prefix) and name not in excluded
        }
        if not types:
            rows.append(f"| {label} | — | {target:.0f} | no matching types |")
            continue
        # Average accuracy over the gate's own types, which is what the
        # benchmarks report -- not the overall, which a large type would skew.
        accuracy = 100 * sum(v["accuracy"] for v in types.values()) / len(types)
        n = sum(v["n"] for v in types.values())
        verdict = "**PASS**" if accuracy >= target else "FAIL"
        following = score.get("instruction_following")
        suffix = f" (n={n}"
        if following is not None:
            suffix += f", IF={100 * following:.1f}%"
        rows.append(f"| {label} | {accuracy:.2f} | {target:.0f} | {verdict}{suffix}) |")
    return rows


def _bakeoff_markdown(report: dict) -> str:
    lines = [
        "# Phase 0 item 9 — base model bake-off",
        "",
        f"**Date:** {report['date']}",
        "",
        f"**Adapter:** `{report.get('adapter') or 'none (zero-shot)'}`",
        "",
        f"> {report['caveat']}",
        "",
        "| model | benchmark | AA / acc@0.5 | overall / mIoU | n |",
        "|---|---|---|---|---|",
    ]
    for model_id, benchmarks in report["models"].items():
        for name, score in benchmarks.items():
            if "acc@0.5" in score:
                lines.append(
                    f"| `{model_id}` | {name} | {score['acc@0.5']:.3f} | "
                    f"{score['mean_iou']:.3f} | — |"
                )
            else:
                lines.append(
                    f"| `{model_id}` | {name} | {score['average_accuracy']:.3f} | "
                    f"{score['overall_accuracy']:.3f} | {score['samples']} |"
                )
    for model_id, benchmarks in report["models"].items():
        lines += [
            "",
            f"## Section 6.1 gates — `{model_id}`",
            "",
            f"Adapter: `{report.get('adapter') or 'none (zero-shot)'}`",
            "",
            "| gate | score | target | verdict |",
            "|---|---|---|---|",
            *_gate_rows(benchmarks),
            "",
            f"> {HR_AREA_CAVEAT}",
        ]

    lines += [
        "",
        "## Deciding",
        "",
        "Section 5.1: commit to **one** base. Splitting bases across adapters "
        "doubles VRAM and removes the multi-LoRA serving argument (C1).",
        "",
        "Grounding carries gate G3 and is scored on the hidden set whichever "
        "option we elect (C17), so a grounding win outranks a VQA win of similar "
        "margin. If the 2B leads on VQA but trails on grounding, take the 4B.",
        "",
        "Record the decision and the numbers behind it in `TEAM_CONTEXT.md`, and "
        "tick 'Base model committed' in the Phase 0 checklist.",
    ]
    return "\n".join(lines)


def run_d2(args) -> dict:
    """Phase 0 item 10 — does the centroid point prior help, and how to frame D2.

    Same samples, same model, two arms: the question alone, and the question
    with a deterministic centroid prior injected as text. The prior comes from
    ``satquery.tools.deterministic``, so the ablation measures the tool that
    would actually run rather than an idealised oracle point.
    """
    from satquery.tools.deterministic import prior_for
    from satquery.training.generate import VLMRunner

    dataset = _load(args.samples, args.image_root)
    limit = min(args.limit, len(dataset)) if args.limit else len(dataset)
    print(f"D2 ablation on {limit} samples")

    # A caller that already holds a matching model passes it in rather than
    # paying a second 8 GB load -- this is how `run_ablations --experiment all`
    # shares one model across every experiment including this one.
    runner = getattr(args, "_shared_runner", None) or VLMRunner(
        args.model, max_pixels=args.max_pixels or None
    )
    plain, primed = [], []
    for index in range(limit):
        item = dataset[index]
        plain.append(item)
        sample = dataset.samples[index]
        target = sample.raw.get("target") or sample.task
        prior = sample.raw.get("point_prior") or prior_for(str(target))
        hint = _prior_sentence(prior)
        primed.append({**item, "question": f"{item['question']} {hint}"})

    references = [dataset.samples[i].answer for i in range(limit)]
    types = [dataset.samples[i].raw.get("question_type", "all") for i in range(limit)]

    print("  arm 1/2: no prior")
    without = score_predictions(
        runner.answer_all(plain, batch_size=args.batch_size), references, types
    )
    print("  arm 2/2: with centroid prior")
    with_prior = score_predictions(
        runner.answer_all(primed, batch_size=args.batch_size), references, types
    )

    delta = with_prior.average_accuracy - without.average_accuracy
    report = {
        "run": "phase0_item10_d2_ablation",
        "date": datetime.now(UTC).isoformat(),
        "model": args.model,
        "samples": limit,
        "without_prior": without.as_dict(),
        "with_prior": with_prior.as_dict(),
        "delta_average_accuracy": round(delta, 4),
        "framing": _d2_framing(delta),
        "caveat": OFFICIAL_SCORER_CAVEAT,
    }
    _write(args.out, report, _d2_markdown(report))
    print(f"\nDelta AA {delta:+.4f} -> {report['framing']}")
    return report


def _prior_sentence(prior: dict) -> str:
    """Render a prior as text the model can use, without inventing precision."""
    if "cx" in prior and "cy" in prior:
        return (
            f"A deterministic detector places the target near normalised "
            f"coordinates ({prior['cx']:.3f}, {prior['cy']:.3f})."
        )
    parts = ", ".join(f"{k}={v:g}" for k, v in sorted(prior.items()))
    return f"A deterministic detector reports a shape prior for the target ({parts})."


def _d2_framing(delta: float) -> str:
    """Section 1.3: the ablation decides how D2 is presented, not whether it ships."""
    if delta >= 0.02:
        return (
            "D2 framed as an accuracy differentiator: the prior measurably helps, so "
            "productionise centroid_prior in the rs_ground_caption prompt mix (Phase 2)."
        )
    if delta <= -0.02:
        return (
            "D2 framed as evidence only: the prior HURTS generation, so keep "
            "centroid_prior as a trace artifact and do not inject it into prompts."
        )
    return (
        "D2 framed as an explainability differentiator, not an accuracy one: the "
        "prior is neutral within noise at this sample count. It still earns its "
        "place in the trace as a deterministic, checkable point, but the deck must "
        "not claim an accuracy gain."
    )


def _d2_markdown(report: dict) -> str:
    return "\n".join(
        [
            "# Phase 0 item 10 — D2 centroid-prior ablation",
            "",
            f"**Date:** {report['date']}  ",
            f"**Model:** `{report['model']}`  ",
            f"**Samples:** {report['samples']}",
            "",
            f"> {report['caveat']}",
            "",
            "| arm | average accuracy | overall |",
            "|---|---|---|",
            f"| no prior | {report['without_prior']['average_accuracy']:.4f} | "
            f"{report['without_prior']['overall_accuracy']:.4f} |",
            f"| centroid prior | {report['with_prior']['average_accuracy']:.4f} | "
            f"{report['with_prior']['overall_accuracy']:.4f} |",
            f"| **delta** | **{report['delta_average_accuracy']:+.4f}** | |",
            "",
            "## Verdict",
            "",
            report["framing"],
        ]
    )


def _write(out: str, report: dict, markdown: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {path} and {path.with_suffix('.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    bake = sub.add_parser("bakeoff", help="Phase 0 item 9")
    bake.add_argument("--benchmark", action="append", required=True, metavar="name=path")
    bake.add_argument("--models", nargs="*", default=None)
    bake.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter directory; omit for the zero-shot baseline",
    )
    bake.add_argument("--image-root", default=None)
    bake.add_argument(
        "--composites",
        type=int,
        default=0,
        help=(
            "image views per sample; must equal the adapter's training value "
            "(satquery.training.config.adapter_composites). 0 leaves rows as "
            "they are, which is only right for a corpus already at that width."
        ),
    )
    bake.add_argument("--batch-size", type=int, default=8)
    bake.add_argument("--max-new-tokens", type=int, default=64)
    bake.add_argument(
        "--keep-types",
        action="append",
        metavar="name=prefix[,prefix]",
        help="restrict one benchmark to question types with these prefixes",
    )
    bake.add_argument("--max-pixels", type=int, default=0)
    bake.add_argument("--dump-predictions", default=None)
    bake.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N rows of each benchmark (0 = all)",
    )
    bake.add_argument(
        "--ablate-images",
        default="none",
        choices=("none", "shuffle", "blank"),
        help="break the question-image link to test whether imagery is used",
    )
    bake.add_argument(
        "--show",
        type=int,
        default=0,
        help="print the first N raw predictions per benchmark for inspection",
    )
    bake.add_argument("--out", default="logs/phase0_item9_bakeoff.md")
    bake.set_defaults(func=run_bakeoff)

    d2 = sub.add_parser("d2", help="Phase 0 item 10")
    d2.add_argument("--samples", required=True)
    d2.add_argument("--model", default=CANDIDATE_BASES[0])
    d2.add_argument("--image-root", default=None)
    d2.add_argument("--limit", type=int, default=200)
    d2.add_argument("--batch-size", type=int, default=8)
    d2.add_argument("--max-pixels", type=int, default=0)
    d2.add_argument("--out", default="logs/phase0_item10_d2.md")
    d2.set_defaults(func=run_d2)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
