"""Phase 0 item 11 — the three-arm router bake-off on the 300-query set.

    python training/eval/router_zero_shot.py --arms rules
    python training/eval/router_zero_shot.py --arms rules llm hybrid --model Qwen/Qwen3.5-2B

Section 4.5.2 specifies a two-stage router: deterministic rules first, an LLM
tie-breaker only where the rules cannot separate two readings. That design is an
assertion until the three arms are measured against each other on the same set:

* **rules** -- the deterministic router alone. Already scored by
  ``score_routing.py``; re-run here so all three arms share one report.
* **llm** -- the LLM classifies every query with no rules at all. The arm that
  says whether the rules are earning their keep.
* **hybrid** -- rules, deferring to the LLM exactly where ``route()`` returns
  ``router_path=llm``. The shipping design.

The risk-register trigger lives here: **hybrid below 90% on unambiguous cases
means the rules get fixed in Week 1**, not discovered in Week 5. The rules arm
already clears it, so the question this run actually answers is narrower and
more useful -- does adding the LLM stage *help*, and on which cases does it
hurt? An LLM tie-breaker that degrades a 100% rules arm is a stage to delete,
and deleting it removes a serving dependency from the critical path.

The LLM arm is text-only on purpose. Routing decides which tools and which
adapter run; it happens before any imagery is loaded, so a router that needs
pixels is a router that cannot be called first.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.agent.router import route  # noqa: E402
from satquery.agent.task_enum import RouterPath, Task  # noqa: E402
from satquery.agent.validator import validate  # noqa: E402
from training.eval.score_routing import context_from_row, load_dataset  # noqa: E402

ARMS = ("rules", "llm", "hybrid")

#: The risk register's number. Below this on unambiguous cases, the rules are
#: broken and Week 1 is where that gets fixed.
UNAMBIGUOUS_GATE = 0.90

TASK_VALUES = tuple(task.value for task in Task)

_PROMPT = """You route satellite-imagery queries to exactly one task type.

Task types:
- single_vqa: a question about one image
- single_caption: describe one image
- single_grounding: locate or point to an object in one image
- change_description: describe what changed between two images
- change_vqa: a question about what changed between two images
- change_map: produce a change mask between two images
- crossmodal_extraction: extract information using optical and SAR together
- crossmodal_vqa: a question answered using optical and SAR together

Query: {query}
Images available: {image_count}
Modalities: {modalities}
Bands: {bands}

Answer with the task type only, no explanation.
Task:"""


class LLMTaskClassifier:
    """Text-only zero-shot task classification.

    Kept separate from :class:`satquery.training.generate.VLMRunner` because
    this one must not take images: a router runs before ingestion.
    """

    def __init__(self, model_id: str, *, device: str | None = None, max_new_tokens: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )

    @staticmethod
    def prompt_for(row: dict[str, Any]) -> str:
        ctx = row["input_context"]
        return _PROMPT.format(
            query=row["query_text"],
            image_count=ctx.get("image_count", 1),
            modalities=", ".join(ctx.get("modalities", [])) or "unknown",
            bands=", ".join(ctx.get("bands", []) or ctx.get("polarisations", [])) or "unknown",
        )

    def classify(self, rows: list[dict[str, Any]], *, batch_size: int = 16) -> list[str | None]:
        """Return a task value per row, or ``None`` where nothing parsed.

        ``None`` is kept rather than defaulted to ``single_vqa``. A silent
        default would score as a correct answer on the most common class and
        make the arm look better than it is.
        """
        import torch

        answers: list[str | None] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = self.tokenizer(
                [self.prompt_for(row) for row in chunk],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            trimmed = generated[:, encoded["input_ids"].shape[1] :]
            for text in self.tokenizer.batch_decode(trimmed, skip_special_tokens=True):
                answers.append(parse_task(text))
        return answers


def parse_task(text: str) -> str | None:
    """Match the longest task name present in the output.

    Longest-first matters: ``change_vqa`` contains no substring trap but
    ``single_vqa`` and ``crossmodal_vqa`` both end in ``vqa``, and a shortest-
    match scan would mislabel them.
    """
    lowered = text.strip().lower()
    for value in sorted(TASK_VALUES, key=len, reverse=True):
        if value in lowered:
            return value
    return None


def _score(rows: list[dict], predictions: list[str | None]) -> dict[str, Any]:
    hits = unambiguous_hits = unambiguous_total = unparsable = 0
    misroutes = []
    for row, predicted in zip(rows, predictions, strict=True):
        truth = row["ground_truth"]
        expected = truth["expected_task"]
        correct = predicted == expected
        hits += correct
        unparsable += predicted is None
        if truth["unambiguous"]:
            unambiguous_total += 1
            unambiguous_hits += correct
            if not correct:
                misroutes.append(
                    {
                        "id": row["id"],
                        "query": row["query_text"],
                        "expected": expected,
                        "predicted": predicted,
                        "category": truth["expected_category"],
                    }
                )
    total = len(rows)
    return {
        "n": total,
        "task_accuracy": round(hits / total, 4) if total else 0.0,
        "unambiguous_accuracy": (
            round(unambiguous_hits / unambiguous_total, 4) if unambiguous_total else 0.0
        ),
        "unambiguous_n": unambiguous_total,
        "unparsable": unparsable,
        "misroutes": misroutes,
    }


def run_rules(rows: list[dict]) -> dict[str, Any]:
    predictions, deferred, caught, refusal_total = [], 0, 0, 0
    for row in rows:
        context = context_from_row(row)
        decision = route(context)
        predictions.append(decision.task.value)
        deferred += decision.router_path is RouterPath.LLM
        if row["ground_truth"]["expected_category"] == "refusal":
            refusal_total += 1
            caught += not validate(context, decision).passed
    report = _score(rows, predictions)
    report["deferred_to_llm"] = deferred
    report["defer_rate"] = round(deferred / len(rows), 4) if rows else 0.0
    report["invalid_config_catch_rate"] = (
        round(caught / refusal_total, 4) if refusal_total else 1.0
    )
    report["predictions"] = predictions
    return report


def run_llm(rows: list[dict], classifier: LLMTaskClassifier, batch_size: int) -> dict[str, Any]:
    predictions = classifier.classify(rows, batch_size=batch_size)
    report = _score(rows, predictions)
    report["predictions"] = predictions
    return report


def run_hybrid(rows: list[dict], rules: dict, llm: dict | None) -> dict[str, Any]:
    """Rules everywhere; the LLM only where ``route()`` actually deferred.

    Without an LLM arm this is identical to the rules arm -- which is itself the
    finding, and is reported as such rather than skipped.
    """
    predictions: list[str | None] = []
    substitutions = 0
    for index, row in enumerate(rows):
        decision = route(context_from_row(row))
        if decision.router_path is RouterPath.LLM and llm is not None:
            candidate = llm["predictions"][index]
            # A tie-breaker that returns nothing must not erase a rules answer
            # that was at least a legal task.
            if candidate is not None:
                predictions.append(candidate)
                substitutions += 1
                continue
        predictions.append(rules["predictions"][index])
    report = _score(rows, predictions)
    report["llm_substitutions"] = substitutions
    report["predictions"] = predictions
    return report


def _verdict(report: dict[str, Any]) -> str:
    rules = report["arms"].get("rules")
    hybrid = report["arms"].get("hybrid")
    llm = report["arms"].get("llm")
    lines = []

    if rules and rules["unambiguous_accuracy"] < UNAMBIGUOUS_GATE:
        lines.append(
            f"**Rules arm is at {rules['unambiguous_accuracy']:.1%} on unambiguous cases, "
            f"below the {UNAMBIGUOUS_GATE:.0%} risk-register trigger. Fix the rules now.**"
        )
    elif rules:
        lines.append(
            f"Rules arm clears the {UNAMBIGUOUS_GATE:.0%} trigger at "
            f"{rules['unambiguous_accuracy']:.1%} on {rules['unambiguous_n']} unambiguous cases."
        )

    if rules and hybrid:
        delta = hybrid["unambiguous_accuracy"] - rules["unambiguous_accuracy"]
        if hybrid.get("llm_substitutions", 0) == 0:
            lines.append(
                "The hybrid arm is numerically identical to the rules arm: the LLM was "
                "never consulted, because `route()` did not defer on this set. Stage 2 "
                "is untested by these 300 queries -- author ambiguous cases that force "
                "a defer, or accept that the tie-breaker is unexercised."
            )
        elif delta < 0:
            lines.append(
                f"**The LLM tie-breaker HURTS: {delta:+.1%} on unambiguous cases across "
                f"{hybrid['llm_substitutions']} substitution(s). Delete stage 2** -- it "
                "costs a serving dependency on the critical path and buys negative "
                "accuracy."
            )
        elif delta == 0:
            lines.append(
                f"The LLM tie-breaker changed {hybrid['llm_substitutions']} decision(s) "
                "and moved accuracy not at all. Keep stage 2 only if the deferred cases "
                "are ones the rules should not be asked to answer; otherwise it is a "
                "dependency with no measured benefit."
            )
        else:
            lines.append(
                f"The LLM tie-breaker helps: {delta:+.1%} on unambiguous cases over "
                f"{hybrid['llm_substitutions']} substitution(s). Stage 2 stays."
            )

    if rules and llm:
        gap = rules["task_accuracy"] - llm["task_accuracy"]
        lines.append(
            f"Rules beat the bare LLM by {gap:+.1%} overall "
            f"({llm['unparsable']} unparsable LLM output(s)). "
            + (
                "Deterministic-first is justified."
                if gap > 0
                else "The rules are NOT beating the LLM; re-read the misroute list before "
                "defending the rules-first design."
            )
        )

    return "\n\n".join(lines)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 0 item 11 — router zero-shot, three arms",
        "",
        f"**Date:** {report['date']}  ",
        f"**Set:** {report['dataset']} ({report['n']} queries)  ",
        f"**LLM:** `{report['model'] or 'not run'}`",
        "",
        "| arm | task accuracy | unambiguous | n unambiguous | unparsable |",
        "|---|---|---|---|---|",
    ]
    for name in ARMS:
        arm = report["arms"].get(name)
        if not arm:
            continue
        lines.append(
            f"| {name} | {arm['task_accuracy']:.1%} | {arm['unambiguous_accuracy']:.1%} | "
            f"{arm['unambiguous_n']} | {arm['unparsable']} |"
        )
    lines += ["", "## Verdict", "", report["verdict"]]

    for name in ARMS:
        arm = report["arms"].get(name)
        if not arm or not arm["misroutes"]:
            continue
        lines += ["", f"### {name}: {len(arm['misroutes'])} misrouted unambiguous case(s)", ""]
        for entry in arm["misroutes"][:20]:
            lines.append(
                f"- `[{entry['category']}]` {entry['expected']} -> {entry['predicted']}: "
                f"{entry['query']}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["rules"])
    parser.add_argument("--model", default=None, help="required for the llm and hybrid arms")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="logs/phase0_item11_router.md")
    args = parser.parse_args()

    dataset = Path(args.dataset) if args.dataset else None
    rows = load_dataset(dataset) if dataset else load_dataset()
    print(f"Router set: {len(rows)} queries")

    arms: dict[str, Any] = {}
    print("arm: rules")
    arms["rules"] = run_rules(rows)

    classifier = None
    if "llm" in args.arms or "hybrid" in args.arms:
        if not args.model:
            raise SystemExit(
                "--model is required for the llm and hybrid arms. Without it the "
                "hybrid arm would silently collapse to the rules arm and be reported "
                "as if the tie-breaker had been tested."
            )
        print(f"loading {args.model}")
        classifier = LLMTaskClassifier(args.model)

    if "llm" in args.arms:
        print("arm: llm")
        arms["llm"] = run_llm(rows, classifier, args.batch_size)
    if "hybrid" in args.arms:
        print("arm: hybrid")
        llm_arm = arms.get("llm") or (
            run_llm(rows, classifier, args.batch_size) if classifier else None
        )
        arms["hybrid"] = run_hybrid(rows, arms["rules"], llm_arm)

    report = {
        "run": "phase0_item11_router_zero_shot",
        "date": datetime.now(UTC).isoformat(),
        "dataset": str(dataset or "training/eval/routing_300.jsonl"),
        "n": len(rows),
        "model": args.model,
        "arms": {name: arm for name, arm in arms.items() if name in ("rules", *args.arms)},
    }
    report["verdict"] = _verdict(report)

    # Predictions are bulky and only useful for a re-score; drop them from the
    # published report but keep them beside it.
    predictions = {name: arm.pop("predictions", None) for name, arm in report["arms"].items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(report), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out.with_name(out.stem + "_predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    print("\n" + report["verdict"])
    print(f"\nReport written to {out}")

    rules = report["arms"]["rules"]
    return 0 if rules["unambiguous_accuracy"] >= UNAMBIGUOUS_GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
