"""Score the rules router against the 300-query set (master plan section 6.2).

Reports the routing confusion matrix over the task enum, split by
``router_path``, plus the invalid-config catch rate. The risk register's trigger
is a number this script prints: **hybrid routing below 90% on unambiguous cases
means the rules get fixed now**, in Week 1, not discovered in Week 5.

Run: ``python training/eval/score_routing.py``
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from satquery.agent.router import QueryContext, route
from satquery.agent.validator import validate
from satquery.ingest.band_inventory import BandInventory

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "training" / "eval" / "routing_300.jsonl"

__all__ = ["evaluate", "load_dataset", "context_from_row"]


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def context_from_row(row: dict[str, Any]) -> QueryContext:
    """Rebuild the router's view of one authored case."""
    ctx = row["input_context"]
    inventories = []
    for modality in ctx["modalities"]:
        if modality == "optical":
            bands = {
                name: index + 1
                for index, name in enumerate(ctx["bands"])
                if name in ("blue", "green", "red", "nir", "swir", "pan")
            }
            inventories.append(
                BandInventory(
                    bands=bands,
                    has_swir="swir" in bands,
                    has_nir="nir" in bands,
                    is_pan_only=ctx["bands"] == ["pan"],
                    computable_indices=list(ctx.get("computable_indices", [])),
                )
            )
        else:
            inventories.append(
                BandInventory(
                    bands={p.lower(): i + 1 for i, p in enumerate(ctx.get("polarisations", []))},
                    polarisations=list(ctx.get("polarisations", [])),
                    sar_band=ctx.get("sar_band"),
                    computable_indices=[],
                )
            )
    return QueryContext(
        query_text=row["query_text"],
        modalities=list(ctx["modalities"]),
        inventories=inventories,
        image_count=int(ctx.get("image_count", 1)),
        dates=list(ctx.get("dates", [])),
    )


def evaluate(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else load_dataset()
    confusion: dict[tuple[str, str], int] = Counter()
    by_path: dict[str, Counter] = defaultdict(Counter)
    task_hits = 0
    unambiguous_total = 0
    unambiguous_hits = 0
    refusal_total = 0
    refusal_caught = 0
    refusal_category_correct = 0
    tool_hits = 0
    tool_total = 0
    misroutes: list[dict[str, Any]] = []

    for row in rows:
        truth = row["ground_truth"]
        context = context_from_row(row)
        decision = route(context)
        result = validate(context, decision)

        predicted = decision.task.value
        expected = truth["expected_task"]
        confusion[(expected, predicted)] += 1
        by_path[decision.router_path.value][predicted == expected] += 1
        if predicted == expected:
            task_hits += 1
        elif truth["unambiguous"]:
            misroutes.append(
                {
                    "id": row["id"],
                    "query": row["query_text"],
                    "expected": expected,
                    "predicted": predicted,
                    "category": truth["expected_category"],
                }
            )
        if truth["unambiguous"]:
            unambiguous_total += 1
            unambiguous_hits += int(predicted == expected)

        if truth["expected_category"] == "refusal":
            refusal_total += 1
            if not result.passed:
                refusal_caught += 1
                if result.refusal and result.refusal.category == truth["expected_refusal_category"]:
                    refusal_category_correct += 1
        elif result.passed and truth["expected_tools"]:
            tool_total += 1
            planned = set(decision.tool_names)
            tool_hits += int(set(truth["expected_tools"]).issubset(planned))

    total = len(rows)
    return {
        "n": total,
        "task_accuracy": task_hits / total if total else 0.0,
        "unambiguous_accuracy": unambiguous_hits / unambiguous_total if unambiguous_total else 0.0,
        "unambiguous_n": unambiguous_total,
        "rules_share": sum(by_path["rules"].values()) / total if total else 0.0,
        "rules_accuracy": (
            by_path["rules"][True] / sum(by_path["rules"].values())
            if sum(by_path["rules"].values())
            else 0.0
        ),
        "llm_share": sum(by_path["llm"].values()) / total if total else 0.0,
        "invalid_config_catch_rate": refusal_caught / refusal_total if refusal_total else 1.0,
        "refusal_category_accuracy": (
            refusal_category_correct / refusal_total if refusal_total else 1.0
        ),
        "plan_coverage": tool_hits / tool_total if tool_total else 1.0,
        "confusion": {f"{e}->{p}": c for (e, p), c in sorted(confusion.items())},
        "misroutes": misroutes,
    }


def main() -> int:
    report = evaluate()
    print(f"queries                     {report['n']}")
    print(f"task accuracy (all)         {report['task_accuracy']:.1%}")
    print(
        f"task accuracy (unambiguous) {report['unambiguous_accuracy']:.1%}"
        f"  <- risk-register trigger at 90% (n={report['unambiguous_n']})"
    )
    print(f"routed by rules             {report['rules_share']:.1%}")
    print(f"deferred to LLM tie-break   {report['llm_share']:.1%}")
    print(f"invalid-config catch rate   {report['invalid_config_catch_rate']:.1%}")
    print(f"refusal category accuracy   {report['refusal_category_accuracy']:.1%}")
    print(f"plan coverage               {report['plan_coverage']:.1%}")
    if report["misroutes"]:
        print(f"\n{len(report['misroutes'])} misrouted unambiguous case(s):")
        for entry in report["misroutes"][:20]:
            print(
                f"  [{entry['category']}] {entry['expected']} -> "
                f"{entry['predicted']}: {entry['query']}"
            )
    return 0 if report["unambiguous_accuracy"] >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
