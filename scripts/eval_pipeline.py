"""Score the measurement pipeline on the same benchmark the VLM was scored on.

    python scripts/eval_pipeline.py --benchmark /data/eval/change_vqa_val.jsonl \
        --sn7-root /data/eval/sn7 --out logs/pipeline_eval.json

Same 2,012 held-out rows, same answer contracts, same per-type accuracy. The
adapter scored **AA 43.5%** on this file; anything reported here is directly
comparable because nothing about the questions or the scoring changed -- only
what produces the answer.

**Perception is a parameter, deliberately.** With ``--source truth`` the
building sets come from SpaceNet 7's own footprints, so the number is the
*ceiling*: what this architecture achieves if detection were perfect. Any gap
from 100% there is the router or the arithmetic being wrong, and is our bug.
With a detector plugged in instead, the drop from that ceiling is exactly what
the detector costs. The VLM offers no such decomposition -- one number, and no
way to tell which half is failing.

**Abstentions are reported, never guessed.** A router that cannot classify a
question returns None and the row is counted as unanswered rather than wrong,
because in the real system it would fall through to the VLM. Scoring an
abstention as a failure would understate the pipeline; silently guessing would
overstate it.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.pipeline.change_pipeline import answer, route  # noqa: E402

#: The month a SpaceNet 7 filename encodes.
_MONTH = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic")


def month_of(name: str) -> str:
    match = _MONTH.search(name)
    return f"{match.group(1)}-{match.group(2)}" if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--sn7-root", default="/data/eval/sn7")
    parser.add_argument(
        "--source",
        default="truth",
        choices=["truth"],
        help="where building sets come from. 'truth' uses SpaceNet 7's own "
        "footprints and so reports the architecture's ceiling",
    )
    parser.add_argument("--out", default="logs/pipeline_eval.json")
    args = parser.parse_args()

    from scripts.gen_change import buildings_in

    rows = [
        json.loads(line)
        for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = Path(args.sn7_root)

    # One read per label file, not per row: the benchmark asks several questions
    # of the same pair, and re-parsing 11M polygons for each would dominate.
    cache: dict[Path, dict] = {}

    def load(aoi: str, month: str):
        matches = list(
            (root / aoi / "labels").glob(
                f"global_monthly_{month.replace('-', '_')}_mosaic_*"
            )
        )
        if not matches:
            return None
        path = matches[0]
        if path not in cache:
            cache[path] = buildings_in(path)
        return cache[path]

    per_type: dict[str, Counter] = {}
    unrouted: Counter = Counter()
    abstained: Counter = Counter()
    missing = 0

    for row in rows:
        kind = row.get("question_type") or row.get("task")
        bucket = per_type.setdefault(kind, Counter())
        operation = route(row["question"])
        if operation is None:
            unrouted[kind] += 1
            bucket["unrouted"] += 1
            continue

        months = [month_of(Path(p).name) for p in row["images"]]
        if not all(months):
            missing += 1
            continue
        before, after = (load(row["aoi"], m) for m in months)
        if before is None or after is None:
            missing += 1
            continue

        predicted = answer(operation, before, after)
        if predicted is None:
            abstained[kind] += 1
            bucket["abstained"] += 1
            continue
        bucket["answered"] += 1
        bucket["correct"] += str(predicted) == str(row["answer"])

    summary = {}
    for kind, bucket in sorted(per_type.items()):
        total = sum(bucket[k] for k in ("answered", "abstained", "unrouted"))
        answered = bucket["answered"]
        summary[kind] = {
            "rows": total,
            "answered": answered,
            "abstained": bucket["abstained"],
            "unrouted": bucket["unrouted"],
            # Accuracy over rows the pipeline actually answered, and coverage
            # separately. One number hiding both would let a pipeline that
            # answers three easy rows perfectly look better than one that
            # answers everything well.
            "accuracy_answered": round(answered and bucket["correct"] / answered or 0.0, 4),
            "accuracy_all_rows": round(total and bucket["correct"] / total or 0.0, 4),
        }

    covered = [v for v in summary.values() if v["answered"]]
    report = {
        "benchmark": args.benchmark,
        "source": args.source,
        "rows": len(rows),
        "missing_labels": missing,
        "average_accuracy_answered": round(
            sum(v["accuracy_answered"] for v in covered) / max(1, len(covered)), 4
        ),
        "average_accuracy_all_rows": round(
            sum(v["accuracy_all_rows"] for v in summary.values()) / max(1, len(summary)), 4
        ),
        "per_type": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{len(rows)} row(s), {missing} unresolvable label(s)\n")
    print(f"{'type':22}{'rows':>6}{'answered':>10}{'acc(ans)':>10}{'acc(all)':>10}")
    for kind, v in summary.items():
        print(
            f"{kind:22}{v['rows']:6}{v['answered']:10}"
            f"{100 * v['accuracy_answered']:9.1f}%{100 * v['accuracy_all_rows']:9.1f}%"
        )
    print(
        f"\nAA over answered rows : {100 * report['average_accuracy_answered']:.1f}%"
        f"\nAA over all rows      : {100 * report['average_accuracy_all_rows']:.1f}%"
        f"\n(the adapter scored 43.5% on this same file)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
