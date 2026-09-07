"""Cut a scoreable benchmark file out of a training manifest's held-out split.

    python scripts/make_eval_split.py --manifest /data/manifests/change_vqa_train.jsonl \
        --split val --out /data/eval/change_vqa_val.jsonl

The evaluator reads ``question_type``; the per-adapter generators write
``task``. They are the same thing under two names -- ``merge_corpus`` uses the
first, the canonical schema uses the second -- and every consumer written
against one has needed teaching about the other. Rather than patch the scorer,
this writes both onto each row, so the benchmark file satisfies the evaluator
without the generator having to know it exists.

Held-out only. Scoring a model on rows it trained on measures memorisation and
reports it as accuracy, which is the one number that must never be wrong.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--per-type",
        type=int,
        default=0,
        help="cap rows per question type; 0 = all. A cap makes the run cheap "
        "without letting one large type dominate the average",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    held = [r for r in rows if r.get("split") == args.split]
    if not held:
        raise SystemExit(
            f"no rows with split={args.split!r} in {args.manifest}. Scoring the "
            "training split would measure memorisation and call it accuracy."
        )

    import random

    rng = random.Random(args.seed)
    rng.shuffle(held)
    if args.per_type:
        seen: Counter = Counter()
        capped = []
        for row in held:
            kind = row.get("question_type") or row.get("task")
            if seen[kind] >= args.per_type:
                continue
            seen[kind] += 1
            capped.append(row)
        held = capped

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in held:
            # Both names on every row: the evaluator filters and scores on
            # `question_type`, the schema and the generators use `task`.
            row = dict(row)
            row.setdefault("question_type", row.get("task"))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = Counter(r.get("question_type") or r.get("task") for r in held)
    print(f"{len(held)} row(s) -> {out}")
    for kind, count in sorted(by_type.items()):
        answers = Counter(str(r["answer"]) for r in held if
                          (r.get("question_type") or r.get("task")) == kind)
        _, top = answers.most_common(1)[0]
        print(
            f"  {kind:20} {count:6}  {len(answers):3} answers  "
            f"majority {100 * top / count:5.1f}%"
        )
    summary = {
        "manifest": args.manifest,
        "split": args.split,
        "rows": len(held),
        "by_type": dict(by_type),
    }
    Path("logs/eval_split_summary.json").parent.mkdir(parents=True, exist_ok=True)
    Path("logs/eval_split_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
