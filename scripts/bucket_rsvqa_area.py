"""Quantise RSVQA area answers into the dataset's own five classes.

    python scripts/bucket_rsvqa_area.py --manifest /data/eval/rsvqa_hr/rsvqa_hr_train.jsonl

RSVQA's area answers are stored as exact integers (``10631m2``) but the paper
scores them as a five-way classification: ``0m2``, ``between 1m2 and 10m2``,
``between 11m2 and 100m2``, ``between 101m2 and 1000m2``, ``more than 1000m2``
(Lobry et al., 2020).

Comparing the raw strings is therefore not a stricter version of the benchmark,
it is a different and unwinnable one: 167 distinct values across 518 HR test
rows, most appearing once. A model that understands the image perfectly still
scores near zero, because it cannot guess ``10631`` rather than ``10630``.

Bucketing is what makes the type learnable *and* what makes our number
comparable to a published one. It has to happen on train and test alike -- a
model trained on exact values and scored on buckets would be measured on a
format it was never shown.

This also absorbs the type's known data defects. 66% of HR area answers are
``0m2`` where OpenStreetMap simply has nothing mapped, and a further 8% exceed
the tile's own 6,088 m2 ground extent. Under exact match both are noise; under
the official classes the first is a legitimate and visually decidable answer,
and the second usually still lands in ``more than 1000m2``.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

#: The paper's classes, in order. The upper bound is exclusive of the next
#: class's floor, and ``0`` is its own class rather than the bottom of the
#: first interval -- "nothing here" is a different answer from "a little".
BUCKETS = (
    (0, "0m2"),
    (10, "between 1m2 and 10m2"),
    (100, "between 11m2 and 100m2"),
    (1000, "between 101m2 and 1000m2"),
    (None, "more than 1000m2"),
)

_AREA = re.compile(r"^\s*(\d+)\s*m2\s*$", re.I)

#: RSVQA-LR's counting classes, from the same paragraph of the paper that
#: defines the area classes: "numerical answers are quantized into the
#: following categories: '0'; 'between 1 and 10'; 'between 11 and 100';
#: 'between 101 and 1000'; 'more than 1000'."
#:
#: RSVQA-**HR** counts are deliberately NOT quantised -- the paper keeps them
#: exact because the maximum count there is 89 -- so this must never be applied
#: to the HR split. Quantising those would raise our score against a task
#: nobody set.
COUNT_BUCKETS = (
    (0, "0"),
    (10, "between 1 and 10"),
    (100, "between 11 and 100"),
    (1000, "between 101 and 1000"),
    (None, "more than 1000"),
)

_COUNT = re.compile(r"^\s*(\d[\d,]*)\s*$")


def bucket(answer: str) -> str | None:
    """The class for one answer, or None when it is not an area value at all.

    Idempotent: an answer that is already one of the classes is returned
    unchanged, so re-running over a partly-converted manifest is safe rather
    than reporting the converted rows as unparsable.
    """
    text = str(answer).strip()
    if text in {label for _, label in BUCKETS}:
        return text
    match = _AREA.match(text)
    if not match:
        return None
    value = int(match.group(1))
    for ceiling, label in BUCKETS:
        if ceiling is None or value <= ceiling:
            return label
    return BUCKETS[-1][1]


def bucket_count(answer: str) -> str | None:
    """The LR counting class for one raw answer, or None if not a count."""
    text = str(answer).strip()
    if text in {label for _, label in COUNT_BUCKETS}:
        return text
    match = _COUNT.match(text)
    if not match:
        return None
    value = int(match.group(1).replace(",", ""))
    for ceiling, label in COUNT_BUCKETS:
        if ceiling is None or value <= ceiling:
            return label
    return COUNT_BUCKETS[-1][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--question-type",
        default="area",
        help="only rows of this question_type are rewritten",
    )
    parser.add_argument(
        "--out",
        default="",
        help="defaults to rewriting the manifest in place",
    )
    args = parser.parse_args()

    path = Path(args.manifest)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    changed = 0
    unparsed: Counter = Counter()
    after: Counter = Counter()
    for row in rows:
        if row.get("question_type") != args.question_type:
            continue
        label = (
            bucket_count(row["answer"])
            if args.question_type == "count"
            else bucket(row["answer"])
        )
        if label is None:
            unparsed[str(row["answer"])] += 1
            continue
        if label != row["answer"]:
            changed += 1
        row["answer"] = label
        after[label] += 1

    out = Path(args.out or args.manifest)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    total = sum(after.values())

    # An artifact, not just stdout: the caller treats a job that wrote nothing
    # as a job whose result cannot be trusted, and this is also the record of
    # what the rewrite actually did to the corpus.
    summary = {
        "manifest": str(path),
        "question_type": args.question_type,
        "rows_rewritten": changed,
        "rows_of_type": total,
        "classes": dict(after),
        "unparsed": dict(unparsed),
    }
    report = Path("logs/bucket_area.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{path.name}: {changed} of {total} {args.question_type} answer(s) rewritten")
    for label, count in sorted(after.items(), key=lambda kv: -kv[1]):
        print(f"  {label:28} {count:5}  {100 * count / max(1, total):5.1f}%")
    if unparsed:
        # Loud, not silent: an answer that does not parse is one the scorer will
        # mark wrong for a formatting reason, which is the failure this script
        # exists to remove.
        print(f"  UNPARSED: {len(unparsed)} distinct, e.g. {list(unparsed)[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
