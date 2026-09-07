"""Re-derive every answer from the labels and check the manifest agrees.

    python scripts/verify_change_rows.py --manifest /data/manifests/change_vqa_train.jsonl \
        --root /data/eval/sn7 --sample 2000

The corpus checker proves a manifest is internally consistent: fields present,
images loadable, answers balanced. It cannot prove the answer on a row is the
answer for *those two images in that order*, and that is the only property that
matters -- a corpus where every question is well-formed and every answer belongs
to a different pair trains perfectly and teaches nothing.

So this ignores the manifest's answer entirely, reads the two mosaics' month
codes out of the image filenames, opens the matching label files, recomputes
the answer with the same functions the generator used, and compares. Three
distinct failures are separable this way:

* **order** -- the images are the right two but presented the wrong way round,
  which would invert every directional answer while looking perfectly valid;
* **pairing** -- the answer belongs to a different date pair;
* **drift** -- the generator and this checker disagree, which means one of them
  changed and the corpus is stale.

A disagreement here is a hard stop. Everything else is a warning about quality;
this is a claim that the data is wrong.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_change import buildings_in, questions_for  # noqa: E402

#: The month a SpaceNet 7 filename encodes: global_monthly_2019_09_mosaic_<aoi>
_MONTH = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic")


def month_of(name: str) -> str:
    match = _MONTH.search(name)
    if not match:
        raise ValueError(f"no month in filename {name!r}")
    return f"{match.group(1)}-{match.group(2)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default="/data/eval/sn7")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--out", default="logs/row_verification.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # Stratify by task: a uniform sample of a corpus that is 29% counting would
    # barely touch change_direction, which has 594 rows in total.
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    per_task = max(1, args.sample // max(1, len(by_task)))
    checked = [r for group in by_task.values() for r in group[:per_task]]

    root = Path(args.root)
    agree = 0
    mismatches: list[dict] = []
    order_errors: list[dict] = []
    missing: list[str] = []
    by_task_counts: Counter = Counter()

    for row in checked:
        images = [Path(p) for p in row["images"]]
        try:
            months = [month_of(p.name) for p in images]
        except ValueError as error:
            missing.append(f"{row['sample_id']}: {error}")
            continue

        # The manifest's own claim about presentation order, checked against the
        # filenames rather than trusted.
        if months != row.get("months"):
            order_errors.append(
                {
                    "sample_id": row["sample_id"],
                    "months_field": row.get("months"),
                    "months_from_filenames": months,
                }
            )
            continue

        aoi = row["aoi"]
        labels = {}
        for month in months:
            candidates = list(
                (root / aoi / "labels").glob(
                    f"global_monthly_{month.replace('-', '_')}_mosaic_*"
                )
            )
            if not candidates:
                missing.append(f"{row['sample_id']}: no label file for {month}")
                break
            labels[month] = buildings_in(candidates[0])
        if len(labels) != 2:
            continue

        # Recomputed with the generator's own functions, in the manifest's
        # presented order. Matching on question text, because one ordering
        # produces several questions and only the identical one is comparable.
        recomputed = {
            item["question"]: item["answer"]
            for item in questions_for(
                labels[months[0]], labels[months[1]], months[0], months[1]
            )
        }
        expected = recomputed.get(row["question"])
        by_task_counts[row["task"]] += 1
        if expected is None:
            mismatches.append(
                {
                    "sample_id": row["sample_id"],
                    "reason": "question not produced for this pair in this order",
                    "question": row["question"],
                }
            )
        elif str(expected) != str(row["answer"]):
            mismatches.append(
                {
                    "sample_id": row["sample_id"],
                    "task": row["task"],
                    "question": row["question"],
                    "manifest_answer": row["answer"],
                    "recomputed_answer": expected,
                    "months": months,
                }
            )
        else:
            agree += 1

    summary = {
        "checked": len(checked),
        "verified": agree,
        "mismatched": len(mismatches),
        "order_errors": len(order_errors),
        "unresolved": len(missing),
        "by_task": dict(by_task_counts),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "summary": summary,
                "mismatches": mismatches[:50],
                "order_errors": order_errors[:50],
                "unresolved": missing[:50],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))
    for item in mismatches[:5]:
        print(f"  MISMATCH {item}")
    for item in order_errors[:5]:
        print(f"  ORDER    {item}")

    if mismatches or order_errors:
        raise SystemExit(
            f"{len(mismatches)} answer(s) and {len(order_errors)} ordering(s) "
            "disagree with the labels. Training on this would teach the wrong "
            "answer to a correctly-formed question, which no loss curve shows."
        )
    print("\nEvery sampled row's answer matches its own imagery, in its own order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
