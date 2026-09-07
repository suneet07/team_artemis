"""Combine staged sources into one adapter corpus, with a quota per source.

    python scripts/merge_corpus.py --adapter rs_vqa \
        --source ben=data/ben_canonical.jsonl:chips:40000 \
        --source rsvqa_lr=data/eval/rsvqa_lr/rsvqa_lr_train.jsonl:eval/rsvqa_lr:10000 \
        --source rsvqa_hr=data/eval/rsvqa_hr/rsvqa_hr_train.jsonl:eval/rsvqa_hr:24000 \
        --out data/manifests/rs_vqa_train.jsonl

Each ``--source`` is ``name=path:prefix:quota``.

**Why a prefix.** Every staged source writes image paths relative to its own
directory, and ``RealChipDataset`` resolves them against a single root. Merged
without rewriting, the loader would look for ``Images_LR/46.tif`` beside the BEN
composites and fail on the first batch. The prefix is where that source lives
under the shared root (``/data`` on the Volume), so the merged manifest is
loadable with one ``--image-root``.

**Why a quota per source rather than "use everything".** The mix is a decision.
Concretely for ``rs_vqa``: BEN.txt supplies task variety across 10 countries and
12 question types but is 10 m Sentinel-2, while RSVQA-HR is the **only**
sub-metre imagery in the project and the hidden set is sub-metre Cartosat. Left
unweighted, the corpus would be 84% 10 m data and the adapter would never see
the resolution it is graded on.

Sampling inside a source is stratified by question type and seeded, so the same
command produces the same corpus on every teammate's machine -- four adapters
trained on four different samples cannot be compared.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_source(spec: str) -> tuple[str, Path, str, int]:
    name, _, rest = spec.partition("=")
    path, _, tail = rest.partition(":")
    prefix, _, quota = tail.partition(":")
    if not (name and path and quota):
        raise SystemExit(
            f"--source wants name=path:prefix:quota, got {spec!r}. The prefix is "
            "where that source sits under the shared image root."
        )
    return name, Path(path), prefix.strip("/"), int(quota)


def take(rows: list[dict], quota: int, seed: str) -> list[dict]:
    """Quota rows, spread evenly across question types."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type", "other")].append(row)

    per_type = max(1, quota // max(1, len(by_type)))
    picked: list[dict] = []
    for question_type, group in sorted(by_type.items()):
        rng = random.Random(f"{seed}:{question_type}")
        rng.shuffle(group)
        picked.extend(group[:per_type])

    # Round out to the quota from whatever is left, rather than returning short
    # because the types divided unevenly.
    if len(picked) < quota:
        chosen = {id(r) for r in picked}
        spare = [r for r in rows if id(r) not in chosen]
        random.Random(f"{seed}:spare").shuffle(spare)
        picked.extend(spare[: quota - len(picked)])
    return picked[:quota]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SOURCE:TYPE",
        help="drop one question type from one source, e.g. rsvqa_hr:area. The "
        "rows stay in the source manifest; only the training corpus omits them, "
        "so the exclusion is reversible and visible in this command.",
    )
    parser.add_argument(
        "--exclude-matching",
        action="append",
        default=[],
        metavar="SOURCE:REGEX",
        help="drop rows from one source whose question matches a pattern. Needed "
        "where the thing to exclude is not a question type: RSVQA's size words "
        "live inside presence/count/comp questions, and LR and HR define them "
        "30x apart (small = under 3,000 m2 in LR, under 100 m2 in HR).",
    )
    args = parser.parse_args()

    merged: list[dict] = []
    report = {}
    excluded: dict[str, set[str]] = defaultdict(set)
    for rule in args.exclude:
        source_name, _, question_type = rule.partition(":")
        if not question_type:
            raise SystemExit(f"--exclude wants SOURCE:TYPE, got {rule!r}")
        excluded[source_name].add(question_type)

    patterns: dict[str, list] = defaultdict(list)
    for rule in args.exclude_matching:
        source_name, _, expression = rule.partition(":")
        if not expression:
            raise SystemExit(f"--exclude-matching wants SOURCE:REGEX, got {rule!r}")
        patterns[source_name].append(re.compile(expression, re.I))

    for spec in args.source:
        name, path, prefix, quota = parse_source(spec)
        if not path.exists():
            raise SystemExit(f"{path} does not exist; stage {name} first")

        rows = [
            row
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
            if row.get("adapter") == args.adapter and row.get("split") == args.split
        ]
        if not rows:
            raise SystemExit(
                f"{path} has no rows for adapter={args.adapter!r} split={args.split!r}. "
                "A source contributing nothing is a staging mistake, not an empty "
                "quota."
            )
        dropped = 0
        if excluded.get(name):
            before = len(rows)
            rows = [r for r in rows if r.get("question_type") not in excluded[name]]
            dropped = before - len(rows)
            print(
                f"  {name:12} excluded {dropped} row(s) of type "
                f"{sorted(excluded[name])}"
            )
        if patterns.get(name):
            before = len(rows)
            rows = [
                r
                for r in rows
                if not any(p.search(r["question"]) for p in patterns[name])
            ]
            matched = before - len(rows)
            dropped += matched
            print(
                f"  {name:12} excluded {matched} row(s) matching "
                f"{[p.pattern for p in patterns[name]]}"
            )
        picked = take(rows, quota, seed=f"merge:{args.adapter}:{name}")

        for row in picked:
            row["images"] = [f"{prefix}/{image}" for image in row["images"]]
            row["sample_id"] = f"{name}_{row['sample_id']}"
            row["corpus_source"] = name
        merged.extend(picked)
        report[name] = {
            "available": len(rows),
            "taken": len(picked),
            "types": len({r.get("question_type") for r in picked}),
            "excluded_types": sorted(excluded.get(name, ())),
            "excluded_rows": dropped,
        }
        print(f"  {name:12} {len(rows):7} available -> {len(picked):6} taken")

    random.Random(f"merge:{args.adapter}").shuffle(merged)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row) + "\n")

    gsds = Counter(round(g, 2) for r in merged for g in r.get("effective_gsd_m", []))

    # A summary file, not just stdout: a run on Modal has to leave an artifact
    # behind or there is nothing to report and nothing to check afterwards.
    summary = {
        "adapter": args.adapter,
        "rows": len(merged),
        "sources": report,
        "resolutions_m": {str(k): v for k, v in gsds.items()},
        "distinct_image_sets": len({tuple(r["images"]) for r in merged}),
        "question_types": len({r.get("question_type") for r in merged}),
    }
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n{len(merged)} row(s) -> {out}")
    print(f"sources     : { {k: v['taken'] for k, v in report.items()} }")
    print(f"resolutions : {dict(gsds)}")
    print(f"images      : {len({tuple(r['images']) for r in merged})}")
    print(f"types       : {len({r.get('question_type') for r in merged})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
