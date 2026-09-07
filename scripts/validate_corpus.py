"""Refuse a training corpus that has any of the failure modes we have already hit.

    python scripts/validate_corpus.py --manifest data/chips/canonical.jsonl \
        --image-root data/chips

Every check here exists because the failure it catches is **invisible in a loss
curve**. A corpus can be entirely wrong and train smoothly to a beautiful
number; these are the specific ways that has happened, or nearly happened, in
this project:

* **Black images.** 21 all-black PNGs were written by a missing-image fallback
  and a loss curve was read off them. Checked by sampling pixels, not by
  trusting that a file exists.
* **Wrong split.** All 21 fetched chips were reBEN *test* patches labelled
  `train`. Training on them and quoting a BEN.txt number inflates it silently.
* **Answer skew.** A presence-question generator that only asks about classes
  that are present produces a corpus that is 100% "yes". The model learns to
  say yes and scores well until it meets a balanced eval set.
* **One image dominating.** A 200-building scene generates hundreds of rows and
  a 2-building scene generates three, so the corpus reflects scene complexity
  rather than the task and the effective dataset is far smaller than the row
  count suggests.
* **Missing question types.** Average accuracy is the mean of *per-type*
  accuracies. A row without a type silently collapses into one bucket and
  changes the headline metric.
* **Missing provenance.** C45 makes the provenance chain a hard gate: a row
  that cannot say where its pixels came from cannot be cleared for training.
* **Duplicate questions.** The same question against the same image, repeated,
  inflates the row count and biases the loss toward whatever it asks.

Exit code is non-zero when any **error** fires. Warnings are printed and do not
block, because some are judgement calls -- but they are printed every time, not
hidden behind a flag.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: A chip whose pixels vary less than this is not imagery. The fabricated PNGs
#: measured exactly 0.0; a real 120x120 Sentinel-2 composite measures ~50-70.
MIN_PIXEL_STD = 1.0

#: Above this share of one answer within a question type, the type is teaching a
#: prior rather than a skill.
MAX_ANSWER_SHARE = 0.80

#: More rows than this from one image SET and the corpus is asking a single
#: scene the same things over and over. Set, not image: a change corpus pairs
#: each mosaic with many others, and every pairing is a different comparison.
#: 16 leaves room for a handful of question types across both orderings while
#: still catching a generator that has run away on one scene.
MAX_ROWS_PER_GROUP = 16

#: Sampling cap for the pixel check -- opening every image in a 500k-row corpus
#: costs more than the check is worth.
DEFAULT_IMAGE_SAMPLE = 200


class Findings:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def load(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{manifest}:{number}: {error}") from error
    if not rows:
        raise SystemExit(f"{manifest} is empty.")
    return rows


def check_schema(rows: list[dict], findings: Findings) -> None:
    required = ("sample_id", "adapter", "task", "images", "question", "answer")
    for row in rows:
        missing = [field for field in required if not row.get(field)]
        if missing:
            findings.error(
                f"{row.get('sample_id', '<no id>')}: missing required field(s) "
                f"{missing}. A row without these cannot be trained on."
            )
            break

    # Either name satisfies this: `question_type` from merge_corpus, `task`
    # from the canonical schema. Demanding the first alone rejected every row a
    # per-adapter generator writes, which is now most of the corpus.
    untyped = [r for r in rows if not (r.get("question_type") or r.get("task"))]
    if untyped:
        findings.error(
            f"{len(untyped)} row(s) carry no question_type. Average accuracy is the "
            "mean of per-type accuracies, so untyped rows collapse into one bucket "
            f"and change the headline metric. First: {untyped[0].get('sample_id')}"
        )

    unsourced = [r for r in rows if not r.get("licence")]
    if unsourced:
        findings.error(
            f"{len(unsourced)} row(s) carry no licence. C45 makes provenance a hard "
            "gate on manifest build; a row that cannot say where its pixels came "
            "from cannot be cleared for training."
        )


def check_duplicates(rows: list[dict], findings: Findings) -> None:
    seen: Counter = Counter()
    for row in rows:
        seen[(tuple(row.get("images", [])), row.get("question", ""))] += 1
    repeated = {k: v for k, v in seen.items() if v > 1}
    if repeated:
        worst = max(repeated.values())
        findings.error(
            f"{len(repeated)} question(s) are asked more than once against the same "
            f"image (worst repeats {worst}x). Duplicates inflate the row count and "
            "bias the loss toward whatever they ask."
        )

    ids = Counter(r.get("sample_id") for r in rows)
    clashes = [k for k, v in ids.items() if v > 1]
    if clashes:
        findings.error(
            f"{len(clashes)} duplicate sample_id(s), e.g. {clashes[0]!r}. Resuming a "
            "run or joining a prediction dump keys off this."
        )


def check_twins(rows: list[dict], findings: Findings) -> dict[str, Any]:
    """Complementary pairs: both halves present, in one split, with a real split.

    A twin is the mechanism that stops a question's text from predicting its
    answer. It only works whole. A half whose partner was dropped by a filter,
    a cap or a merge is an ordinary unbalanced row wearing a `twin_id`, and it
    is precisely the row a language prior wins -- so the loss is silent unless
    something looks for it.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("twin_id"):
            groups[row["twin_id"]].append(row)
    if not groups:
        return {"twins": 0}

    straddling = [k for k, g in groups.items() if len({r.get("split") for r in g}) > 1]
    if straddling:
        findings.error(
            f"{len(straddling)} twin(s) straddle the train/val boundary, e.g. "
            f"{straddling[0]!r}. One half in each split means the validation row's "
            "answer is inferable from its training partner, and the val score is "
            "no longer measuring generalisation."
        )

    # A same-answer pair carries no balancing information: it is two rows
    # agreeing, not a counterexample. Structural singletons are legitimate --
    # some questions only apply in one direction -- so this counts rather than
    # errors, and the number is what tells you whether the design held.
    paired = {k: g for k, g in groups.items() if len(g) > 1}
    flipped = sum(1 for g in paired.values() if len({str(r["answer"]) for r in g}) > 1)
    singles = len(groups) - len(paired)
    if paired and flipped / len(paired) < 0.5:
        findings.warn(
            f"only {flipped}/{len(paired)} complete twin(s) carry opposing "
            "answers. The pairing is present but not doing its job; check the "
            "generator is emitting both orderings."
        )
    return {
        "twins": len(groups),
        "complete": len(paired),
        "opposing": flipped,
        "single": singles,
    }


def check_balance(rows: list[dict], findings: Findings) -> dict[str, Any]:
    by_type: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        # `question_type` is what merge_corpus writes; `task` is the canonical
        # schema field the per-adapter generators write. Reading only the first
        # filed every change_vqa row under "?" and turned the balance check --
        # the one check that would catch a corpus a blind model can win -- into
        # a single meaningless bucket.
        kind = row.get("question_type") or row.get("task") or "?"
        by_type[kind][str(row.get("answer", "")).lower()] += 1

    summary = {}
    for question_type, answers in sorted(by_type.items()):
        total = sum(answers.values())
        answer, count = answers.most_common(1)[0]
        share = count / total
        summary[question_type] = {
            "n": total,
            "distinct_answers": len(answers),
            "most_common": answer,
            "share": round(share, 3),
        }
        if len(answers) == 1 and total > 1:
            findings.error(
                f"question_type '{question_type}': every one of {total} rows has the "
                f"same answer ({answer!r}). This teaches a constant, not a skill."
            )
        elif share > MAX_ANSWER_SHARE and total >= 20:
            findings.warn(
                f"question_type '{question_type}': {share:.0%} of {total} rows answer "
                f"{answer!r}. Above {MAX_ANSWER_SHARE:.0%} the type is teaching a prior."
            )
    return summary


def check_image_load(rows: list[dict], findings: Findings) -> dict[str, Any]:
    """Reuse, counted per image *set* rather than per image.

    For a single-image corpus these are the same thing and one image is one
    scene, so 100 rows off it means 100 near-identical samples. For a
    multi-image adapter they are not: a SpaceNet 7 mosaic takes part in up to
    23 different date pairings, and each pairing is a genuinely distinct visual
    comparison. Counting per image reported one mosaic in 188 rows and read as
    heavy duplication; counted per pair the same corpus asks a median of 4
    questions per comparison, which is ordinary VQA density.

    So the warning fires on the unit the model actually sees -- the combination
    of images -- and the per-image figure is reported without a threshold, as
    context rather than as a finding.
    """
    per_image: Counter = Counter()
    per_group: Counter = Counter()
    for row in rows:
        images = row.get("images", [])
        for image in images:
            per_image[image] += 1
        # Order-insensitive: (A, B) and (B, A) are the same comparison shown
        # two ways, and the whole complementary-pair design depends on both
        # being present, so they must not read as duplication.
        per_group[frozenset(images)] += 1

    if per_group:
        worst_group, worst_count = per_group.most_common(1)[0]
        if worst_count > MAX_ROWS_PER_GROUP:
            findings.warn(
                f"one image set carries {worst_count} rows (cap "
                f"{MAX_ROWS_PER_GROUP}): {sorted(worst_group)[:1]}. Past this the "
                "corpus is asking one scene the same things repeatedly, and its "
                "effective size is well below its row count."
            )
    return {
        "distinct_images": len(per_image),
        "distinct_image_sets": len(per_group),
        "rows_per_image_max": max(per_image.values()) if per_image else 0,
        "rows_per_set_max": max(per_group.values()) if per_group else 0,
    }


def check_pixels(
    rows: list[dict], root: Path, findings: Findings, limit: int
) -> dict[str, Any]:
    """Open a sample of images and confirm they contain actual imagery."""
    import numpy as np
    from PIL import Image

    images = []
    for row in rows:
        for image in row.get("images", []):
            images.append(image)
    unique = sorted(set(images))
    step = max(1, len(unique) // limit)
    checked = unique[::step][:limit]

    missing, black = [], []
    for relative in checked:
        path = Path(relative)
        path = path if path.is_absolute() else root / path
        if not path.exists():
            missing.append(relative)
            continue
        try:
            array = np.asarray(Image.open(path).convert("RGB"))
        except Exception as error:  # noqa: BLE001 - unreadable is a finding
            findings.error(f"{relative}: cannot be opened ({error!r})")
            continue
        if float(array.std()) < MIN_PIXEL_STD:
            black.append(relative)

    if missing:
        findings.error(
            f"{len(missing)} of {len(checked)} sampled image(s) do not exist, e.g. "
            f"{missing[0]}. The manifest and the staged imagery have drifted."
        )
    if black:
        findings.error(
            f"{len(black)} of {len(checked)} sampled image(s) are blank "
            f"(pixel std < {MIN_PIXEL_STD}), e.g. {black[0]}. This project has "
            "already read a loss curve off 21 all-black PNGs."
        )
    return {"images_checked": len(checked), "missing": len(missing), "blank": len(black)}


def check_splits(rows: list[dict], findings: Findings) -> dict[str, int]:
    splits = Counter(r.get("split", "") for r in rows)
    if "" in splits:
        findings.error(f"{splits['']} row(s) have no split.")
    return dict(splits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--sample-images", type=int, default=DEFAULT_IMAGE_SAMPLE)
    parser.add_argument("--skip-pixels", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    root = Path(args.image_root) if args.image_root else manifest.parent
    rows = load(manifest)
    findings = Findings()

    check_schema(rows, findings)
    check_duplicates(rows, findings)
    twins = check_twins(rows, findings)
    balance = check_balance(rows, findings)
    images = check_image_load(rows, findings)
    splits = check_splits(rows, findings)
    pixels = (
        {"skipped": True}
        if args.skip_pixels
        else check_pixels(rows, root, findings, args.sample_images)
    )

    adapters = Counter(r.get("adapter", "") for r in rows)
    sources = Counter(r.get("source", "") for r in rows)

    print(f"rows        : {len(rows)}")
    print(f"adapters    : {dict(adapters)}")
    print(f"sources     : {dict(sources)}")
    print(f"splits      : {splits}")
    print(f"images      : {images}")
    print(f"pixels      : {pixels}")
    if twins.get("twins"):
        print(f"twins       : {twins}")
    print("question types:")
    for name, info in balance.items():
        print(
            f"  {name:22} n={info['n']:7}  distinct={info['distinct_answers']:4}  "
            f"most_common={info['most_common']!r} @ {info['share']:.0%}"
        )

    for message in findings.warnings:
        print(f"\nWARN  {message}")
    for message in findings.errors:
        print(f"\nERROR {message}")

    report = {
        "manifest": str(manifest),
        "rows": len(rows),
        "adapters": dict(adapters),
        "sources": dict(sources),
        "splits": splits,
        "images": images,
        "pixels": pixels,
        "twins": twins,
        "question_types": balance,
        "errors": findings.errors,
        "warnings": findings.warnings,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if findings.errors:
        print(f"\nFAILED with {len(findings.errors)} error(s).")
        return 1
    print(f"\nOK ({len(findings.warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
