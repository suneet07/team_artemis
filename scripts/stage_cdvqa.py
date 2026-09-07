"""Turn CDVQA's published question/answer JSON into change_vqa training rows.

    python scripts/stage_cdvqa.py build \
        --second /data/eval/second \
        --split Train \
        --out /data/manifests/change_vqa_cdvqa_train.jsonl

**Why this exists.** The organisers nominated CDVQA as the change-VQA evaluation
and our adapter has never seen anything like it. ``change_vqa`` was trained
entirely on SpaceNet 7 / MUDS -- 4 m imagery, building appearance and
disappearance, answers about counts and compass directions. CDVQA asks about six
land-cover classes at 0.5 m over 19 answer categories. The vocabularies barely
overlap, so a zero-shot score would measure the mismatch rather than the model.

**Licence.** CDVQA is **Apache-2.0**, from ``YZHJessica/CDVQA``, which is where
this script downloads from.

**The evaluation split is untouched.** ``stage_benchmarks.py`` holds CDVQA as
``eval_only`` and ``tests/test_phase0_harnesses.py`` asserts it, for a
methodological reason rather than a legal one: a resampled eval set is no longer
comparable to published numbers. This writes a *separate* train manifest and
changes no spec, so that assertion keeps holding and Test/Test2 stay pristine.

**Three files, joined by id.** CDVQA publishes ``*_questions.json``,
``*_answers.json`` and ``*_images.json`` separately -- questions carry
``img_id`` and ``answers_ids``, images carry ``file_name``. The generic reader in
``stage_benchmarks.py`` reads one file and cannot join them, which is why this is
its own script rather than another ``--dataset``.
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
SOURCE = "CDVQA (Yuan et al., TGRS 2022)"

#: Written into the ``licence`` field of every row this script emits, so it is a
#: factual record rather than commentary -- keep it accurate.
LICENCE = "Apache-2.0"
PROVENANCE = [
    "CDVQA, Apache-2.0, github.com/YZHJessica/CDVQA",
    "organiser-nominated evaluation benchmark for change-based VQA",
]

#: CDVQA's own question types, kept verbatim rather than mapped onto SpaceNet 7's.
#: Average accuracy is the mean of per-type accuracies, so renaming types to look
#: like ours would silently merge two different taxonomies into one bucket and
#: change the headline number.
CLOSED_ANSWERS = {"change_or_not": ["yes", "no"]}

#: CDVQA publishes four splits. ``Test`` and ``Test2`` are the ones the paper
#: reports on, so a number scored against them is comparable to published work
#: in a way a ``Val`` number is not. They stay eval-only: the manifests carry a
#: split name that is neither ``train`` nor ``val``, and the licence blocklist
#: test asserts no training manifest ever contains them.
_SPLIT_NAMES = {
    "Train": "train",
    "Val": "val",
    "Test": "test",
    "Test2": "test2",
}


def fetch(split: str, kind: str, cache: Path) -> dict:
    """One CDVQA JSON, from the cache if present."""
    name = f"{split}_{kind}.json"
    local = cache / name
    if not local.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"fetching {name}", flush=True)
        urllib.request.urlretrieve(f"{RAW}/{name}", local)  # noqa: S310
    return json.loads(local.read_text(encoding="utf-8"))


def index_pairs(image_root: Path) -> dict[str, tuple[Path, Path]]:
    """Map each image file name to a pair of images that actually decode.

    Two complications, both learned the hard way. The archive nests ``im1``/
    ``im2`` under release directories whose names differ between mirrors, so the
    depth cannot be assumed. And the same name now exists more than once: p7zip
    could not read the RAR5 leg and left 9,871 damaged files in place, then
    ``unar`` extracted good copies into a separate tree beside them. A dict keyed
    on file name would let whichever path ``rglob`` yielded last win, which is a
    coin flip between a valid image and a truncated one.

    So candidates are collected per name and the first that PIL can decode is
    kept. A file count proved nothing here -- the damaged files occupied
    directory entries and only failed at DataLoader time, three commands later.
    """
    from PIL import Image

    def readable(path: Path) -> bool:
        try:
            with Image.open(path) as handle:
                handle.verify()
            return True
        except Exception:  # noqa: BLE001 - any decode failure disqualifies it
            return False

    candidates: dict[str, dict[str, list[Path]]] = {}
    for path in image_root.rglob("*.png"):
        side = path.parent.name.lower()
        if side not in ("im1", "im2"):
            continue
        candidates.setdefault(path.name, {}).setdefault(side, []).append(path)

    pairs: dict[str, tuple[Path, Path]] = {}
    rejected = 0
    for name, sides in candidates.items():
        first = next((p for p in sides.get("im1", []) if readable(p)), None)
        second = next((p for p in sides.get("im2", []) if readable(p)), None)
        if first is None or second is None:
            rejected += 1
            continue
        pairs[name] = (first, second)

    print(
        f"imagery: {len(candidates)} distinct name(s), {len(pairs)} pair(s) whose "
        f"both dates decode, {rejected} rejected",
        flush=True,
    )
    return pairs


def build(split: str, second_root: Path, cache: Path, limit: int):
    pairs = index_pairs(second_root)
    images = {i["id"]: i for i in fetch(split, "images", cache)["images"]}
    questions = fetch(split, "questions", cache)["questions"]
    answers = {a["question_id"]: a for a in fetch(split, "answers", cache)["answers"]}
    print(
        f"{split}: {len(images)} image entr(ies), {len(questions)} question(s), "
        f"{len(answers)} answer(s)",
        flush=True,
    )

    rows, stats = [], Counter()
    missing_images = set()
    for question in questions:
        if not question.get("active", True):
            stats["inactive"] += 1
            continue
        record = images.get(question["img_id"])
        answer = answers.get(question["id"])
        if record is None or answer is None:
            stats["unjoined"] += 1
            continue

        pair = pairs.get(record["file_name"])
        if pair is None:
            missing_images.add(record["file_name"])
            stats["image_missing"] += 1
            continue

        kind = question.get("type", "unknown")
        row = {
            "sample_id": f"cdvqa_{split.lower()}_{question['id']:07d}",
            "adapter": "change_vqa",
            "task": kind,
            "images": [str(pair[0]), str(pair[1])],
            "image_roles": ["first", "second"],
            "modality": ["optical", "optical"],
            # CDVQA's own metadata says 0.1524 m for every patch, which is the
            # RSVQA-HR template value carried over rather than this imagery's.
            # The source is 0.5-3 m, so the declared figure is not trustworthy
            # and 0.5 is recorded as its own stated best resolution.
            "effective_gsd_m": [0.5, 0.5],
            "question": question["question"],
            "answer": str(answer["answer"]),
            "answer_type": "yesno" if kind in CLOSED_ANSWERS else "text",
            "question_type": kind,
            "split": _SPLIT_NAMES[split],
            "source": SOURCE,
            "licence": LICENCE,
            "provenance_chain": list(PROVENANCE),
        }
        if kind in CLOSED_ANSWERS:
            row["answer_options"] = CLOSED_ANSWERS[kind]
        rows.append(row)
        stats[f"type/{kind}"] += 1
        if limit and len(rows) >= limit:
            break

    if missing_images:
        stats["distinct_images_missing"] = len(missing_images)
    return rows, stats, sorted(missing_images)[:10]


def cap_per_type(rows: list[dict], cap: int, seed: int) -> tuple[list[dict], dict]:
    """Even out the question types, because average accuracy averages over them.

    CDVQA ships ``change_or_not`` with 23,048 rows and ``change_ratio`` with
    3,200 -- a 7.2x gap between two types that count the same toward the score.
    ``change_ratio`` is also the hardest type, with a 16.5% blind ceiling, so
    the metric's weakest term is the one with the least signal behind it.

    Sampled rather than truncated: questions arrive grouped by image, so taking
    the first N would take a handful of scenes rather than a spread of them.
    """
    import random as _random

    rng = _random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["task"], []).append(row)

    kept: list[dict] = []
    report: dict[str, dict] = {}
    for kind, group in sorted(by_type.items()):
        take = group if len(group) <= cap else rng.sample(group, cap)
        kept.extend(take)
        report[kind] = {"available": len(group), "kept": len(take)}
    rng.shuffle(kept)
    return kept, report


def cmd_build(args) -> int:
    second_root = Path(args.second)
    rows, stats, missing_examples = build(
        args.split, second_root, Path(args.cache), args.limit
    )
    cap_report = None
    if args.cap_per_type:
        before = len(rows)
        rows, cap_report = cap_per_type(rows, args.cap_per_type, args.seed)
        print(f"capped {before} -> {len(rows)} at {args.cap_per_type}/type", flush=True)

    if not rows:
        # Staging nothing is the right outcome when the imagery is absent, and
        # saying so loudly beats writing an empty manifest that a training run
        # discovers at step zero.
        print(
            f"\nFAILED: no row could be built. {stats.get('image_missing', 0)} "
            f"question(s) referenced imagery that is not under {second_root}. "
            f"Expected {second_root}/im1/<name>.png and im2/<name>.png. "
            f"Missing e.g. {missing_examples[:3]}",
            flush=True,
        )
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    answers = Counter(r["answer"] for r in rows)
    per_type_floor = {}
    by_type: dict[str, Counter] = {}
    for row in rows:
        by_type.setdefault(row["task"], Counter())[row["answer"]] += 1
    for kind, counts in by_type.items():
        per_type_floor[kind] = round(max(counts.values()) / sum(counts.values()), 4)

    summary = {
        "split": args.split,
        "rows": len(rows),
        "source": SOURCE,
        "licence": LICENCE,
        "question_types": {k: v for k, v in sorted(stats.items()) if k.startswith("type/")},
        "distinct_answers": len(answers),
        "top_answers": dict(answers.most_common(10)),
        # Average accuracy is the mean over types, so the per-type blind ceiling
        # is what each type's score has to beat -- not the corpus-wide one.
        "blind_ceiling_per_type": per_type_floor,
        "skipped": {k: v for k, v in sorted(stats.items()) if not k.startswith("type/")},
        "cap_per_type": args.cap_per_type or None,
        "capped": cap_report,
        "out": str(out),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument(
        "--second", required=True, help="imagery root containing im1/ and im2/"
    )
    build_cmd.add_argument("--split", default="Train", choices=["Train", "Val", "Test", "Test2"])
    build_cmd.add_argument("--cache", default="/data/eval/cdvqa_qa")
    build_cmd.add_argument("--out", default="/data/manifests/change_vqa_cdvqa_train.jsonl")
    build_cmd.add_argument("--summary", default="logs/cdvqa_corpus.json")
    build_cmd.add_argument("--limit", type=int, default=0)
    build_cmd.add_argument("--cap-per-type", type=int, default=0,
                           help="max rows per question type; 0 disables")
    build_cmd.add_argument("--seed", type=int, default=0)
    build_cmd.set_defaults(func=cmd_build)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
