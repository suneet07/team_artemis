"""Turn the RSVQA-LR release into canonical rows.

    python scripts/stage_rsvqa_lr.py --root data/eval/rsvqa_lr --split test

The release is not one file. Question and answer *text* live in
``all_questions.json`` / ``all_answers.json``, which cover every split together;
the per-split files (``LR_split_test_questions.json`` and friends) carry only
``{"id": N, "active": true|false}`` flags saying which ids belong to that split.
Reading the split files alone gets you 33,212 rows of nothing, which is the
shape of the mistake this script exists to not make.

Two things it is strict about:

**Question type is carried through.** RSVQA-LR rows are typed ``rural_urban``,
``presence``, ``count`` or ``comp``, and the reported metric is average accuracy
-- the mean of per-type accuracies, not overall accuracy. The types are wildly
unbalanced (200 ``rural_urban`` against 7,823 ``comp`` in the first 20k), so a
row that loses its type silently moves the headline number.

**One image means one view.** RSVQA-LR ships a single RGB tif per sample, not a
three-composite stack. ``RealChipDataset`` repeats a view when a caller pins a
composite count, so running the C22 two-vs-three ablation against this corpus
would compare a picture with a copy of itself. That ablation belongs on the BEN
chips, which have three genuinely different composites.

Licence: CC-BY-4.0, confirmed from the Zenodo record metadata (6344334), not
from the paper.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Sentinel-2 at 10 m, 256 px tiles -> 2,560 m on a side. Staged as shipped: the
#: eval preprocessing has to match how the published numbers were produced.
GSD_M = 10.0
SOURCE = "RSVQA-LR (Zenodo 6344334)"
LICENCE = "CC-BY-4.0"


def _load(root: Path, name: str) -> list[dict]:
    path = root / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Download the full release -- the split files alone "
            "carry no question text. Needed: all_questions.json, all_answers.json, "
            "LR_split_<split>_questions.json and Images_LR/."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[next(iter(payload))]


def build(root: Path, split: str) -> list[dict]:
    questions = {q["id"]: q for q in _load(root, "all_questions.json")}
    answers = {a["question_id"]: a for a in _load(root, "all_answers.json")}
    active = {
        row["id"] for row in _load(root, f"LR_split_{split}_questions.json") if row.get("active")
    }
    if not active:
        raise SystemExit(
            f"no active question ids in LR_split_{split}_questions.json. The split "
            "files mark membership with an `active` flag; an empty set means the "
            "wrong file or the wrong split name."
        )

    rows = []
    missing_answers = 0
    for qid in sorted(active):
        question = questions.get(qid)
        answer = answers.get(qid)
        if question is None or answer is None:
            missing_answers += 1
            continue
        image = f"Images_LR/{question['img_id']}.tif"
        if not (root / image).exists():
            raise SystemExit(
                f"{root / image} is missing. Unzip Images_LR.zip into {root} before "
                "staging -- a manifest pointing at absent images fails at step 1 of "
                "training rather than here, where the cause is obvious."
            )
        rows.append(
            {
                "sample_id": f"rsvqa_lr_{qid}",
                "adapter": "rs_vqa",
                "task": "single_vqa",
                "images": [image],
                "question": question["question"],
                "answer": str(answer["answer"]),
                "answer_type": "text",
                # The metric is the mean of per-type accuracies. Losing this
                # field turns average accuracy into overall accuracy silently.
                "question_type": question.get("type", "unknown"),
                "image_roles": ["true_colour"],
                "modality": ["optical"],
                "effective_gsd_m": [GSD_M],
                "split": split,
                "source": SOURCE,
                "licence": LICENCE,
            }
        )

    if missing_answers:
        raise SystemExit(
            f"{missing_answers} active question(s) had no matching answer. The "
            "release is inconsistent or the wrong all_answers.json was downloaded; "
            "staging a partial corpus would understate the benchmark size."
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/eval/rsvqa_lr")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=0,
        help="cap questions per image; 0 = uncapped. Use ~24 for a training "
        "corpus and 0 for evaluation, where every published question counts",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = build(root, args.split)

    if args.max_per_image:
        # RSVQA-LR annotates 572 images with 57,223 training questions -- about
        # **101 rows per image**. BEN.txt averages 6. Merged untouched, half an
        # rs_vqa corpus would come from 572 pictures seen a hundred times each,
        # and the plan asks for RSVQA-LR as a "train minority" rather than half
        # the data. Capping trades row count for scene diversity, which is the
        # thing the row count was standing in for.
        import random
        from collections import defaultdict

        by_image: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_image[row["images"][0]].append(row)
        capped: list[dict] = []
        for image, group in sorted(by_image.items()):
            rng = random.Random(f"rsvqa_lr:{args.split}:{image}")
            rng.shuffle(group)
            capped.extend(group[: args.max_per_image])
        rows = sorted(capped, key=lambda r: r["sample_id"])

    if args.limit:
        # Sampled, not truncated. Rows are ordered by question id, which groups
        # by image, so `rows[:limit]` returns every question from the
        # lowest-numbered images and none from the rest -- a subset of the
        # scenes rather than a subset of the questions. Seeded so the same limit
        # gives the same corpus on every machine.
        import random

        rng = random.Random(f"rsvqa_lr:{args.split}:{args.limit}")
        rng.shuffle(rows)
        rows = rows[: args.limit]
        rows.sort(key=lambda r: r["sample_id"])

    out = Path(args.out or root / f"rsvqa_lr_{args.split}.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    types: dict[str, int] = {}
    for row in rows:
        types[row["question_type"]] = types.get(row["question_type"], 0) + 1
    images = len({row["images"][0] for row in rows})

    print(f"{len(rows)} row(s) over {images} image(s) -> {out}")
    print("per-type counts (average accuracy is the mean over these):")
    for name, count in sorted(types.items(), key=lambda item: -item[1]):
        print(f"  {name:14} {count:6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
