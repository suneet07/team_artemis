"""Turn staged SpaceNet 7 date pairs into canonical change_vqa questions.

    python scripts/gen_change.py --root /data/eval/sn7 \
        --out /data/manifests/change_vqa_train.jsonl --limit 30000

**The labels answer the question; the script only phrases it.** Every month of
SpaceNet 7 carries a matched label file in which a building keeps the same
persistent ``Id`` across dates. Set arithmetic on those ID sets between two
months yields an exact change inventory -- appeared, demolished, persisted --
so a count here is ground truth, not a threshold on a spectral index.

**Every pair is emitted in both orders, and that is the point of the file.**
The adapter-1 shuffled-image ablation put IO_AdTest at 33-43%: about a third of
the score came from vision and the rest from knowing what questions like this
usually answer. Chappuis et al. (2023) measure the same failure on RSVQA and
show it survives instructions -- their models did not change their answers when
forest was manually erased from the image. Wording cannot fix a statistical
regularity; removing the regularity can.

So this follows Goyal et al. (2017): every question ships with a twin carrying
the opposite answer. Presenting (A, B) and (B, A) gives identical question text
over identical pixels with a flipped answer, because ``questions_for`` reads
its two arguments directionally -- what appeared going forwards was demolished
going backwards. A text-only model scores exactly 50% on the result by
construction, which is the strongest guarantee available without collecting
more imagery. The swap is not a claim about time: the question asks about the
first and second image shown, which is a well-posed visual comparison in either
order.

**Why counts split at 20.** An exact count is fair when the answer is small
enough to arrive at by looking. Across a year a busy AOI gains hundreds, and
"how many exactly" stops being answerable from a 4 m image by any observer.
Above the split the question changes form rather than getting harder.
"""

import argparse
import json
import random
import sys
from collections import Counter
from hashlib import sha1
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ADAPTER = "change_vqa"
SOURCE = "SpaceNet 7 / MUDS"
LICENCE = "CC BY-SA 4.0"
GSD_M = 4.0

#: Above this many new buildings, ask for a band instead of an exact number.
EXACT_COUNT_MAX = 20

#: Bands for change_magnitude. Ordered; first match wins.
MAGNITUDE_BANDS: list[tuple[int, str]] = [
    (0, "none"),
    (10, "a few"),
    (50, "dozens"),
    (10**9, "many"),
]

#: A direction question is only asked when the appeared buildings actually
#: cluster. Below these thresholds the true answer is a coin toss between two
#: quadrants, and training on an arbitrary label teaches noise.
WHERE_MIN_BUILDINGS = 6
#: With a scene-centred frame a genuine cluster clears this comfortably; the
#: old 0.5 was satisfied by any split at all, because the frame guaranteed it.
WHERE_MIN_SHARE = 0.65

#: The closed answer set per question type, carried on every row. The eval
#: harness constrains generation to this set -- the published Qwen change-VQA
#: study does the same -- and a formatter that has to recover the options by
#: parsing them back out of the question text is one rephrasing away from
#: silently scoring everything wrong. ``change_count`` is open by design.
ANSWER_OPTIONS: dict[str, list[str] | None] = {
    "change_presence": ["yes", "no"],
    "change_direction": ["increased", "decreased", "unchanged"],
    "change_compare": ["first", "second"],
    "change_magnitude": [band for _, band in MAGNITUDE_BANDS],
    "change_where": ["north", "south", "east", "west"],
    "change_count": None,
}

#: Hard cap per (split, question type, answer).
DEFAULT_CAP = 400

#: Hard cap on a whole question type, which the per-answer cap alone does not
#: give. The per-answer cap grants a type as many slots as it has answers, so
#: ``change_count`` -- 21 distinct integers -- took 21x400 while yes/no types
#: took 2x400. Measured on the first real run that was 9,052 of 15,892 rows:
#: 57% of the corpus was one question type, and the adapter would have spent
#: most of its training counting.
DEFAULT_TASK_CAP = 2500


def months_between(month0: str, month1: str) -> int:
    """Absolute gap in months between two ``YYYY-MM`` strings."""
    y0, m0 = (int(part) for part in month0.split("-"))
    y1, m1 = (int(part) for part in month1.split("-"))
    return abs((y1 * 12 + m1) - (y0 * 12 + m0))


def band_for(count: int) -> str:
    for ceiling, label in MAGNITUDE_BANDS:
        if count <= ceiling:
            return label
    raise AssertionError("MAGNITUDE_BANDS must end with a catch-all")


def _centroid(geometry: dict | None) -> tuple[float, float] | None:
    """Rough centroid of a footprint: the mean of its exterior ring.

    Exact area-weighted centroids are unnecessary here. The only consumer asks
    which half of the tile a building sits in, and a 4 m building's ring mean
    and true centroid never fall on opposite sides of that line.
    """
    if not geometry:
        return None
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "Polygon":
        rings = coordinates[:1]
    elif kind == "MultiPolygon":
        rings = [polygon[0] for polygon in coordinates if polygon]
    else:
        return None
    points = [point for ring in rings for point in ring]
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def buildings_in(path: Path) -> dict:
    """Persistent building ID -> centroid, for one matched label file.

    An empty file is a real state (a month with no buildings), but a missing
    ``Id`` is an error: without it a building cannot be tracked between dates,
    and treating it as absent would fabricate a demolition.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for feature in document.get("features", []):
        properties = feature.get("properties") or {}
        identifier = properties.get("Id", properties.get("id"))
        if identifier is None:
            raise ValueError(
                f"{path.name} has a feature with no persistent 'Id'. This is a "
                "labels_match file requirement -- without it, buildings cannot "
                "be tracked between dates and every count here is fiction."
            )
        out[identifier] = _centroid(feature.get("geometry"))
    return out


#: The largest share of a mosaic that may be masked out before its labels stop
#: describing what the model can see.
#:
#: SpaceNet 7's ``images_masked`` sets cloud-covered regions to pure black and
#: keeps the label file whole, so every building under the mask is still
#: counted. A fully black month was already rejected; a *partial* mask was not,
#: and the corpus browser turned up a tile with a hole through the middle of it
#: under the question "which image contains more buildings".
#:
#: ``scripts/audit_masking.py`` measured all 1,423 mosaics: 837 clean, 400
#: below 1%, and 131 above 5% -- of which 52 are masked between a fifth and all
#: of the frame. 1% of a 1024 px tile at 4 m is about 0.17 km2, enough to hide
#: perhaps twenty small buildings, so the line is drawn there rather than
#: higher. It costs 186 mosaics and buys labels the imagery can support.
MAX_MASKED_SHARE = 0.01


def masked_share(path: Path, _cache: dict[Path, bool] = {}) -> bool:  # noqa: B006
    """True when too much of a mosaic is masked out. Cached per file.

    Exactly-black pixels, not merely dark ones: the mask writes zero across all
    three channels while genuine shadow and deep water sit above it. Strided by
    4 -- a mask large enough to hide a building dwarfs a 4 px sampling grid.
    """
    if path in _cache:
        return _cache[path]
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as source:
            array = np.asarray(source.convert("RGB"))[::4, ::4, :]
        share = float((array.sum(axis=2) == 0).mean())
        bad = share > MAX_MASKED_SHARE
    except Exception as error:  # noqa: BLE001 - unreadable is unusable
        print(f"  [warn] {path.name}: {error!r} -- treating as masked", flush=True)
        bad = True
    _cache[path] = bad
    return bad


def _where(
    appeared: list[tuple[float, float]], frame: list[tuple[float, float]]
) -> str | None:
    """The compass half the new buildings cluster in, if they cluster at all.

    ``frame`` establishes where the middle of the scene is and must NOT be the
    appeared buildings themselves. The first version of this took the midpoint
    as the mean of the very points it was classifying, which puts about half of
    them on each side by construction: on a real AOI, 39 new buildings split
    17/22 east-west and 23/16 north-south, and the 59% majority cleared the 50%
    threshold on noise. Every answer this produced was close to a coin flip,
    and because a coin flip is uniform it looked like a perfectly balanced
    question type in the corpus summary.
    """
    points = [p for p in appeared if p]
    scene = [p for p in frame if p]
    if len(points) < WHERE_MIN_BUILDINGS or len(scene) < WHERE_MIN_BUILDINGS:
        return None
    # Centre of the scene's bounding box, from every building at either date.
    # Independent of which subset appeared, which is the whole point.
    xs = [p[0] for p in scene]
    ys = [p[1] for p in scene]
    mid_x = (min(xs) + max(xs)) / 2
    mid_y = (min(ys) + max(ys)) / 2
    tallies = Counter()
    for x, y in points:
        tallies["east" if x >= mid_x else "west"] += 1
        tallies["north" if y >= mid_y else "south"] += 1
    # Compare within an axis, never across: every point votes once per axis, so
    # a global argmax would just pick whichever axis split more evenly.
    best = None
    for axis in (("north", "south"), ("east", "west")):
        label = max(axis, key=lambda k: tallies[k])
        share = tallies[label] / len(points)
        if share >= WHERE_MIN_SHARE and (best is None or share > best[1]):
            best = (label, share)
    return best[0] if best else None


def questions_for(before: dict, after: dict, month0: str, month1: str) -> list[dict]:
    """Every question this ordering of a date pair can honestly support.

    Directional by design: swapping ``before`` and ``after`` is what produces
    the complementary twin, so nothing in here may be made order-agnostic.
    """
    new_ids = set(after) - set(before)
    gone_ids = set(before) - set(after)
    appeared, demolished = len(new_ids), len(gone_ids)
    total0, total1 = len(before), len(after)
    out: list[dict] = []

    out.append(
        {
            "task": "change_presence",
            "question": (
                "Comparing the first image to the second, did any new buildings "
                "appear?"
            ),
            "answer": "yes" if appeared else "no",
            "answer_type": "yesno",
        }
    )
    out.append(
        {
            "task": "change_presence",
            "question": (
                "Comparing the first image to the second, were any buildings "
                "demolished?"
            ),
            "answer": "yes" if demolished else "no",
            "answer_type": "yesno",
        }
    )

    if total0 == total1:
        direction = "unchanged"
    elif total1 > total0:
        direction = "increased"
    else:
        direction = "decreased"
    out.append(
        {
            "task": "change_direction",
            "question": (
                "Did the number of buildings increase, decrease, or stay "
                "unchanged between the first and second image?"
            ),
            "answer": direction,
            "answer_type": "text",
        }
    )

    # Perfectly balanced across the two orderings by construction, whatever the
    # underlying scene does: whichever image holds more, the twin says the
    # other one. This is the cheapest complementary pair in the corpus.
    if total0 != total1:
        out.append(
            {
                "task": "change_compare",
                "question": (
                    "Which image contains more buildings, the first or the "
                    "second?"
                ),
                "answer": "second" if total1 > total0 else "first",
                "answer_type": "text",
            }
        )

    # Magnitude is asked of EVERY pair, not only the large ones. Restricting it
    # would make "none" and "a few" unreachable answers to a question that
    # offers them, teaching the model two of its four options are never right.
    out.append(
        {
            "task": "change_magnitude",
            "question": (
                "How many new buildings appeared in the second image: none, "
                "a few, dozens, or many?"
            ),
            "answer": band_for(appeared),
            "answer_type": "text",
        }
    )

    if appeared <= EXACT_COUNT_MAX:
        out.append(
            {
                "task": "change_count",
                "question": "How many new buildings appeared in the second image?",
                "answer": str(appeared),
                "answer_type": "number",
            }
        )
    if demolished <= EXACT_COUNT_MAX:
        out.append(
            {
                "task": "change_count",
                "question": "How many buildings were demolished by the second image?",
                "answer": str(demolished),
                "answer_type": "number",
            }
        )

    # No answer prior can reach this one: the compass half the growth landed in
    # is a property of the pixels and nothing else.
    side = _where([after[i] for i in new_ids], list(after.values()))
    if side:
        out.append(
            {
                "task": "change_where",
                "question": (
                    "In which part of the second image did the new buildings "
                    "mostly appear: north, south, east, or west?"
                ),
                "answer": side,
                "answer_type": "text",
            }
        )

    for item in out:
        item["answer_options"] = ANSWER_OPTIONS[item["task"]]
        item["gap_months"] = months_between(month0, month1)
        item["months"] = [month0, month1]
        item["appeared"] = appeared
        item["demolished"] = demolished
        item["buildings"] = [total0, total1]
    return out


def twin_id(aoi: str, month_a: str, month_b: str, item: dict) -> str:
    """Stable id shared by the two orderings of one question.

    Keyed on the *unordered* month pair plus the question text, so (A, B) and
    (B, A) collide deliberately while the two presence questions on the same
    pair -- which share a task but ask opposite things -- do not.
    """
    months = "_".join(sorted((month_a, month_b)))
    digest = sha1(item["question"].encode("utf-8")).hexdigest()[:8]
    return f"{aoi}_{months}_{item['task']}_{digest}"


def _write_summary(path: Path, rows: list[dict]) -> Path:
    """A small stats file beside the manifest.

    The manifest itself lives on the Volume and is far too large to pull back,
    so without this a remote run has no artifact -- and the job wrapper treats
    a run that produced no artifact as a failure, which is how a successful
    generation reported as a RuntimeError. The blind ceiling per question type
    is the number worth reading anyway.
    """
    by_task: dict[str, Counter] = {}
    for row in rows:
        by_task.setdefault(row["task"], Counter())[str(row["answer"])] += 1
    summary = {
        "rows": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "types": {
            task: {
                "rows": sum(answers.values()),
                "distinct_answers": len(answers),
                "blind_ceiling": round(
                    100 * answers.most_common(1)[0][1] / sum(answers.values()), 2
                ),
                "top": dict(answers.most_common(6)),
            }
            for task, answers in sorted(by_task.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path


def _balance(rows: list[dict], rng: random.Random) -> list[dict]:
    """Equalise answers within each (split, task), dropping whole twin groups.

    Removing the blank-mosaic pairs took change_presence from a 53.9% blind
    ceiling to 74.3%, because most of the corpus's "nothing changed" examples
    were months with no valid imagery -- they read as no-change because there
    was no picture, not because nothing happened. The honest balance is much
    tighter than the raw one, so it has to be imposed rather than hoped for.

    Twin groups are the unit, never rows: dropping half a pair would leave the
    orphan this generator works to avoid. Groups whose two halves give the SAME
    answer go first, because a twin that does not flip carries no balancing
    information -- it is two rows agreeing, not a counterexample.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["twin_id"], []).append(row)

    keep: list[dict] = []
    by_bucket: dict[tuple[str, str], list[list[dict]]] = {}
    for members in groups.values():
        by_bucket.setdefault(
            (members[0]["split"], members[0]["task"]), []
        ).append(members)

    for (_split, task), bucket in sorted(by_bucket.items()):
        if ANSWER_OPTIONS.get(task) is None:
            # Open-vocabulary counting: 21 near-uniform answers already, and
            # equalising them would throw away most of the type for nothing.
            keep.extend(r for members in bucket for r in members)
            continue
        # Flipping groups first, so what survives is maximally informative.
        rng.shuffle(bucket)
        bucket.sort(key=lambda m: len({r["answer"] for r in m}) < 2)
        tally: Counter = Counter()
        target = min(
            Counter(
                r["answer"] for members in bucket for r in members
            ).values()
        )
        for members in bucket:
            demand = Counter(r["answer"] for r in members)
            if any(tally[a] + n > target for a, n in demand.items()):
                continue
            tally.update(demand)
            keep.extend(members)
    return keep


def cmd_build(args) -> int:
    root = Path(args.root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    aois = sorted(plan["aois"])
    # Hold whole AOIs out, never individual pairs: two pairs from one AOI share
    # imagery, so a pair-level split leaks the validation scenes into training
    # and every score after that is inflated.
    rng.shuffle(aois)
    if args.aois:
        # Sampled after the shuffle, so a small run is a random slice of the
        # world rather than the first few AOIs alphabetically -- which on
        # SpaceNet 7 would be a run of adjacent tiles from one continent.
        aois = aois[: args.aois]
    holdout = set(aois[: max(1, round(len(aois) * args.val_fraction))])
    print(f"{len(aois)} AOI(s); {len(holdout)} held out for validation")

    kept: list[dict] = []
    caps: Counter = Counter()
    skipped_missing = 0
    skipped_blank = 0
    considered = 0

    for aoi in aois:
        entry = plan["aois"][aoi]
        split = "val" if aoi in holdout else "train"
        labels: dict[str, dict] = {}
        pairs = list(combinations(entry["months"], 2))
        # Shuffle before capping so the discarded surplus is not systematically
        # the long-gap pairs at the tail of the month list.
        rng.shuffle(pairs)

        for month_a, month_b in pairs:
            paths = {}
            for month in (month_a, month_b):
                paths[month] = (
                    root / aoi / "images" / Path(entry["images"][month]).name,
                    root / aoi / "labels" / Path(entry["labels"][month]).name,
                )
            if not all(p.exists() for pair in paths.values() for p in pair):
                skipped_missing += 1
                continue
            if any(masked_share(paths[m][0]) for m in (month_a, month_b)):
                skipped_blank += 1
                continue
            considered += 1
            # Cached per AOI, not per pair. Each month takes part in 23 pairs on
            # a full 24-month AOI, and re-reading its footprints every time
            # turned 24 label parses into 552 -- against files that hold 11M
            # polygons across the dataset.
            for month in (month_a, month_b):
                if month not in labels:
                    labels[month] = buildings_in(paths[month][1])

            # Both orderings, admitted or rejected as a unit. Capping each half
            # independently left half the corpus holding one side of a pair:
            # the marginal balance survived that, but paired scoring did not,
            # and a lone half is exactly the row a language prior can win.
            groups: dict[str, list[tuple[str, str, dict]]] = {}
            for first, second in ((month_a, month_b), (month_b, month_a)):
                for item in questions_for(
                    labels[first], labels[second], first, second
                ):
                    groups.setdefault(
                        twin_id(aoi, month_a, month_b, item), []
                    ).append((first, second, item))

            cap = args.cap if split == "train" else max(1, args.cap // 5)
            for members in groups.values():
                # Count the group's own demand before admitting it: two halves
                # answering "no" need two slots, not one.
                demand: Counter = Counter(
                    (split, item["task"], item["answer"]) for _, _, item in members
                )
                if any(caps[k] + n > cap for k, n in demand.items()):
                    continue
                caps.update(demand)
                for first, second, item in members:
                    kept.append(
                        {
                            # Deterministic, so a rerun with the same plan
                            # produces the same ids and a resumed or joined
                            # prediction dump still lines up. The digest is over
                            # the question text, which is what separates the two
                            # same-task questions asked of one ordering.
                            "sample_id": (
                                f"sn7_{aoi}_{first}_{second}_{item['task']}_"
                                f"{sha1(item['question'].encode()).hexdigest()[:8]}"
                            ),
                            # Shared by both orderings of one question, so the
                            # complementary pair survives shuffling and merging.
                            # Paired scoring needs it: a model is only credited
                            # with seeing when it answers BOTH halves correctly,
                            # and without this the halves cannot be rejoined.
                            "twin_id": twin_id(aoi, month_a, month_b, item),
                            "adapter": ADAPTER,
                            "aoi": aoi,
                            "task": item["task"],
                            "images": [
                                str(paths[first][0].relative_to(args.image_root)),
                                str(paths[second][0].relative_to(args.image_root)),
                            ],
                            # Presentation order, not chronology: the question
                            # says "first" and "second", and after a swap those
                            # no longer line up with the dates.
                            "image_roles": ["first", "second"],
                            "modality": ["optical", "optical"],
                            "effective_gsd_m": [GSD_M, GSD_M],
                            "question": item["question"],
                            "answer": item["answer"],
                            "answer_type": item["answer_type"],
                            "answer_options": item["answer_options"],
                            "gap_months": item["gap_months"],
                            "split": split,
                            "source": SOURCE,
                            "licence": LICENCE,
                            "months": item["months"],
                            "appeared": item["appeared"],
                            "demolished": item["demolished"],
                            "buildings": item["buildings"],
                        }
                    )

    # Balance BEFORE the per-type ceiling, never after. Applied first, the
    # ceiling truncates a type in shuffled order and takes the rare answer down
    # with it: change_direction's "unchanged" is 206 cases in 31,958, so a
    # 12,000-row cut left ~77 of them and balancing to that gave 192 rows where
    # 618 were available.
    if args.balance:
        kept = _balance(kept, rng)
    if args.task_cap:
        by_task: Counter = Counter()
        capped: list[dict] = []
        for row in sorted(kept, key=lambda r: r["twin_id"]):
            key = (row["split"], row["task"])
            if by_task[key] >= args.task_cap:
                continue
            by_task[key] += 1
            capped.append(row)
        kept = capped

    rng.shuffle(kept)
    if args.limit:
        kept = kept[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_summary(Path(args.summary_out) if args.summary_out
                   else out.with_suffix('.summary.json'), kept)
    by_task = Counter(r["task"] for r in kept)
    print(f"\n{considered} usable pair(s), {skipped_missing} skipped for missing files")
    print(f"{len(kept)} row(s) -> {out}")
    print(f"\n{'task':20} {'rows':>6} {'answers':>8}  most common (blind ceiling)")
    for task, count in sorted(by_task.items()):
        answers = Counter(r["answer"] for r in kept if r["task"] == task)
        # The single most useful audit line in this script. A text-only model
        # that always guesses the majority answer scores exactly this, so it is
        # the floor any real score has to beat before vision is demonstrated.
        _, top_n = answers.most_common(1)[0]
        top = ", ".join(f"{a}={n}" for a, n in answers.most_common(4))
        print(f"{task:20} {count:6} {len(answers):8}  {100 * top_n / count:5.1f}%  {top}")
    by_split = sorted(Counter(r["split"] for r in kept).items())
    print("  " + "  ".join(f"{k}={v}" for k, v in by_split))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="/data/eval/sn7")
    parser.add_argument("--out", default="/data/manifests/change_vqa_train.jsonl")
    parser.add_argument(
        "--image-root",
        default="/data",
        help="paths in the manifest are written relative to this",
    )
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument(
        "--task-cap",
        type=int,
        default=DEFAULT_TASK_CAP,
        help="ceiling on a whole question type; 0 disables",
    )
    parser.add_argument(
        "--aois", type=int, default=0, help="0 = every AOI; small = test run"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="equalise answers within each question type, by whole twin groups",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--summary-out",
        default="",
        help="where the stats file goes; defaults to beside the manifest. Modal "
        "runs point this at logs/ so the report comes back to the workstation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
