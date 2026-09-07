"""Turn self-generated Indian Sentinel-2 tile pairs into change_vqa questions.

    python scripts/gen_india.py --root /data/eval/india \
        --out /data/manifests/change_vqa_india.jsonl

**These labels are weaker than SpaceNet 7's and the file should say so.** There
are no annotations here. Every answer comes from thresholding a spectral index
difference between two dates, which is a measurement of reflectance, not of
what is on the ground. A field that was harvested and a field that was flooded
both drop NDVI. SpaceNet 7 remains the source of record for change; this one
exists for Indian domain match, for land-cover classes SpaceNet 7 does not
carry, and for volume.

**The dead band is the whole honesty mechanism.** An index delta near zero
cannot be called either way -- atmospheric correction residue, sun angle and
sensor noise all live in that range. Rather than round it to "unchanged" and
train on a coin flip, tiles inside the band are skipped for that index. This
costs rows and buys labels that mean something.

**Both orderings, as in gen_change.py.** Every delta flips sign under a swap,
so presenting (t0, t1) and (t1, t0) gives identical question text over
identical pixels with the opposite answer. See that file's header for why this
matters more than any wording of the prompt.

**Six views per sample, and the questions require them.** Each date carries a
true-colour, a false-colour (near-infrared) and a short-wave composite.
Vegetation is legible in the false-colour view and close to invisible in true
colour; moisture and built-up surfaces need the short-wave one. A question
about vegetation is therefore not answerable from the RGB view alone, which is
the property that makes the extra views earn their cost.

Licence: Copernicus open -- contains modified Copernicus Sentinel data.
"""

import argparse
import json
import random
import sys
from collections import Counter
from hashlib import sha1
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ADAPTER = "change_vqa"
SOURCE = "Sentinel-2 L2A (self-generated, Indian AOIs)"
LICENCE = "Copernicus open - contains modified Copernicus Sentinel data"
GSD_M = 10.0

#: The three composites written per date, in the order they are presented.
VIEWS = ("true_colour", "false_colour", "short_wave")

#: Index -> (what it measures, the noun the question uses).
INDICES = {
    "ndvi": "vegetation cover",
    "ndwi": "surface water",
    "ndbi": "built-up surface",
}

#: Below this, a delta is not distinguishable from atmospheric and sensor
#: noise; above it, the change is real enough to name. Deltas that fall in
#: between are skipped rather than rounded, so no row carries a label the
#: measurement cannot support. 0.10 is the threshold commonly used for
#: meaningful NDVI change; the dead band runs from 0.05 to it.
CHANGE_MIN = 0.10
NOISE_MAX = 0.05

ANSWER_OPTIONS: dict[str, list[str] | None] = {
    "index_direction": ["increased", "decreased", "unchanged"],
    "index_presence": ["yes", "no"],
    "index_compare": ["first", "second"],
    "dominant_change": sorted(INDICES),
}

#: ``dominant_change`` is written but suppressed. On the first real run it
#: returned a single answer -- ndbi, 8 rows, a 100% blind ceiling -- because
#: NDBI is the noisiest of the three indices and therefore "wins" the ranking
#: on tiles where nothing built-up happened at all. The forest AOI showed mean
#: |NDBI delta| of 0.222 against 0.046 for NDVI, which is backwards for forest.
#: The index needs low-signal pixels masked before averaging; until then this
#: question type would teach the model that everything is a construction site.
DOMINANT_CHANGE_ENABLED = False

DEFAULT_CAP = 400


def images_for(root: Path, tile: dict, date_key: str) -> list[str]:
    return [
        str(root / tile["aoi"] / f"{tile['tile_id']}_{date_key}_{view}.png")
        for view in VIEWS
    ]


def questions_for(tile: dict, forward: bool) -> list[dict]:
    """Questions for one ordering of one tile pair.

    ``forward`` False means t1 is shown first, which negates every delta. This
    is the complementary twin and nothing here may be made sign-agnostic.
    """
    sign = 1 if forward else -1
    out: list[dict] = []
    magnitudes: dict[str, float] = {}

    for index, noun in INDICES.items():
        raw = tile.get(f"{index}_delta")
        if raw is None:
            continue
        delta = sign * raw
        magnitudes[index] = abs(raw)

        if abs(delta) >= CHANGE_MIN:
            direction = "increased" if delta > 0 else "decreased"
            changed = "yes"
        elif abs(delta) <= NOISE_MAX:
            direction = "unchanged"
            changed = "no"
        else:
            # Inside the dead band. Skipped, not rounded.
            continue

        out.append(
            {
                "task": "index_direction",
                "index": index,
                "question": (
                    f"Between the first and second image, did {noun} increase, "
                    "decrease, or stay unchanged?"
                ),
                "answer": direction,
                "answer_type": "text",
            }
        )
        out.append(
            {
                "task": "index_presence",
                "index": index,
                "question": (
                    f"Is there a noticeable change in {noun} between the first "
                    "and second image?"
                ),
                "answer": changed,
                "answer_type": "yesno",
            }
        )
        if direction != "unchanged":
            out.append(
                {
                    "task": "index_compare",
                    "index": index,
                    "question": f"Which image shows more {noun}, the first or the second?",
                    "answer": "second" if delta > 0 else "first",
                    "answer_type": "text",
                }
            )

    # Which band moved most. Asked only when one clearly leads, because a tie
    # between two indices has no defensible answer. Order-invariant by nature:
    # swapping dates negates every delta and leaves the ranking untouched, so
    # this type carries no twin and the cap alone governs it.
    if DOMINANT_CHANGE_ENABLED and len(magnitudes) >= 2:
        ranked = sorted(magnitudes.items(), key=lambda kv: kv[1], reverse=True)
        (lead, top), (_, second) = ranked[0], ranked[1]
        if top >= CHANGE_MIN and top >= 2 * second:
            out.append(
                {
                    "task": "dominant_change",
                    "index": lead,
                    "question": (
                        "Which changed most between the two images: vegetation "
                        "cover, surface water, or built-up surface?"
                    ),
                    "answer": lead,
                    "answer_type": "text",
                }
            )
    return out


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


def cmd_build(args) -> int:
    root = Path(args.root)
    tiles = json.loads((root / "tiles.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    # Held out by AOI, never by tile: tiles from one AOI overlap in season,
    # sensor geometry and land cover, so a tile-level split leaks the
    # validation scenes into training.
    aois = sorted({t["aoi"] for t in tiles})
    rng.shuffle(aois)
    holdout = set(aois[: max(1, round(len(aois) * args.val_fraction))])
    print(f"{len(tiles)} tile(s) across {len(aois)} AOI(s); {len(holdout)} held out")

    kept: list[dict] = []
    caps: Counter = Counter()
    rng.shuffle(tiles)

    for tile in tiles:
        split = "val" if tile["aoi"] in holdout else "train"
        cap = args.cap if split == "train" else max(1, args.cap // 5)

        groups: dict[str, list[tuple[bool, dict]]] = {}
        for forward in (True, False):
            for item in questions_for(tile, forward):
                key = (
                    f"{tile['tile_id']}_{item['task']}_{item['index']}_"
                    f"{sha1(item['question'].encode()).hexdigest()[:8]}"
                )
                groups.setdefault(key, []).append((forward, item))

        # Admitted as a unit, so the cap can never keep one half of a twin.
        for key, members in groups.items():
            demand: Counter = Counter(
                (split, item["task"], item["answer"]) for _, item in members
            )
            if any(caps[k] + n > cap for k, n in demand.items()):
                continue
            caps.update(demand)
            for forward, item in members:
                first, second = ("t0", "t1") if forward else ("t1", "t0")
                dates = [tile["t0_date"], tile["t1_date"]]
                kept.append(
                    {
                        "sample_id": f"india_{key}_{first}{second}",
                        "twin_id": key,
                        "adapter": ADAPTER,
                        "aoi": tile["aoi"],
                        "tile_id": tile["tile_id"],
                        "task": item["task"],
                        "images": [
                            str(Path(p).relative_to(args.image_root))
                            for p in images_for(root, tile, first)
                            + images_for(root, tile, second)
                        ],
                        "image_roles": [f"first_{v}" for v in VIEWS]
                        + [f"second_{v}" for v in VIEWS],
                        "modality": ["optical"] * 6,
                        "effective_gsd_m": [GSD_M] * 6,
                        "question": item["question"],
                        "answer": item["answer"],
                        "answer_type": item["answer_type"],
                        "answer_options": ANSWER_OPTIONS[item["task"]],
                        "split": split,
                        "source": SOURCE,
                        "licence": LICENCE,
                        "stratum": tile["stratum"],
                        "season": tile["season"],
                        "index": item["index"],
                        "dates": dates if forward else dates[::-1],
                        # Kept so a suspicious label can be traced to the number
                        # that produced it without re-running the fetch.
                        "deltas": {
                            k: tile.get(f"{k}_delta") for k in INDICES
                        },
                    }
                )

    rng.shuffle(kept)
    if args.limit:
        kept = kept[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(kept)} row(s) -> {out}")
    print(f"\n{'task':20} {'rows':>6} {'answers':>8}  blind ceiling")
    for task, count in sorted(Counter(r["task"] for r in kept).items()):
        answers = Counter(r["answer"] for r in kept if r["task"] == task)
        _, top_n = answers.most_common(1)[0]
        top = ", ".join(f"{a}={n}" for a, n in answers.most_common(4))
        print(f"{task:20} {count:6} {len(answers):8}  {100 * top_n / count:5.1f}%  {top}")
    strata = Counter(r["stratum"] for r in kept)
    print("\nby stratum: " + "  ".join(f"{k}={v}" for k, v in sorted(strata.items())))
    print("  " + "  ".join(
        f"{k}={v}" for k, v in sorted(Counter(r["split"] for r in kept).items())
    ))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="/data/eval/india")
    parser.add_argument("--out", default="/data/manifests/change_vqa_india.jsonl")
    parser.add_argument("--image-root", default="/data")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
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
