"""Turn staged object annotations into canonical rs_ground_caption questions.

    python scripts/gen_ground.py --rareplanes /data/eval/rareplanes \
        --out /data/manifests/rs_ground_caption_train.jsonl

**This adapter answers with coordinates, and that changes every rule.** A
change_vqa answer is one of a handful of words, so a language prior can win it
and the blind ceiling is the number that matters. A box cannot be guessed the
same way -- but it can be *approximated*, and that is the failure to design
against here. BEN.txt's referring boxes have a median area of 19.8% of the
image and not one is under 1%; SpaceNet 6's buildings measure 0.03-0.22%. A
model trained on the first alone learns to emit a large box, scores well on
BEN's own mIoU, and lands near zero on a 30 cm aircraft.

So the corpus is built from three sources with deliberately different object
scales, every box is normalised to the same convention, and the eval reports
accuracy **per source** -- because one number averaged across a 100x scale
range would hide exactly the failure it needs to surface.

**Three box formats, one convention.** BEN already writes ``[x0 y0, x1 y1]``
normalised 0-1 and is passed through unchanged, since matching the primary
source's format is what keeps the model's output parseable. RarePlanes ships
COCO ``[x, y, w, h]`` in pixels on a 512 px tile. SpaceNet 6 ships polygons in
EPSG:32631 metres, which need the image's own geotransform to become pixels.
All three end up in BEN's convention.

**A referring expression must refer to exactly one thing.** "The aircraft" on a
tile holding nine of them names nothing, and a model cannot be right. Every
referring question here is checked for uniqueness against the tile's own
annotations before it is emitted, and dropped when the phrase matches more than
one object. That check is why the corpus is smaller than the annotation count.

**Truncated objects are excluded.** Both sources flag objects clipped by the
tile edge. Their true box extends past what the model can see, so grounding
them asks for something the pixels do not contain.
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

ADAPTER = "rs_ground_caption"

#: Answer vocabularies. Grounding tasks answer with a box and so have none --
#: the harness cannot constrain generation to a list, which is precisely why
#: the format has to be learned rather than enforced.
ANSWER_OPTIONS: dict[str, list[str] | None] = {
    "ground_reference": None,
    "ground_point": None,
    "ground_presence": ["yes", "no"],
    "caption": None,
}

#: Short, readable role names. RarePlanes' own strings carry slashes and
#: "Utility" suffixes that read badly inside a referring phrase.
ROLE_PHRASES = {
    "Small Civil Transport/Utility": "small civil aircraft",
    "Medium Civil Transport/Utility": "medium civil aircraft",
    "Large Civil Transport/Utility": "large civil airliner",
    "Military Transport/Utility/AWAC": "military transport aircraft",
    "Military Fighter/Interceptor/Attack": "military fighter jet",
    "Military Trainer": "military trainer aircraft",
    "Military Bomber": "military bomber",
}

DEFAULT_CAP = 4000


#: Smallest extent two decimal places can express. At 512 px that is ~5 px.
_MIN_EXTENT = 0.01


def box_string(x0: float, y0: float, x1: float, y1: float) -> str:
    """BEN's convention, which the model's output has to match exactly.

    **Collapsed dimensions are widened to ``_MIN_EXTENT``, centred.** Two
    decimals cannot express an object thinner than 0.01, so a SpaceNet 6
    building under ~5 px rounded to a zero-area box: 273 of 19,896 rows in the
    first corpus, all from SpaceNet 6, none from RarePlanes, whose aircraft are
    larger relative to the tile. A zero-area reference is unscoreable -- IoU
    against it is 0 for every prediction including a perfect one -- and in
    training it teaches the model to emit degenerate boxes.

    Widening rather than dropping the row keeps the supervision: the centre
    stays correct and the extent becomes the quantisation floor, which is the
    most the format can honestly say about an object that small. Raising the
    precision instead would be more faithful but would break the shared
    convention with the BEN grounding rows, which is the one thing the model's
    output has to match.
    """
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    x0, y0, x1, y1 = clamp(x0), clamp(y0), clamp(x1), clamp(y1)

    def widen(lo: float, hi: float) -> tuple[float, float]:
        if round(hi, 2) - round(lo, 2) >= _MIN_EXTENT:
            return lo, hi
        centre = (lo + hi) / 2
        lo, hi = centre - _MIN_EXTENT / 2, centre + _MIN_EXTENT / 2
        # Push back inside the frame rather than clipping, so the box keeps its
        # full width when the object sits on an edge.
        if lo < 0:
            lo, hi = 0.0, _MIN_EXTENT
        if hi > 1:
            lo, hi = 1.0 - _MIN_EXTENT, 1.0
        return lo, hi

    x0, x1 = widen(x0, x1)
    y0, y1 = widen(y0, y1)
    return f"[{x0:.2f} {y0:.2f}, {x1:.2f} {y1:.2f}]"


def coco_to_box(bbox: list[float], width: int, height: int) -> str:
    """COCO ``[x, y, w, h]`` pixels -> normalised ``[x0 y0, x1 y1]``."""
    x, y, w, h = bbox
    return box_string(x / width, y / height, (x + w) / width, (y + h) / height)


def _touches_edge(bbox: list[float], width: int, height: int, margin: float = 1.0) -> bool:
    """Does this box run into the tile boundary, and so past what is visible?"""
    x, y, w, h = bbox
    return (
        x <= margin
        or y <= margin
        or (x + w) >= width - margin
        or (y + h) >= height - margin
    )


def _quadrant_key(bbox: list[float], width: int, height: int) -> tuple[str, str]:
    """Canonical (vertical, horizontal) cell. Vocabulary-independent."""
    x, y, w, h = bbox
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    return ("top" if cy < 0.5 else "bottom", "left" if cx < 0.5 else "right")


def _quadrant(bbox: list[float], width: int, height: int) -> str:
    """The canonical cell as one string -- used for uniqueness and ``detail``.

    Kept vocabulary-free so a row's identity does not change when the surface
    phrasing does: two renderings of the same cell must still collide in the
    uniqueness check, or a target that is ambiguous in one vocabulary would
    quietly become "unique" in another.
    """
    vertical, horizontal = _quadrant_key(bbox, width, height)
    return f"{vertical} {horizontal}"


#: VRSBench phrases position **screen-relative** -- "bottom-left corner",
#: "middle-left side", "top of the image" -- while this generator used
#: **compass** terms. Measured on VRSBench_EVAL_referring.json, **87.4%** of its
#: 16,159 referring expressions turn on a position word and 72.6% carry no
#: relational clause at all, so position is the single largest signal in the
#: benchmark. A model that has only ever seen "northern eastern" has no reason
#: to map "bottom-left corner" onto the same region -- the same format-compliance
#: failure as the CDVQA answer vocabulary and the VRSBench 0-100 box scale, and
#: as with those, it costs nothing to fix and is worth more than a better model.
#:
#: Both vocabularies are generated so the model learns the mapping rather than
#: one dialect of it. The ISRO set's phrasing is undisclosed, which is a second
#: reason not to bet on either alone.
_COMPASS = {"top": "northern", "bottom": "southern", "left": "western", "right": "eastern"}

_POSITION_FORMS = (
    "{v}-{h} corner of the image",
    "{v} {h} part of the image",
    "{v}-{h} side of the image",
    "{v} {h} of the image",
)


def _position_phrase(key: tuple[str, str], rng: random.Random) -> str:
    """Render a quadrant in one of the two vocabularies, screen or compass."""
    vertical, horizontal = key
    if rng.random() < 0.5:
        return rng.choice(_POSITION_FORMS).format(v=vertical, h=horizontal)
    return f"{_COMPASS[vertical]} {_COMPASS[horizontal]} part of the image"


#: Surface forms for a referring expression. VRSBench gives an **untagged
#: declarative phrase** ("The large vehicle located at the bottom-left corner of
#: the image."); this generator only ever produced a tagged interrogative
#: ("Where can the <ref>...</ref> be found?"). Both are generated: the tagged
#: form is what our own agent pipeline emits, the untagged declarative is what
#: the graded benchmark supplies, and a model that has seen only one has to
#: generalise across both a punctuation change and a grammatical mood change at
#: evaluation time.
_REFERENCE_FORMS = (
    "Where can the <ref>{p}</ref> be found?",
    "Provide the bounding box for the <ref>{p}</ref>.",
    "Locate the {p}.",
    "The {p}.",
    "{p}",
    "Give the region containing the {p}.",
)


def render_reference(phrase: str, rng: random.Random) -> str:
    """One surface form for a referring target, chosen deterministically."""
    return rng.choice(_REFERENCE_FORMS).format(p=phrase)


def rareplanes_questions(tile: dict, rng: random.Random) -> list[dict]:
    """Grounding, presence and referring questions for one RarePlanes tile."""
    width = tile.get("width") or 512
    height = tile.get("height") or 512
    # Truncated aircraft carry a box that runs off the tile; asking the model to
    # reproduce it is asking for something not in the picture.
    #
    # The flag alone is not enough. Measured across all 18,393 annotations:
    # 4,756 are flagged and every one of those also touches an edge -- but 105
    # touch an edge WITHOUT being flagged. So the flag is a subset of the real
    # condition, and the geometry is the check that closes it.
    boxes = [
        b
        for b in tile["boxes"]
        if not b.get("truncated") and not _touches_edge(b["bbox"], width, height)
    ]
    if not boxes:
        return []

    out: list[dict] = []
    roles = Counter(b.get("role") for b in boxes)

    # A phrase that matches one object is a question; a phrase that matches
    # three is a trick. Only unique roles become referring targets.
    for box in boxes:
        role = box.get("role")
        phrase = ROLE_PHRASES.get(role)
        if not phrase or roles[role] != 1:
            continue
        out.append(
            {
                "task": "ground_reference",
                "question": render_reference(phrase, rng),
                "answer": coco_to_box(box["bbox"], width, height),
                "answer_type": "box",
                "detail": role,
            }
        )

    # Position disambiguates where class does not: with several aircraft of one
    # role, the one in a given corner is still a single object -- provided no
    # other shares that corner.
    quadrants = Counter(_quadrant(b["bbox"], width, height) for b in boxes)
    for box in boxes:
        key = _quadrant_key(box["bbox"], width, height)
        where = _quadrant(box["bbox"], width, height)
        if quadrants[where] != 1:
            continue
        out.append(
            {
                "task": "ground_reference",
                "question": render_reference(
                    f"aircraft at the {_position_phrase(key, rng)}", rng
                ),
                "answer": coco_to_box(box["bbox"], width, height),
                "answer_type": "box",
                "detail": where,
            }
        )

    # Point-prompted grounding: the centre is given, the extent is the answer.
    # Unambiguous by construction, so it needs no uniqueness check and is the
    # one grounding question every annotated tile can support.
    box = rng.choice(boxes)
    x, y, w, h = box["bbox"]
    out.append(
        {
            "task": "ground_point",
            "question": (
                "Create a bounding box enclosing the object located at "
                f"<point>({(x + w / 2) / width:.2f}, {(y + h / 2) / height:.2f})"
                "</point> in the image."
            ),
            "answer": coco_to_box(box["bbox"], width, height),
            "answer_type": "box",
            "detail": box.get("role"),
        }
    )

    # Presence, with adversarial negatives. Every tile holds aircraft, so "is
    # there an aircraft?" has one answer and cannot be asked. A *class* that is
    # plausibly present and absent is a real question -- POPE's adversarial
    # setting, and a harder negative than empty sky.
    present = {r for r in roles if r in ROLE_PHRASES}
    absent = [r for r in ROLE_PHRASES if r not in present]
    for role in ([rng.choice(sorted(present))] if present else []):
        out.append(
            {
                "task": "ground_presence",
                "question": f"Is there a {ROLE_PHRASES[role]} in this image?",
                "answer": "yes",
                "answer_type": "yesno",
                "detail": role,
            }
        )
    if absent:
        role = rng.choice(sorted(absent))
        out.append(
            {
                "task": "ground_presence",
                "question": f"Is there a {ROLE_PHRASES[role]} in this image?",
                "answer": "no",
                "answer_type": "yesno",
                "detail": role,
            }
        )
    return out


def sn6_questions(
    image_path: Path, label_path: Path, rng: random.Random
) -> list[dict]:
    """Building grounding questions for one SpaceNet 6 tile.

    SpaceNet 6 ships polygons in EPSG:32631 metres, so the tile's own
    geotransform is what turns them into pixels. That is a per-tile read of the
    GeoTIFF header -- not its pixels -- which is cheap and unavoidable: the
    label file alone cannot say where in the image a building sits.

    One object class means no class-absence negatives here, unlike RarePlanes.
    Disambiguation comes from size and position instead: "the largest building"
    is unique by construction, and a quadrant is unique when nothing shares it.
    """
    import rasterio
    from rasterio.transform import rowcol

    document = json.loads(label_path.read_text(encoding="utf-8"))
    features = document.get("features", [])
    if not features:
        return []

    with rasterio.open(image_path) as src:
        width, height, transform = src.width, src.height, src.transform

    buildings: list[dict] = []
    for feature in features:
        properties = feature.get("properties") or {}
        # Same reasoning as RarePlanes: a footprint clipped by the tile edge
        # has a true box the model cannot see.
        #
        # `partialDec` is a visible FRACTION, not a flag -- 1.0 means fully
        # visible, which is the good case. Reading it as truthy rejected every
        # building in the dataset and produced a silent zero-row source.
        # Confirmed against RarePlanes, where the two fields agree exactly:
        # 4,756 annotations are truncated AND below 1.0, and neither condition
        # ever occurs without the other.
        if properties.get("truncated"):
            continue
        visible = properties.get("partialDec")
        if visible is not None and float(visible) < 1.0:
            continue
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            continue
        ring = geometry["coordinates"][0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        top, left = rowcol(transform, min(xs), max(ys))
        bottom, right = rowcol(transform, max(xs), min(ys))
        if _touches_edge([left, top, right - left, bottom - top], width, height):
            continue
        buildings.append(
            {
                "box": box_string(
                    left / width, top / height, right / width, bottom / height
                ),
                "pixels": [left, top, right - left, bottom - top],
                "area": properties.get("origarea") or 0.0,
                "centre": ((left + right) / 2 / width, (top + bottom) / 2 / height),
            }
        )
    if not buildings:
        return []

    out: list[dict] = []

    # Point-prompted: unambiguous by construction, so every tile supports it.
    target = rng.choice(buildings)
    cx, cy = target["centre"]
    out.append(
        {
            "task": "ground_point",
            "question": (
                "Create a bounding box enclosing the building located at "
                f"<point>({cx:.2f}, {cy:.2f})</point> in the image."
            ),
            "answer": target["box"],
            "answer_type": "box",
            "detail": "point",
        }
    )

    # Superlatives are unique by definition -- but only worth asking when the
    # winner is clearly bigger than the runner-up. Two near-identical buildings
    # make "the largest" a coin toss the model cannot be blamed for losing.
    if len(buildings) >= 2:
        ranked = sorted(buildings, key=lambda b: b["area"], reverse=True)
        if ranked[0]["area"] >= 1.5 * max(ranked[1]["area"], 1e-6):
            out.append(
                {
                    "task": "ground_reference",
                    # "large"/"small" as well as the superlative: 35.3% of
                    # VRSBench referring carries a size or colour attribute, and
                    # "the large building" is the phrasing it actually uses.
                    "question": render_reference(
                        rng.choice(("largest building", "large building")), rng
                    ),
                    "answer": ranked[0]["box"],
                    "answer_type": "box",
                    "detail": "largest",
                }
            )

    # Position, where it identifies exactly one footprint.
    quadrants = Counter(
        _quadrant(b["pixels"], width, height) for b in buildings
    )
    for building in buildings:
        key = _quadrant_key(building["pixels"], width, height)
        where = _quadrant(building["pixels"], width, height)
        if quadrants[where] != 1:
            continue
        out.append(
            {
                "task": "ground_reference",
                "question": render_reference(
                    f"building at the {_position_phrase(key, rng)}", rng
                ),
                "answer": building["box"],
                "answer_type": "box",
                "detail": where,
            }
        )
    return out


def cmd_build(args) -> int:
    rng = random.Random(args.seed)
    rows: list[dict] = []

    if args.rareplanes:
        root = Path(args.rareplanes)
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        tiles = plan["tiles"]
        # Held out by source scene, not by tile: tiles cut from one WorldView-3
        # image share airport, aircraft and sun angle, so a tile-level split
        # puts near-duplicates on both sides of the boundary.
        scenes = sorted({t["file_name"].split("_tile_")[0] for t in tiles})
        rng.shuffle(scenes)
        holdout = set(scenes[: max(1, round(len(scenes) * args.val_fraction))])
        print(f"RarePlanes: {len(tiles)} tile(s), {len(scenes)} scene(s), "
              f"{len(holdout)} held out")

        caps: Counter = Counter()
        for tile in tiles:
            scene = tile["file_name"].split("_tile_")[0]
            split = "val" if scene in holdout else "train"
            cap = args.cap if split == "train" else max(1, args.cap // 5)
            image = str(root / "tiles" / tile["file_name"])
            for item in rareplanes_questions(tile, rng):
                key = (split, item["task"], item["answer_type"])
                if caps[key] >= cap:
                    continue
                caps[key] += 1
                digest = sha1(
                    f"{tile['file_name']}{item['question']}".encode()
                ).hexdigest()[:10]
                rows.append(
                    {
                        "sample_id": f"rp_{digest}",
                        "adapter": ADAPTER,
                        "task": item["task"],
                        "images": [str(Path(image).relative_to(args.image_root))],
                        "image_roles": ["scene"],
                        "modality": ["optical"],
                        "effective_gsd_m": [plan.get("gsd_m", 0.3)],
                        "question": item["question"],
                        "answer": item["answer"],
                        "answer_type": item["answer_type"],
                        "answer_options": ANSWER_OPTIONS[item["task"]],
                        "split": split,
                        "source": plan.get("source", "RarePlanes"),
                        "licence": plan.get("licence", "CC BY-SA 4.0"),
                        "scene": scene,
                        "detail": item.get("detail"),
                    }
                )

    if args.spacenet6:
        root = Path(args.spacenet6)
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        pairs = []
        for tile, modalities in plan["tiles"].items():
            image = root / "PS-RGB" / Path(modalities["PS-RGB"]).name
            label = root / "geojson_buildings" / Path(
                modalities["geojson_buildings"]
            ).name
            if image.exists() and label.exists():
                pairs.append((tile, image, label))
        # Rotterdam is one city, so there is no scene to hold out by -- the
        # split is by tile, and the honest caveat is that neighbouring tiles
        # share a street. Stated rather than hidden: SpaceNet 6's diversity
        # limit is the reason OpenEarthMap-SAR is in the plan at all.
        rng.shuffle(pairs)
        cut = max(1, round(len(pairs) * args.val_fraction))
        holdout = {t for t, _, _ in pairs[:cut]}
        print(f"SpaceNet 6: {len(pairs)} tile pair(s), {len(holdout)} held out")

        caps: Counter = Counter()
        for tile, image, label in pairs:
            split = "val" if tile in holdout else "train"
            cap = args.cap if split == "train" else max(1, args.cap // 5)
            for item in sn6_questions(image, label, rng):
                key = (split, item["task"], item["answer_type"])
                if caps[key] >= cap:
                    continue
                caps[key] += 1
                digest = sha1(f"{tile}{item['question']}".encode()).hexdigest()[:10]
                rows.append(
                    {
                        "sample_id": f"sn6_{digest}",
                        "adapter": ADAPTER,
                        "task": item["task"],
                        "images": [str(image.relative_to(args.image_root))],
                        "image_roles": ["scene"],
                        "modality": ["optical"],
                        "effective_gsd_m": [plan.get("gsd_m", 0.5)],
                        "question": item["question"],
                        "answer": item["answer"],
                        "answer_type": item["answer_type"],
                        "answer_options": ANSWER_OPTIONS[item["task"]],
                        "split": split,
                        "source": plan.get("source", "SpaceNet 6 / MSAW"),
                        "licence": plan.get("licence", "CC BY-SA 4.0"),
                        "scene": "AOI_11_Rotterdam",
                        "detail": item.get("detail"),
                    }
                )

    if not rows:
        raise SystemExit("no rows produced; pass at least one source root")

    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(rows)} row(s) -> {out}")
    print(f"\n{'task':20} {'rows':>6} {'sources':>8}  note")
    for task, count in sorted(Counter(r["task"] for r in rows).items()):
        answers = Counter(r["answer"] for r in rows if r["task"] == task)
        if ANSWER_OPTIONS.get(task) is None:
            # Boxes: the useful number is how often the SAME box answers a
            # question, since that is what a lazy model would learn to emit.
            _, top = answers.most_common(1)[0]
            note = f"{len(answers)} distinct boxes, most common {100 * top / count:.1f}%"
        else:
            _, top = answers.most_common(1)[0]
            note = f"blind ceiling {100 * top / count:.1f}%  " + ", ".join(
                f"{a}={n}" for a, n in answers.most_common(3)
            )
        print(f"{task:20} {count:6} {len(set(r['source'] for r in rows)):8}  {note}")
    print("  " + "  ".join(
        f"{k}={v}" for k, v in sorted(Counter(r["split"] for r in rows).items())
    ))
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "rows": len(rows),
                    "splits": dict(Counter(r["split"] for r in rows)),
                    "tasks": dict(Counter(r["task"] for r in rows)),
                    "sources": dict(Counter(r["source"] for r in rows)),
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    # Box scale per source. BEN regions and SpaceNet 6 buildings differ by two
    # orders of magnitude, and a single averaged accuracy would hide whichever
    # end the model fails at -- so the corpus reports the spread it contains.
    import re as _re
    import statistics as _st

    print(f"\n{'source':34} {'rows':>6} {'median box area':>16}")
    for source in sorted({r["source"] for r in rows}):
        areas = []
        for row in rows:
            if row["source"] != source or row["answer_type"] != "box":
                continue
            values = [float(v) for v in _re.findall(r"[\d.]+", row["answer"])]
            if len(values) == 4:
                areas.append((values[2] - values[0]) * (values[3] - values[1]))
        count = sum(1 for r in rows if r["source"] == source)
        median = f"{_st.median(areas):.2%}" if areas else "n/a"
        print(f"{source:34} {count:6} {median:>16}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rareplanes", default="", help="staged RarePlanes root")
    parser.add_argument("--spacenet6", default="", help="staged SpaceNet 6 root")
    parser.add_argument("--out", default="/data/manifests/rs_ground_caption_train.jsonl")
    parser.add_argument("--image-root", default="/data")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--summary-out",
        default="",
        help="stats file; a remote run needs an artifact or the wrapper calls "
        "the job a failure",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
