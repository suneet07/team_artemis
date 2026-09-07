"""Stage the RarePlanes real split from AWS Open Data into the Volume.

    python scripts/stage_rareplanes.py plan  --out /data/eval/rareplanes --tiles 4000
    python scripts/stage_rareplanes.py fetch --plan /data/eval/rareplanes/plan.json

**Why this source exists in the mix.** ``rs_ground_caption`` is scored on how
well a predicted box overlaps a real one, and its primary source -- BEN.txt --
carries only CORINE land-cover regions. Measured on the 19,996 staged reference
boxes: the median covers 19.8% of the image and *not one* is under 1%. A model
trained on that alone learns to draw large boxes, which scores well on BEN and
near zero on a 30 cm aircraft. RarePlanes is the object half: 14,700 hand-
annotated real aircraft, and at 30 cm it is the closest GSD match to Cartosat-2S
anywhere in the inventory.

**The tiled variant, not the full scenes.** ``PS-RGB`` holds whole WorldView-3
scenes; ``PS-RGB_tiled`` holds them cut into tiles with
``geojson_aircraft_tiled`` and a COCO annotation file to match. Tiles are what
the model trains on, and taking them pre-cut means the box coordinates already
refer to the image the model sees rather than to a scene it never receives.

**There are no empty tiles, and that shapes the question design.** All 5,815
tiled images carry at least one aircraft, so "is there an aircraft here?" has
exactly one answer and cannot be asked. Negatives have to come from *class*
absence instead: 10,328 of the aircraft are small civil transports against 185
military fighters, so asking for a fighter over an apron of light aircraft is a
real question with a real "no". That is POPE's adversarial negative -- a
plausible co-occurring class that happens to be absent -- and it is a harder,
more useful negative than empty sky would have been.

**The attributes are the point, not just the boxes.** Each annotation carries
``role``, ``num_engines``, ``propulsion``, ``wingspan``, ``wing_type``,
``wing_position`` and ``faa_wingspan_class``. Those are what make a referring
expression discriminative -- "the twin-engine jet" picks one aircraft out of a
line of propeller singles, where "the aircraft" picks nothing. 3,243 tiles hold
two or more aircraft, which is the precondition for asking at all.

Licence: CC BY-SA 4.0, AWS Open Data Program. Imagery is Maxar WorldView-3 at
~30 cm, pan-sharpened RGB.
"""

import argparse
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUCKET = "https://rareplanes-public.s3.amazonaws.com"
TILE_PREFIX = "real/train/PS-RGB_tiled/"
COCO_KEY = "real/metadata_annotations/RarePlanes_Train_Coco_Annotations_tiled.json"

GSD_M = 0.30
SOURCE = "RarePlanes (real split)"
LICENCE = "CC BY-SA 4.0"


def _get(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _list_all(prefix: str) -> list[str]:
    """Every key under ``prefix``, paginated.

    Paginated and not capped at one page: the tile prefix holds 11,630 objects
    against S3's 1,000-per-response limit, so a single request would silently
    return 9% of the dataset and look like a complete listing.
    """
    keys: list[str] = []
    token: str | None = None
    while True:
        url = (
            f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        )
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        body = _get(url).decode("utf-8", "replace")
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", body))
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not match:
            return keys
        token = match.group(1)


def cmd_plan(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("listing tiles...", flush=True)
    keys = [k for k in _list_all(TILE_PREFIX) if k.lower().endswith(".png")]
    if not keys:
        raise SystemExit(f"no PNG tiles under {TILE_PREFIX}; the bucket layout changed")
    by_name = {Path(k).name: k for k in keys}
    print(f"{len(keys):,} tile(s) available", flush=True)

    print("fetching COCO annotations...", flush=True)
    coco = json.loads(_get(f"{BUCKET}/{urllib.parse.quote(COCO_KEY)}"))
    (out / "coco_train_tiled.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )

    # `categories` in this file is degenerate -- one entry per annotation rather
    # than a class vocabulary -- and every annotation's `category_id` is null.
    # The real class label is `role`, so read that and ignore the rest.
    boxes_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        boxes_by_image.setdefault(annotation["image_id"], []).append(annotation)

    # The COCO file indexes every tile the dataset defines; the bucket holds the
    # pixels. Only their intersection is usable, and reporting the gap matters:
    # a rename upstream would otherwise show up as a quietly smaller corpus.
    usable, missing = [], 0
    for image in coco.get("images", []):
        name = Path(image["file_name"]).name
        if name not in by_name:
            missing += 1
            continue
        annotations = boxes_by_image.get(image["id"], [])
        usable.append(
            {
                "key": by_name[name],
                "file_name": name,
                "width": image.get("width"),
                "height": image.get("height"),
                "aircraft": len(annotations),
                "boxes": [
                    {
                        "bbox": a["bbox"],  # COCO: [x, y, w, h] in pixels
                        "role": a.get("role"),
                        "engines": a.get("num_engines"),
                        "propulsion": a.get("propulsion"),
                        "wingspan": a.get("wingspan"),
                        "wing_type": a.get("wing_type"),
                        "wing_position": a.get("wing_position"),
                        "size_class": a.get("faa_wingspan_class"),
                        # Partly-visible aircraft make an unfair grounding
                        # target: the true box is clipped by the tile edge and
                        # the visible object does not match it.
                        "truncated": a.get("truncated"),
                        "location": a.get("location"),
                    }
                    for a in annotations
                ],
            }
        )
    print(f"{len(usable):,} tile(s) with annotations; {missing:,} indexed but absent")

    rng = random.Random(args.seed)
    positives = [t for t in usable if t["aircraft"] > 0]
    negatives = [t for t in usable if t["aircraft"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Positives lead: they are the only rows that can carry a grounding target,
    # and they are the scarcer half. Negatives are taken to a ratio rather than
    # in full, so "is there an aircraft?" has two answers without the corpus
    # becoming mostly empty sky.
    take_positive = positives if not args.tiles else positives[: args.tiles]
    take_negative = negatives[: int(len(take_positive) * args.negative_ratio)]
    chosen = take_positive + take_negative
    rng.shuffle(chosen)

    counts = Counter(b["role"] for t in chosen for b in t["boxes"])
    document = {
        "source": SOURCE,
        "licence": LICENCE,
        "gsd_m": GSD_M,
        "tiles": chosen,
        "counts": {
            "tiles": len(chosen),
            "with_aircraft": len(take_positive),
            "without_aircraft": len(take_negative),
            "aircraft_total": sum(t["aircraft"] for t in chosen),
        },
    }
    (out / "plan.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(
        f"\n{len(chosen):,} tile(s) planned "
        f"({len(take_positive):,} with aircraft, {len(take_negative):,} without), "
        f"{document['counts']['aircraft_total']:,} aircraft -> {out / 'plan.json'}"
    )
    for name, count in counts.most_common(8):
        print(f"  {name:28} {count:6}")
    return 0


def cmd_fetch(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.plan).parent
    target_dir = root / "tiles"
    target_dir.mkdir(parents=True, exist_ok=True)

    wanted = [(t["key"], target_dir / t["file_name"]) for t in plan["tiles"]]
    todo = [(k, p) for k, p in wanted if not p.exists()]
    print(f"{len(wanted) - len(todo)} present; {len(todo)} to fetch", flush=True)

    done = failed = 0
    for key, target in todo:
        try:
            payload = _get(f"{BUCKET}/{urllib.parse.quote(key)}")
        except Exception as error:  # noqa: BLE001 - one bad file is a result
            print(f"  FAILED {key}: {error!r}", flush=True)
            failed += 1
            continue
        # Through a temporary name: a half-written file already exists as far as
        # the resume check above is concerned, and would be skipped forever.
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        done += 1
        if done % 250 == 0:
            print(f"  {done}/{len(todo)} fetched", flush=True)

    summary = {
        "source": SOURCE,
        "licence": LICENCE,
        "gsd_m": GSD_M,
        "tiles_wanted": len(wanted),
        "tiles_present": sum(1 for _, p in wanted if p.exists()),
        "failed": failed,
    }
    (root / "fetch_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="choose tiles and pull the COCO annotations")
    plan.add_argument("--out", default="/data/eval/rareplanes")
    plan.add_argument(
        "--tiles", type=int, default=0, help="cap on annotated tiles; 0 = all"
    )
    plan.add_argument(
        "--negative-ratio",
        type=float,
        default=0.5,
        help="empty tiles per annotated tile, so presence questions have two answers",
    )
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(func=cmd_plan)

    fetch = sub.add_parser("fetch", help="pull the planned tiles, resumably")
    fetch.add_argument("--plan", default="/data/eval/rareplanes/plan.json")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
