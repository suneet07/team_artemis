"""Select an RSVQA-HR corpus and pull only the images it needs from the tar.

    # 1. choose images and questions (needs only the small JSON files)
    python scripts/stage_rsvqa_hr.py plan --images 3000 --max-per-image 24

    # 2. stream Images.tar and keep just those images
    python scripts/stage_rsvqa_hr.py fetch --plan data/eval/rsvqa_hr/plan.json

RSVQA-HR is **the only sub-metre imagery in the rs_vqa mix**. BEN.txt and
RSVQA-LR are both 10 m Sentinel-2; the hidden set is Cartosat-2S at well under a
metre. Without this, the adapter has never seen the resolution it is graded on.

**Resolution.** Native is 0.1524 m (recorded per image as ``res_x``). Section
5.2 downsamples 2x to ~0.30 m, which is Cartosat-scale, and the manifest records
``effective_gsd_m: 0.3048`` rather than the native value -- GSD-conditioned
prompting reads that field, so a manifest claiming 0.15 m tells the model a
resolution its pixels do not have.

**Two passes on purpose.** The questions live in small JSON files; the imagery
is a 12.9 GB tar holding a ``.tif`` *and* a redundant ``.png`` per image. Doing
the selection first means the tar is streamed once and only the chosen ``.tif``
members are decoded -- the same trick that made BigEarthNet 26x faster than
rebuilding it.

**Sampling.** Images are picked spread over the survey area using their recorded
map coordinates, not taken from the head of the file, and questions are capped
per image. RSVQA-HR carries ~100 questions per image; uncapped, a few thousand
images would supply hundreds of thousands of rows and the corpus would measure
annotation density rather than scene variety.
"""

import argparse
import io
import json
import random
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = Path("data/eval/rsvqa_hr")
IMAGES_TAR = "https://zenodo.org/records/6344367/files/Images.tar?download=1"
SOURCE = "RSVQA-HR (Zenodo 6344367)"
LICENCE = "CC-BY-4.0 (USGS HRO imagery, public domain)"

#: Native 0.1524 m, downsampled 2x by section 5.2. Recorded as the *effective*
#: value because that is what the model actually sees.
NATIVE_GSD_M = 0.1524
DOWNSAMPLE = 2
EFFECTIVE_GSD_M = NATIVE_GSD_M * DOWNSAMPLE

#: Spatial bins for stratification, in projected metres. ~50 km cells spread the
#: sample over the survey area instead of clustering it in one city.
GRID_M = 50_000.0


def _load(name: str, root: Path) -> list[dict]:
    path = root / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. The split files carry only `active` flags; the "
            "question and answer text is in USGSquestions.json and "
            "USGSanswers.json, which must be downloaded too."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[next(iter(payload))]


def choose_images(images: list[dict], active_ids: set[int], target: int) -> list[int]:
    """Pick image ids spread across the survey area, deterministically."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for image in images:
        if image["id"] not in active_ids:
            continue
        x, y = image.get("upperleft_map_x"), image.get("upperleft_map_y")
        cell = (
            (int(x // GRID_M), int(y // GRID_M))
            if x is not None and y is not None
            else ("unknown", "unknown")
        )
        buckets[cell].append(image["id"])

    for cell, ids in buckets.items():
        random.Random(f"rsvqa_hr:{cell}").shuffle(ids)

    chosen: list[int] = []
    order = sorted(buckets, key=str)
    index = 0
    while len(chosen) < target:
        progressed = False
        for cell in order:
            pool = buckets[cell]
            if index < len(pool):
                chosen.append(pool[index])
                progressed = True
                if len(chosen) >= target:
                    break
        if not progressed:
            break
        index += 1
    print(f"{len(chosen)} image(s) chosen from {len(buckets)} spatial cell(s)")
    return chosen


def cmd_plan(args) -> int:
    root = Path(args.root)
    questions = {q["id"]: q for q in _load("USGSquestions.json", root)}
    answers = {a["question_id"]: a for a in _load("USGSanswers.json", root)}
    images = _load("USGSimages.json", root)
    active = {
        row["id"]
        for row in _load(f"USGS_split_{args.split}_questions.json", root)
        if row.get("active")
    }
    if not active:
        raise SystemExit(f"no active questions in the {args.split!r} split")
    print(f"{len(active)} active question(s) in split={args.split}")

    active_images = {questions[q]["img_id"] for q in active if q in questions}
    chosen = set(choose_images(images, active_images, args.images))

    by_image: dict[int, list[dict]] = defaultdict(list)
    for qid in sorted(active):
        question = questions.get(qid)
        answer = answers.get(qid)
        if question is None or answer is None or question["img_id"] not in chosen:
            continue
        by_image[question["img_id"]].append(
            {
                "qid": qid,
                "question": question["question"],
                "answer": str(answer["answer"]),
                "question_type": question.get("type", "unknown"),
            }
        )

    rows = []
    for image_id, group in sorted(by_image.items()):
        rng = random.Random(f"rsvqa_hr:{args.split}:{image_id}")
        rng.shuffle(group)
        for pair in group[: args.max_per_image] if args.max_per_image else group:
            rows.append({"img_id": image_id, **pair})

    plan = {
        "split": args.split,
        "images": sorted({r["img_id"] for r in rows}),
        "rows": rows,
        "downsample": DOWNSAMPLE,
        "effective_gsd_m": EFFECTIVE_GSD_M,
    }
    out = Path(args.out or root / "plan.json")
    out.write_text(json.dumps(plan), encoding="utf-8")
    print(f"{len(rows)} row(s) over {len(plan['images'])} image(s) -> {out}")
    for name, count in Counter(r["question_type"] for r in rows).most_common():
        print(f"  {name:12} {count:7}")
    return 0


def cmd_fetch(args) -> int:
    from PIL import Image

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    out_root = Path(args.out)
    (out_root / "images").mkdir(parents=True, exist_ok=True)

    # Resume rather than restart. Streaming 12.9 GB takes the better part of an
    # hour, and an interrupted run that has to redo every image already on disk
    # turns one interruption into two hours.
    already = {
        int(path.stem)
        for path in (out_root / "images").glob("*.png")
        if path.stem.isdigit() and path.stat().st_size > 0
    }
    wanted = {int(i) for i in plan["images"]} - already
    if already:
        print(f"{len(already)} image(s) already present; {len(wanted)} to go", flush=True)
    if not wanted:
        print("every planned image is already extracted", flush=True)

    written = 0
    if wanted:
        print(f"streaming Images.tar for {len(wanted)} image(s)", flush=True)
    request = urllib.request.Request(args.url, headers={"User-Agent": "satquery/0"})
    if wanted:
        with urllib.request.urlopen(request) as raw:  # noqa: S310 - fixed URL
            with tarfile.open(fileobj=raw, mode="r|") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".tif"):
                        continue
                    try:
                        image_id = int(Path(member.name).stem)
                    except ValueError:
                        continue
                    if image_id not in wanted:
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    with Image.open(io.BytesIO(handle.read())) as source:
                        image = source.convert("RGB")
                        # LANCZOS, not nearest: this is a *resolution* change on
                        # optical imagery, where an averaged pixel is the honest
                        # representation of the coarser sensor being simulated.
                        target = (image.width // DOWNSAMPLE, image.height // DOWNSAMPLE)
                        image.resize(target, Image.LANCZOS).save(
                            out_root / "images" / f"{image_id}.png"
                        )
                    written += 1
                    if written % 200 == 0:
                        print(f"  {written}/{len(wanted)}", flush=True)
                    if written == len(wanted):
                        print("all wanted images found; stopping the stream", flush=True)
                        break

    rows = []
    for row in plan["rows"]:
        path = f"images/{row['img_id']}.png"
        if not (out_root / path).exists():
            continue
        rows.append(
            {
                "sample_id": f"rsvqa_hr_{row['qid']}",
                "adapter": "rs_vqa",
                "task": "single_vqa",
                "images": [path],
                "question": row["question"],
                "answer": row["answer"],
                "answer_type": "text",
                "question_type": row["question_type"],
                "image_roles": ["true_colour"],
                "modality": ["optical"],
                "effective_gsd_m": [plan["effective_gsd_m"]],
                "split": plan["split"],
                "source": SOURCE,
                "licence": LICENCE,
                "native_gsd_m": NATIVE_GSD_M,
                "downsample": plan["downsample"],
            }
        )

    out = out_root / f"rsvqa_hr_{plan['split']}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"{written} image(s), {len(rows)} row(s) -> {out}")
    return 0 if rows else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="choose images and questions; no imagery")
    plan.add_argument("--root", default=str(ROOT))
    plan.add_argument("--split", default="train", choices=("train", "val", "test", "test_phili"))
    plan.add_argument("--images", type=int, default=3000)
    plan.add_argument("--max-per-image", type=int, default=24)
    plan.add_argument("--out", default=None)
    plan.set_defaults(func=cmd_plan)

    fetch = sub.add_parser("fetch", help="stream the tar, keep the planned images")
    fetch.add_argument("--plan", default=str(ROOT / "plan.json"))
    fetch.add_argument("--out", default=str(ROOT))
    fetch.add_argument("--url", default=IMAGES_TAR)
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
