"""Can a text-prompted detector even *see* what VRSBench asks about?

    python scripts/eval_grounding_recall.py \
        --benchmark /data/eval/vrsbench_val/VRSBench_EVAL_referring.json \
        --images    /data/eval/vrsbench_val/Images_val \
        --samples 600 --out logs/grounding_recall.json

**This measures recall, deliberately, and not accuracy.** The proposed design is
Grounding DINO proposes candidate boxes, then something -- positional rules, or
the VLM picking a numbered mark -- chooses which candidate the referring phrase
means. That second stage can only ever *lose* boxes. So the ceiling of the whole
design is: how often is the correct box present among the detector's top-N at
all? If recall@N is high the design is worth building; if it is low, no amount
of clever selection recovers it, and the honest move is to stop.

**Why this is the right first experiment.** It needs no VLM, no training and no
selection logic, so it cannot be confounded by any of them. One number decides
whether G3 grounding becomes a detector-plus-rules pipeline or stays a
fine-tuned adapter.

**The naive setup would fail for known reasons, so it is not the setup used.**
Both the Text2Seg paper and the ScienceDirect follow-up report that
off-the-shelf prompting on overhead imagery suffers scale drift and poor recall
on small objects -- and VRSBench is 73.6% small objects. The fixes they report
are synonym prompts and tiled inference. Synonyms are on by default here;
``--tiles`` adds the sliding window. Measuring the naive configuration and
concluding "zero-shot does not work" would be measuring the wrong thing.

Licence: Grounding DINO is Apache-2.0 (IDEA-Research), served through
``transformers``, which needs no custom CUDA extension. Nothing here trains.
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.training.metrics import box_iou  # noqa: E402

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"

#: VRSBench references are "{<25><40><33><60>}" on a 0-100 scale -- verified
#: across all 64,636 coordinates in the referring split (min 0, max 100).
_COORD = re.compile(r"<(\d+)>")
VRSBENCH_SCALE = 100.0

#: Synonym expansion per VRSBench class. The ScienceDirect study reports a
#: "synonym-based prompt strategy to maximize recall for overhead objects", and
#: the reason is visible in the class names: Grounding DINO was trained on
#: natural-image caption text, where nobody writes "ground-track-field". Each
#: entry is what a photo caption would plausibly call the thing seen from above.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "vehicle": ("vehicle", "car", "truck", "van", "parked car"),
    "ship": ("ship", "boat", "vessel", "cargo ship"),
    "airplane": ("airplane", "aircraft", "plane", "jet"),
    "helicopter": ("helicopter", "chopper"),
    "tennis-court": ("tennis court",),
    "basketball-court": ("basketball court",),
    "baseball-diamond": ("baseball field", "baseball diamond"),
    "soccer-ball-field": ("soccer field", "football pitch"),
    "ground-track-field": ("running track", "athletics track", "stadium track"),
    "golffield": ("golf course", "golf green"),
    "swimming-pool": ("swimming pool", "pool"),
    "harbor": ("harbor", "harbour", "dock", "pier", "marina"),
    "bridge": ("bridge",),
    "overpass": ("overpass", "highway overpass", "flyover"),
    "roundabout": ("roundabout", "traffic circle"),
    "storage-tank": ("storage tank", "oil tank", "silo", "round tank"),
    "chimney": ("chimney", "smokestack", "cooling tower"),
    "windmill": ("windmill", "wind turbine"),
    "stadium": ("stadium", "arena"),
    "airport": ("airport", "airfield", "runway"),
    "trainstation": ("train station", "railway station"),
    "dam": ("dam",),
    "container-crane": ("container crane", "port crane", "gantry crane"),
    "expressway-service-area": ("service area", "rest stop"),
    "expressway-toll-station": ("toll station", "toll booth"),
    "helipad": ("helipad", "helicopter pad"),
}


def reference_box(ground_truth: str) -> tuple[float, float, float, float] | None:
    """VRSBench's ``{<x1><y1><x2><y2>}`` as a normalised ``(x1,y1,x2,y2)``."""
    values = [int(v) for v in _COORD.findall(ground_truth)]
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = (v / VRSBENCH_SCALE for v in values)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # A zero-area reference is unscoreable -- IoU against it is 0 for every
    # prediction including a perfect one. 13 of 16,159 rows carry one.
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def prompt_for(obj_cls: str, use_synonyms: bool) -> str:
    """Grounding DINO wants lowercase phrases separated by ' . '."""
    if not use_synonyms:
        return obj_cls.replace("-", " ").lower() + " ."
    words = SYNONYMS.get(obj_cls, (obj_cls.replace("-", " "),))
    return " . ".join(w.lower() for w in words) + " ."


def _tile_boxes(width: int, height: int, tiles: int, overlap: float):
    """Sliding window origins. ``tiles=1`` is the whole image."""
    if tiles <= 1:
        yield (0, 0, width, height)
        return
    step_x, step_y = width / tiles, height / tiles
    pad_x, pad_y = step_x * overlap, step_y * overlap
    for row in range(tiles):
        for col in range(tiles):
            left = max(0, int(col * step_x - pad_x))
            top = max(0, int(row * step_y - pad_y))
            right = min(width, int((col + 1) * step_x + pad_x))
            bottom = min(height, int((row + 1) * step_y + pad_y))
            if right > left and bottom > top:
                yield (left, top, right, bottom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument(
        "--tiles",
        type=int,
        default=1,
        help="sliding-window grid per side; 2 means a 2x2 grid. 1 = whole image",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--no-synonyms", action="store_true")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs/grounding_recall.json")
    parser.add_argument(
        "--dump-dir",
        default="",
        help="write annotated PNGs here -- reference box in green, top "
        "candidates in orange. A recall number says how often the box was "
        "found; only the picture says whether the misses are near-misses or "
        "the detector looking at something else entirely.",
    )
    parser.add_argument("--dump-count", type=int, default=0)
    parser.add_argument("--dump-top", type=int, default=5)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    images_root = Path(args.images)

    usable = [r for r in rows if reference_box(r.get("ground_truth", "")) is not None]
    dropped = len(rows) - len(usable)

    # Stratify by class so a 600-row sample still says something about
    # storage-tank, not just about vehicle (28% of the split).
    by_class: dict[str, list] = collections.defaultdict(list)
    for row in usable:
        by_class[row.get("obj_cls", "?")].append(row)
    rng = random.Random(args.seed)
    for bucket in by_class.values():
        rng.shuffle(bucket)

    sample: list = []
    index = 0
    while len(sample) < min(args.samples, len(usable)):
        added = False
        for name in sorted(by_class):
            if index < len(by_class[name]) and len(sample) < args.samples:
                sample.append(by_class[name][index])
                added = True
        if not added:
            break
        index += 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    model.eval()
    print(
        f"{len(rows)} row(s); {dropped} degenerate reference(s) dropped; "
        f"scoring {len(sample)} sampled row(s) on {device}\n"
        f"model={args.model} tiles={args.tiles} synonyms={not args.no_synonyms}",
        flush=True,
    )

    # Deduplicated: a repeated k double-counts its hits and reports a recall
    # above 1.0, which is how this bug announced itself on the first run.
    ks = tuple(sorted({k for k in (1, 3, 5, 10, args.topk) if k <= args.topk}))
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    best_ious: list[float] = []
    dumped: list[dict] = []
    missing_images = 0

    for position, row in enumerate(sample, 1):
        path = images_root / row["image_id"]
        if not path.exists():
            missing_images += 1
            continue
        reference = reference_box(row["ground_truth"])
        image = Image.open(path).convert("RGB")
        width, height = image.size
        text = prompt_for(row.get("obj_cls", ""), not args.no_synonyms)

        candidates: list[tuple[float, tuple[float, float, float, float]]] = []
        for left, top, right, bottom in _tile_boxes(
            width, height, args.tiles, args.tile_overlap
        ):
            crop = image.crop((left, top, right, bottom))
            inputs = processor(images=crop, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[crop.size[::-1]],
            )[0]
            scores_and_boxes = zip(
                result["scores"].tolist(), result["boxes"].tolist(), strict=False
            )
            for score, box in scores_and_boxes:
                # Tile pixels -> full-image pixels -> normalised, so every
                # candidate lands in the reference's coordinate space.
                x1, y1, x2, y2 = box
                candidates.append(
                    (
                        float(score),
                        (
                            (x1 + left) / width,
                            (y1 + top) / height,
                            (x2 + left) / width,
                            (y2 + top) / height,
                        ),
                    )
                )

        candidates.sort(key=lambda c: c[0], reverse=True)
        ious = [box_iou(box, reference) for _, box in candidates[: args.topk]]
        best_ious.append(max(ious, default=0.0))

        group = row.get("size_group") or "unspecified"
        keys = ("all", f"class:{row.get('obj_cls','?')}", f"size:{group}",
                f"unique:{bool(row.get('unique'))}")
        for key in keys:
            stats[key]["n"] += 1
            stats[key]["candidates"] += len(candidates)
            for k in ks:
                if any(i >= args.iou for i in ious[:k]):
                    stats[key][f"hit@{k}"] += 1

        if args.dump_dir and len(dumped) < args.dump_count:
            from PIL import ImageDraw

            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            # Candidates first so the reference box is never hidden under one.
            for rank, (score, box) in enumerate(candidates[: args.dump_top], 1):
                x1, y1, x2, y2 = (box[0] * width, box[1] * height,
                                  box[2] * width, box[3] * height)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
                draw.text((x1 + 3, max(0, y1 - 12)), f"{rank} {score:.2f}",
                          fill=(255, 140, 0))
            rx1, ry1, rx2, ry2 = (reference[0] * width, reference[1] * height,
                                  reference[2] * width, reference[3] * height)
            draw.rectangle([rx1, ry1, rx2, ry2], outline=(0, 220, 90), width=3)
            best = max(ious, default=0.0)
            name = f"{'HIT' if best >= args.iou else 'MISS'}_{best:.2f}_{position}.png"
            target = Path(args.dump_dir)
            target.mkdir(parents=True, exist_ok=True)
            canvas.save(target / name)
            dumped.append(
                {
                    "file": name,
                    "image_id": row["image_id"],
                    "question": row.get("question", ""),
                    "obj_cls": row.get("obj_cls", ""),
                    "size_group": row.get("size_group") or "unspecified",
                    "unique": bool(row.get("unique")),
                    "prompt": text,
                    "candidates": len(candidates),
                    "best_iou": round(best, 4),
                    "hit": bool(best >= args.iou),
                }
            )

        if position % 50 == 0:
            hit = stats["all"][f"hit@{args.topk}"] / max(1, stats["all"]["n"])
            print(f"  {position}/{len(sample)}  recall@{args.topk} so far {hit:.3f}",
                  flush=True)

    def summarise(counter: collections.Counter) -> dict:
        n = counter["n"] or 1
        out = {"n": counter["n"], "mean_candidates": round(counter["candidates"] / n, 1)}
        for k in ks:
            out[f"recall@{k}"] = round(counter[f"hit@{k}"] / n, 4)
        return out

    report = {
        "model": args.model,
        "iou_threshold": args.iou,
        "topk": args.topk,
        "tiles": args.tiles,
        "synonyms": not args.no_synonyms,
        "box_threshold": args.box_threshold,
        "sampled": len(sample),
        "missing_images": missing_images,
        "degenerate_references_dropped": dropped,
        "mean_best_iou": round(sum(best_ious) / max(1, len(best_ious)), 4),
        "overall": summarise(stats["all"]),
        "by_size": {k[5:]: summarise(v) for k, v in sorted(stats.items())
                    if k.startswith("size:")},
        "by_unique": {k[7:]: summarise(v) for k, v in sorted(stats.items())
                      if k.startswith("unique:")},
        "by_class": {k[6:]: summarise(v) for k, v in sorted(stats.items())
                     if k.startswith("class:")},
        "dumped": dumped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{'':22}{'n':>6}{'cands':>8}" + "".join(f"{'R@'+str(k):>9}" for k in ks))
    for label, key in (("OVERALL", "all"),):
        v = summarise(stats[key])
        print(f"{label:22}{v['n']:6}{v['mean_candidates']:8.1f}"
              + "".join(f"{100*v['recall@'+str(k)]:8.1f}%" for k in ks))
    for prefix, title in (("size:", "size"), ("unique:", "unique")):
        for key in sorted(k for k in stats if k.startswith(prefix)):
            v = summarise(stats[key])
            print(f"  {title} {key[len(prefix):]:15}{v['n']:6}{v['mean_candidates']:8.1f}"
                  + "".join(f"{100*v['recall@'+str(k)]:8.1f}%" for k in ks))
    print(f"\nmean best IoU among top-{args.topk}: {report['mean_best_iou']:.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
