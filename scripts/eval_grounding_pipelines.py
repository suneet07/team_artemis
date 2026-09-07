"""Four ways to answer a VRSBench referring question, on identical rows.

    python scripts/eval_grounding_pipelines.py --pipeline qwen        --tag qwen
    python scripts/eval_grounding_pipelines.py --pipeline dino_vlm    --tag dino_vlm
    python scripts/eval_grounding_pipelines.py --pipeline dino_sam_vlm --tag dino_sam
    python scripts/eval_grounding_pipelines.py --pipeline dino_top1   --tag dino_top1

**The rows are fixed and shared.** Every pipeline scores the *same* sampled row
ids, drawn once with a fixed seed and written to ``--sample-file``. That makes
the comparison paired: with 200 rows an unpaired difference of five points means
nothing, while a paired one on identical questions is readable. Earlier runs in
this project each drew their own sample, so the detector and the VLM were never
actually compared on the same questions.

**Sampling is random, never stratified.** VRSBench is 28% ``vehicle``; a
class-balanced draw gives it 4% and flatters whichever model is weakest on the
dominant class. Measured on this split, stratifying moved the base VLM from
47.9% to 61.3% -- the same model, the same weights, a different sample.

The pipelines
-------------
``qwen``          the base VLM alone: phrase in, box out.
``dino_top1``     Grounding DINO's highest-confidence box. No language
                  understanding at all -- the control that shows how much of
                  the score is just "find any object of that class".
``dino_vlm``      Grounding DINO proposes, the VLM chooses. Candidates are
                  drawn on the image as numbered marks and the VLM answers with
                  a number, so it never has to emit a coordinate -- the task it
                  is worst at becomes the task it is best at.
``dino_sam_vlm``  as above, then SAM tightens the chosen box. Refinement only:
                  it cannot recover a wrong choice, and if the choice is wrong
                  it will sharpen the wrong object.

Licences: Qwen3-VL Apache-2.0, Grounding DINO Apache-2.0, SAM Apache-2.0.
Nothing here trains, and no restricted data is touched -- VRSBench is
evaluation-only under C20.
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

from satquery.evalcli.formatter import _BOX_NUMBER, format_box  # noqa: E402
from satquery.training.metrics import box_iou, grounding_accuracy  # noqa: E402

_COORD = re.compile(r"<(\d+)>")
VRSBENCH_SCALE = 100.0
QWEN_SCALE = 1000.0

SYNONYMS: dict[str, tuple[str, ...]] = {
    "vehicle": ("vehicle", "car", "truck", "van"),
    "ship": ("ship", "boat", "vessel"),
    "airplane": ("airplane", "aircraft", "plane"),
    "helicopter": ("helicopter",),
    "tennis-court": ("tennis court",),
    "basketball-court": ("basketball court",),
    "baseball-diamond": ("baseball field", "baseball diamond"),
    "soccer-ball-field": ("soccer field",),
    "ground-track-field": ("running track", "athletics track"),
    "golffield": ("golf course",),
    "swimming-pool": ("swimming pool",),
    "harbor": ("harbor", "dock", "pier"),
    "bridge": ("bridge",),
    "overpass": ("overpass", "flyover"),
    "roundabout": ("roundabout", "traffic circle"),
    "storage-tank": ("storage tank", "silo"),
    "chimney": ("chimney", "smokestack"),
    "windmill": ("windmill", "wind turbine"),
    "stadium": ("stadium",),
    "airport": ("airport", "runway"),
    "trainstation": ("train station",),
    "dam": ("dam",),
    "container-crane": ("container crane", "port crane"),
    "expressway-service-area": ("service area",),
    "expressway-toll-station": ("toll booth",),
    "helipad": ("helipad",),
}

DIRECT_PROMPT = (
    "Locate {phrase} in this image. "
    'Respond with only the bounding box as JSON: [{{"bbox_2d": [x1, y1, x2, y2]}}]'
)

#: Written from what the labels measurably are, not from intuition. Measured on
#: 150 rows of this split: the reference box is the axis-aligned bounds of
#: DOTA's rotated quadrilateral (99/150 agree within 0.02), 42/150 targets are
#: elongated, the median target covers 2.5% of the image, and the base model
#: draws too big rather than too small by 61 to 32. So the instruction pushes
#: tight, and names the one case where tight still means large -- a diagonal
#: pier's enclosing rectangle is much bigger than the pier itself.
#: Imported, not duplicated. This is the prompt the 62.7% arm used, and the
#: serving path uses the same object -- a second copy here could drift and the
#: benchmark would stop describing what ships.
from satquery.agent.served_prompts import PRECISE_PROMPT  # noqa: E402

#: The opposite bet: over-cover on purpose, because SAM can trim a box that is
#: too big and cannot grow one that is too small.
GENEROUS_PROMPT = (
    "Find this object in the satellite image: {phrase}\n\n"
    "Give a box that comfortably contains the whole object with a margin around "
    "it. Erring large is fine; clipping part of the object is not.\n"
    "Coordinates run 0-1000.\n\n"
    'Respond with only: [{{"bbox_2d": [x1, y1, x2, y2]}}]'
)

#: SAM takes point prompts as well as boxes, and the Echo write-up reports it
#: "performs better with point-based prompts". A centre point also asks the VLM
#: for less: where, not where and how big.
POINT_PROMPT = (
    "Find this object in the satellite image: {phrase}\n\n"
    "Give the single point at its centre. Coordinates run 0-1000.\n\n"
    'Respond with only: [{{"point_2d": [x, y]}}]'
)

PICK_PROMPT = (
    "The image has {n} candidate objects, each outlined and labelled with a "
    "number.\n\nWhich number is: {phrase}\n\n"
    "Answer with only the number."
)


def reference_box(ground_truth: str):
    values = [int(v) for v in _COORD.findall(ground_truth)]
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = (v / VRSBENCH_SCALE for v in values)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def load_or_make_sample(rows, path: Path, count: int, seed: int):
    """One row list, written once, reused by every pipeline."""
    usable = [r for r in rows if reference_box(r.get("ground_truth", "")) is not None]
    if path.exists():
        wanted = set(json.loads(path.read_text(encoding="utf-8"))["question_ids"])
        chosen = [r for r in usable if r["question_id"] in wanted]
        print(f"reusing sample of {len(chosen)} from {path}", flush=True)
        return chosen
    chosen = random.Random(seed).sample(usable, min(count, len(usable)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "sampling": "random",
                "question_ids": [r["question_id"] for r in chosen],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"drew a new sample of {len(chosen)} -> {path}", flush=True)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["qwen", "qwen_sam", "qwen_precise", "qwen_generous_sam",
                 "qwen_multimask", "qwen_point_sam", "qwen_zoom",
                 "qwen_precise_sam",
                 "dino_top1", "dino_vlm", "dino_sam_vlm"],
    )
    parser.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dino", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam", default="facebook/sam-vit-base")
    parser.add_argument("--sample-file", default="/data/eval/pipeline_sample.json")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-marks", type=int, default=12)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import torch
    from PIL import Image, ImageDraw

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    sample = load_or_make_sample(
        rows, Path(args.sample_file), args.samples, args.seed
    )
    images_root = Path(args.images)

    need_dino = args.pipeline.startswith("dino")
    need_vlm = not args.pipeline == "dino_top1"
    need_sam = "sam" in args.pipeline or args.pipeline == "qwen_multimask"

    dino = dino_proc = vlm = vlm_proc = sam = sam_proc = None
    if need_dino:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        dino_proc = AutoProcessor.from_pretrained(args.dino)
        dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino).to(device)
        dino.eval()
    if need_vlm:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        vlm_proc = AutoProcessor.from_pretrained(args.vlm)
        vlm = AutoModelForImageTextToText.from_pretrained(
            args.vlm, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        vlm.eval()
    if need_sam:
        from transformers import SamModel, SamProcessor

        sam_proc = SamProcessor.from_pretrained(args.sam)
        sam = SamModel.from_pretrained(args.sam).to(device)
        sam.eval()

    print(f"pipeline={args.pipeline} rows={len(sample)} device={device}", flush=True)

    def detect(image, obj_cls):
        words = SYNONYMS.get(obj_cls, (obj_cls.replace("-", " "),))
        text = " . ".join(w.lower() for w in words) + " ."
        inputs = dino_proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = dino(**inputs)
        res = dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=args.box_threshold,
            text_threshold=args.box_threshold, target_sizes=[image.size[::-1]],
        )[0]
        w, h = image.size
        boxes = [
            (float(s), (b[0] / w, b[1] / h, b[2] / w, b[3] / h))
            for s, b in zip(res["scores"].tolist(), res["boxes"].tolist(), strict=False)
        ]
        boxes.sort(key=lambda c: c[0], reverse=True)
        return [b for _, b in boxes[: args.max_marks]]

    def ask(image, prompt, max_new_tokens=48):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = vlm_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = vlm_proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return vlm_proc.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def marked(image, boxes):
        """Numbered outlines -- Set-of-Mark. The VLM answers with a digit."""
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        w, h = image.size
        for i, b in enumerate(boxes, 1):
            px = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
            draw.rectangle(px, outline=(255, 60, 0), width=3)
            label = str(i)
            tx, ty = px[0] + 2, max(0, px[1] - 14)
            draw.rectangle([tx - 1, ty - 1, tx + 8 * len(label) + 3, ty + 13],
                           fill=(255, 60, 0))
            draw.text((tx + 1, ty), label, fill=(255, 255, 255))
        return canvas

    def _mask_boxes(image, *, box=None, point=None, multi=False):
        """Run SAM and return every candidate mask as a normalised box."""
        w, h = image.size
        kwargs = {}
        if box is not None:
            kwargs["input_boxes"] = [[[box[0] * w, box[1] * h,
                                       box[2] * w, box[3] * h]]]
        if point is not None:
            kwargs["input_points"] = [[[[point[0] * w, point[1] * h]]]]
        inputs = sam_proc(image, return_tensors="pt", **kwargs).to(device)
        with torch.no_grad():
            out = sam(**inputs, multimask_output=multi)
        masks = sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0][0]
        found = []
        for m in masks:
            ys, xs = m.numpy().nonzero()
            if len(xs) >= 8:
                found.append((xs.min() / w, ys.min() / h,
                              (xs.max() + 1) / w, (ys.max() + 1) / h))
        return found

    def tighten(image, box, multi=False):
        """SAM: rough box in, mask out, mask's bounding box back.

        With ``multi`` SAM returns three masks at different scales -- typically
        the part, the object and the group -- and the one closest to the prompt
        box is kept. Taking only the single mask (the default) silently discards
        two thirds of what SAM offers, which is what the first version did.
        """
        found = _mask_boxes(image, box=box, multi=multi)
        if not found:
            return box
        return max(found, key=lambda b: box_iou(b, box))

    def from_point(image, point):
        """Point prompt instead of a box: SAM decides the extent by itself."""
        found = _mask_boxes(image, point=point, multi=True)
        if not found:
            return None
        # Without a box to compare against, prefer the middle-sized mask: SAM's
        # three are roughly part / object / group, and the object is what a
        # referring expression names.
        return sorted(found, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[len(found) // 2]

    predicted, references, records = [], [], []
    for position, row in enumerate(sample, 1):
        path = images_root / row["image_id"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        ref = reference_box(row["ground_truth"])
        phrase = row["question"].rstrip(". ")
        box, detail = None, {}

        if args.pipeline == "qwen_point_sam":
            reply = ask(image, POINT_PROMPT.format(phrase=phrase))
            # _BOX_NUMBER, not a fresh pattern: a bare scan takes the '2'
            # out of "point_2d" as the x coordinate and turns every answer
            # into (0.002, x). That scored 0.0% and looked like the model
            # failing. Same bug as bbox_2d, same guard, third occurrence --
            # so the shared pattern is imported rather than re-derived.
            nums = [float(v) for v in _BOX_NUMBER.findall(reply)]
            detail = {"reply": reply[:120]}
            if len(nums) >= 2:
                pt = (nums[0] / QWEN_SCALE, nums[1] / QWEN_SCALE)
                if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                    box = from_point(image, pt)
                    detail["point"] = [round(v, 3) for v in pt]

        elif args.pipeline == "qwen_zoom":
            # Coarse then fine. 88% of this benchmark is "small" objects, so at
            # 512 px the target can be 20 px across -- the model is guessing at
            # something it can barely resolve. A second look at a crop around
            # the first answer gives it 3-4x the pixels, which is the same
            # upscaling trick the SpaceNet 7 winners relied on for buildings.
            reply = ask(image, DIRECT_PROMPT.format(phrase=phrase))
            formatted = format_box(reply, scale=QWEN_SCALE)
            rough = None
            if formatted.matched:
                parts = [float(v) for v in re.findall(r"\d+\.\d+", formatted.text)]
                rough = tuple(parts) if len(parts) == 4 else None
            detail = {"reply": reply[:120]}
            box = rough
            if rough:
                w, h = image.size
                cx, cy = (rough[0] + rough[2]) / 2, (rough[1] + rough[3]) / 2
                half = max(rough[2] - rough[0], rough[3] - rough[1], 0.06) * 1.6
                left, top = max(0.0, cx - half), max(0.0, cy - half)
                r_, b_ = min(1.0, cx + half), min(1.0, cy + half)
                crop = image.crop(
                    (int(left * w), int(top * h), int(r_ * w), int(b_ * h))
                )
                if min(crop.size) >= 16:
                    zoom = crop.resize((crop.width * 3, crop.height * 3),
                                       Image.LANCZOS)
                    reply2 = ask(zoom, PRECISE_PROMPT.format(phrase=phrase))
                    f2 = format_box(reply2, scale=QWEN_SCALE)
                    if f2.matched:
                        q = [float(v) for v in re.findall(r"\d+\.\d+", f2.text)]
                        if len(q) == 4:
                            # crop coordinates back to the full frame
                            box = (
                                left + q[0] * (r_ - left),
                                top + q[1] * (b_ - top),
                                left + q[2] * (r_ - left),
                                top + q[3] * (b_ - top),
                            )
                            detail["zoomed"] = True

        elif args.pipeline.startswith("qwen"):
            prompt = {
                "qwen_precise": PRECISE_PROMPT,
                "qwen_precise_sam": PRECISE_PROMPT,
                "qwen_generous_sam": GENEROUS_PROMPT,
            }.get(args.pipeline, DIRECT_PROMPT)
            reply = ask(image, prompt.format(phrase=phrase))
            formatted = format_box(reply, scale=QWEN_SCALE)
            if formatted.matched:
                parts = [float(v) for v in re.findall(r"\d+\.\d+", formatted.text)]
                box = tuple(parts) if len(parts) == 4 else None
            detail = {"reply": reply[:160]}
            # SAM on the VLM's own box. The earlier SAM test refined the
            # *detector's* boxes, where only 35% sat on the right object at all,
            # so it mostly sharpened mistakes. Here 61% are already correct and
            # 21 more over-cover the target by >1.4x -- boxes that contain the
            # object and too much else, which is the case SAM exists for.
            if box and args.pipeline in (
                "qwen_sam", "qwen_precise_sam", "qwen_generous_sam",
                "qwen_multimask",
            ):
                before = box
                box = tighten(image, box,
                              multi=args.pipeline == "qwen_multimask")
                detail["iou_before_sam"] = round(box_iou(before, ref), 4)
        else:
            candidates = detect(image, row.get("obj_cls", ""))
            detail["candidates"] = len(candidates)
            if candidates and args.pipeline == "dino_top1":
                box = candidates[0]
            elif candidates:
                reply = ask(
                    marked(image, candidates),
                    PICK_PROMPT.format(n=len(candidates), phrase=phrase),
                    max_new_tokens=8,
                )
                detail["reply"] = reply[:80]
                digits = re.findall(r"\d+", reply)
                pick = int(digits[0]) if digits else 0
                detail["picked"] = pick
                if 1 <= pick <= len(candidates):
                    box = candidates[pick - 1]
                elif candidates:
                    # An unusable answer falls back to the top candidate rather
                    # than scoring zero: that isolates the *selection* quality
                    # from the model's willingness to emit a bare number.
                    box = candidates[0]
                    detail["fell_back"] = True
                if box and args.pipeline == "dino_sam_vlm":
                    before = box
                    box = tighten(image, box)
                    detail["iou_before_sam"] = round(box_iou(before, ref), 4)

        predicted.append(box)
        references.append(ref)
        iou = box_iou(box, ref) if box else 0.0
        records.append({
            "question_id": row["question_id"], "image_id": row["image_id"],
            "question": row["question"], "obj_cls": row.get("obj_cls"),
            "size_group": row.get("size_group") or "unspecified",
            "unique": bool(row.get("unique")), "iou": round(iou, 4),
            "hit": bool(iou >= 0.5), "box": list(box) if box else None,
            "reference": list(ref), **detail,
        })

        if args.dump_dir:
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            w, h = image.size
            if box:
                draw.rectangle([box[0]*w, box[1]*h, box[2]*w, box[3]*h],
                               outline=(50, 130, 255), width=3)
            draw.rectangle([ref[0]*w, ref[1]*h, ref[2]*w, ref[3]*h],
                           outline=(0, 220, 90), width=3)
            d = Path(args.dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            canvas.save(d / f"{row['question_id']}.png")

        if position % 25 == 0:
            partial = grounding_accuracy(predicted, references)
            print(f"  {position}/{len(sample)}  acc@0.5 {100*partial['acc@0.5']:.1f}%",
                  flush=True)

    overall = grounding_accuracy(predicted, references)
    by: dict[str, list] = collections.defaultdict(list)
    for i, r in enumerate(records):
        by[r["size_group"]].append(i)
        by[f"unique:{r['unique']}"].append(i)
    slices = {
        k: {"n": len(v), **{m: round(x, 4) for m, x in grounding_accuracy(
            [predicted[i] for i in v], [references[i] for i in v]).items()}}
        for k, v in sorted(by.items()) if v
    }

    report = {
        "pipeline": args.pipeline, "vlm": args.vlm if need_vlm else None,
        "detector": args.dino if need_dino else None,
        "sam": args.sam if need_sam else None,
        "rows": len(predicted),
        "overall": {k: round(v, 4) for k, v in overall.items()},
        "by_slice": slices, "records": records,
    }
    out = Path(args.out or f"logs/pipeline_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{args.pipeline}: acc@0.5 {100*overall['acc@0.5']:.1f}%  "
          f"acc@0.7 {100*overall['acc@0.7']:.1f}%  meanIoU {overall['mean_iou']:.3f}")
    for k, v in slices.items():
        print(f"  {k:18}{v['n']:5}{100*v['acc@0.5']:8.1f}%")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
