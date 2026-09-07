"""What does the base VLM score on VRSBench referring, with no fine-tuning?

    python scripts/eval_vlm_grounding.py \
        --benchmark /data/eval/vrsbench_val/VRSBench_EVAL_referring.json \
        --images    /data/eval/vrsbench_val/Images_val \
        --samples 600 --out logs/vlm_grounding_base.json

**The number nobody measured.** The whole G3 argument has been "fine-tune our
corpus" versus "bolt on Grounding DINO", and both were compared against each
other rather than against the model we already load. Qwen-VL has grounding in
its pretraining; the base checkpoint may already do this. If it does, the
detector and the selection stage are solving a problem we do not have. If it
does not, the detector route is established as necessary rather than assumed.

**Coordinate space is the trap here, and it was measured rather than reasoned
about.** Qwen answers on a **0-1000 grid**, not in pixels: its replies on 512 px
inputs run to exactly 1000. The first version of this script normalised by the
model's resized image size instead, which clamped every coordinate above 512 to
1.0, collapsed the box, and reported **75% "unparsable"** -- indistinguishable
from a model that cannot ground at all. Only dumping the raw replies showed the
answers were fine and the harness was wrong.

Two lessons worth keeping, because this is the third coordinate-convention bug
in this project: VRSBench references are 0-100, Qwen predictions are 0-1000, and
our own generator writes 0-1 floats. Never infer a scale from magnitude, and
always look at the raw output before trusting a score.

Scored with the repo's own ``grounding_accuracy`` and parsed by the repo's own
``format_box``, so this number is directly comparable to the adapter's whenever
one exists.
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

from satquery.evalcli.formatter import format_box  # noqa: E402
from satquery.training.metrics import grounding_accuracy  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_COORD = re.compile(r"<(\d+)>")

#: VRSBench references are 0-100 integers -- verified across all 64,636
#: coordinates in the referring split.
VRSBENCH_SCALE = 100.0

#: Qwen's grounding convention: ask for JSON, get pixel coordinates back. The
#: phrase is inserted verbatim, because the benchmark's phrasing *is* the task
#: -- rewriting it into our own template would measure a different question.
PROMPT = (
    "Locate {phrase} in this image. "
    "Respond with only the bounding box as JSON: "
    '[{{"bbox_2d": [x1, y1, x2, y2]}}]'
)


def reference_box(ground_truth: str):
    values = [int(v) for v in _COORD.findall(ground_truth)]
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = (v / VRSBENCH_SCALE for v in values)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="", help="optional LoRA to load on top")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--sampling",
        default="random",
        choices=["random", "stratified"],
        help="random matches the benchmark's class mix and is the only "
        "mode comparable to a published score; stratified is for per-class "
        "diagnosis on a small sample",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs/vlm_grounding.json")
    parser.add_argument("--dump", type=int, default=0, help="record N raw replies")
    parser.add_argument(
        "--dump-dir",
        default="",
        help="write annotated PNGs here -- reference in green, the model's "
        "answer in blue. A score says how often it was right; only the "
        "picture says whether a wrong answer was the wrong object or the "
        "right object loosely boxed, and those need different fixes.",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1000.0,
        help="divisor for the model's box coordinates. Qwen-VL uses 0-1000",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    images_root = Path(args.images)
    usable = [r for r in rows if reference_box(r.get("ground_truth", "")) is not None]

    # **Sampling mode changes what the number means, so it is explicit.**
    #
    # "random" reproduces the benchmark's own class mix and is the only mode
    # whose score is comparable to a published one. "stratified" gives every
    # class roughly equal weight, which is what you want to diagnose per-class
    # behaviour on a small sample -- and is badly wrong as a headline: measured
    # on this split, stratifying under-represents `vehicle` by 7x (28.1% of the
    # benchmark, 4.0% of the sample) while over-weighting classes with a handful
    # of rows. The first version of this harness stratified and reported the
    # result as the score, which flattered the model on its hardest class.
    rng = random.Random(args.seed)
    if args.sampling == "random":
        sample = rng.sample(usable, min(args.samples, len(usable)))
    else:
        by_class: dict[str, list] = collections.defaultdict(list)
        for row in usable:
            by_class[row.get("obj_cls", "?")].append(row)
        for bucket in by_class.values():
            rng.shuffle(bucket)
        sample, index = [], 0
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
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(
        f"{len(rows)} row(s); scoring {len(sample)} on {device}\n"
        f"model={args.model} adapter={args.adapter or 'none (base)'}",
        flush=True,
    )

    predicted, reference, replies = [], [], []
    per_type: dict[str, list] = collections.defaultdict(list)

    for position, row in enumerate(sample, 1):
        path = images_root / row["image_id"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.format(phrase=row["question"].rstrip(". "))},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        reply = processor.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Qwen answers on a 0-1000 grid, not in pixels. Measured directly from
        # its replies: values run to exactly 1000 ("[58, 840, 345, 1000]") on
        # 512 px inputs. Normalising by the image size instead clamps every
        # coordinate above 512 to 1.0 and collapses the box, which reads as the
        # model refusing to answer -- it was 75% "unparsable" until this was
        # measured rather than assumed.
        formatted = format_box(reply, scale=args.coord_scale)

        box = None
        if formatted.matched:
            parts = [float(v) for v in re.findall(r"\d+\.\d+", formatted.text)]
            if len(parts) == 4:
                box = tuple(parts)
        predicted.append(box)
        reference.append(reference_box(row["ground_truth"]))
        per_type[row.get("size_group") or "unspecified"].append(len(predicted) - 1)
        per_type[f"unique:{bool(row.get('unique'))}"].append(len(predicted) - 1)
        per_type[f"class:{row.get('obj_cls', '?')}"].append(len(predicted) - 1)
        if args.dump_dir and box and len(replies) < args.dump:
            from PIL import ImageDraw

            from satquery.training.metrics import box_iou

            width, height = image.size
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            px = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            draw.rectangle(px, outline=(50, 130, 255), width=3)
            ref = reference_box(row["ground_truth"])
            rx = [ref[0] * width, ref[1] * height, ref[2] * width, ref[3] * height]
            draw.rectangle(rx, outline=(0, 220, 90), width=3)
            iou = box_iou(box, ref)
            target = Path(args.dump_dir)
            target.mkdir(parents=True, exist_ok=True)
            canvas.save(target / f"{'HIT' if iou >= 0.5 else 'MISS'}_{iou:.2f}_{position}.png")

        if len(replies) < args.dump:
            replies.append(
                {
                    "question": row["question"],
                    "obj_cls": row.get("obj_cls"),
                    "reply": reply[:300],
                    "parsed": formatted.text,
                    "iou": round(
                        __import__("satquery.training.metrics", fromlist=["box_iou"]).box_iou(
                            box, reference_box(row["ground_truth"])
                        ), 4) if box else 0.0,
                    "image_id": row["image_id"],
                    "size_group": row.get("size_group") or "unspecified",
                    "unique": bool(row.get("unique")),
                    "position": position,
                    "basis": formatted.basis,
                    "coord_scale": args.coord_scale,
                }
            )

        if position % 50 == 0:
            partial = grounding_accuracy(predicted, reference)
            print(
                f"  {position}/{len(sample)}  acc@0.5 {100*partial['acc@0.5']:.1f}%  "
                f"unparsable {100*partial['unparsable']:.1f}%",
                flush=True,
            )

    overall = grounding_accuracy(predicted, reference)
    slices = {}
    for name, idx in sorted(per_type.items()):
        if not idx:
            continue
        slices[name] = {
            "n": len(idx),
            **{
                k: round(v, 4)
                for k, v in grounding_accuracy(
                    [predicted[i] for i in idx], [reference[i] for i in idx]
                ).items()
            },
        }

    report = {
        "model": args.model,
        "adapter": args.adapter or None,
        "sampling": args.sampling,
        "sampled": len(predicted),
        "overall": {k: round(v, 4) for k, v in overall.items()},
        "by_slice": slices,
        "replies": replies,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{'slice':26}{'n':>6}{'acc@0.5':>10}{'acc@0.7':>10}{'meanIoU':>10}{'unparsed':>10}")
    print(f"{'OVERALL':26}{len(predicted):6}{100*overall['acc@0.5']:9.1f}%"
          f"{100*overall['acc@0.7']:9.1f}%{overall['mean_iou']:10.3f}"
          f"{100*overall['unparsable']:9.1f}%")
    for name, v in slices.items():
        print(f"  {name:24}{v['n']:6}{100*v['acc@0.5']:9.1f}%{100*v['acc@0.7']:9.1f}%"
              f"{v['mean_iou']:10.3f}{100*v['unparsable']:9.1f}%")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
