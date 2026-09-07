"""Florence-2 on VRSBench captioning, scored the same way as everything else.

    python scripts/eval_florence.py \
        --benchmark /data/eval/vrsbench_val/VRSBench_EVAL_Cap.json \
        --images    /data/eval/vrsbench_val/Images_val \
        --samples 300 --out logs/caption_florence.json

**Florence-2 cannot be prompted.** Its input is one of a fixed set of task
tokens, not a sentence. There is no place to put the register mined from the
9,350 references -- the thing worth +0.05 ROUGE-L on Qwen -- so this measures a
model answering in its own voice against a house style it was never told. Read
the number with that in mind: a low score here is partly register, not sight.
``--task`` sweeps the three caption tokens because they differ mostly in length,
and length is what the brevity penalty grades.

Separate from ``eval_captioning.py`` because the API genuinely differs: Florence
needs ``trust_remote_code``, takes a task token rather than a chat template, and
its output must go through ``post_process_generation``. Bending the Qwen script
around that would make both harder to read.

Blind floor is computed exactly as in ``eval_captioning.py`` -- each prediction
also scored against a different image's reference -- so the two reports compare.
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.training.metrics import (  # noqa: E402
    OFFICIAL_SCORER_CAVEAT,
    caption_scores,
)

DEFAULT_MODEL = "microsoft/Florence-2-large"

#: The three captioning task tokens, shortest first. They are not prompts and
#: take no arguments; the model was trained to associate each with a length.
TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--task",
        default="<MORE_DETAILED_CAPTION>",
        choices=TASKS + ["all"],
        help="'all' runs every task token on the same images.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump", type=int, default=10)
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--out", default="logs/caption_florence.json")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    print(f"{len(rows)} caption(s); scoring {args.samples} on cuda", flush=True)

    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.samples, len(rows)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    tasks = TASKS if args.task == "all" else [args.task]
    images_dir = Path(args.images)
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    report = {"model": args.model, "samples": len(picked), "by_task": {}}

    for task in tasks:
        print(f"\nmodel={args.model} task={task}", flush=True)
        preds, refs, examples = [], [], []
        for i, row in enumerate(picked, 1):
            image_id = row.get("image_id") or row.get("image")
            reference = row.get("caption") or row.get("answer") or ""
            path = images_dir / image_id
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")

            inputs = processor(text=task, images=image, return_tensors="pt")
            inputs = {k: v.to(device, dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=3,
                    do_sample=False,
                )
            raw = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                raw, task=task, image_size=(image.width, image.height)
            )
            prediction = str(parsed.get(task, "")).strip()

            preds.append(prediction)
            refs.append(reference)
            if len(examples) < args.dump:
                examples.append(
                    {
                        "image_id": image_id,
                        "reference": reference,
                        "prediction": prediction,
                        "pred_words": len(prediction.split()),
                    }
                )
                if dump_dir:
                    image.save(dump_dir / image_id)
            if i % 50 == 0:
                s = caption_scores(preds, refs)
                print(
                    f"  {i}/{len(picked)}  ROUGE-L {s['rouge_l']:.3f}"
                    f"  BLEU {s['bleu_1']:.3f}  words {s['mean_pred_words']:.0f}",
                    flush=True,
                )

        scores = caption_scores(preds, refs)
        # Rotating the references by one scores every caption against a
        # different image: what fluent, on-topic nonsense earns for free.
        floor = caption_scores(preds, refs[1:] + refs[:1])
        for e in examples:
            e["rouge_l"] = caption_scores([e["prediction"]], [e["reference"]])["rouge_l"]

        print(
            f"  FINAL  ROUGE-L {scores['rouge_l']:.3f}"
            f"  (floor {floor['rouge_l']:.3f},"
            f" gain {scores['rouge_l'] - floor['rouge_l']:+.3f})"
            f"  BLEU-1 {scores['bleu_1']:.3f}  CIDEr {scores['cider_d']:.3f}"
            f"  words {scores['mean_pred_words']:.0f}",
            flush=True,
        )
        report["by_task"][task] = {
            "scores": scores,
            "blind_floor": floor,
            "examples": examples,
        }

    report["caveat"] = OFFICIAL_SCORER_CAVEAT
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
