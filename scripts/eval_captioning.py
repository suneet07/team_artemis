"""What does the model score on VRSBench captioning, and what is the floor?

    python scripts/eval_captioning.py \
        --benchmark /data/eval/vrsbench_val/VRSBench_EVAL_Cap.json \
        --images    /data/eval/vrsbench_val/Images_val \
        --samples 300 --out logs/caption_base.json

**Captioning is the half a detector cannot touch.** Grounding is settled --
the base model plus a written prompt scores 62.7% on VRSBench referring with no
training. Nothing about that helps here: `VRSBench_EVAL_Cap.json` asks for prose,
our G3 corpus contains zero captions, and BEN.txt's 19,983 are the only source we
hold. This measures the starting point before anyone trains anything.

**Every run reports a blind floor next to the score.** Each prediction is also
scored against a *different image's* reference. That number is what fluent,
style-matched nonsense earns, and on this benchmark it is not small: measured on
200 references, describing the wrong image scores ROUGE-L 0.244. A caption model
reporting 0.30 has barely cleared writing about something else, and a report
without the floor beside it hides that.

**Style is part of the task, and is stated in the prompt rather than discovered.**
VRSBench references open "The aerial image from GoogleEarth features ..." and run
about 50 words. A model answering in two sentences is punished by the brevity
penalty no matter how accurate it is -- the same format-compliance trap that cost
this project three coordinate-convention bugs. ``--style plain`` turns the
guidance off to measure how much of the score is style rather than seeing.

Scored with the in-repo caption metrics. The OFFICIAL_SCORER_CAVEAT applies with
extra force: caption metrics are far more tokenisation-sensitive than accuracy.
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
    bleu,
    caption_scores,
    rouge_l,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

PLAIN_PROMPT = "Describe the image in detail."

#: Derived from the references, not invented: they open with the sensor phrasing,
#: run ~50 words, name object types with rough counts, and place things by
#: position in the frame. Telling the model the house style is not gaming the
#: metric -- a caption in the wrong register is marked down for register, and we
#: want to measure sight, not house style.
STYLED_PROMPT = (
    "Describe this aerial satellite image in detail, in one paragraph of about "
    "50 words.\n\n"
    "Follow this style:\n"
    "- Open with what the image shows overall.\n"
    "- Name the object types present and roughly how many.\n"
    "- Say where each sits in the frame (top-left, bottom-right, centre).\n"
    "- Mention surfaces and surroundings: roads, vegetation, water, buildings.\n"
    "- Plain declarative sentences. No speculation about purpose or location."
)


#: **Mined from the 9,350 references, not written from taste.** Every figure
#: below was measured: openings ("The image, sourced from GoogleEarth" 17.3%,
#: "The high-resolution image from GoogleEarth" 9.4%), shape (median 47 words,
#: 3 sentences, ~16 words each; 10th percentile 29, 90th 70), attributes
#: ("small" in **53.8%**, "large" 15.8%, "green" 12.3%), counts as words ("one"
#: 25.8%, "two" 25.9%), and positional phrasing ("in the" 37.8%, "on the"
#: 27.1%, "at the" 24.7%, "towards the" 19.1%).
#:
#: Stating the register is not gaming the metric. BLEU and ROUGE score shared
#: words, so a caption in the wrong register is marked down for register while
#: reading as a failure of sight. Fixing the register is how we get to measure
#: sight at all.
#: Imported, not duplicated -- the 0.252 ROUGE-L run used this exact string and
#: so does the serving path.
from satquery.agent.served_prompts import STRONG_PROMPT  # noqa: E402

#: Detect-then-narrate with one model instead of two. The captioning literature
#: repeatedly extracts structure before generating (Region-Driven,
#: multilabel-then-caption, mask-guided), and a detector was the obvious way --
#: but Grounding DINO measured 8-67% recall on the classes captions actually
#: mention, and four of the top six are regions it barely handles. The VLM
#: already localises objects 84.7% of the time, so it inventories its own view
#: and writes from that. No second model, no wrong counts injected.
INVENTORY_PROMPT = (
    "List what is visible in this aerial satellite image.\n\n"
    "One line per object type, in this form:\n"
    "<count> <small|large> <object type> - <position in the frame>\n\n"
    "Then a final line: SURROUNDINGS: <roads, vegetation, water, buildings, "
    "bare ground -- whichever are present>\n\n"
    "Count carefully. List only what you can actually see."
)

NARRATE_PROMPT = (
    "Here is an inventory of this image:\n\n{inventory}\n\n"
    "Write one caption from it.\n\n"
    "Format, followed exactly:\n"
    "- Open with: \"The image, sourced from GoogleEarth,\" then shows, "
    "features or captures.\n"
    "- About 47 words across 3 sentences.\n"
    "- Use every item in the inventory, with its count and position.\n"
    "- Counts spelled as words. Objects called small or large.\n"
    "- Place things using: in the, on the, at the, towards the.\n"
    "- Close with the surroundings.\n"
    "- Plain prose. No bullet points, no bold, no headings."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--style",
        default="styled",
        choices=["styled", "plain", "strong", "twopass"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump", type=int, default=10)
    parser.add_argument(
        "--dump-dir",
        default="",
        help="copy the scored images here. A caption metric cannot tell a "
        "wrong description from a correct one in different words, so the "
        "only way to know which we have is to read them beside the picture.",
    )
    parser.add_argument("--out", default="logs/caption_eval.json")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    images_root = Path(args.images)
    sample = random.Random(args.seed).sample(rows, min(args.samples, len(rows)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt = {
        "styled": STYLED_PROMPT,
        "plain": PLAIN_PROMPT,
        "strong": STRONG_PROMPT,
        "twopass": INVENTORY_PROMPT,
    }[args.style]
    print(
        f"{len(rows)} caption(s); scoring {len(sample)} on {device}\n"
        f"model={args.model} adapter={args.adapter or 'none (base)'} "
        f"style={args.style}",
        flush=True,
    )

    def vlm_generate(prepared):
        return model.generate(
            **prepared, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    predictions, references, records = [], [], []
    for position, row in enumerate(sample, 1):
        path = images_root / row["image_id"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = processor.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        inventory = ""
        if args.style == "twopass":
            inventory = reply
            second = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": NARRATE_PROMPT.format(inventory=inventory)}]}]
            text2 = processor.apply_chat_template(
                second, tokenize=False, add_generation_prompt=True)
            inputs2 = processor(
                images=image, text=text2, return_tensors="pt").to(device)
            with torch.no_grad():
                gen2 = vlm_generate(inputs2)
            reply = processor.decode(
                gen2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

        predictions.append(reply)
        references.append(row["ground_truth"])
        if len(records) < args.dump:
            # Per-caption scores, so a low aggregate can be traced to specific
            # captions and judged. ROUGE-L and BLEU are computed per row here;
            # CIDEr is corpus-relative and has no meaningful single-row value.
            one = rouge_l([reply], [row["ground_truth"]])
            one.update(bleu([reply], [row["ground_truth"]]))
            records.append({
                "image_id": row["image_id"],
                "reference": row["ground_truth"],
                "prediction": reply,
                "rouge_l": one["rouge_l"],
                "bleu_1": one["bleu_1"],
                "inventory": inventory,
                "pred_words": len(reply.split()),
                "ref_words": len(row["ground_truth"].split()),
            })
            if args.dump_dir:
                target = Path(args.dump_dir)
                target.mkdir(parents=True, exist_ok=True)
                image.save(target / row["image_id"])
        if position % 50 == 0:
            partial = caption_scores(predictions, references)
            print(f"  {position}/{len(sample)}  ROUGE-L {partial['rouge_l']:.3f}  "
                  f"BLEU {partial['bleu']:.3f}  words {partial['mean_pred_words']:.0f}",
                  flush=True)

    scored = caption_scores(predictions, references)

    # The floor: same predictions, each judged against a different image's
    # reference. Whatever a fluent caption of the wrong scene earns.
    rotated = references[1:] + references[:1]
    floor = caption_scores(predictions, rotated)

    report = {
        "model": args.model,
        "adapter": args.adapter or None,
        "style": args.style,
        "samples": len(predictions),
        "scores": scored,
        "blind_floor": floor,
        "headroom": {
            k: round(scored[k] - floor[k], 4)
            for k in ("bleu", "rouge_l", "cider_d")
        },
        "caveat": OFFICIAL_SCORER_CAVEAT,
        "examples": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{'metric':16}{'scored':>10}{'blind floor':>14}{'gain':>9}")
    for k in ("bleu", "bleu_1", "rouge_l", "cider_d"):
        gap = scored[k] - floor[k]
        print(f"{k:16}{scored[k]:10.4f}{floor[k]:14.4f}{gap:+9.4f}")
    print(f"\nwords: predicted {scored['mean_pred_words']:.0f} vs "
          f"reference {scored['mean_ref_words']:.0f}   "
          f"empty {scored['empty_predictions']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
