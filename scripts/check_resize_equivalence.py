"""Is a pre-resized mosaic the same input as the native one?

    python scripts/check_resize_equivalence.py --root /data/eval/sn7 --sample 12

Pre-resizing the staged imagery is the one large speed-up available without
touching the frozen ``max_pixels``: the run is input-bound, and a 1024 px
mosaic is read, decoded and downsampled on every epoch to produce pixels the
processor would have produced anyway.

"Anyway" is the claim this file checks rather than assumes. Qwen's processor
does not resize to a round number -- it snaps each side to a multiple of 28
under ``max_pixels``, which for a 1024 px square is 504, not 512. Pre-resizing
to 512 would therefore be resampled a *second* time, which is worse than doing
it once. Pre-resizing to exactly the processor's own target should instead make
its resize a no-op.

So this runs the real processor over a sample of real mosaics both ways and
compares the pixel tensors it produces. Anything but an exact match is reported
with its magnitude, because "near enough" is a judgement for a person to make
with the number in front of them, not for this script to make silently.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FACTOR = 28


def smart_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """The processor's own target size, reimplemented to plan the pre-resize."""
    w = round(width / FACTOR) * FACTOR
    h = round(height / FACTOR) * FACTOR
    if w * h > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        w = math.floor(width / beta / FACTOR) * FACTOR
        h = math.floor(height / beta / FACTOR) * FACTOR
    return max(FACTOR, w), max(FACTOR, h)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="/data/eval/sn7")
    parser.add_argument("--sample", type=int, default=12)
    parser.add_argument("--out", default="logs/resize_equivalence.json")
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    from satquery.training.config import TrainingConfig

    cfg = TrainingConfig.for_adapter("change_vqa")
    processor = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)
    print(f"max_pixels={cfg.max_pixels:,}  model={cfg.model_id}")

    images = sorted(Path(args.root).glob("*/images/*.tif"))[: args.sample]
    if not images:
        raise SystemExit(f"no mosaics under {args.root}")

    results = []
    for path in images:
        with Image.open(path) as source:
            native = source.convert("RGB")
            a = processor.image_processor(images=[native], return_tensors="pt")
            # Ask the processor what size it chose rather than reimplementing
            # its rule. A first attempt reproduced the documented 28-multiple
            # formula and produced 504x504 where the processor had chosen a
            # smaller grid, so the "equivalent" pre-resize was 7% larger and
            # every tensor came back a different shape.
            grid = a["image_grid_thw"][0].tolist()  # [t, h, w] in merged units
            merge = getattr(processor.image_processor, "merge_size", 2)
            patch = getattr(processor.image_processor, "patch_size", 14)
            target = (grid[2] * merge * patch, grid[1] * merge * patch)
            pre = native.resize(target, Image.BICUBIC)
            b = processor.image_processor(images=[pre], return_tensors="pt")

        same_shape = a["pixel_values"].shape == b["pixel_values"].shape
        entry = {
            "image": path.name,
            "native": [native.width, native.height],
            "target": list(target),
            "grid_thw": grid,
            "shape_native": list(a["pixel_values"].shape),
            "shape_pre": list(b["pixel_values"].shape),
            "same_shape": bool(same_shape),
        }
        if same_shape:
            diff = (a["pixel_values"] - b["pixel_values"]).abs()
            entry["max_abs_diff"] = round(float(diff.max()), 6)
            entry["mean_abs_diff"] = round(float(diff.mean()), 6)
            # The tensors are normalised, so express the gap against their own
            # spread rather than as a bare number nobody can size.
            entry["rel_to_std"] = round(
                float(diff.mean() / a["pixel_values"].std()), 6
            )
        results.append(entry)
        print(f"  {path.name[:52]:52} {entry.get('max_abs_diff', 'SHAPE MISMATCH')}")

    matched = [r for r in results if r["same_shape"]]
    summary = {
        "checked": len(results),
        "same_shape": len(matched),
        "worst_max_abs_diff": max((r["max_abs_diff"] for r in matched), default=None),
        "worst_mean_abs_diff": max((r["mean_abs_diff"] for r in matched), default=None),
        "worst_rel_to_std": max((r["rel_to_std"] for r in matched), default=None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "per_image": results}, indent=1),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, indent=1))
    if len(matched) != len(results):
        raise SystemExit(
            "a pre-resized mosaic produced a different tensor SHAPE. The "
            "pre-resize target is wrong and would change what the model sees."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
