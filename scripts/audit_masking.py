"""How much of each SpaceNet 7 mosaic is masked out, and what that costs.

    python scripts/audit_masking.py --root /data/eval/sn7 --out logs/masking.json

SpaceNet 7 ships ``images_masked``: cloud-covered regions are set to pure black
rather than left as cloud. A month that was entirely obscured comes back fully
black -- ``gen_change.is_blank`` already rejects those -- but a *partial* mask
leaves a legible tile with a hole in it, and the label file still carries every
building underneath.

That makes the question unanswerable from the pixels. "Which image contains
more buildings" has a ground-truth answer counted over the whole footprint,
while the model can only see what survived the mask. The corpus browser turned
one up immediately, which is the sort of thing a row count never shows.

This measures the distribution so the rejection threshold is chosen from the
data rather than guessed.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def masked_fraction(path: Path) -> float:
    """Share of pixels that are exactly black across all channels.

    Exactly zero, not merely dark: the mask writes 0 while genuine shadow and
    water sit above it. Strided by 4 for speed -- a mask large enough to hide a
    building is far larger than a 4 px sampling grid.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"))[::4, ::4, :]
    return float((array.sum(axis=2) == 0).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="/data/eval/sn7")
    parser.add_argument("--out", default="logs/masking.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    images = sorted(Path(args.root).glob("*/images/*.tif"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no mosaics under {args.root}")

    fractions: dict[str, float] = {}
    for index, path in enumerate(images, 1):
        try:
            fractions[str(path.relative_to(args.root))] = masked_fraction(path)
        except Exception as error:  # noqa: BLE001 - one bad file is a result
            print(f"  FAILED {path.name}: {error!r}", flush=True)
        if index % 200 == 0:
            print(f"  {index}/{len(images)} measured", flush=True)

    values = sorted(fractions.values())
    bands = Counter()
    for value in values:
        if value == 0:
            bands["clean (0%)"] += 1
        elif value < 0.01:
            bands["trace (<1%)"] += 1
        elif value < 0.05:
            bands["light (1-5%)"] += 1
        elif value < 0.20:
            bands["moderate (5-20%)"] += 1
        elif value < 0.99:
            bands["heavy (20-99%)"] += 1
        else:
            bands["blank (>=99%)"] += 1

    def percentile(share: float) -> float:
        return values[min(len(values) - 1, int(share * len(values)))]

    summary = {
        "images": len(values),
        "bands": dict(bands),
        "median": round(percentile(0.50), 4),
        "p90": round(percentile(0.90), 4),
        "p99": round(percentile(0.99), 4),
        "max": round(values[-1], 4),
        # The rows that matter: anything a person would call "has a hole in it".
        "over_5pct": sum(1 for v in values if v > 0.05),
        "over_20pct": sum(1 for v in values if v > 0.20),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "per_image": fractions}, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
