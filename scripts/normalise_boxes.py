"""Put every grounding answer in a manifest on one coordinate scale.

    python scripts/normalise_boxes.py --manifest data/chips/canonical.jsonl

BEN.txt writes boxes as normalised floats (``[0.64 0.0, 1.0 0.71]``); Qwen3-VL
was pretrained on 0-1000 integers, and generated grounding rows from RarePlanes
and LS-SSDD emit that scale. Left mixed, one adapter would be taught two
coordinate systems for the same task -- and the one it already knows is the one
we would be teaching it to abandon.

Only rows whose task is grounding are touched, and only their answers. A
caption that happens to contain four numbers is not a box, which is why the
row's task decides rather than the text pattern.

Idempotent: a coordinate already above 1 is left alone, so running twice cannot
scale a box to 640,000.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.qgen.boxes import PROMPT_BOX_SCALE, normalise_box_answer  # noqa: E402

GROUNDING_TASKS = {"single_grounding", "grounding", "referring"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None, help="defaults to rewriting in place")
    args = parser.parse_args()

    path = Path(args.manifest)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    changed = untouched = skipped = 0
    for row in rows:
        if row.get("task") not in GROUNDING_TASKS:
            skipped += 1
            continue
        before = row.get("answer", "")
        after = normalise_box_answer(before, scale=PROMPT_BOX_SCALE)
        if after != before:
            row["answer"] = after
            row.setdefault("raw_answer", before)
            changed += 1
        else:
            untouched += 1

    out = Path(args.out or path)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"{len(rows)} row(s): {changed} rescaled, {untouched} already on scale, "
          f"{skipped} not grounding")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
