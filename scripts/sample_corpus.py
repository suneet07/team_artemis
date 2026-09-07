"""Pull a browsable sample out of a corpus: images, questions, answers.

    python scripts/sample_corpus.py --manifest /data/manifests/rs_vqa_train.jsonl \
        --image-root /data --out logs/corpus_sample.json

Validation reports statistics. Statistics cannot tell you whether a question
makes sense for the picture beside it -- a corpus can pass every automated check
and still ask about pastures in a photograph of open sea. That needs eyes.

So this emits the sample as JSON with the images base64-encoded inline, which
travels back from a container as one artifact and needs no second lookup.
Stratified across sources and question types, so the sample shows the range
rather than twenty near-identical rows from whichever source sorted first.
"""

import argparse
import base64
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Images are shown at this width; anything larger is wasted bytes in a page
#: that displays a grid of thumbnails.
THUMB_PX = 320


def encode(path: Path, width: int = THUMB_PX, quality: int = 0) -> str | None:
    """A data: URI for one image, downscaled. None when it cannot be read.

    ``quality`` above zero switches to JPEG. PNG is lossless and correct for
    inspecting a chip, and ruinous for a page of them: 54 six-view samples came
    to 63 MB as PNG, four times what a published page may carry. Satellite
    imagery is continuous-tone, which is what JPEG is for, and the artefacts
    land well below the scale anyone reviews these at.
    """
    from PIL import Image

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.width > width:
                ratio = width / image.width
                image = image.resize(
                    (width, max(1, int(image.height * ratio))), Image.LANCZOS
                )
            buffer = io.BytesIO()
            if quality:
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
            else:
                image.save(buffer, format="PNG")
    except Exception:  # noqa: BLE001 - an unreadable image is a finding, not a crash
        return None
    kind = "jpeg" if quality else "png"
    return f"data:image/{kind};base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", default="logs/corpus_sample.json")
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument(
        "--group-by",
        default="",
        help="extra row field to bucket on, e.g. stratum, so the sample spans it",
    )
    parser.add_argument("--thumb-px", type=int, default=THUMB_PX)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=0,
        help="0 keeps lossless PNG; 70-85 makes a page small enough to publish",
    )
    args = parser.parse_args()

    root = Path(args.image_root)
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Stratify by (source, question type) so the sample shows the range rather
    # than whichever source happens to sort first.
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        # `corpus_source`/`question_type` are what merge_corpus writes; `source`
        # and `task` are the canonical fields the per-adapter generators write.
        # Keying on the first pair alone collapsed every change_vqa row into one
        # bucket, so the "sample per type" this function exists to produce came
        # back as one type repeated.
        source = row.get("corpus_source") or row.get("source") or "?"
        kind = row.get("question_type") or row.get("task") or "?"
        # A third key, when asked for. Bucketing on (source, type) alone gives
        # a spread of question types and says nothing about scene variety: for
        # change_vqa every row shares one source, so nine land-cover strata
        # would have been represented by whatever the shuffle happened to pick.
        extra = str(row.get(args.group_by)) if args.group_by else ""
        buckets[(source, kind, extra)].append(row)

    picked = []
    for key in sorted(buckets, key=str):
        group = buckets[key]
        random.Random(f"sample:{key}").shuffle(group)
        picked.extend(group[: args.per_type])

    samples = []
    for row in picked:
        images = []
        for relative in row["images"]:
            uri = encode(root / relative, args.thumb_px, args.jpeg_quality)
            if uri:
                images.append({"role": relative.rsplit("_", 1)[-1].split(".")[0], "uri": uri})
        samples.append(
            {
                "sample_id": row["sample_id"],
                "source": row.get("corpus_source") or row.get("source"),
                "question_type": row.get("question_type") or row.get("task"),
                "task": row.get("task"),
                "question": row["question"],
                "answer": row["answer"],
                "gsd_m": (row.get("effective_gsd_m") or [None])[0],
                "roles": row.get("image_roles", []),
                # Whatever context makes a row judgeable by eye. For change_vqa
                # that is the land-cover stratum and the season: "vegetation
                # decreased" is unremarkable over rain-fed cropland between
                # seasons and worth a second look over forest.
                "context": {
                    k: row[k]
                    for k in ("stratum", "season", "aoi", "gap_months", "index")
                    if row.get(k) is not None
                },
                "images": images,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    print(f"{len(samples)} sample(s) from {len(buckets)} bucket(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
