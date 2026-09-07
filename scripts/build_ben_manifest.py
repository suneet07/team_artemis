"""Phase 0 item 8 — the BigEarthNet patch-ID manifest.

    python scripts/build_ben_manifest.py --source local --limit 500
    python scripts/build_ben_manifest.py --source hf --limit 20000

A manifest of *patch identifiers and question-answer pairs*, with no pixels.
That separation is the point. ``SPECULATIONS.md`` item 2: BEN.txt ships
annotations only and its imagery lives in a 600 GB+ LMDB that cannot be streamed
selectively, while BEN.txt is the majority source for all four adapters. The
manifest is buildable today from annotations alone; the pixels are fetched
afterwards, per patch, by ``scripts/fetch_copernicus_patches.py``.

Two sources:

``local``
    ``ben-micro-split/train_metadata.jsonl`` -- already on disk, no network, and
    the right thing for testing the pipeline end to end.

``hf``
    Streams ``BIFOLD-BigEarthNetv2-0/BigEarthNet.txt``. Streaming, so nothing
    near 600 GB is downloaded; only the annotation records are read.

Every patch id is parsed before it is written. An unparsable id is a hard error
because the fetcher keys the entire geometry off that string, and a row that
cannot be located is a row that will either fail loudly later or -- worse --
match the wrong granule.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not emit a canonical training manifest. Canonical rows carry image
paths, and no image exists yet. Producing rows that point at files nobody has
fetched is how a corpus ends up half-real.
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.ingest.copernicus import parse_patch_id  # noqa: E402
from satquery.ingest.copernicus.composites import REQUIRED_BANDS  # noqa: E402

LOCAL_METADATA = REPO_ROOT / "ben-micro-split" / "train_metadata.jsonl"
HF_DATASET = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"


def _patch_id_from(value: str) -> str:
    """Strip a file extension if the record carries an image filename."""
    return str(value).rsplit(".", 1)[0] if "." in str(value) else str(value)


def read_local(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Either run training/data/extract_ben_micro.py "
            "first, or use --source hf."
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                {
                    "patch_id": _patch_id_from(record.get("image_id") or record["patch_id"]),
                    "question": record["question"],
                    "answer": record["answer"],
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def read_hf(limit: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "--source hf needs `pip install datasets`. Streaming reads only the "
            "annotation records, so this does not pull the 600 GB archive."
        ) from error

    stream = load_dataset(HF_DATASET, split="all_data", streaming=True)
    rows = []
    for record in stream:
        patch_id = record.get("patch_id")
        if not patch_id:
            continue
        rows.append(
            {
                "patch_id": _patch_id_from(patch_id),
                "question": record.get("input", ""),
                "answer": record.get("output", ""),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


#: reBEN's own split assignment, from the release metadata. Kept out of the
#: signature so callers cannot pass a hand-made mapping.
REBEN_METADATA = REPO_ROOT / "data" / "reben" / "metadata.parquet"


def official_splits(path: Path | None = None) -> dict[str, str]:
    """patch_id -> reBEN's own split, or {} when the metadata is absent."""
    path = path or REBEN_METADATA
    if not path.exists():
        return {}
    import pandas as pd

    frame = pd.read_parquet(path, columns=["patch_id", "split"])
    return dict(zip(frame.patch_id, frame.split, strict=True))



#: BEN.txt ships four task families in one annotation file, and they belong to
#: two different adapters. Treating the file as one VQA corpus mislabels roughly
#: two thirds of it: the grounding and captioning rows would train `rs_vqa` on
#: tasks it never serves, while `rs_ground_caption` -- the adapter gate G3 turns
#: on -- would be starved of exactly the supervision it needs.
#:
#: The questions are template-generated, so matching on their markup is reliable
#: rather than a heuristic: `<point>` and "bounding box" mark grounding, `<ref>`
#: marks referring, and the rest split on whether the answer is yes/no.
_BBOX_MARKERS = ("bounding box", "<point>")
_REFER_MARKERS = ("<ref>", "where is", "locate the", "identify the location")

_MCQ_ANSWER = re.compile(r"^[a-dA-D]$")
_MCQ_OPTIONS = re.compile(r"(^|\s)a\)\s")

#: Dropped from training by the plan's MCQ rule. Kept as distinct question types
#: rather than deleted at parse time, so the corpus can report the MCQ number
#: twice -- with and without them -- instead of silently omitting a category.
_MCQ_EXCLUDED = {
    "country": ("country",),
    "season": ("season",),
    "climate": ("climate zone", "climate"),
}


def classify_question(question: str, answer: str) -> tuple[str, str, str]:
    """(adapter, task, question_type) for one BEN.txt pair.

    Returns the *adapter that serves the task*, not the adapter that happens to
    be staging the file. A row whose answer is a box is grounding supervision
    wherever it was found.
    """
    lowered = question.lower()
    if any(marker in lowered for marker in _BBOX_MARKERS):
        return "rs_ground_caption", "single_grounding", "bbox"
    if any(marker in lowered for marker in _REFER_MARKERS):
        return "rs_ground_caption", "single_grounding", "referring"
    if answer.strip().lower() in {"yes", "no"}:
        return "rs_vqa", "single_vqa", "presence"
    if len(answer.split()) > 12:
        return "rs_ground_caption", "single_caption", "caption"

    # Multiple choice: a single-letter answer against lettered options. Typed by
    # subject, because the plan drops three subjects on purpose (§ the MCQ
    # note): country, season and climate zone. A model that has learned to guess
    # European countries from a Sentinel-2 chip has learned something actively
    # wrong for Indian imagery, and it would score well doing it.
    if _MCQ_ANSWER.match(answer.strip()) and _MCQ_OPTIONS.search(question):
        for subject, markers in _MCQ_EXCLUDED.items():
            if any(marker in lowered for marker in markers):
                return "rs_vqa", "single_vqa", f"mcq_{subject}"
        return "rs_vqa", "single_vqa", "mcq"

    return "rs_vqa", "single_vqa", "other"


def build(
    rows: list[dict],
    *,
    adapter: str,
    split: str,
    enforce_official_split: bool = True,
) -> tuple[list[dict], dict]:
    """Group question-answer pairs by patch and validate every identifier.

    **The requested split is checked against reBEN's own, not trusted.**
    ``ben-micro-split/train_metadata.jsonl`` labels its patches as training
    data; all 21 of them are in reBEN's **test** split. Building a training
    manifest from them produces a corpus that trains on the benchmark's test
    set, and every later BEN.txt number would be inflated with no visible
    symptom -- no crash, no warning, just a better score than the model earned.

    So a disagreement is a hard stop. Pass ``enforce_official_split=False`` only
    when deliberately building a corpus whose split is known to differ, and say
    why in the caller.
    """
    by_patch: dict[str, list[dict]] = {}
    unparsable: set[str] = set()
    for row in rows:
        patch_id = row["patch_id"]
        if patch_id in unparsable:
            continue
        if patch_id not in by_patch:
            try:
                parse_patch_id(patch_id)
            except ValueError:
                unparsable.add(patch_id)
                continue
            by_patch[patch_id] = []
        adapter_for, task, question_type = classify_question(
            row["question"], row["answer"]
        )
        by_patch[patch_id].append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "adapter": adapter_for,
                "task": task,
                "question_type": question_type,
            }
        )

    if unparsable:
        raise SystemExit(
            f"{len(unparsable)} patch id(s) do not parse as reBEN identifiers, "
            f"starting with {sorted(unparsable)[0]!r}. The fetcher derives granule and "
            "window geometry from this string; an id it cannot parse is a row that "
            "will silently match nothing or, worse, the wrong granule."
        )

    if enforce_official_split:
        official = official_splits()
        if official:
            wrong = {
                patch_id: official[patch_id]
                for patch_id in by_patch
                if patch_id in official and official[patch_id] != split
            }
            if wrong:
                sample = sorted(wrong.items())[:3]
                raise SystemExit(
                    f"{len(wrong)} of {len(by_patch)} patch(es) are not in reBEN's "
                    f"'{split}' split. Examples: {sample}. Building a '{split}' "
                    "manifest from them would train on the benchmark's own "
                    "evaluation data and inflate every BEN.txt number reported "
                    "afterwards, silently. Use --split with the split these "
                    "patches actually belong to, or pass "
                    "enforce_official_split=False and record why."
                )

    entries = []
    for patch_id, pairs in by_patch.items():
        patch = parse_patch_id(patch_id)
        entries.append(
            {
                "patch_id": patch_id,
                "tile": patch.tile,
                "sensing": patch.sensing,
                "orbit": patch.orbit,
                "grid_first": patch.first,
                "grid_second": patch.second,
                "adapter": adapter,
                "split": split,
                "source": "BigEarthNet.txt",
                "licence": "CDLA-Permissive-1.0",
                "bands_required": list(REQUIRED_BANDS),
                # 10 m native, 120 px patch -- the resolution policy keeps BEN at
                # native GSD and does not resample it.
                "effective_gsd_m": 10.0,
                "qa": pairs,
            }
        )

    tiles = Counter(entry["tile"] for entry in entries)
    stats = {
        "patches": len(entries),
        "qa_pairs": sum(len(entry["qa"]) for entry in entries),
        "tiles": len(tiles),
        "granule_fetches": len({(e["tile"], e["sensing"], e["orbit"]) for e in entries}),
        "top_tiles": tiles.most_common(10),
    }
    return entries, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=("local", "hf"), default="local")
    parser.add_argument("--limit", type=int, default=0, help="0 = everything available")
    parser.add_argument("--adapter", default="rs_vqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/manifests/ben_patches.jsonl")
    args = parser.parse_args()

    rows = read_local(LOCAL_METADATA, args.limit) if args.source == "local" else read_hf(args.limit)
    print(f"{len(rows)} annotation row(s) from {args.source}")

    entries, stats = build(rows, adapter=args.adapter, split=args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    summary = {
        "built": datetime.now(UTC).isoformat(),
        "source": args.source,
        "manifest": str(out),
        **stats,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"{stats['patches']} patch(es), {stats['qa_pairs']} QA pair(s), "
        f"{stats['tiles']} tile(s)"
    )
    print(
        f"{stats['granule_fetches']} distinct granule(s) to resolve. Each granule is "
        "opened once and read many times by the windowed fetcher, so this number, not "
        "the patch count, sets the network cost."
    )
    print(f"Manifest written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
