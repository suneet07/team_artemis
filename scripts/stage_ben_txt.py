"""Select a BigEarthNet.txt training corpus, stratified and quota-balanced.

    python scripts/stage_ben_txt.py --patches 20000 --rows-per-adapter 60000

BEN.txt is 9.55M annotations over 464k image pairs; the training split alone is
**4.67M rows over 229,114 patches**. You can afford roughly 60k rows per
adapter. So the whole job is choosing well, and the default -- take the first N
rows -- chooses badly.

**Why not the head of the file.** Rows are ordered by patch id, which sorts by
tile, then date, then grid position. Measured on the existing micro-split: 21
patches, **1 tile of 54, 1 date, a contiguous 4x8 grid block** -- a single
5x10 km rectangle in Austria on one June morning, from a dataset spanning 10
countries, 10 climate zones, 4 seasons and 54 tiles.

**Four things this does instead.**

1. *Stratify.* Patches are drawn round-robin across country x season x climate
   zone, so Finland's 155k does not swamp Kosovo's few thousand. Coverage, not
   proportion: the Indian hidden set resembles neither distribution.
2. *Space them out.* Neighbouring patches are 1.2 km apart on the same day --
   near-duplicate imagery with near-duplicate labels.
3. *Quota by task.* BEN.txt's own `type` and `category` columns are used rather
   than inferred, and each adapter gets a deliberate mix instead of the file's
   native proportions.
4. *Drop what the plan drops.* `country`, `season` and `climate zone` MCQs are
   excluded -- 687,342 rows -- because a model that has learned to guess
   European countries from a Sentinel-2 chip has learned something actively
   wrong for Indian imagery, and would score well doing it.

**Routing.** BEN.txt is not one task. `bounding box` and `captioning` rows are
grounding and captioning supervision and belong to ``rs_ground_caption`` -- the
adapter gate G3 is scored on. `binary` and `mcq` go to ``rs_vqa``. Sending all
of it to one adapter starves the other, which is what the first pass did.

Output is the same manifest shape ``fetch_copernicus_patches.py`` already
consumes: patch identifiers plus their questions, and **no image paths**, since
no image has been fetched yet.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.ingest.copernicus import parse_patch_id  # noqa: E402
from satquery.qgen.sampling import (  # noqa: E402
    balance_question_types,
    coverage_report,
    stratified_patches,
)

SOURCE = "BigEarthNet.txt"
LICENCE = "CDLA-Permissive-1.0"
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
GSD_M = 10.0

#: Dropped by the plan's MCQ rule. Named, not keyword-matched: the dataset
#: labels them, so there is no reason to guess.
EXCLUDED_CATEGORIES = {"country", "season", "climate zone"}

#: BEN.txt task type -> the adapter that serves it.
ADAPTER_FOR_TYPE = {
    "binary": "rs_vqa",
    "mcq": "rs_vqa",
    "bounding box": "rs_ground_caption",
    "captioning": "rs_ground_caption",
}

#: Canonical task name per BEN.txt type.
TASK_FOR_TYPE = {
    "binary": "single_vqa",
    "mcq": "single_vqa",
    "bounding box": "single_grounding",
    "captioning": "single_caption",
}

#: Questions kept from any one patch. A patch that supplies 40 rows crowds a
#: batch and makes the corpus reflect annotation density rather than the task.
MAX_QA_PER_PATCH = 12


def load(path: Path, split: str):
    import pandas as pd

    frame = pd.read_parquet(
        path,
        columns=[
            "patch_id", "input", "output", "type", "category",
            "split", "country", "season", "climate_zone",
        ],
    )
    frame = frame[frame.split == split]
    if frame.empty:
        raise SystemExit(f"no rows with split={split!r} in {path}")
    # Captioning rows carry no category -- as a real null in some rows and the
    # literal string "None" in others. Both become "caption": question_type is
    # what average accuracy buckets on, and "captioning_None" is a bucket name
    # that will be read as a bug later.
    frame["category"] = (
        frame["category"].fillna("caption").replace({"None": "caption", "": "caption"})
    )
    before = len(frame)
    frame = frame[~frame.category.str.lower().isin(EXCLUDED_CATEGORIES)]
    print(f"{before} row(s) in split={split}; {before - len(frame)} excluded by the "
          f"MCQ rule ({sorted(EXCLUDED_CATEGORIES)}); {len(frame)} remain")
    return frame


def choose_patches(frame, target: int) -> list[str]:
    patches = (
        frame[["patch_id", "country", "season", "climate_zone"]]
        .drop_duplicates("patch_id")
        .to_dict("records")
    )
    print(f"{len(patches)} distinct patch(es) available; selecting {target}")
    return stratified_patches(
        patches, target, strata_keys=("country", "season", "climate_zone")
    )


def build(
    frame, patch_ids: list[str], rows_per_adapter: int, split: str
) -> tuple[list[dict], dict]:
    chosen = set(patch_ids)
    subset = frame[frame.patch_id.isin(chosen)]

    rows = []
    for record in subset.itertuples(index=False):
        adapter = ADAPTER_FOR_TYPE.get(record.type)
        if adapter is None:
            continue
        rows.append(
            {
                "patch_id": record.patch_id,
                "question": record.input,
                "answer": record.output,
                "adapter": adapter,
                "task": TASK_FOR_TYPE[record.type],
                "question_type": f"{record.type}_{record.category}".replace(" ", "_"),
            }
        )

    # Quota per (type, category) within each adapter, so the mix is chosen
    # rather than inherited from whatever proportions the file happens to have.
    kept: list[dict] = []
    for adapter in sorted({r["adapter"] for r in rows}):
        pool = [r for r in rows if r["adapter"] == adapter]
        types = sorted({r["question_type"] for r in pool})
        per_type = {name: rows_per_adapter // max(1, len(types)) for name in types}
        taken = balance_question_types(pool, per_type)
        print(f"  {adapter:18} {len(pool):8} available -> {len(taken):7} kept "
              f"across {len(types)} type(s)")
        kept.extend(taken)

    # Cap per patch last: quotas are global, and without this a densely
    # annotated patch can still win a large share of its category.
    by_patch: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(kept, key=lambda r: (r["patch_id"], r["question_type"])):
        if len(by_patch[row["patch_id"]]) < MAX_QA_PER_PATCH:
            by_patch[row["patch_id"]].append(row)

    entries = []
    unparsable = []
    for patch_id, pairs in by_patch.items():
        try:
            patch = parse_patch_id(patch_id)
        except ValueError:
            unparsable.append(patch_id)
            continue
        entries.append(
            {
                "patch_id": patch_id,
                "tile": patch.tile,
                "sensing": patch.sensing,
                "orbit": patch.orbit,
                "grid_first": patch.first,
                "grid_second": patch.second,
                # From the caller, never hardcoded. A hardcoded "train" here
                # is how the bench split -- 1,082 human-verified pairs meant
                # only for evaluation -- becomes training data that inflates
                # every number reported against it afterwards.
                "split": split,
                "source": SOURCE,
                "licence": LICENCE,
                "bands_required": list(BANDS),
                "effective_gsd_m": GSD_M,
                "qa": [
                    {k: v for k, v in pair.items() if k != "patch_id"} for pair in pairs
                ],
            }
        )
    if unparsable:
        raise SystemExit(
            f"{len(unparsable)} patch id(s) do not parse, starting with "
            f"{unparsable[0]!r}. The fetcher derives all its geometry from this "
            "string; an id it cannot parse fetches nothing or the wrong granule."
        )

    stats = {
        "patches": len(entries),
        "qa_pairs": sum(len(e["qa"]) for e in entries),
        "tiles": len({e["tile"] for e in entries}),
        "granule_fetches": len({(e["tile"], e["sensing"], e["orbit"]) for e in entries}),
        "by_adapter": dict(
            Counter(p["adapter"] for e in entries for p in e["qa"])
        ),
        "by_type": dict(
            Counter(p["question_type"] for e in entries for p in e["qa"])
        ),
    }
    return entries, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet", default="data/eval/ben_txt/BigEarthNet.txt.parquet")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test", "bench"),
    )
    parser.add_argument("--patches", type=int, default=20000)
    parser.add_argument("--rows-per-adapter", type=int, default=60000)
    parser.add_argument("--out", default="data/manifests/ben_txt_train.jsonl")
    args = parser.parse_args()

    frame = load(Path(args.parquet), args.split)
    patch_ids = choose_patches(frame, args.patches)

    lookup = {
        r["patch_id"]: r
        for r in frame[["patch_id", "country", "season", "climate_zone"]]
        .drop_duplicates("patch_id")
        .to_dict("records")
    }
    coverage = coverage_report(patch_ids, lookup)
    print("\ncoverage of the selected patches:")
    print(f"  countries : {len(coverage['countries'])} {coverage['countries']}")
    print(f"  seasons   : {coverage['seasons']}")
    print(f"  tiles     : {coverage['tiles']}  "
          f"largest tile {coverage['largest_tile_share']:.1%} of the corpus")

    print("\nquotas:")
    entries, stats = build(frame, patch_ids, args.rows_per_adapter, args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(
        json.dumps({**stats, "coverage": coverage}, indent=2), encoding="utf-8"
    )

    print(f"\n{stats['patches']} patch(es), {stats['qa_pairs']} QA pair(s), "
          f"{stats['tiles']} tile(s)")
    print(f"by adapter: {stats['by_adapter']}")
    print(f"{stats['granule_fetches']} granule(s) to resolve -- this, not the patch "
          "count, sets the network cost")
    print(f"Manifest written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
