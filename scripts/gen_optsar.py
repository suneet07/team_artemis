"""Cross-modal optical/SAR questions from the staged reBEN pairs.

    python scripts/gen_optsar.py build \
        --reben /data/eval/reben/patches/BEN_14k \
        --metadata /data/eval/reben/_hf/.../metadata.parquet \
        --out /data/manifests/optsar_fusion.jsonl

This is the first caller ``satquery.qgen.gen_crossmodal.gen_crossmodal_qa`` has
ever had. The generator emits one question three ways -- optical alone, SAR
alone, and both -- so the adapter cannot learn that the answer always lives in
one input slot, and so a SAR-only scene at inference is in distribution rather
than out of it (C26). The problem statement admits "one optical/multispectral
**or** SAR image", so that arm is not a nicety.

**Not every class earns all three arms, and this is the whole design.** The
three-arm contract asserts the SAR-only row is answerable from radar. At C-band,
10 m, single date, that is true of surfaces with distinct scattering behaviour
and false of everything separated by colour or phenology:

* water is specular and reads near-black;
* built-up is a corner reflector and reads bright;
* forest scatters through its volume and separates from bare ground.

But *arable land* against *pastures* against *complex cultivation* differ by crop
type and season, not by backscatter -- single-date C-band cannot tell them apart,
and a SAR row asking about them teaches the model to answer from the label prior.
That row would score well while learning nothing, which is the failure the blind
floor exists to catch and which is cheaper to not create. So
:data:`SAR_ANSWERABLE` gates which questions get radar arms; everything else is
optical-only, and says so in ``question_type``.

**Answers are balanced by construction.** Asking only about classes that are
present yields a corpus that is 100% "yes", which ``stage_ben_txt`` already
records as a defect worth naming. Each patch contributes one present class and
one absent one.

Licence: reBEN is CDLA-Permissive 1.0. Imagery and labels both.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.qgen.gen_crossmodal import gen_crossmodal_qa  # noqa: E402
from satquery.qgen.primitives import P6CrossModal  # noqa: E402

SOURCE = "reBEN (BigEarthNet v2.0)"
LICENCE = "CDLA-Permissive-1.0"
PROVENANCE = [
    "BigEarthNet v2.0 (reBEN), CDLA-Permissive-1.0",
    "Sentinel-1 GRD / Sentinel-2 L2A, Copernicus open access",
    "CORINE Land Cover 2018, 19-class nomenclature",
]

#: Classes a single-date C-band scene can genuinely be asked about, with the
#: scattering mechanism that makes each one legible. Anything absent from this
#: mapping is emitted on the optical arm only.
SAR_ANSWERABLE = {
    "Inland waters": "specular reflection away from the sensor",
    "Marine waters": "specular reflection away from the sensor",
    "Urban fabric": "double-bounce from wall-and-ground corners",
    "Industrial or commercial units": "double-bounce from wall-and-ground corners",
}

#: Forest is legible to radar as a *category* -- volume scattering separates it
#: from bare ground and from water -- while the species split that CORINE draws
#: is not. So the grouped question gets radar arms and the three member classes
#: do not.
SAR_GROUPS = {
    "forest": (
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
    ),
}

ALL_ARMS = ("optical", "sar", "fused")
OPTICAL_ONLY = ("optical",)


def index_patches(root: Path) -> dict[str, dict[str, Path]]:
    """Map each modality's file stem to its path.

    The archive nests by split, and the two modalities use different naming
    conventions, so this indexes by stem rather than assuming a layout.
    """
    index: dict[str, dict[str, Path]] = {}
    for modality in ("S1", "S2"):
        directory = root / f"BigEarthNet-{modality}"
        index[modality] = {p.stem: p for p in directory.rglob("*.tif")}
        print(f"{modality}: {len(index[modality])} raster(s)", flush=True)
    return index


def presence_questions(labels: set[str], vocabulary: list[str], rng):
    """One present class and one absent class, so the corpus is not all 'yes'.

    Radar-answerable classes are excluded here and asked as a deliberate pair
    instead. Left in, they leak an unbalanced subset into the radar arm: the
    positive draw reaches one of them only 38.6% of the time while the negative
    draw reaches one almost always, which measured 1,068 yes against 2,879 no
    and put the arm's majority-answer floor at 53.9% instead of 50%.
    """
    present = sorted(labels - set(SAR_ANSWERABLE))
    absent = [c for c in vocabulary if c not in labels and c not in SAR_ANSWERABLE]
    out = []
    if present:
        out.append((rng.choice(present), "yes"))
    if absent:
        out.append((rng.choice(absent), "no"))
    return out


def arms_for(class_name: str) -> tuple[str, ...]:
    return ALL_ARMS if class_name in SAR_ANSWERABLE else OPTICAL_ONLY


def build_rows(frame, index, rng, limit: int) -> tuple[list[dict], Counter]:
    vocabulary = sorted({c for labels in frame["labels"] for c in labels})
    rows: list[dict] = []
    stats: Counter = Counter()

    forest_members = set(SAR_GROUPS["forest"])

    # Which patches may carry the grouped forest question. 63.7% of reBEN holds
    # some forest, so emitting it everywhere makes "yes" the right answer nearly
    # two thirds of the time on the question that would otherwise dominate the
    # radar arm. Every forest-free patch is kept and an equal number of
    # forest-bearing ones are sampled, which costs rows and buys a question that
    # cannot be answered without looking.
    with_forest, without_forest = [], []
    for record in frame.itertuples():
        (with_forest if set(record.labels) & forest_members else without_forest).append(
            record.patch_id
        )
    rng.shuffle(with_forest)
    forest_eligible = set(without_forest) | set(with_forest[: len(without_forest)])
    stats["forest_pool"] = len(forest_eligible)

    for record in frame.itertuples():
        s2_path = index["S2"].get(record.patch_id)
        s1_path = index["S1"].get(record.s1_name)
        if s2_path is None or s1_path is None:
            stats["unpaired"] += 1
            continue

        labels = set(record.labels)
        primitive = P6CrossModal(
            sample_id=f"optsar_{record.patch_id}",
            # Optical first, SAR second: gen_crossmodal_qa's default indices.
            image_paths=[str(s2_path), str(s1_path)],
            effective_gsd_m=[10.0, 10.0],
            source=SOURCE,
            source_ann_id=record.patch_id,
            licence=LICENCE,
            provenance_chain=list(PROVENANCE),
            split="train" if record.split == "train" else "val",
            labels=sorted(labels),
        )

        for class_name, answer in presence_questions(labels, vocabulary, rng):
            arms = arms_for(class_name)
            stats[f"presence/{'3arm' if len(arms) == 3 else 'optical'}"] += 1
            rows.extend(
                gen_crossmodal_qa(
                    primitive,
                    question=f"Is there {class_name.lower()} in this image?",
                    answer=answer,
                    question_type="landcover_presence",
                    answer_type="binary",
                    arms=arms,
                )
            )

        # A radar presence *pair*, drawn from the classes radar can answer. The
        # uniform draw above reaches these four only about one time in five, so
        # without this the arm meant to teach radar is mostly a single question.
        # The pair is emitted only where a positive exists, which is what keeps
        # yes and no equal rather than flooding the arm with "no".
        sar_present = sorted(labels & set(SAR_ANSWERABLE))
        sar_absent = [c for c in SAR_ANSWERABLE if c not in labels]
        if sar_present and sar_absent:
            for class_name, answer in (
                (rng.choice(sar_present), "yes"),
                (rng.choice(sar_absent), "no"),
            ):
                stats["sar_presence/3arm"] += 1
                rows.extend(
                    gen_crossmodal_qa(
                        primitive,
                        question=f"Is there {class_name.lower()} in this image?",
                        answer=answer,
                        question_type="landcover_presence",
                        answer_type="binary",
                        arms=ALL_ARMS,
                    )
                )

        # The grouped radar question, on the balanced pool only.
        if record.patch_id in forest_eligible:
            for group, members in SAR_GROUPS.items():
                answer = "yes" if labels & set(members) else "no"
                stats["group/3arm"] += 1
                rows.extend(
                    gen_crossmodal_qa(
                        primitive,
                        question=f"Is there {group} in this image?",
                        answer=answer,
                        question_type=f"landcover_{group}",
                        answer_type="binary",
                        arms=ALL_ARMS,
                    )
                )

        if limit and len(rows) >= limit:
            break

    return rows, stats


def resolve_metadata(given: str, cache: Path) -> Path:
    """The official index, fetched if not handed to us.

    Not defaulted to a path under the HuggingFace cache: that path carries a
    snapshot hash, and pinning one here would break the first time the mirror
    republishes. ``hf_hub_download`` hits the same cache when it is already
    there, so this costs nothing on a warm volume.
    """
    if given:
        return Path(given)
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="torchgeo/bigearthnet",
            filename="V2/metadata.parquet",
            repo_type="dataset",
            cache_dir=str(cache),
        )
    )


def cmd_build(args) -> int:
    import pandas as pd

    rng = random.Random(args.seed)
    root = Path(args.reben)
    index = index_patches(root)

    metadata = resolve_metadata(args.metadata, root.parent.parent / "_hf")
    print(f"index: {metadata}", flush=True)
    frame = pd.read_parquet(metadata)
    staged = set(index["S2"])
    frame = frame[frame["patch_id"].isin(staged)]
    print(f"{len(frame)} staged patch(es) carry metadata", flush=True)

    rows, stats = build_rows(frame, index, rng, args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    arm_counts = Counter(r.get("modality_arm") for r in rows)
    answer_counts = Counter(r["answer"] for r in rows)
    # Per-arm answer balance is the number that matters. A corpus balanced
    # overall can still have one arm answerable from the answer prior alone,
    # which is exactly what the first build did.
    per_arm_answers: dict[str, dict[str, int]] = {}
    per_arm_questions: dict[str, dict[str, int]] = {}
    for row in rows:
        arm = row.get("modality_arm", "?")
        per_arm_answers.setdefault(arm, Counter())[row["answer"]] += 1
        kind = row["question_type"].rsplit("_", 1)[0]
        per_arm_questions.setdefault(arm, Counter())[kind] += 1
    summary = {
        "rows": len(rows),
        "source": SOURCE,
        "licence": LICENCE,
        "arms": dict(arm_counts),
        "answers": dict(answer_counts),
        "answers_per_arm": {k: dict(v) for k, v in per_arm_answers.items()},
        # What a model scores by always giving that arm's commonest answer. It
        # is the number to read every score against, and the first build hid a
        # 60.6% radar floor behind an aggregate that looked balanced.
        "majority_answer_floor": {
            arm: round(max(counts.values()) / sum(counts.values()), 4)
            for arm, counts in per_arm_answers.items()
        },
        "questions_per_arm": {k: dict(v) for k, v in per_arm_questions.items()},
        "questions": dict(stats),
        "splits": dict(Counter(r["split"] for r in rows)),
        "sar_answerable_classes": sorted(SAR_ANSWERABLE),
        "out": str(out),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--reben", required=True, help="the BEN_14k directory")
    build.add_argument(
        "--metadata", default="", help="official metadata.parquet; fetched if omitted"
    )
    build.add_argument("--out", default="/data/manifests/optsar_fusion.jsonl")
    build.add_argument("--summary", default="logs/optsar_corpus.json")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
