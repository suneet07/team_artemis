"""Assemble the testing-corpus gallery: 200 items, 50 per segment.

Every item comes from a **test or held-out split** of the benchmark its adapter
was scored on, joined back to its question, its gold answer and the imagery
itself. Nothing here is from a training split, and each item records the Modal
account and corpus it came from so the claim can be checked rather than trusted.

**Benchmark conditions travel with the item.** Each row carries the
``benchmark``/``question_type`` pair that ``contract_for`` resolves into an
answer contract, so ``verify_gallery.py`` can score a served answer through the
*same* formatter the benchmark used. Without that the comparison is against a
different function than the one that produced the published number -- an
`rsvqa_hr` count of "about 12" is wrong raw and right formatted, and the whole
+7.3 that count quantisation was worth would silently reappear as failures.

Three statuses, because they carry different weight and conflating them would be
the dishonest move:

``verified``   a per-item benchmark record exists and the model got it right.
               change_vqa (1,200-row CDVQA Val dump) and grounding (150-row
               VRSBench referring run).
``scored``     captioning. ROUGE-L is a score, not a verdict; the run's blind
               floor travels with each item so the number has a reference.
``candidate``  no per-item record was ever dumped, so correctness is unknown
               until it is run. rs_vqa and SAR. ``verify_gallery.py`` promotes
               or drops each one.

    python scripts/build_gallery.py select
    python scripts/build_gallery.py fetch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gallery"
SRC = OUT / "_src"
IMAGES = OUT / "images"
MANIFEST = OUT / "manifest.json"

#: Which Modal workspace holds each segment's imagery, as
#: ``{"segment": "workspace"}``. Read from a local file rather than written
#: here: workspace names identify the operator's own accounts and do not belong
#: in a public repository. The file is gitignored; see the error below for its
#: shape. Only ``--fetch`` needs it, so building a manifest from cached images
#: works without it.
ACCOUNTS_FILE = ROOT / "configs" / "gallery_accounts.json"


def _accounts() -> dict[str, str]:
    if not ACCOUNTS_FILE.exists():
        raise SystemExit(
            f"{ACCOUNTS_FILE} is absent. It maps each segment to the Modal "
            'workspace holding its imagery, e.g. {"rs_vqa": "<workspace>"}. '
            "It is gitignored because workspace names identify an account."
        )
    return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))

PER_SEGMENT = 50
SEED = 20260906


def _load_json(name: str):
    return json.loads((SRC / name).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced(rows: list[dict], key: str, want: int, seed: int = SEED) -> list[dict]:
    """``want`` rows spread as evenly as the pool allows across ``key``.

    Round-robin, not proportional. Mirroring the benchmark's own distribution
    would make the gallery three-quarters presence questions, and the reason to
    show it at all is that every question type is represented.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    order = sorted(buckets)
    while len(picked) < want and any(buckets[name] for name in order):
        for name in order:
            if buckets[name] and len(picked) < want:
                picked.append(buckets[name].pop())
    return picked


def _contract(benchmark: str, question_type: str | None) -> str | None:
    from satquery.evalcli.formatter import contract_for

    return contract_for(benchmark, question_type or "")


# --------------------------------------------------------------- change_vqa
def select_change_vqa() -> list[dict]:
    """CDVQA Val rows the adapter answered correctly in the AA 68.0% run.

    The dump carries ``sample_id``/``reference``/``formatted`` and no question
    text or imagery, so each row is joined back through CDVQA's own question ->
    image tables to the image pair it was asked about.
    """
    dump = _jsonl(ROOT / "logs/dump_heldout/Qwen_Qwen3-VL-4B-Instruct_cdvqa.jsonl")
    correct = [
        row
        for row in dump
        if str(row["formatted"]).strip().lower() == str(row["reference"]).strip().lower()
    ]

    questions = {q["id"]: q for q in _load_json("Val_questions.json")["questions"]}
    images = {i["id"]: i for i in _load_json("Val_images.json")["images"]}

    joined: list[dict] = []
    for row in correct:
        # "cdvqa_val_0016419" -> question id 16419, as stage_cdvqa.py wrote it.
        qid = int(str(row["sample_id"]).rsplit("_", 1)[-1])
        question = questions.get(qid)
        if question is None:
            continue
        record = images.get(question["img_id"])
        if record is None:
            continue
        name = record["file_name"]
        kind = row["question_type"]
        joined.append(
            {
                "item_id": f"change_vqa_{qid:07d}",
                "segment": "change_vqa",
                "status": "verified",
                "question": question["question"],
                "gold": str(row["reference"]),
                "benchmark_answer": str(row["formatted"]),
                "question_type": kind,
                "benchmark": "cdvqa",
                "contract": _contract("cdvqa", kind),
                "images": [
                    {"role": "first", "remote": f"/eval/second/im1/{name}"},
                    {"role": "second", "remote": f"/eval/second/im2/{name}"},
                ],
                "corpus": "CDVQA Val",
                "split": "val",
                "evidence": "logs/dump_heldout/Qwen_Qwen3-VL-4B-Instruct_cdvqa.jsonl",
                "headline": "AA 68.0% over 1,200 rows (45.0% blind ceiling)",
            }
        )
    return _balanced(joined, "question_type", PER_SEGMENT)


# ----------------------------------------------------------------- grounding
def select_grounding(want: int) -> list[dict]:
    """VRSBench referring items the base model localised at IoU >= 0.5."""
    run = json.loads((ROOT / "logs/pipeline_qwen_precise.json").read_text(encoding="utf-8"))
    hits = [record for record in run["records"] if record.get("hit")]
    return [
        {
            "item_id": f"grounding_{record['question_id']}",
            "segment": "grounding",
            "status": "verified",
            "question": record["question"],
            "gold": record["reference"],
            "benchmark_answer": record.get("box"),
            "iou": round(float(record["iou"]), 4),
            "question_type": record.get("obj_cls"),
            "benchmark": "vrsbench_referring",
            "contract": "box_iou_0.5",
            "images": [
                {
                    "role": "primary",
                    "remote": f"/eval/vrsbench_val/Images_val/{record['image_id']}",
                }
            ],
            "corpus": "VRSBench referring, EVAL split",
            "split": "val",
            "evidence": "logs/pipeline_qwen_precise.json",
            "headline": "62.7% acc@0.5, above fine-tuned GeoChat's 60.6%",
        }
        for record in _balanced(hits, "obj_cls", want)
    ]


# ---------------------------------------------------------------- captioning
def select_captioning(want: int) -> list[dict]:
    """VRSBench captions, ranked by ROUGE-L.

    ``scored``, never ``verified``: a caption is not right or wrong, and
    presenting a ROUGE-L as a verdict would misrepresent what was measured.
    """
    run = json.loads((ROOT / "logs/caption_strong.json").read_text(encoding="utf-8"))
    floor = float(run["blind_floor"]["rouge_l"])
    ranked = sorted(run["examples"], key=lambda e: -float(e.get("rouge_l") or 0))
    return [
        {
            "item_id": f"captioning_{Path(example['image_id']).stem}",
            "segment": "captioning",
            "status": "scored",
            "question": "Describe this image.",
            "gold": example["reference"],
            "benchmark_answer": example["prediction"],
            "rouge_l": round(float(example["rouge_l"]), 4),
            "blind_floor_rouge_l": round(floor, 4),
            "question_type": "caption",
            "benchmark": "vrsbench_caption",
            "contract": "rouge_l",
            "images": [
                {
                    "role": "primary",
                    "remote": f"/eval/vrsbench_val/Images_val/{example['image_id']}",
                }
            ],
            "corpus": "VRSBench captioning, EVAL split",
            "split": "val",
            "evidence": "logs/caption_strong.json",
            "headline": "ROUGE-L 0.252 against a 0.222 blind floor",
        }
        for example in ranked[:want]
    ]


# -------------------------------------------------------------------- rs_vqa
def select_rs_vqa() -> list[dict]:
    """RSVQA test rows. Candidates -- no per-item dump was ever written.

    LR and HR together, balanced across question type so ``count`` and ``area``
    are represented rather than crowded out by the presence questions that
    dominate both sets.
    """
    rows: list[dict] = []
    for relative, corpus, benchmark in (
        ("data/eval/rsvqa_lr/rsvqa_lr_test.jsonl", "RSVQA-LR test", "rsvqa_lr"),
        ("data/eval/rsvqa_hr/rsvqa_hr_test.jsonl", "RSVQA-HR test", "rsvqa_hr"),
    ):
        path = ROOT / relative
        if not path.exists():
            continue
        for index, row in enumerate(_jsonl(path)):
            images = row.get("images") or []
            if not images:
                continue
            kind = row.get("question_type") or row.get("type") or "unknown"
            rows.append(
                {
                    "_qtype": f"{benchmark}/{kind}",
                    "item_id": f"rs_vqa_{benchmark}_{row.get('sample_id') or index}",
                    "segment": "rs_vqa",
                    "status": "candidate",
                    "question": row["question"],
                    "gold": str(row["answer"]),
                    "question_type": kind,
                    "benchmark": benchmark,
                    "contract": _contract(benchmark, kind),
                    "images": [
                        {
                            "role": "primary",
                            "local_src": f"{Path(relative).parent.as_posix()}/{images[0]}",
                        }
                    ],
                    "corpus": corpus,
                    "split": "test",
                    "evidence": None,
                    "headline": "RSVQA-HR 85.06 / RSVQA-LR 83.08",
                }
            )
    return [
        {k: v for k, v in row.items() if k != "_qtype"}
        for row in _balanced(rows, "_qtype", PER_SEGMENT)
    ]


# ----------------------------------------------------------------------- SAR
def select_sar() -> list[dict]:
    """reBEN held-out rows for the BIFOLD radar path. Candidates.

    ``val``, because that is the split ``eval_bifold.py`` scored 74.95% on. We
    never trained BIFOLD -- it is a published MIT-licensed model -- so val is
    held out for us in the strongest sense: no part of our stack has seen it.
    """
    rows = [
        row
        for row in _jsonl(SRC / "optsar_fusion.jsonl")
        if row.get("split") == "val" and row.get("modality_arm") in {"sar", "fused"}
    ]
    out: list[dict] = []
    for row in _balanced(rows, "question_type", PER_SEGMENT):
        out.append(
            {
                "item_id": f"sar_{row['sample_id']}",
                "segment": "sar",
                "status": "candidate",
                "question": row["question"],
                "gold": str(row["answer"]),
                "question_type": row.get("question_type"),
                "benchmark": "bifold_reben",
                "contract": "yes_no",
                # The corpus already names them: a fused row carries the S2
                # optical patch and the S1 radar patch, in that order. Calling
                # both "primary" -- which this did -- collapsed the pair onto
                # one filename on the way down, so the radar half overwrote the
                # optical half and the cards rendered blank.
                "images": [
                    {"role": role, "remote": str(image).replace("/data", "", 1)}
                    for image, role in zip(
                        row.get("images") or [],
                        row.get("image_roles") or [],
                        strict=True,
                    )
                ],
                "corpus": "reBEN (BigEarthNet v2.0), held-out",
                "split": "val",
                "evidence": None,
                "headline": "resnet50-s1 74.95% against a 50.1% floor",
            }
        )
    return out


def cmd_select(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    items += select_change_vqa()
    items += select_grounding(25)
    items += select_captioning(25)
    items += select_rs_vqa()
    items += select_sar()

    for item in items:
        item["trained_on"] = False

    manifest = {
        "generated_for": "SatQuery testing-corpus gallery",
        "seed": SEED,
        "counts": dict(Counter(item["segment"] for item in items)),
        "statuses": dict(Counter(item["status"] for item in items)),
        "items": items,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"}, indent=1))
    print(f"{len(items)} item(s) -> {MANIFEST}")
    return 0


def cmd_fetch(args) -> int:
    """Download only the images the manifest names, from each segment's workspace."""
    accounts = _accounts()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    copies: list[tuple[Path, Path]] = []

    for item in manifest["items"]:
        for image in item["images"]:
            source_name = image.get("remote") or image.get("local_src") or ""
            suffix = Path(source_name).suffix or ".png"
            target = IMAGES / item["segment"] / f"{item['item_id']}_{image['role']}{suffix}"
            image["path"] = str(target.relative_to(ROOT)).replace("\\", "/")
            if target.exists():
                continue
            if image.get("remote"):
                wanted[accounts[item["segment"]]].append((image["remote"], target))
            elif image.get("local_src"):
                copies.append((ROOT / image["local_src"], target))

    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            target.write_bytes(source.read_bytes())
        else:
            print(f"  missing local source: {source}")

    def _get(account: str, remote: str, target: Path) -> str | None:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["modal", "volume", "get", "satquery-data", remote, str(target), "--force"],
            env={**os.environ, "MODAL_PROFILE": account, "MSYS_NO_PATHCONV": "1"},
            capture_output=True,
            text=True,
            # Modal draws a box-drawing progress frame; on Windows the default
            # cp1252 decode raises inside subprocess's reader thread and the
            # transfer is reported as a failure it never had.
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return None
        tail = (result.stderr or "").strip().splitlines()
        return f"  FAIL {remote}: {tail[-1] if tail else 'unknown'}"

    for account, entries in wanted.items():
        print(f"== {account}: {len(entries)} file(s)", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_get, account, remote, target) for remote, target in entries]
            for done, future in enumerate(as_completed(futures), 1):
                problem = future.result()
                if problem:
                    print(problem, flush=True)
                if done % 25 == 0:
                    print(f"  {done}/{len(entries)}", flush=True)

    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    have = sum(
        1 for i in manifest["items"] for im in i["images"] if (ROOT / im["path"]).exists()
    )
    total = sum(len(i["images"]) for i in manifest["items"])
    print(f"{have}/{total} image(s) present")
    return 0


def cmd_merge(args) -> int:
    """Fold a verification run back into the manifest.

    Statuses are *earned* here, never assumed: an item the run answered
    correctly becomes ``verified``, one it got wrong becomes ``failed`` and
    keeps the reason, and captioning stays ``scored`` because a ROUGE-L above a
    blind floor is not a verdict. The per-segment pass rate is recorded at the
    top so the page can state what fraction of its own sample actually held up.

    Merging rather than replacing, so a run over one segment does not discard
    what is known about the others.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    by_id = {row["item_id"]: row for row in results.get("items", [])}

    for item in manifest["items"]:
        row = by_id.get(item["item_id"])
        if row is None:
            continue
        item["served_answer"] = row.get("served_answer")
        item["passed"] = row.get("passed")
        item["reason"] = row.get("reason")
        item["tools"] = row.get("tools")
        if item["status"] != "scored":
            item["status"] = "verified" if row.get("passed") else "failed"

    verification = manifest.setdefault("verification", {})
    verification["build"] = results.get("build") or verification.get("build")
    verification["base"] = results.get("base")
    verification.setdefault("per_segment", {}).update(results.get("per_segment", {}))
    manifest["statuses"] = dict(Counter(item["status"] for item in manifest["items"]))

    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"statuses": manifest["statuses"], "verification": verification}, indent=1))
    return 0


def cmd_prune(args) -> int:
    """Drop items that failed verification, keeping the count honest.

    The gallery is meant to show rows the system answers correctly. Keeping a
    failure in it and labelling it ``failed`` is more honest but not what the
    page is for; dropping it silently would hide the pass rate. So the pruned
    count stays in ``verification`` and the page states it.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before = len(manifest["items"])
    kept = [item for item in manifest["items"] if item.get("status") != "failed"]
    manifest["items"] = kept
    manifest["counts"] = dict(Counter(item["segment"] for item in kept))
    manifest["statuses"] = dict(Counter(item["status"] for item in kept))
    manifest.setdefault("verification", {})["pruned"] = before - len(kept)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"kept {len(kept)}/{before}; dropped {before - len(kept)} that failed")
    print(json.dumps(manifest["counts"], indent=1))
    return 0


#: What each segment should hold once failures are dropped.
TARGETS = {"rs_vqa": 50, "change_vqa": 50, "grounding": 25, "captioning": 25, "sar": 50}


def cmd_topup(args) -> int:
    """Replace pruned items with fresh ones from the same pools.

    Verification drops whatever the system actually got wrong, which leaves each
    segment short. Rather than re-run all 200 to refill -- GPU time on an
    account with a budget -- this picks only as many new rows as were lost, and
    marks them ``candidate`` so the next verification pass touches those alone.

    Replacements are drawn from the same balanced pools and exclude everything
    already in the manifest, so a failed row is never silently reselected.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {item["item_id"] for item in manifest["items"]}

    # Everything that has ever failed, read back from the verification runs.
    #
    # Excluding only what is in the manifest is not enough and was actively
    # wrong: `prune` has just removed the failures, so they are absent from the
    # manifest and a balanced pool hands back the same rows in the same order.
    # The first run of this reselected all 31 failures verbatim -- a top-up that
    # replaced each dropped row with itself.
    for results in sorted(OUT.glob("verified*.json")):
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        have |= {
            row["item_id"] for row in data.get("items", []) if not row.get("passed")
        }

    counts = Counter(item["segment"] for item in manifest["items"])

    pools = {
        "change_vqa": select_change_vqa,
        "grounding": lambda: select_grounding(TARGETS["grounding"] * 4),
        "rs_vqa": select_rs_vqa,
        "sar": select_sar,
    }

    added: list[dict] = []
    for segment, target in TARGETS.items():
        short = target - counts.get(segment, 0)
        if short <= 0 or segment not in pools:
            continue
        # PER_SEGMENT is what the selectors size to; widen it so the balanced
        # pool is deeper than the manifest and there is something new to take.
        global PER_SEGMENT
        original, PER_SEGMENT = PER_SEGMENT, target * 4
        try:
            candidates = [row for row in pools[segment]() if row["item_id"] not in have]
        finally:
            PER_SEGMENT = original
        chosen = candidates[:short]
        for row in chosen:
            row["trained_on"] = False
            row["status"] = "candidate"
        added.extend(chosen)
        print(f"{segment}: short {short}, adding {len(chosen)}")

    manifest["items"].extend(added)
    manifest["counts"] = dict(Counter(i["segment"] for i in manifest["items"]))
    manifest["statuses"] = dict(Counter(i["status"] for i in manifest["items"]))
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "statuses": manifest["statuses"]}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").set_defaults(func=cmd_select)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--workers", type=int, default=12)
    fetch.set_defaults(func=cmd_fetch)
    merge = sub.add_parser("merge")
    merge.add_argument("--results", default=str(OUT / "verified.json"))
    merge.set_defaults(func=cmd_merge)
    sub.add_parser("prune").set_defaults(func=cmd_prune)
    sub.add_parser("topup").set_defaults(func=cmd_topup)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
