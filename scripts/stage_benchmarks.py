"""Stage a downloaded benchmark into canonical JSONL at its policy GSD.

    python scripts/stage_benchmarks.py --list
    python scripts/stage_benchmarks.py --dataset rsvqa_lr --source ~/downloads/RSVQA-LR \
        --out data/eval/rsvqa_lr
    python scripts/stage_benchmarks.py --dataset rsvqa_hr --source ~/downloads/RSVQA-HR \
        --out data/eval/rsvqa_hr --split test

This is the *staging* half only. It does not download anything, and that is
deliberate rather than unfinished: ``docs/02-data/phase-0-resolution-policy.md``
cites these sources as papers and project pages, not as machine-fetchable
archives, and several carry access conditions a script must not click through on
someone's behalf. ``--list`` prints the documented source and the directory
layout expected for each; the download is an operator action, the conversion is
this script's job.

What the conversion actually does is the part that matters, because it is where
the resolution policy stops being a document and becomes pixels:

* **RSVQA-HR is downsampled 2x, to 0.30 m.** Section 4 of the policy: a 10 m
  object is 33 px at 0.30 m, far above the 12--16 px a VQA question needs, and
  0.15 m is *finer than Cartosat-2S pan* (0.65 m) -- training there widens the
  very domain gap C27 exists to close. It also cuts that source from 256 vision
  tokens to 64, on 20% of the ``rs_vqa`` mix.
* **RSVQA-LR is staged as shipped**, 256 px at 10 m.
* **CDVQA and VRSBench eval imagery is never altered.** The policy is explicit:
  preprocessing at eval must match training per-source, and an eval set someone
  has resampled is no longer comparable to the published numbers.
* **VRSBench is evaluation-only.** C55/C56 exclude it from training outright, so
  every row it emits is stamped ``split: test`` regardless of what is asked for,
  and the licence field says so.

Each staged row carries ``effective_gsd_m`` -- the GSD after any resample, not
the native one. Section 5.4 conditions the prompt on that number, and
conditioning on 0.15 m for pixels that are now 0.30 m teaches the model a scale
its imagery does not have.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark's staging policy, straight out of the resolution doc."""

    name: str
    task: str
    adapter: str
    native_gsd_m: float
    #: Divisor applied to the long side. 1 means "as shipped".
    downsample: int
    effective_gsd_m: float
    target_px: int
    licence: str
    source: str
    layout: str
    #: True for eval-only sets, which are never resampled and never train.
    eval_only: bool = False
    train_forbidden: bool = False

    def note(self) -> str:
        if self.downsample == 1:
            return "staged as shipped"
        return (
            f"downsampled {self.downsample}x, {self.native_gsd_m:g} m -> "
            f"{self.effective_gsd_m:g} m"
        )


SPECS: dict[str, BenchmarkSpec] = {
    "rsvqa_lr": BenchmarkSpec(
        name="RSVQA-LR",
        task="single_vqa",
        adapter="rs_vqa",
        native_gsd_m=10.0,
        downsample=1,
        effective_gsd_m=10.0,
        target_px=256,
        licence="CC-BY-4.0 (Sentinel-2 derived)",
        source="arXiv 2003.07333 — LR 256^2 Sentinel-2 @ 10 m",
        layout="images/*.tif plus the published questions/answers JSON",
    ),
    "rsvqa_hr": BenchmarkSpec(
        name="RSVQA-HR",
        task="single_vqa",
        adapter="rs_vqa",
        native_gsd_m=0.15,
        downsample=2,
        effective_gsd_m=0.30,
        target_px=256,
        licence="USGS HRO — public domain imagery",
        source='arXiv 2003.07333 — HR 512^2 @ "15cm resolution aerial RGB, USGS" HRO',
        layout="Data/*.tif plus the published questions/answers JSON",
    ),
    "cdvqa": BenchmarkSpec(
        name="CDVQA",
        task="change_vqa",
        adapter="change_vqa",
        native_gsd_m=0.5,
        downsample=1,
        effective_gsd_m=0.5,
        target_px=512,
        licence="see the CDVQA release terms",
        source="CDVQA change-detection VQA release",
        layout="im1/*.png, im2/*.png and the published question JSON",
        eval_only=True,
    ),
    "vrsbench": BenchmarkSpec(
        name="VRSBench",
        task="single_grounding",
        adapter="rs_ground_caption",
        native_gsd_m=0.3,
        downsample=1,
        effective_gsd_m=0.3,
        target_px=512,
        licence="EVALUATION ONLY — excluded from training by C55/C56",
        source="VRSBench referring-grounding and captioning benchmark",
        layout="Images/*.png plus the published annotation JSON",
        eval_only=True,
        train_forbidden=True,
    ),
}


def _find_annotations(root: Path) -> list[Path]:
    """Every JSON/JSONL in the download, shallowest first.

    Layouts differ between releases and mirrors, so the annotation file is
    located rather than assumed. Nothing is guessed about its *contents*: the
    reader below requires named fields and refuses anything else.
    """
    found = [p for p in root.rglob("*.json")] + [p for p in root.rglob("*.jsonl")]
    return sorted(found, key=lambda p: (len(p.relative_to(root).parts), p.name))


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("questions", "annotations", "data", "images"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(
        f"{path} is a JSON object with no recognised list under "
        "'questions', 'annotations', 'data' or 'images'. Point --annotations at "
        "the file that holds the question list."
    )


_IMAGE_KEYS = ("img_id", "image_id", "image", "img", "file_name", "filename")
_QUESTION_KEYS = ("question", "input", "query", "text")
_ANSWER_KEYS = ("answer", "output", "label", "response")
_TYPE_KEYS = ("type", "question_type", "category")


def _first(record: dict, keys: tuple[str, ...]):
    for key in keys:
        if record.get(key) not in (None, ""):
            return record[key]
    return None


def _resolve_image(root: Path, value: Any, index: dict[str, Path]) -> Path | None:
    """Match an annotation's image reference against the files on disk."""
    if value is None:
        return None
    name = Path(str(value).replace("\\", "/")).name
    stem = name.rsplit(".", 1)[0]
    direct = root / str(value).replace("\\", "/")
    if direct.exists():
        return direct
    return index.get(name) or index.get(stem)


def _resample(source: Path, target: Path, spec: BenchmarkSpec) -> None:
    """Write the staged chip, downsampling only where the policy says to."""
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if spec.downsample > 1:
            # Area-averaging, not nearest. A 2x nearest decimation throws away
            # three of every four measurements; LANCZOS keeps the signal that
            # justified calling 0.30 m sufficient in the first place.
            size = (image.width // spec.downsample, image.height // spec.downsample)
            image = image.resize(size, Image.LANCZOS)
        image.save(target)


def stage(args, spec: BenchmarkSpec) -> dict[str, Any]:
    root = Path(args.source).expanduser()
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Download the release first; see --list.")

    annotations = (
        [Path(args.annotations)] if args.annotations else _find_annotations(root)
    )
    if not annotations:
        raise SystemExit(
            f"No JSON or JSONL annotation file under {root}. Expected layout: "
            f"{spec.layout}"
        )

    records: list[dict] = []
    used: list[Path] = []
    for candidate in annotations:
        try:
            found = _read_records(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
        if found and isinstance(found[0], dict) and _first(found[0], _QUESTION_KEYS):
            records.extend(found)
            used.append(candidate)
            if args.annotations:
                break
    if not records:
        raise SystemExit(
            f"Found {len(annotations)} JSON file(s) under {root} but none holds records "
            f"with a question field ({_QUESTION_KEYS}). Pass --annotations explicitly."
        )
    print(f"{len(records)} record(s) from {[p.name for p in used]}")

    index: dict[str, Path] = {}
    for pattern in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        for path in root.rglob(pattern):
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)

    out_root = Path(args.out)
    split = "test" if spec.train_forbidden else args.split

    rows: list[dict] = []
    missing = 0
    staged_images: dict[Path, str] = {}
    for number, record in enumerate(records):
        if args.limit and len(rows) >= args.limit:
            break
        question = _first(record, _QUESTION_KEYS)
        answer = _first(record, _ANSWER_KEYS)
        if question is None or answer is None:
            continue
        source_image = _resolve_image(root, _first(record, _IMAGE_KEYS), index)
        if source_image is None:
            missing += 1
            continue

        if source_image not in staged_images:
            relative = f"images/{source_image.stem}.png"
            _resample(source_image, out_root / relative, spec)
            staged_images[source_image] = relative
        relative = staged_images[source_image]

        rows.append(
            {
                "sample_id": f"{spec.name.lower()}_{number:06d}",
                "adapter": spec.adapter,
                "task": spec.task,
                "images": [relative],
                "question": str(question),
                "answer": str(answer),
                "answer_type": "text",
                "image_roles": ["true_colour"],
                "modality": ["optical"],
                "effective_gsd_m": [spec.effective_gsd_m],
                "split": split,
                "source": spec.name,
                "licence": spec.licence,
                "question_type": str(_first(record, _TYPE_KEYS) or "all"),
                "native_gsd_m": spec.native_gsd_m,
                "staging": spec.note(),
                "eval_only": spec.eval_only,
            }
        )

    if not rows:
        raise SystemExit(
            "Every record was dropped: no annotation could be matched to an image on "
            f"disk ({missing} unmatched). Check that --source points at the release "
            f"root. Expected layout: {spec.layout}"
        )

    manifest = out_root / f"{args.dataset}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "dataset": spec.name,
        "manifest": str(manifest),
        "rows": len(rows),
        "images_staged": len(staged_images),
        "records_without_image": missing,
        "native_gsd_m": spec.native_gsd_m,
        "effective_gsd_m": spec.effective_gsd_m,
        "staging": spec.note(),
        "split": split,
        "licence": spec.licence,
        "eval_only": spec.eval_only,
    }
    (out_root / f"{args.dataset}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"{len(rows)} row(s), {len(staged_images)} image(s) — {spec.note()}")
    if missing:
        print(
            f"{missing} record(s) referenced an image not present on disk and were "
            "dropped. They are counted, not silently ignored: a benchmark quietly "
            "short by 30% scores differently from the published one."
        )
    if spec.train_forbidden:
        print(
            f"{spec.name} is evaluation-only under C55/C56. Every row is stamped "
            "split=test regardless of --split, and it must never enter a training mix."
        )
    print(f"Manifest written to {manifest}")
    return summary


def _print_list() -> None:
    print("Download these yourself, then stage with --dataset/--source.\n")
    for key, spec in SPECS.items():
        flags = []
        if spec.eval_only:
            flags.append("eval-only")
        if spec.train_forbidden:
            flags.append("TRAINING FORBIDDEN (C55/C56)")
        print(f"{key}")
        print(f"  name       {spec.name}{'  [' + ', '.join(flags) + ']' if flags else ''}")
        print(f"  source     {spec.source}")
        print(f"  layout     {spec.layout}")
        print(f"  staging    {spec.note()} -> {spec.target_px}px, adapter {spec.adapter}")
        print(f"  licence    {spec.licence}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="print sources and layouts")
    parser.add_argument("--dataset", choices=sorted(SPECS))
    parser.add_argument("--source", help="the downloaded release root")
    parser.add_argument("--annotations", default=None, help="explicit annotation file")
    parser.add_argument("--out", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.list:
        _print_list()
        return 0
    if not args.dataset or not args.source:
        parser.error("--dataset and --source are required (or --list)")
    args.out = args.out or f"data/eval/{args.dataset}"

    stage(args, SPECS[args.dataset])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
