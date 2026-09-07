"""Stage SpaceNet 7 / MUDS from AWS Open Data into the Volume.

    python scripts/stage_spacenet7.py plan  --aois 40 --months 12 --out /data/eval/sn7
    python scripts/stage_spacenet7.py fetch --plan /data/eval/sn7/plan.json

**Two passes, because a fetch that cannot resume is a fetch that must not be
interrupted.** The plan is a small JSON naming exactly which AOIs, months and
label files to pull; the fetch reads it and skips whatever is already on disk.
The RSVQA-HR staging died overnight when a laptop slept, having got 1,160 of
3,000 images, and only survived because the plan existed to resume from.

**The extracted tree, not the tarball.** SpaceNet publishes both
``tarballs/SN7_buildings_train.tar.gz`` (9.16 GB, sequential) and an extracted
``train/`` prefix. Taking the tree means fetching only the AOIs and months the
plan chose, at 59-62 MB/s measured from Modal to this bucket -- the fastest
source in the inventory, against Zenodo's 1-5 MB/s.

**Why ``labels_match`` and not ``labels``.** Only the matched labels carry the
persistent ``Id`` that makes a building the *same* building across months. That
identity is the entire reason this dataset is the primary change source: set
arithmetic on ID sets between two dates yields an exact change inventory -- new,
demolished, unchanged -- rather than a thresholded guess. Without it these are
just 25 unrelated building maps.

Licence: CC BY-SA 4.0, AWS Open Data Program. Imagery is 4 m Planet, RGB only:
there is no near-infrared or SWIR band here, so a SpaceNet 7 sample carries one
composite per date and not three.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUCKET = "https://spacenet-dataset.s3.amazonaws.com"
PREFIX = "spacenet/SN7_buildings/train"

#: 4 m Planet mosaics. Stated once here rather than inferred per-file, because
#: the manifest's ``effective_gsd_m`` drives the scale prefix the model is given
#: and a wrong value there is worse than no value.
GSD_M = 4.0
SOURCE = "SpaceNet 7 / MUDS"
LICENCE = "CC BY-SA 4.0"

#: The month embedded in every filename: global_monthly_2018_01_mosaic_<aoi>.tif
_MONTH = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic")


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _list(prefix: str, delimiter: str = "/") -> tuple[list[str], list[str]]:
    """One page of keys and common prefixes under ``prefix``.

    S3 caps a listing at 1000 entries and SpaceNet 7 fits comfortably inside
    that at every level we walk, so this deliberately does not paginate: a
    silent truncation would look like a smaller dataset rather than an error.
    """
    url = (
        f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
        f"&delimiter={delimiter}&max-keys=1000"
    )
    body = _get(url).decode("utf-8", "replace")
    if "<IsTruncated>true</IsTruncated>" in body:
        raise SystemExit(
            f"listing of {prefix!r} was truncated at 1000 entries. Paginate "
            "before trusting this result -- a short list here reads as a small "
            "dataset rather than a bug."
        )
    keys = re.findall(r"<Key>([^<]+)</Key>", body)
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", body)
    return keys, [p for p in prefixes if p != prefix]


def month_of(name: str) -> str:
    match = _MONTH.search(name)
    return f"{match.group(1)}-{match.group(2)}" if match else ""


def cmd_plan(args) -> int:
    _, aoi_prefixes = _list(f"{PREFIX}/")
    aois = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in aoi_prefixes)
    if not aois:
        raise SystemExit(f"no AOIs found under {PREFIX}; the bucket layout changed")
    print(f"{len(aois)} AOI(s) available")
    chosen = aois[: args.aois] if args.aois else aois

    plan: dict[str, dict] = {}
    for index, aoi in enumerate(chosen, 1):
        images, _ = _list(f"{PREFIX}/{aoi}/images_masked/", delimiter="")
        labels, _ = _list(f"{PREFIX}/{aoi}/labels_match/", delimiter="")
        by_month = {month_of(Path(k).name): k for k in images if month_of(Path(k).name)}
        label_by_month = {
            month_of(Path(k).name): k for k in labels if month_of(Path(k).name)
        }
        # Only months with BOTH an image and a matched label are usable: a date
        # without labels cannot take part in a change pair, and keeping it would
        # inflate the plan with files that generate nothing.
        months = sorted(set(by_month) & set(label_by_month))
        if args.months:
            months = months[: args.months]
        if len(months) < 2:
            print(f"  {aoi}: {len(months)} usable month(s) -- skipped, needs 2 to pair")
            continue
        plan[aoi] = {
            "months": months,
            "images": {m: by_month[m] for m in months},
            "labels": {m: label_by_month[m] for m in months},
        }
        print(f"  [{index}/{len(chosen)}] {aoi}: {len(months)} month(s)")

    pairs = sum(len(v["months"]) * (len(v["months"]) - 1) // 2 for v in plan.values())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "source": SOURCE,
        "licence": LICENCE,
        "gsd_m": GSD_M,
        "composites_per_date": 1,
        "aois": plan,
    }
    (out / "plan.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    files = sum(len(v["images"]) + len(v["labels"]) for v in plan.values())
    print(
        f"\n{len(plan)} AOI(s), {files} file(s) to fetch, "
        f"{pairs:,} possible date pair(s) -> {out / 'plan.json'}"
    )
    return 0


def cmd_fetch(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.plan).parent
    wanted: list[tuple[str, Path]] = []
    for aoi, entry in plan["aois"].items():
        for kind in ("images", "labels"):
            for key in entry[kind].values():
                wanted.append((key, root / aoi / kind / Path(key).name))

    todo = [(k, p) for k, p in wanted if not p.exists()]
    print(f"{len(wanted) - len(todo)} file(s) already present; {len(todo)} to go",
          flush=True)

    done = 0
    for key, target in todo:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = _get(f"{BUCKET}/{urllib.parse.quote(key)}")
        except Exception as error:  # noqa: BLE001 - one bad file is a result
            print(f"  FAILED {key}: {error!r}", flush=True)
            continue
        # Write via a temporary name: a half-written file that already exists
        # would be skipped by the resume check above and silently corrupt a run.
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(todo)} fetched", flush=True)

    summary = {
        "source": SOURCE,
        "licence": LICENCE,
        "files_wanted": len(wanted),
        "files_present": sum(1 for _, p in wanted if p.exists()),
    }
    (root / "fetch_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="choose AOIs, months and label files")
    plan.add_argument("--aois", type=int, default=0, help="0 = every AOI")
    plan.add_argument("--months", type=int, default=0, help="0 = every month")
    plan.add_argument("--out", default="/data/eval/sn7")
    plan.set_defaults(func=cmd_plan)

    fetch = sub.add_parser("fetch", help="pull the planned files, resumably")
    fetch.add_argument("--plan", default="/data/eval/sn7/plan.json")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
