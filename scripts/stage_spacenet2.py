"""Stage SpaceNet 2 building footprints for ``rs_ground_caption``.

    python scripts/stage_spacenet2.py plan  --tiles 6000
    python scripts/stage_spacenet2.py fetch --plan /data/eval/sn2/plan.json

**Why this exists alongside SpaceNet 6.** SpaceNet 6 is one city. Its 3,401
Rotterdam tiles were exhausted by the first G3 corpus -- 2,627 and 2,459 rows
against a 4,000 cap -- so the building half of grounding cannot grow without a
second source. SpaceNet 2 is 0.3 m across **four cities** (Vegas, Paris,
Shanghai, Khartoum) with 302,701 footprints against SpaceNet 6's ~48,000.

**Diversity is the point, so tiles are drawn evenly across AOIs.** Vegas alone
carries more tiles than Khartoum; a proportional draw would teach American
suburban tract housing and call it buildings. The hidden set is Cartosat over
India, which resembles none of the four -- so coverage across building
vernaculars matters more than matching any one distribution. This is the same
reasoning ``stage_ben_txt.py`` applies to country x season strata.

**Tile ids are namespaced by AOI.** Every AOI numbers its images from img1, so
a bare id collides four ways. Filenames on disk already carry the AOI, so only
the plan's keys need qualifying.

SpaceNet 6 stays regardless: it is the only co-registered optical-SAR source we
hold, and ``optsar_fusion`` needs it whatever happens to G3.

Licence: CC BY-SA 4.0, SpaceNet Partners, via the AWS Open Data Program.
"""

import argparse
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUCKET = "https://spacenet-dataset.s3.amazonaws.com"
PREFIX = "spacenet/SN2_buildings/train"

#: The four AOIs, in the dataset's own numbering.
AOIS = ("AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum")

GSD_M = 0.3
SOURCE = "SpaceNet 2 (Vegas/Paris/Shanghai/Khartoum)"
LICENCE = "CC BY-SA 4.0"

#: Filenames end ``..._img1234.tif`` / ``..._img1234.geojson``. SpaceNet 6 uses
#: ``_tile_``; this series uses ``_img``, and reusing the wrong pattern silently
#: matches nothing and plans an empty fetch.
_IMG = re.compile(r"_img(\d+)\.")

#: Labels always come along: a tile without footprints carries no grounding
#: target and is dead weight on the Volume.
LABELS = "geojson_buildings"


def _get(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _list_all(prefix: str) -> list[tuple[str, int]]:
    """Every key and size under ``prefix``, paginated.

    Paginated because an AOI holds several thousand objects against S3's 1,000
    per response, and a truncated listing plans a silently partial fetch.
    """
    out: list[tuple[str, int]] = []
    token = ""
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        body = _get(url).decode("utf-8", "replace")
        out += [
            (key, int(size))
            for key, size in re.findall(
                r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", body, re.S
            )
        ]
        match = re.search(r"<NextContinuationToken>([^<]+)</", body)
        if not match:
            return out
        token = match.group(1)


def image_id(key: str) -> str:
    match = _IMG.search(key)
    return match.group(1) if match else ""


def cmd_plan(args) -> int:
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if LABELS not in modalities:
        modalities.append(LABELS)

    per_aoi: dict[str, list[str]] = {}
    listings: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)

    for aoi in AOIS:
        found: dict[str, dict[str, tuple[str, int]]] = {}
        for modality in modalities:
            entries = _list_all(f"{PREFIX}/{aoi}/{modality}/")
            found[modality] = {
                image_id(k): (k, s) for k, s in entries if image_id(k)
            }
        # Only images present in EVERY modality are usable: one with pixels but
        # no footprints cannot carry a box.
        common = set.intersection(*(set(v) for v in found.values()))
        for modality, by_id in found.items():
            for img in common:
                listings[modality][f"{aoi}/{img}"] = by_id[img]
        per_aoi[aoi] = sorted(common, key=int)
        size = sum(found[m][i][1] for i in common for m in modalities)
        print(
            f"  {aoi:18} {len(common):6} image(s) in all "
            f"{len(modalities)} modality/ies  {size / 1073741824:6.2f} GB"
        )

    # Round-robin across AOIs so --tiles splits evenly between cities rather
    # than by however many each happens to contain.
    rng = random.Random(args.seed)
    pools = {}
    for aoi, ids in per_aoi.items():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        pools[aoi] = shuffled

    chosen: list[str] = []
    limit = args.tiles or sum(len(v) for v in pools.values())
    index = 0
    while len(chosen) < limit and any(index < len(v) for v in pools.values()):
        for aoi in AOIS:
            if index < len(pools[aoi]) and len(chosen) < limit:
                chosen.append(f"{aoi}/{pools[aoi][index]}")
        index += 1

    plan = {tile: {m: listings[m][tile][0] for m in modalities} for tile in chosen}
    planned_bytes = sum(listings[m][t][1] for t in chosen for m in modalities)

    by_city: dict[str, int] = defaultdict(int)
    for tile in chosen:
        by_city[tile.split("/")[0]] += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "source": SOURCE,
        "licence": LICENCE,
        "gsd_m": GSD_M,
        "modalities": modalities,
        "aois": dict(by_city),
        "tiles": plan,
    }
    (out / "plan.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"\nper AOI: {dict(by_city)}")
    print(
        f"{len(chosen)} image(s) x {len(modalities)} modality/ies = "
        f"{len(chosen) * len(modalities)} file(s), "
        f"{planned_bytes / 1073741824:.2f} GB -> {out / 'plan.json'}"
    )
    return 0


def cmd_fetch(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.plan).parent

    wanted: list[tuple[str, Path]] = []
    for tile, modalities in plan["tiles"].items():
        aoi = tile.split("/")[0]
        for modality, key in modalities.items():
            # Kept under the AOI so a city can be inspected, counted or dropped
            # without parsing every filename.
            wanted.append((key, root / aoi / modality / Path(key).name))

    todo = [(k, p) for k, p in wanted if not p.exists()]
    print(f"{len(wanted) - len(todo)} present; {len(todo)} to fetch", flush=True)

    done = failed = 0
    for key, target in todo:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = _get(f"{BUCKET}/{urllib.parse.quote(key)}")
        except Exception as error:  # noqa: BLE001 - one bad file is a result
            print(f"  FAILED {key}: {error!r}", flush=True)
            failed += 1
            continue
        # Through a temporary name: a half-written file already exists as far as
        # the resume check is concerned and would be skipped for good.
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(todo)} fetched", flush=True)

    summary = {
        "source": SOURCE,
        "licence": LICENCE,
        "gsd_m": GSD_M,
        "modalities": plan["modalities"],
        "aois": plan.get("aois", {}),
        "files_wanted": len(wanted),
        "files_present": sum(1 for _, p in wanted if p.exists()),
        "failed": failed,
    }
    (root / "fetch_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="choose images evenly across the four AOIs")
    plan.add_argument(
        "--modalities",
        default="PS-RGB",
        help="comma-separated; geojson_buildings is always added",
    )
    plan.add_argument(
        "--tiles",
        type=int,
        default=0,
        help="total across all AOIs, split evenly; 0 takes everything",
    )
    plan.add_argument("--out", default="/data/eval/sn2")
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(func=cmd_plan)

    fetch = sub.add_parser("fetch", help="pull the planned images, resumably")
    fetch.add_argument("--plan", default="/data/eval/sn2/plan.json")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
