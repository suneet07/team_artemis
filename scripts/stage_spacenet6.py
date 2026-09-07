"""Stage SpaceNet 6 / MSAW from AWS Open Data into the Volume.

    python scripts/stage_spacenet6.py plan  --modalities PS-RGB --tiles 2000
    python scripts/stage_spacenet6.py fetch --plan /data/eval/sn6/plan.json

**One dataset, two adapters, and they want different bytes.**
``rs_ground_caption`` needs building boxes on optical imagery: ``PS-RGB`` plus
``geojson_buildings``, 7.9 GB. ``optsar_fusion`` needs the same ground seen two
ways, which adds ``SAR-Intensity`` at 41 GB. Modal Volumes do not cross
accounts, so those are separate stagings on separate accounts -- hence
``--modalities`` rather than a fixed list. Pulling 41 GB of quad-pol SAR onto
the grounding account would be 5x the download for bytes that adapter cannot
use.

**Every modality is tile-aligned.** All four directories hold 3,401 files with
matching names, so a tile's optical, SAR and footprints are the same ground and
need no registration step. That co-registration is the whole reason this
dataset exists and the reason it is the only sub-metre optical-SAR supervision
in the inventory.

**Rotterdam, and only Rotterdam.** 120 km2 of one city. That is a real
diversity limit -- SpaceNet 7 spans 60 AOIs across six continents by
comparison -- and it is why OpenEarthMap-SAR (35 regions, Japan/France/USA) is
in the plan as the second cross-modal source. Stated here so the number this
produces is read with it in mind.

Licence: CC BY-SA 4.0, AWS Open Data Program. ~0.5 m Capella X-band quad-pol
SAR co-registered with ~0.5 m Maxar optical; ~48,000 building footprints.
"""

import argparse
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUCKET = "https://spacenet-dataset.s3.amazonaws.com"
PREFIX = "spacenet/SN6_buildings/train/AOI_11_Rotterdam"

GSD_M = 0.5
SOURCE = "SpaceNet 6 / MSAW (Rotterdam)"
LICENCE = "CC BY-SA 4.0"

#: Every modality carries the same tile id in its filename after this token, so
#: files can be matched across directories without parsing dates or indices.
_TILE = re.compile(r"_tile_(\d+)\.")

#: Labels always come along: a tile without its footprints can carry no
#: grounding target and is dead weight on the Volume.
LABELS = "geojson_buildings"


def _get(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _list_all(prefix: str) -> list[tuple[str, int]]:
    """Every key and size under ``prefix``, paginated.

    Paginated because each modality holds 3,401 objects against S3's 1,000 per
    response: one page would return a third of the dataset and look complete.
    """
    out: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        body = _get(url).decode("utf-8", "replace")
        keys = re.findall(r"<Key>([^<]+)</Key>", body)
        sizes = [int(s) for s in re.findall(r"<Size>(\d+)</Size>", body)]
        out.extend(zip(keys, sizes, strict=False))
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not match:
            return out
        token = match.group(1)


def tile_of(key: str) -> str:
    match = _TILE.search(key)
    return match.group(1) if match else ""


def cmd_plan(args) -> int:
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if LABELS not in modalities:
        modalities.append(LABELS)

    listings: dict[str, dict[str, tuple[str, int]]] = {}
    for modality in modalities:
        entries = _list_all(f"{PREFIX}/{modality}/")
        by_tile = {tile_of(k): (k, s) for k, s in entries if tile_of(k)}
        listings[modality] = by_tile
        total = sum(s for _, s in by_tile.values())
        print(f"  {modality:20} {len(by_tile):5} tile(s)  {total / 1073741824:6.2f} GB")

    # Only tiles present in EVERY requested modality are usable: a tile with
    # optical but no SAR cannot answer a cross-modal question, and one with no
    # footprints cannot carry a box.
    common = set.intersection(*(set(v) for v in listings.values()))
    dropped = {m: len(v) - len(common) for m, v in listings.items()}
    print(f"\n{len(common)} tile(s) present in all {len(modalities)} modality/ies")
    for modality, count in dropped.items():
        if count:
            print(f"  [warn] {modality}: {count} tile(s) not present in every modality")

    tiles = sorted(common, key=int)
    rng = random.Random(args.seed)
    rng.shuffle(tiles)
    if args.tiles:
        tiles = tiles[: args.tiles]

    plan = {
        tile: {m: listings[m][tile][0] for m in modalities} for tile in tiles
    }
    planned_bytes = sum(
        listings[m][t][1] for t in tiles for m in modalities
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "source": SOURCE,
        "licence": LICENCE,
        "gsd_m": GSD_M,
        "modalities": modalities,
        "tiles": plan,
    }
    (out / "plan.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(
        f"\n{len(tiles)} tile(s) x {len(modalities)} modality/ies = "
        f"{len(tiles) * len(modalities)} file(s), "
        f"{planned_bytes / 1073741824:.2f} GB -> {out / 'plan.json'}"
    )
    return 0


def cmd_fetch(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.plan).parent

    wanted: list[tuple[str, Path]] = []
    for _tile, modalities in plan["tiles"].items():
        for modality, key in modalities.items():
            wanted.append((key, root / modality / Path(key).name))

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

    plan = sub.add_parser("plan", help="choose tiles and modalities")
    plan.add_argument(
        "--modalities",
        default="PS-RGB",
        help="comma-separated; geojson_buildings is always added. PS-RGB for "
        "grounding, add SAR-Intensity for cross-modal work",
    )
    plan.add_argument("--tiles", type=int, default=0, help="0 = every tile")
    plan.add_argument("--out", default="/data/eval/sn6")
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(func=cmd_plan)

    fetch = sub.add_parser("fetch", help="pull the planned files, resumably")
    fetch.add_argument("--plan", default="/data/eval/sn6/plan.json")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
