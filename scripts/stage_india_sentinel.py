"""Acquire bi-temporal Sentinel-2 chips over Indian AOIs (C46, India holdout v0).

    python scripts/stage_india_sentinel.py plan  --out /data/eval/india --per-aoi 24
    python scripts/stage_india_sentinel.py fetch --plan /data/eval/india/plan.json

**Why this exists.** Every cleared training source is Western: BigEarthNet is
European, RSVQA-HR is the US northeast, SpaceNet 7 is 60 places worldwide with
little India, SpaceNet 6 is one city in the Netherlands. The hidden set is
Cartosat-2S and RISAT over India. This is the only source that can add Indian
geography *and* arbitrary scene diversity, because we choose the AOIs.

**Earth Search rather than CDSE.** The Copernicus Data Space fetcher in
``satquery.ingest.copernicus`` needs OIDC and S3 credentials, which are not
attached to any Modal Secret and already failed one staging run. Element 84's
Earth Search indexes the same Sentinel-2 L2A scenes as Cloud-Optimised GeoTIFFs
in a public bucket: no credentials, and a windowed read pulls one 256x256 chip
out of a 10,980x10,980 scene instead of downloading the granule.

**Three composites, matching BEN.txt.** The COGs carry blue/green/red/nir/
swir16/swir22, which is exactly enough for the corpus's existing true-colour,
false-colour and short-wave recipe. Indian rows therefore drop into the same
convention as everything else rather than being a special case -- unlike
SpaceNet 7, which is RGB-only and yields one view per date.

**Labels come from the imagery, not from a third party.** Change is measured by
differencing spectral indices between the two dates (NDVI vegetation, NDWI
water, NDBI built-up). That keeps the licence position to Copernicus alone and
follows the project's rule that answers are computed, never authored. It is
weaker supervision than SpaceNet 7's expert footprints -- it says "the built-up
index rose here", not "three buildings appeared" -- so this source carries
presence-of-change and change-type questions while SpaceNet 7 carries counts.

Licence: Copernicus open data. Contains modified Copernicus Sentinel data.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

#: Bands the three composites need. Named as Earth Search names them, so a
#: rename upstream fails loudly here rather than silently fetching the wrong
#: wavelength into a composite the model reads as true colour.
BANDS = ("red", "green", "blue", "nir", "swir16", "swir22")

#: Sentinel-2 L2A ships a per-pixel Scene Classification Layer. Fetched
#: alongside the reflectance bands but kept out of ``BANDS`` so an AOI missing
#: it is still usable -- it gates tiles, it does not enter a composite.
SCL_BAND = "scl"

#: SCL classes that make a pixel unusable: saturated, cloud shadow, cloud at
#: medium and high probability, thin cirrus. Nodata (0) is already caught by the
#: red-band check below.
SCL_BAD = (1, 3, 8, 9, 10)

#: Above this share of unusable pixels a tile is dropped. Cloud screening was
#: previously scene-level only -- the plan picks the granule with the lowest
#: `eo:cloud_cover` over a 110 km footprint, which says nothing about our 22 km
#: box. A cloud parked on the AOI in one date and not the other reads as change
#: across every index, which is the one failure mode that produces confidently
#: wrong labels rather than merely coarse ones.
MAX_CLOUD_FRACTION = 0.05

#: Sentinel-2 is 10 m; a 256 px chip is 2.56 km across, which is the same ground
#: extent as an RSVQA-LR tile and keeps the scale prefix comparable.
CHIP_PX = 256
GSD_M = 10.0
SOURCE = "Sentinel-2 L2A (self-generated, Indian AOIs)"
LICENCE = "Copernicus open — contains modified Copernicus Sentinel data"

#: Stratified by land cover, climate and change regime -- not by city. Sampling
#: only urban centres would teach the adapter that "India" means dense
#: construction, when the hidden set is ISRO's and their use cases are
#: agriculture, forestry, water and coast as much as cities.
#:
#: Each entry is (name, stratum, season, west, south, east, north).
#:
#: **Season is held constant within a pair and varied across AOIs.** Comparing a
#: dry-season scene against a monsoon scene detects phenology, not change: the
#: whole Deccan greens up and every index moves. But fixing every AOI to the dry
#: season would leave the adapter never having seen wet-season India, and the
#: hidden set carries no such guarantee. So each AOI is pinned to one season and
#: the set spans both.
#:
#: **Stable strata are as important as changing ones.** Forest, arid and
#: mountain AOIs over four years usually show almost nothing, and those are the
#: negatives that stop the model answering "yes, something changed"
#: unconditionally -- the failure the BEN threshold leak already demonstrated.
AOIS = [
    # Urban cores -- dense construction, the strongest positives.
    ("bengaluru_urban", "urban", "dry", 77.50, 12.85, 77.70, 13.05),
    ("delhi_ncr", "urban", "dry", 77.10, 28.52, 77.30, 28.73),
    ("surat_industrial", "urban", "dry", 72.75, 21.10, 72.95, 21.30),
    ("hyderabad_deccan", "urban", "wet", 78.35, 17.32, 78.55, 17.53),
    ("ahmedabad_arid", "urban", "dry", 72.50, 22.97, 72.70, 23.18),
    ("lucknow_gangetic", "urban", "wet", 80.85, 26.77, 81.05, 26.98),
    # Irrigated agriculture -- rabi/kharif cycles, canal command areas.
    ("punjab_irrigated", "agriculture_irrigated", "dry", 75.35, 30.72, 75.55, 30.93),
    ("godavari_delta", "agriculture_irrigated", "wet", 81.65, 16.45, 81.85, 16.65),
    ("cauvery_delta", "agriculture_irrigated", "dry", 79.55, 10.75, 79.75, 10.95),
    # Rain-fed agriculture -- drought signal, far more inter-annual variation.
    ("vidarbha_rainfed", "agriculture_rainfed", "wet", 78.45, 20.65, 78.65, 20.85),
    ("marathwada_rainfed", "agriculture_rainfed", "dry", 76.25, 19.25, 76.45, 19.45),
    # Forest -- mostly negatives, with deforestation where it occurs.
    ("western_ghats_forest", "forest", "wet", 74.35, 13.45, 74.55, 13.65),
    ("similipal_forest", "forest", "wet", 86.25, 21.75, 86.45, 21.95),
    ("bandipur_forest", "forest", "dry", 76.60, 11.70, 76.80, 11.90),
    # Coastal -- erosion, accretion, aquaculture ponds.
    ("sundarbans_coastal", "coastal", "dry", 88.75, 21.75, 88.95, 21.95),
    ("kutch_coastal", "coastal", "dry", 69.65, 22.95, 69.85, 23.15),
    ("kochi_backwater", "coastal", "wet", 76.25, 9.95, 76.45, 10.15),
    # Arid -- stable negatives with spectra unlike anything in BEN.txt.
    ("thar_margin", "arid", "dry", 72.65, 26.85, 72.85, 27.05),
    ("ladakh_cold_desert", "arid", "dry", 77.55, 34.10, 77.75, 34.30),
    # Water bodies -- reservoir level is the clearest annual change signal.
    ("nagarjuna_reservoir", "water", "dry", 79.25, 16.55, 79.45, 16.75),
    ("ukai_reservoir", "water", "wet", 73.55, 21.20, 73.75, 21.40),
    # Mountain -- terrain shadow and snow, the hardest radiometry.
    ("dehradun_foothill", "mountain", "dry", 78.00, 30.30, 78.20, 30.50),
    ("shimla_himalaya", "mountain", "dry", 77.15, 31.05, 77.35, 31.25),
    # Northeast -- distinct land cover, chronically under-represented.
    ("guwahati_northeast", "northeast", "wet", 91.70, 26.15, 91.90, 26.35),
]

#: Same season on both sides of a pair, varied across AOIs. Dry is post-monsoon
#: through pre-summer, when Indian skies are clearest; wet is the tail of the
#: monsoon, which costs cloud-free scenes but is where the phenology lives.
SEASONS = {
    "dry": (("2019-01-01", "2019-03-31"), ("2023-01-01", "2023-03-31")),
    "wet": (("2019-09-01", "2019-11-30"), ("2023-09-01", "2023-11-30")),
}


def _post(url: str, payload: dict, timeout: int = 180, attempts: int = 3) -> dict:
    """POST with retries. A public STAC API is slow before it is broken."""
    import time

    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "satquery/0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001 - retried, then raised
            last = error
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
    raise SystemExit(f"STAC search failed after {attempts} attempt(s): {last!r}")


def _covers(scene: dict, bbox) -> bool:
    """Does the scene's footprint span the whole AOI?

    STAC returns everything that *intersects* the box, so a granule clipping one
    corner scores as a hit. Punjab came back 87% nodata that way -- every tile
    was margin, and the corpus would have been black rectangles teaching the
    model that black means "no change". Rectangle containment is approximate
    (footprints are not rectangles) but it removes the pathological case, and
    the per-tile nodata check downstream catches the rest.
    """
    west, south, east, north = bbox
    scene_bbox = scene.get("bbox") or []
    if len(scene_bbox) < 4:
        return False
    return (
        scene_bbox[0] <= west
        and scene_bbox[1] <= south
        and scene_bbox[2] >= east
        and scene_bbox[3] >= north
    )


def search(bbox, start: str, end: str, cloud: int, limit: int = 50) -> list[dict]:
    """Scenes whose footprint fully covers ``bbox`` in the window."""
    found = _post(
        STAC,
        {
            "collections": [COLLECTION],
            "bbox": list(bbox),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": cloud}},
            "limit": limit,
        },
    )
    return [f for f in found.get("features", []) if _covers(f, bbox)]


def cmd_plan(args) -> int:
    plan: dict[str, dict] = {}
    for name, stratum, season, west, south, east, north in AOIS:
        (t0_start, t0_end), (t1_start, t1_end) = SEASONS[season]
        bbox = (west, south, east, north)
        early = search(bbox, t0_start, t0_end, args.cloud)
        late = search(bbox, t1_start, t1_end, args.cloud)
        if not early or not late:
            print(f"  {name:24} {len(early)} early / {len(late)} late -- skipped")
            continue

        # Clearest in each window rather than nearest in date: an index
        # difference between a hazy scene and a clear one reads as change
        # everywhere, which would poison every label from this AOI.
        def clearest(scenes):
            return min(scenes, key=lambda f: f["properties"].get("eo:cloud_cover", 100))

        a, b = clearest(early), clearest(late)
        plan[name] = {
            "stratum": stratum,
            "season": season,
            "bbox": list(bbox),
            "t0": {
                "id": a["id"],
                "date": a["properties"]["datetime"][:10],
                "cloud": a["properties"].get("eo:cloud_cover"),
                "assets": {
                    k: a["assets"][k]["href"]
                    for k in (*BANDS, SCL_BAND)
                    if k in a["assets"]
                },
            },
            "t1": {
                "id": b["id"],
                "date": b["properties"]["datetime"][:10],
                "cloud": b["properties"].get("eo:cloud_cover"),
                "assets": {
                    k: b["assets"][k]["href"]
                    for k in (*BANDS, SCL_BAND)
                    if k in b["assets"]
                },
            },
        }
        print(
            f"  {name:24} {stratum:22} {season:3} "
            f"{a['properties']['datetime'][:10]} -> {b['properties']['datetime'][:10]}"
        )

    missing = [n for n, e in plan.items() if len(e["t0"]["assets"]) < len(BANDS)]
    if missing:
        raise SystemExit(
            f"{len(missing)} AOI(s) are missing bands ({missing[:3]}). A composite "
            "built from a partial band set is not the composite the model was "
            "trained on -- fix the asset names before fetching."
        )

    by_stratum: dict[str, int] = {}
    for entry in plan.values():
        by_stratum[entry["stratum"]] = by_stratum.get(entry["stratum"], 0) + 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "licence": LICENCE,
                "gsd_m": GSD_M,
                "chip_px": CHIP_PX,
                "composites_per_date": 3,
                "aois": plan,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("\n" + f"{len(plan)} AOI(s) -> {out / 'plan.json'}")
    for stratum, count in sorted(by_stratum.items()):
        print(f"  {stratum:24} {count}")
    return 0


def _index(a, b):
    """Normalised difference (a-b)/(a+b), the shape every index here takes."""

    a = a.astype("float32")
    b = b.astype("float32")
    return (a - b) / (a + b + 1e-6)


def cmd_fetch(args) -> int:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    from satquery.ingest.copernicus.composites import stretch

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.plan).parent
    tiles: list[dict] = []
    cloud_skipped = 0
    written = 0

    for name, entry in plan["aois"].items():
        target = root / name
        target.mkdir(parents=True, exist_ok=True)

        # One read per band per date covering the WHOLE AOI, then cut locally.
        # A 0.2 degree box is ~22 km, which is ~2,200 px at 10 m: taking a
        # single 256 px chip from it used about 1% of what the request already
        # paid for. Tiling turns one download into ~64 chips spanning city core,
        # suburb, periphery and rural fringe -- diversity at no extra cost.
        # Every band is resampled onto ONE grid before anything is cut from it.
        #
        # Sentinel-2 does not ship a single resolution: red, green, blue and nir
        # are 10 m while swir16 and swir22 are 20 m, so the same geographic
        # window returns 2236x2194 for one and 1118x1097 for the other. Reading
        # each natively and then slicing every band with the same array indices
        # made tile [0:256, 0:256] cover 2.56 km of nir and 5.12 km of swir --
        # a different patch of ground. NDBI, which is (swir16 - nir), was
        # therefore differencing pixels that do not overlap, and the short-wave
        # composite shown beside the true-colour one framed a different scene.
        #
        # That is the measured NDBI instability: forest came back with mean
        # |delta| 0.222 against NDVI's 0.046, backwards for forest, and
        # dominant_change answered "built-up" everywhere.
        scene: dict[str, dict[str, object]] = {}
        grid: tuple[int, int] | None = None
        for when in ("t0", "t1"):
            bands = {}
            # 10 m bands first, so the finest grid is established before any
            # coarser band is resampled onto it. Sorting on the known 20 m names
            # rather than on measured resolution keeps this readable; a band
            # list change would need this list changed with it.
            ordered = sorted(
                entry[when]["assets"].items(),
                key=lambda kv: kv[0] in ("swir16", "swir22"),
            )
            for band, href in ordered:
                cached = target / f"{when}_{band}.npy"
                if cached.exists():
                    try:
                        data = np.load(cached)
                    except Exception as error:  # noqa: BLE001 - re-read, not fatal
                        # A cache written by an interrupted run before writes
                        # became atomic. Unreadable is recoverable: the source
                        # COG is still there, so re-read rather than crash a
                        # multi-hour fetch on one bad file.
                        print(
                            f"  {name}: {when}_{band} cache unreadable ({error!r})"
                            " -- re-reading",
                            flush=True,
                        )
                        cached.unlink(missing_ok=True)
                        data = None
                    if data is not None:
                        if grid is None:
                            grid = data.shape
                        if data.shape == grid:
                            bands[band] = data
                            continue
                        # A cached array off the reference grid was written by
                        # the version that read every band natively. Re-reading
                        # is the only repair: keeping it would silently restore
                        # the misalignment this check exists to catch.
                        print(
                            f"  {name}: re-reading {when}_{band} — cached "
                            f"{data.shape} is off the {grid} grid",
                            flush=True,
                        )
                with rasterio.open(href) as src:
                    left, bottom, right, top = transform_bounds(
                        "EPSG:4326", src.crs, *entry["bbox"]
                    )
                    window = from_bounds(left, bottom, right, top, src.transform)
                    if grid is None:
                        data = src.read(1, window=window, boundless=True)
                        grid = data.shape
                    else:
                        # Same geographic window, forced onto the reference
                        # grid. Bilinear rather than nearest: these are
                        # continuous reflectance values, and nearest-neighbour
                        # upsampling would put 20 m blocks into an index that is
                        # then averaged.
                        data = src.read(
                            1,
                            window=window,
                            out_shape=grid,
                            # SCL holds class codes, not reflectance. Bilinear
                            # would average class 4 and class 8 into class 6 --
                            # inventing "water" out of vegetation beside cloud.
                            resampling=(
                                Resampling.nearest
                                if band == SCL_BAND
                                else Resampling.bilinear
                            ),
                            boundless=True,
                        )
                bands[band] = data
                # Written through a temporary name. np.save is not atomic, and
                # this fetch is routinely interrupted -- a half-written .npy
                # would be picked up by the resume check above and either crash
                # the next run or, worse, load as a valid array of garbage.
                temporary = cached.with_suffix(".npy.part")
                # Through a file handle, not a path. np.save appends ".npy" to
                # any path that does not already end in it, so saving to
                # "t0_swir16.npy.part" silently produced
                # "t0_swir16.npy.part.npy" and the rename below then failed on
                # a file that was never written.
                with temporary.open("wb") as handle:
                    np.save(handle, data)
                temporary.replace(cached)
            scene[when] = bands
        rows = min(scene["t0"][b].shape[0] for b in scene["t0"])
        cols = min(scene["t0"][b].shape[1] for b in scene["t0"])
        down, across = rows // CHIP_PX, cols // CHIP_PX
        print(f"  {name:24} {rows}x{cols} -> {down * across} tile(s)", flush=True)

        for row in range(down):
            for col in range(across):
                y, x = row * CHIP_PX, col * CHIP_PX
                cut = {
                    when: {b: v[y : y + CHIP_PX, x : x + CHIP_PX] for b, v in bands.items()}
                    for when, bands in scene.items()
                }
                # A tile that is mostly nodata is a margin artefact, not a
                # scene. Keeping it would put black rectangles in the corpus and
                # teach the model that black means "no change".
                if any((v["red"] == 0).mean() > 0.2 for v in cut.values()):
                    continue

                # Per-pixel cloud gate. The plan already picked the clearest
                # granule in each window, but that figure covers a 110 km
                # footprint and says nothing about this 2.56 km chip. A cloud
                # present at one date and absent at the other moves every index
                # at once and would be labelled as change.
                cloudy = False
                for _when, values in cut.items():
                    scl = values.get(SCL_BAND)
                    if scl is None:
                        continue
                    fraction = float(np.isin(scl, SCL_BAD).mean())
                    if fraction > MAX_CLOUD_FRACTION:
                        cloudy = True
                        cloud_skipped += 1
                        break
                if cloudy:
                    continue

                tile_id = f"{name}_r{row:02d}c{col:02d}"
                for when, bands in cut.items():
                    for label, triple in (
                        ("true_colour", ("red", "green", "blue")),
                        ("false_colour", ("nir", "red", "green")),
                        ("short_wave", ("swir22", "nir", "red")),
                    ):
                        if not all(b in bands for b in triple):
                            continue
                        stack = np.dstack([stretch(bands[b])[0] for b in triple])
                        Image.fromarray(stack).save(
                            target / f"{tile_id}_{when}_{label}.png"
                        )
                        written += 1

                # The change label, computed from the imagery alone. Stored as
                # per-tile index means rather than full arrays: the generator
                # needs the magnitude and direction, not the raster, and 24 AOIs
                # of float32 rasters would be gigabytes for nothing.
                record = {"tile_id": tile_id, "aoi": name,
                          "stratum": entry["stratum"], "season": entry["season"],
                          "t0_date": entry["t0"]["date"], "t1_date": entry["t1"]["date"]}
                for key, (a, b) in (
                    ("ndvi", ("nir", "red")),
                    ("ndwi", ("green", "nir")),
                    ("ndbi", ("swir16", "nir")),
                ):
                    if not all(x in cut["t0"] for x in (a, b)):
                        continue
                    before = float(_index(cut["t0"][a], cut["t0"][b]).mean())
                    after = float(_index(cut["t1"][a], cut["t1"][b]).mean())
                    record[f"{key}_t0"] = round(before, 4)
                    record[f"{key}_t1"] = round(after, 4)
                    record[f"{key}_delta"] = round(after - before, 4)
                tiles.append(record)

    (root / "tiles.json").write_text(json.dumps(tiles, indent=1), encoding="utf-8")
    summary = {
        "source": SOURCE,
        "licence": LICENCE,
        "aois": len(plan["aois"]),
        "tiles": len(tiles),
        "tiles_dropped_cloudy": cloud_skipped,
        "composites_written": written,
    }
    (root / "fetch_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="search Earth Search for a date pair per AOI")
    plan.add_argument("--t0-start", default="2019-01-01")
    plan.add_argument("--t0-end", default="2019-03-31")
    plan.add_argument("--t1-start", default="2023-01-01")
    plan.add_argument("--t1-end", default="2023-03-31")
    plan.add_argument("--cloud", type=int, default=10, help="max cloud cover %%")
    plan.add_argument("--out", default="/data/eval/india")
    plan.set_defaults(func=cmd_plan)

    fetch = sub.add_parser("fetch", help="windowed COG reads into chips + composites")
    fetch.add_argument("--plan", default="/data/eval/india/plan.json")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
