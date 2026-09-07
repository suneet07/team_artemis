"""Fetch patch pixels from Copernicus and emit a canonical training manifest.

    # 1. settle the grid convention. Once, against dataset-supplied bounds.
    python scripts/fetch_copernicus_patches.py verify \
        --patch-id S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57 \
        --bounds 499980 5390220 501180 5391420 --granule-transform-from-tile

    # 2. fetch, once verify has been recorded in patch_grid.py
    python scripts/fetch_copernicus_patches.py fetch \
        --manifest data/manifests/ben_patches.jsonl \
        --out data/chips --limit 50

This is the second half of the BEN.txt unblock. ``build_ben_manifest.py``
produced identifiers; this turns them into imagery via windowed ``/vsis3/``
reads against the Copernicus Data Space, so a 120x120 patch costs a few hundred
kilobytes rather than a 1 GB product download.

Three things it is strict about, each because the failure is silent:

**Grid convention.** ``fetch`` refuses to run until
``GRID_CONVENTION_VERIFIED`` is true. A patch read one cell off is a valid-
looking Sentinel-2 chip carrying its neighbour's land-cover labels, and no
training metric can see it. ``--i-accept-unverified-geometry`` exists for
throwaway experiments and stamps every row it produces with
``geometry_verified: false`` so that corpus can never be mistaken for the real
one.

**Granule reuse.** Patches are grouped by granule and each granule is resolved
and opened once. The manifest summary reports granule count for this reason:
it, not the patch count, is the network cost.

**Partial failure.** A patch that cannot be fetched is recorded in a failures
file and skipped. It is not retried silently and it is not written as a blank
chip -- ``extract_ben_micro.py`` once wrote 21 all-black PNGs on a missing-image
fallback and a loss curve was read off them.

Credentials come from the environment (``CDSE_USERNAME``/``CDSE_PASSWORD`` for
the catalogue, ``CDSE_S3_ACCESS_KEY``/``CDSE_S3_SECRET_KEY`` for the object
store) and are never read from a file in this repo or echoed to a log.
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.ingest.copernicus import (  # noqa: E402
    GRID_CONVENTION_VERIFIED,
    parse_patch_id,
    verify_convention,
)
from satquery.ingest.copernicus.cdse import (  # noqa: E402
    credentials_present,
    fetch_patches,
    resolve_granule,
    write_patch,
)
from satquery.ingest.copernicus.composites import (  # noqa: E402
    COMPOSITES,
    REQUIRED_BANDS,
    build_composites,
    composite_names_for,
)

#: Every composite, written once per patch. Individual rows then reference the
#: subset their adapter uses, so one patch can serve a three-view rs_vqa
#: question and a two-view rs_ground_caption question from the same pixels.
ALL_COMPOSITES = tuple(COMPOSITES)


def _require_credentials() -> None:
    present = credentials_present()
    if not all(present.values()):
        missing = [name for name, ok in present.items() if not ok]
        raise SystemExit(
            f"Copernicus credentials missing for: {missing}. Set CDSE_USERNAME and "
            "CDSE_PASSWORD for the OData catalogue, and CDSE_S3_ACCESS_KEY and "
            "CDSE_S3_SECRET_KEY for the object store. They are two separate "
            "credential sets and one does not substitute for the other."
        )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args) -> int:
    """Settle the grid convention against one patch of known bounds."""
    _require_credentials()
    patch = parse_patch_id(args.patch_id)
    granule = resolve_granule(patch)
    print(f"granule: {granule.name}")

    import rasterio

    from satquery.ingest.copernicus.cdse import _band_url, _open_s3

    env, _ = _open_s3(granule.s3_path)
    with env, rasterio.open(_band_url(granule, "B04")) as source:
        transform = source.transform
        print(f"tile transform: {transform}")
        print(f"tile CRS: {source.crs}")

    bounds = tuple(float(v) for v in args.bounds)
    convention = verify_convention(args.patch_id, bounds, transform, tolerance_m=args.tolerance)

    print(f"\nCONVENTION: {convention}")
    print(
        "\nThis result is not self-applying. Record it by editing "
        "satquery/ingest/copernicus/patch_grid.py:\n"
        f'    DEFAULT_CONVENTION = "{convention}"\n'
        "    GRID_CONVENTION_VERIFIED = True\n"
        "and commit the bounds you verified against in the same change, so the "
        "next reader can check the evidence rather than trust the flag."
    )
    return 0


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _load_manifest(path: Path, limit: int) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
            if limit and len(entries) >= limit:
                break
    if not entries:
        raise SystemExit(f"{path} is empty; build it with scripts/build_ben_manifest.py")
    return entries


def _write_composites(read, entry, out_root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Write each composite as a PNG beside the multi-band GeoTIFF.

    The GeoTIFF is the archival artifact -- it keeps CRS and transform, which
    the whole ingest path keys off. The PNGs are what the processor loads, and
    they are derived from it rather than fetched separately, so the two can
    never disagree about which pixels a sample is made of.

    Returns a name -> path map rather than a list, because a single patch now
    serves questions belonging to **different adapters with different view
    counts**: ``rs_vqa`` takes three composites, ``rs_ground_caption`` takes two
    (Cartosat has no SWIR, C22). Writing one fixed list per patch gave grounding
    rows a short-wave view the target sensor cannot produce.
    """
    written: dict[str, str] = {}
    for composite in build_composites(read.bands, names=names):
        relative = Path("composites") / f"{entry['patch_id']}_{composite.name}.png"
        target = out_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        composite.to_image().save(target)
        written[composite.name] = relative.as_posix()
    return written


def cmd_fetch(args) -> int:
    if not GRID_CONVENTION_VERIFIED and not args.i_accept_unverified_geometry:
        raise SystemExit(
            "GRID_CONVENTION_VERIFIED is False. Run the `verify` subcommand against a "
            "patch whose bounds come from the dataset's own metadata, record the "
            "result in patch_grid.py, then fetch. A corpus fetched on a guessed "
            "convention is real imagery under the wrong labels and no downstream "
            "metric will reveal it. Pass --i-accept-unverified-geometry only for a "
            "throwaway experiment; every row will be stamped geometry_verified: false."
        )
    _require_credentials()

    entries = _load_manifest(Path(args.manifest), args.limit)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    by_granule: dict[tuple, list[dict]] = defaultdict(list)
    for entry in entries:
        by_granule[(entry["tile"], entry["sensing"], entry["orbit"])].append(entry)
    print(f"{len(entries)} patch(es) across {len(by_granule)} granule(s)")

    canonical: list[dict] = []
    failures: list[dict] = []
    fetched = 0

    def process_granule(item):
        """One granule, end to end. Independent of every other granule."""
        key, group = item
        local_rows: list[dict] = []
        local_failures: list[dict] = []
        local_fetched = 0
        print(f"\ngranule {key}: {len(group)} patch(es)")
        try:
            granule = resolve_granule(group[0]["patch_id"])
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            # Every patch in this granule is unreachable, but the other 113
            # granules are unaffected -- return rather than abandoning the run.
            print(f"  {key}: resolve failed: {error!r}", flush=True)
            return (
                [],
                [
                    {"patch_id": e["patch_id"], "stage": "resolve", "error": repr(error)}
                    for e in group
                ],
                0,
            )

        # One pass per granule, not per patch: each band's JP2 is opened once
        # and every window read from that handle. Opening per patch cost 6 x N
        # network opens of a 10,980 px image -- at 175 patches per granule that
        # is 1,050 opens where 6 will do, and it was the difference between a
        # few hours and about 167 of them.
        by_id = {entry["patch_id"]: entry for entry in group}
        reads = fetch_patches(
            list(by_id),
            granule=granule,
            bands=REQUIRED_BANDS,
            require_verified_convention=not args.i_accept_unverified_geometry,
            on_error=lambda pid, err: local_failures.append(
                {"patch_id": pid, "stage": "fetch", "error": repr(err)}
            ),
        )

        for read in reads:
            patch_id = read.patch_id
            entry = by_id[patch_id]
            try:
                tif = out_root / "geotiff" / f"{patch_id}.tif"
                write_patch(read, tif)
                # Write the union once; each question then takes only the
                # views its own adapter uses.
                available = _write_composites(read, entry, out_root, ALL_COMPOSITES)
            except Exception as error:  # noqa: BLE001
                local_failures.append(
                    {"patch_id": patch_id, "stage": "write", "error": repr(error)}
                )
                print(f"  {patch_id}: FAILED {error!r}")
                continue

            local_fetched += 1
            gsd = entry.get("effective_gsd_m", 10.0)
            for index, pair in enumerate(entry.get("qa", [])):
                row_adapter = pair.get(
                    "adapter", entry.get("adapter", args.adapter)
                )
                # The views this question's adapter actually uses -- not the
                # patch's, and not the command line's.
                roles = [
                    name
                    for name in composite_names_for(row_adapter)
                    if name in available
                ]
                images = [available[name] for name in roles]
                local_rows.append(
                    {
                        "sample_id": f"{patch_id}_{index:03d}",
                        # Per-pair, not per-file: BEN.txt mixes VQA, grounding
                        # and captioning, and they serve different adapters.
                        # Stamping the file's adapter onto every row sends
                        # grounding supervision to rs_vqa and starves the
                        # adapter gate G3 turns on.
                        "adapter": row_adapter,
                        "task": pair.get("task", "single_vqa"),
                        # Average accuracy is the mean of per-type accuracies;
                        # an untyped row collapses into one bucket and moves the
                        # headline metric.
                        "question_type": pair.get("question_type", "other"),
                        "images": images,
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "answer_type": "text",
                        "image_roles": roles,
                        "modality": ["optical"] * len(images),
                        "effective_gsd_m": [gsd] * len(images),
                        "split": entry.get("split", "train"),
                        "source": entry.get("source", "BigEarthNet.txt"),
                        "licence": entry.get("licence", ""),
                        "geotiff": str(tif.relative_to(out_root).as_posix()),
                        "geometry_verified": bool(GRID_CONVENTION_VERIFIED),
                        "grid_convention": read.convention,
                        "granule": granule.name,
                    }
                )
            print(f"  {patch_id}: {len(images)} composite(s)")
        return local_rows, local_failures, local_fetched

    # Granules are independent, and the cost is JP2 window decoding plus
    # network latency -- both of which parallelise. Serially, 20,000 patches
    # measured at 9.6 s each: 53 hours. The work is not CPU-bound enough to
    # need processes, and threads keep one GDAL cache warm per worker.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = 0
        for rows_, failures_, fetched_ in pool.map(
            process_granule, by_granule.items()
        ):
            canonical.extend(rows_)
            failures.extend(failures_)
            fetched += fetched_
            done += 1
            print(
                f"granule {done}/{len(by_granule)}: {fetched} patch(es), "
                f"{len(canonical)} row(s), {len(failures)} failure(s)",
                flush=True,
            )

    manifest_out = out_root / "canonical.jsonl"
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in canonical:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "fetched": datetime.now(UTC).isoformat(),
        "source_manifest": args.manifest,
        "patches_requested": len(entries),
        "patches_fetched": fetched,
        "patches_failed": len(failures),
        "canonical_rows": len(canonical),
        "geometry_verified": bool(GRID_CONVENTION_VERIFIED),
        "granules": len(by_granule),
    }
    (out_root / "fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        (out_root / "fetch_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )

    print(
        f"\n{fetched}/{len(entries)} patch(es) fetched, {len(canonical)} canonical row(s) "
        f"-> {manifest_out}"
    )
    if failures:
        print(
            f"{len(failures)} failure(s) recorded in {out_root / 'fetch_failures.json'}. "
            "They are skipped, not retried and not substituted with blanks."
        )
    # A run that fetched nothing exits non-zero: an empty corpus written
    # successfully is the failure mode that looks most like success.
    return 0 if fetched else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="settle the reBEN grid convention")
    verify.add_argument("--patch-id", required=True)
    verify.add_argument(
        "--bounds",
        nargs=4,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="the patch's true bounds, in the granule CRS, from dataset metadata",
    )
    verify.add_argument("--tolerance", type=float, default=1.0)
    verify.add_argument("--granule-transform-from-tile", action="store_true", default=True)
    verify.set_defaults(func=cmd_verify)

    fetch = sub.add_parser("fetch", help="fetch patch pixels and emit canonical rows")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--out", default="data/chips")
    fetch.add_argument("--adapter", default="rs_vqa")
    fetch.add_argument("--limit", type=int, default=0)
    fetch.add_argument(
        "--workers",
        type=int,
        default=8,
        help="granules fetched concurrently; the cost is per-window decode "
        "and network latency, both of which parallelise",
    )
    fetch.add_argument("--i-accept-unverified-geometry", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
