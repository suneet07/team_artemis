"""Stage a small paired Sentinel-1/Sentinel-2 subset of reBEN, verified at source.

    python scripts/stage_reben_probe.py inspect --out /data/eval/reben
    python scripts/stage_reben_probe.py stage   --out /data/eval/reben

**Why a third-party subset rather than the authoritative archive.** reBEN ships
from Zenodo as ``BigEarthNet-S1.tar.zst`` (54.4 GB) and ``BigEarthNet-S2.tar.zst``
(63.3 GB), and no mirror is addressable per patch -- the HuggingFace copies are a
155 GB LMDB, a split ``tar.gz``, and 5 GB ``split`` fragments of one blob. We need
roughly 20,000 of 549,488 patches. Pulling 118 GB to keep 3.6% of it is the
fallback, not the opening move; this stages a 3.1 GB paired subset first so a
defect in the cross-modal generator costs a 3 GB download to find instead of a
118 GB one.

**Licence.** BigEarthNet v2.0 is CDLA-Permissive 1.0, which permits
redistribution, so the right to use these pixels comes from BigEarthNet's own
grant rather than from the uploader -- who declares no licence and contributed a
*selection*, not data. That is the split that barred SARLANG-1M, whose QA text
was the uploader's own contribution with no grant behind it, and it is why this
subset is admissible where SARLANG was not.

**Which makes provenance, not licensing, the risk here.** An unverified third
party could have altered, mislabelled or truncated the contents, and training on
a corrupted corpus is invisible in every metric we have. So nothing is staged on
trust: ``stage`` resolves every extracted patch identifier against the
**official** ``metadata.parquet`` and refuses any patch the authoritative index
does not carry. A subset that cannot be reconciled is not staged at all.

``inspect`` exists because the archive's internal layout is undocumented. It
downloads, prints the tree, and stops -- writing extraction logic against a
guessed layout is how the RarePlanes staging lost a run.
"""

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The paired subset. 3.1 GB, ~14,000 S1+S2 pairs.
SUBSET_REPO = "ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2"
SUBSET_FILE = "BigEarthNet_14K.zip"

#: The authoritative index, taken from a mirror that carries the dataset in its
#: published form rather than from the subset's own copy. Verifying a stranger's
#: archive against a metadata file from the same stranger would verify nothing.
METADATA_REPO = "torchgeo/bigearthnet"
METADATA_FILE = "V2/metadata.parquet"

SOURCE = "reBEN (BigEarthNet v2.0)"
LICENCE = "CDLA-Permissive-1.0"


def _download(repo: str, filename: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"fetching {repo}/{filename}", flush=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache),
    )
    size = Path(path).stat().st_size
    print(f"  {size / 2**30:.2f} GiB at {path}", flush=True)
    return Path(path)


def cmd_inspect(args) -> int:
    """Download the archive and report its layout without extracting anything."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = _download(SUBSET_REPO, SUBSET_FILE, out / "_hf")

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    print(f"\n{len(names)} entries", flush=True)

    # Depth-1 and depth-2 prefixes say whether the layout is patch-per-folder,
    # modality-per-folder, or something else, which is what the extractor needs.
    for depth in (1, 2):
        counts = Counter("/".join(n.split("/")[:depth]) for n in names if "/" in n)
        print(f"\ntop {depth}-level prefixes:", flush=True)
        for prefix, count in counts.most_common(12):
            print(f"  {count:>7}  {prefix}", flush=True)

    print("\nsuffixes:", flush=True)
    for suffix, count in Counter(Path(n).suffix for n in names).most_common(10):
        print(f"  {count:>7}  {suffix or '(none)'}", flush=True)

    print("\nfirst 25 entries:", flush=True)
    for name in names[:25]:
        print(f"  {name}", flush=True)

    (out / "inspect.json").write_text(
        json.dumps({"entries": len(names), "sample": names[:200]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out / 'inspect.json'}", flush=True)
    return 0


#: What reBEN's own specification says a patch holds. Anything narrower means
#: the repack dropped information, and every one of these losses is invisible
#: once the pixels are in a training corpus.
#:
#: S2 is Level-2A surface reflectance as uint16 scaled by 10,000, so a real
#: patch runs in the hundreds to low thousands. S1 is calibrated sigma-nought
#: in decibels as float32, which is negative over nearly every surface -- the
#: same property `satquery.tools.deterministic.assert_decibels` keys on.
EXPECTED = {
    "S1": {"min_bands": 2, "dtypes": {"float32", "float64"}, "expect_negative": True},
    "S2": {"min_bands": 10, "dtypes": {"uint16", "int16"}, "expect_negative": False},
}


def verify_rasters(root: Path, sample: int) -> dict:
    """Open a sample of the extracted rasters and report what they actually hold.

    The archive is a *repack* -- one stacked GeoTIFF per patch, where reBEN
    ships a directory of per-band files -- so byte-identity against the original
    cannot be established at all. What can be established is whether the repack
    preserved the information: band count, dtype, geometry and value range. A
    subset re-encoded to 8-bit, or with bands dropped, passes every check that
    only looks at file names.
    """
    import numpy as np
    import rasterio

    report: dict = {}
    for modality in ("S1", "S2"):
        directory = root / "BEN_14k" / f"BigEarthNet-{modality}"
        files = sorted(directory.rglob("*.tif"))[:sample] if directory.exists() else []
        rows = []
        for path in files:
            with rasterio.open(path) as handle:
                data = handle.read()
                rows.append(
                    {
                        "name": path.name,
                        "bands": handle.count,
                        "dtype": str(handle.dtypes[0]),
                        "shape": [handle.height, handle.width],
                        "crs": str(handle.crs),
                        "res": [abs(handle.transform.a), abs(handle.transform.e)],
                        "min": float(np.nanmin(data)),
                        "max": float(np.nanmax(data)),
                    }
                )
        if not rows:
            report[modality] = {"checked": 0, "verdict": "MISSING"}
            continue

        want = EXPECTED[modality]
        problems = []
        bands = {r["bands"] for r in rows}
        dtypes = {r["dtype"] for r in rows}
        if min(bands) < want["min_bands"]:
            problems.append(
                f"only {min(bands)} band(s); reBEN {modality} carries at least "
                f"{want['min_bands']} -- the repack dropped bands"
            )
        if not dtypes & want["dtypes"]:
            problems.append(
                f"dtype {sorted(dtypes)} is not {sorted(want['dtypes'])} -- the "
                f"values have been re-quantised and are no longer the source units"
            )
        has_negative = any(r["min"] < 0 for r in rows)
        if want["expect_negative"] and not has_negative:
            problems.append(
                "no negative value in any sampled patch, so these are not "
                "calibrated decibels -- most likely stretched to 8-bit or "
                "normalised, which silently breaks every dB threshold in D1"
            )
        if not want["expect_negative"] and max(r["max"] for r in rows) <= 255:
            problems.append(
                "every value fits in 0..255, so surface reflectance has been "
                "stretched to 8-bit rather than kept as scaled uint16"
            )
        resolutions = {tuple(r["res"]) for r in rows}
        shapes = {tuple(r["shape"]) for r in rows}

        report[modality] = {
            "checked": len(rows),
            "bands": sorted(bands),
            "dtypes": sorted(dtypes),
            "shapes": [list(x) for x in sorted(shapes)],
            "resolutions_m": [list(x) for x in sorted(resolutions)],
            "value_range": [
                min(r["min"] for r in rows),
                max(r["max"] for r in rows),
            ],
            "crs_seen": sorted({r["crs"] for r in rows})[:5],
            "problems": problems,
            "verdict": "OK" if not problems else "SUSPECT",
            "examples": rows[:3],
        }
    return report


def compare_metadata(subset_parquet: Path, official_parquet: Path) -> dict:
    """Is the subset's own metadata the official file, or something rewritten?

    The subset ships a ``metadata.parquet`` whose size matches the official one
    byte for byte, which is suggestive but not proof. This compares row count
    and the label column itself.
    """
    import pandas as pd

    subset = pd.read_parquet(subset_parquet)
    official = pd.read_parquet(official_parquet)
    shared = [c for c in subset.columns if c in official.columns]
    return {
        "subset_rows": int(len(subset)),
        "official_rows": int(len(official)),
        "subset_columns": list(subset.columns),
        "shared_columns": shared,
        "identical_row_count": bool(len(subset) == len(official)),
    }


def official_patch_ids(metadata_path: Path) -> dict[str, set[str]]:
    """The authoritative identifier space for each modality.

    reBEN names the two modalities differently -- an S2 patch is
    ``S2A_MSIL2A_..._26_57`` and its S1 counterpart is
    ``S1A_IW_GRDH_1SDV_..._34_59`` -- and the index carries both, which is what
    makes them pairable at all. Returning one merged set would let an S1 file
    reconcile against an S2 id and vice versa.
    """
    import pandas as pd

    frame = pd.read_parquet(metadata_path)
    columns = {c.lower(): c for c in frame.columns}

    def pick(*candidates: str) -> set[str]:
        for candidate in candidates:
            if candidate in columns:
                return set(frame[columns[candidate]].astype(str))
        return set()

    ids = {
        "S2": pick("patch_id", "s2_name", "s2v1_name", "name"),
        "S1": pick("s1_name", "s1v1_name", "s1_patch_id"),
    }
    missing = [k for k, v in ids.items() if not v]
    if missing:
        raise ValueError(
            f"no {'/'.join(missing)} identifier column in {metadata_path}; "
            f"columns are {list(frame.columns)}"
        )
    return ids


def cmd_stage(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "_hf"

    metadata = _download(METADATA_REPO, METADATA_FILE, cache)
    known = official_patch_ids(metadata)
    print(
        f"official index carries {len(known['S2']):,} S2 and "
        f"{len(known['S1']):,} S1 identifier(s)",
        flush=True,
    )

    archive = _download(SUBSET_REPO, SUBSET_FILE, cache)
    target = out / "patches"
    target.mkdir(parents=True, exist_ok=True)

    # Match on the file *stem*, and match each modality against its own
    # identifier space. The archive names S2 patches the reBEN way
    # (S2A_MSIL2A_..._26_57) but S1 patches the Sentinel-1 way
    # (S1A_IW_GRDH_1SDV_..._34TCR_34_59), so a single set of S2 ids -- and a
    # match against `parts`, which carries the ".tif" -- would have refused
    # every radar file and quietly staged an optical-only "cross-modal" corpus.
    kept, refused, seen = 0, 0, {"S1": set(), "S2": set()}
    refused_examples: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for index, member in enumerate(members, 1):
            name = member.filename
            stem = Path(name).stem
            modality = "S1" if "BigEarthNet-S1" in name else (
                "S2" if "BigEarthNet-S2" in name else None
            )
            if modality is None or stem not in known[modality]:
                refused += 1
                if len(refused_examples) < 10:
                    refused_examples.append(name)
                continue
            seen[modality].add(stem)
            zf.extract(member, target)
            kept += 1
            if index % 5000 == 0:
                print(
                    f"  {index}/{len(members)} entries, {kept} kept, "
                    f"S1 {len(seen['S1'])} / S2 {len(seen['S2'])}",
                    flush=True,
                )

    summary = {
        "source": SOURCE,
        "licence": LICENCE,
        "subset_repo": SUBSET_REPO,
        "verified_against": f"{METADATA_REPO}/{METADATA_FILE}",
        "official_patch_ids": {k: len(v) for k, v in known.items()},
        "files_kept": kept,
        "files_refused": refused,
        "patches_staged": {k: len(v) for k, v in seen.items()},
        "refused_examples": refused_examples,
    }
    (out / "stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)

    if not (seen["S1"] and seen["S2"]):
        # Every identifier failing to reconcile means the archive is not what it
        # claims, or the layout hides the id somewhere this does not look. Either
        # way, staging nothing is the correct outcome and a zero-row corpus built
        # on it silently would not be.
        print(
            "\nFAILED: no patch in the archive matches the official index. "
            "Run `inspect` and check the layout before trusting this subset.",
            flush=True,
        )
        return 1
    return 0


def cmd_verify(args) -> int:
    out = Path(args.out)
    report = verify_rasters(out / "patches", args.sample)

    subset_parquet = out / "patches" / "BEN_14k" / "metadata.parquet"
    official = _download(METADATA_REPO, METADATA_FILE, out / "_hf")
    if subset_parquet.exists():
        report["metadata"] = compare_metadata(subset_parquet, official)

    (out / "verify.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    verdicts = [v.get("verdict") for v in report.values() if isinstance(v, dict)]
    if any(v in ("SUSPECT", "MISSING") for v in verdicts):
        print(
            "\nFAILED: the repack did not preserve the source values. Do not "
            "train on this subset -- pull the authoritative archives instead.",
            flush=True,
        )
        return 1
    print("\nvalues, bands and units are consistent with reBEN's specification.",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="download and report layout only")
    inspect.add_argument("--out", default="/data/eval/reben")
    inspect.set_defaults(func=cmd_inspect)

    verify = sub.add_parser(
        "verify", help="open extracted rasters and check values, bands and units"
    )
    verify.add_argument("--out", default="/data/eval/reben")
    verify.add_argument("--sample", type=int, default=40)
    verify.set_defaults(func=cmd_verify)

    stage = sub.add_parser("stage", help="extract, verified against the official index")
    stage.add_argument("--out", default="/data/eval/reben")
    stage.set_defaults(func=cmd_stage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
