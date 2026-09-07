"""Pull one reBEN reference map and read its true bounds, for the grid verify.

    python scripts/fetch_reben_reference_bounds.py

``verify_convention`` needs the real bounds of one patch, from the dataset's own
georeferencing. The obvious source is not the one the code assumed:
``metadata.parquet`` carries labels, split and country and **no geometry at
all**. The georeferencing lives in ``Reference_Maps.tar.zst``, whose members are
per-patch GeoTIFFs carrying CRS and transform.

That archive is 282 MB, but a single patch settles the question, so this streams
the tar and stops at the first usable member rather than downloading the rest.
``zstd`` decompression comes from ``compression.zstd`` (Python 3.14+); older
interpreters get told to use the ``zstandard`` package.

**Off-diagonal only.** A patch whose two indices are equal proves nothing: on a
square grid ``row_col`` and ``col_row`` produce identical bounds wherever
``first == second``, so a diagonal patch is consistent with both conventions and
``verify_convention`` will refuse it. This picks the first member with
``first != second``.

The bounds are printed, not recorded. Recording them is a deliberate edit to
``patch_grid.py`` with the evidence in the same change, so the next reader can
check the reasoning rather than trust a flag someone set.
"""

import argparse
import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARCHIVE_URL = (
    "https://zenodo.org/records/10891137/files/Reference_Maps.tar.zst?download=1"
)

#: A reference-map member looks like
#: ``Reference_Maps/S2A_MSIL2A_..._T33UUP_26_57/S2A_..._26_57_reference_map.tif``
PATCH_MEMBER = re.compile(r"([A-Z0-9_]+_T[0-9]{2}[A-Z]{3}_(\d+)_(\d+))[^/]*\.tiff?$")


def _open_zstd_stream(raw):
    """Return a decompressing file object, or say which package is missing."""
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        try:
            import zstandard
        except ImportError as error:
            raise SystemExit(
                "zstd decompression needs Python 3.14+ (compression.zstd) or the "
                "`zstandard` package. Install it with: pip install zstandard"
            ) from error
        return zstandard.ZstdDecompressor().stream_reader(raw)
    return zstd.ZstdFile(raw)


def find_first_off_diagonal(url: str, *, max_members: int = 400):
    """Stream the archive and return (patch_id, tif_bytes) for the first usable patch.

    Stops at the first off-diagonal member. ``max_members`` bounds the walk so a
    surprise in the archive layout fails loudly instead of streaming 282 MB.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
    with urllib.request.urlopen(request) as raw:  # noqa: S310 - fixed Zenodo URL
        stream = _open_zstd_stream(raw)
        # "r|" is the streaming mode: no seeking, so nothing forces a full read.
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for index, member in enumerate(archive):
                if index >= max_members:
                    raise SystemExit(
                        f"walked {max_members} members without finding an off-diagonal "
                        "reference map. The archive layout is not what this expects; "
                        "inspect it before trusting anything downstream."
                    )
                if not member.isfile():
                    continue
                match = PATCH_MEMBER.search(member.name)
                if not match:
                    continue
                patch_id, first, second = match.group(1), match.group(2), match.group(3)
                if first == second:
                    continue  # diagonal: cannot discriminate the two conventions
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                return patch_id, handle.read()
    raise SystemExit("archive ended without a usable reference map")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument(
        "--save",
        default="data/reben/reference_sample.tif",
        help="where to keep the sampled GeoTIFF as evidence",
    )
    args = parser.parse_args()

    print("streaming Reference_Maps.tar.zst (stops at the first off-diagonal patch)")
    patch_id, payload = find_first_off_diagonal(args.url)

    import rasterio

    with rasterio.open(io.BytesIO(payload)) as source:
        bounds = source.bounds
        crs = source.crs
        transform = source.transform

    target = Path(args.save)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    print(f"\npatch:  {patch_id}")
    print(f"CRS:    {crs}")
    print(f"size:   {source.width} x {source.height} px")
    print(f"bounds: {bounds.left} {bounds.bottom} {bounds.right} {bounds.top}")
    print(f"transform: {transform}")
    print(f"saved:  {target}")
    print(
        "\nSettle the convention with:\n"
        f"  python scripts/fetch_copernicus_patches.py verify \\\n"
        f"      --patch-id {patch_id} \\\n"
        # Fixed-point, not %g: the verify tolerance is 1 m, and %g would render
        # 5330412.5 as 5.33041e+06 and silently drop 2.5 m of it.
        f"      --bounds {bounds.left:.3f} {bounds.bottom:.3f} "
        f"{bounds.right:.3f} {bounds.top:.3f} --granule-transform-from-tile"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
