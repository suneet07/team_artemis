"""Pull the chosen patches out of the BigEarthNet-S2 archive by streaming it.

    python scripts/extract_ben_archive.py \
        --manifest data/manifests/ben_txt_train.jsonl --out data/chips

The windowed Copernicus fetcher rebuilds 120x120 chips by decoding windows out
of 10,980 px JP2 granules. It works, and after three rounds of optimisation it
still measured ~3.3 s per patch: about **18 hours for 20,000 patches**.

BigEarthNet v2.0 already ships those exact chips. ``BigEarthNet-S2.tar.zst``
holds one directory per patch with each band as a small GeoTIFF -- 29,196 bytes
for a 10 m band, 7,572 for a 20 m one. We were reconstructing files that exist.

The archive is 59 GB and tar is sequential, so this streams the whole thing and
writes only the members it wants: ~130 KB per patch across the six bands the
input contract uses, so roughly 2.6 GB kept out of 59 GB read. Network-bound,
and on a fast connection that is tens of minutes rather than tens of hours.

**What this removes, beyond time.** No CDSE credentials. No granule resolution.
No JP2 decoding. And no dependence on our grid convention at all -- the patches
arrive cut by the dataset's own authors, so the geometry is theirs rather than
our reconstruction of it. The verified convention still matters for the India
data, where no pre-cut archive exists and the windowed path is the only option.

**Why not just extract everything.** 549k patches x 12 bands is 6.6M files,
which is well past what a Modal Volume tolerates (50k recommended, 500k inodes
hard). Extracting a chosen subset keeps the file count in the tens of thousands
and skips data no adapter will read.
"""

import argparse
import io
import json
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.ingest.copernicus.composites import (  # noqa: E402
    COMPOSITES,
    REQUIRED_BANDS,
    build_composites,
    composite_names_for,
)

#: Zenodo is the canonical home and it is **slow**: measured 5.28 MB/s from
#: Modal and 3.67 MB/s from a home line, so 59 GiB takes 3-5 hours. The TorchGeo
#: mirror on HuggingFace carries the same archive at a measured 24.83 MB/s --
#: about 41 minutes. Same data, same CDLA-Permissive-1.0 licence; a mirror
#: cannot grant fewer rights than the original, and CDLA-Permissive expressly
#: allows redistribution (C45: the provenance chain is BigEarthNet v2.0 either
#: way, and is recorded as such).
SOURCES = {
    "hf": [
        "https://huggingface.co/datasets/torchgeo/bigearthnet/resolve/main/"
        "V2/BigEarthNet-S2.tar.gzaa",
        "https://huggingface.co/datasets/torchgeo/bigearthnet/resolve/main/"
        "V2/BigEarthNet-S2.tar.gzab",
    ],
    "zenodo": [
        "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
    ],
}
ARCHIVE_URL = SOURCES["hf"][0]
SOURCE = "BigEarthNet.txt"
LICENCE = "CDLA-Permissive-1.0"
ALL_COMPOSITES = tuple(COMPOSITES)


class _Chained(io.RawIOBase):
    """Read several URLs as one continuous byte stream.

    The TorchGeo mirror ships the archive split across ``.tar.gzaa`` and
    ``.tar.gzab``. Those are not two archives -- they are one file cut in half,
    so the halves must be concatenated *before* decompression. Decompressing
    them separately fails on the second part, which has no gzip header.
    """

    def __init__(self, urls: list[str]):
        self._urls = list(urls)
        self._current = None
        self._index = 0

    def readable(self) -> bool:
        return True

    def _advance(self) -> bool:
        if self._index >= len(self._urls):
            return False
        url = self._urls[self._index]
        self._index += 1
        print(f"  part {self._index}/{len(self._urls)}: {url.rsplit('/', 1)[-1]}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
        self._current = urllib.request.urlopen(request)  # noqa: S310 - fixed hosts
        return True

    def readinto(self, buffer) -> int:
        while True:
            if self._current is None and not self._advance():
                return 0
            got = self._current.readinto(buffer)
            if got:
                return got
            self._current.close()
            self._current = None


def _open_stream(url_or_urls) -> object:
    """A decompressed tar stream, picking the codec from the file extension."""
    import gzip

    urls = [url_or_urls] if isinstance(url_or_urls, str) else list(url_or_urls)
    raw = io.BufferedReader(_Chained(urls), buffer_size=8 << 20)

    if urls[0].endswith(".zst") or ".zst" in urls[0]:
        try:
            from compression import zstd  # Python 3.14+

            return zstd.ZstdFile(raw)
        except ImportError:
            try:
                import zstandard
            except ImportError as error:
                raise SystemExit(
                    "zstd support needs Python 3.14+ or `pip install zstandard`."
                ) from error
            return zstandard.ZstdDecompressor().stream_reader(raw)
    return gzip.GzipFile(fileobj=raw)


def _member_key(name: str) -> tuple[str, str] | None:
    """(patch_id, band) for a band GeoTIFF, or None for anything else."""
    stem = Path(name).stem
    if not stem or "_" not in stem:
        return None
    patch_id, _, band = stem.rpartition("_")
    return (patch_id, band) if band.startswith("B") else None


def extract(manifest: Path, out_root: Path, urls, *, bands=REQUIRED_BANDS) -> dict:
    entries = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                entries[entry["patch_id"]] = entry
    if not entries:
        raise SystemExit(f"{manifest} is empty")

    wanted_bands = set(bands)
    pending: dict[str, dict[str, bytes]] = defaultdict(dict)
    written = 0
    canonical: list[dict] = []
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{len(entries)} patch(es) wanted; streaming from source", flush=True)
    stream = _open_stream(urls)
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key = _member_key(member.name)
            if key is None:
                continue
            patch_id, band = key
            if patch_id not in entries or band not in wanted_bands:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            pending[patch_id][band] = handle.read()

            if len(pending[patch_id]) == len(wanted_bands):
                canonical.extend(
                    _finish(patch_id, pending.pop(patch_id), entries[patch_id], out_root)
                )
                written += 1
                if written % 250 == 0:
                    print(f"  {written}/{len(entries)} patches", flush=True)
                if written == len(entries):
                    # Every wanted patch is complete; the rest of the 59 GB has
                    # nothing for us.
                    print("all wanted patches found; stopping the stream", flush=True)
                    break

    incomplete = {p: sorted(b) for p, b in pending.items()}
    manifest_out = out_root / "canonical.jsonl"
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in canonical:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "source_manifest": str(manifest),
        "patches_wanted": len(entries),
        "patches_written": written,
        "patches_incomplete": len(incomplete),
        "canonical_rows": len(canonical),
        "bands": list(bands),
        "geometry": "dataset-provided patch tiles (no reconstruction)",
    }
    (out_root / "extract_summary.json").write_text(
        json.dumps({**summary, "incomplete": incomplete}, indent=2), encoding="utf-8"
    )
    return summary


def _finish(patch_id: str, raw: dict[str, bytes], entry: dict, out_root: Path) -> list[dict]:
    """Build composites for one patch and emit its canonical rows."""
    import numpy as np
    import rasterio

    arrays: dict[str, np.ndarray] = {}
    for band, payload in raw.items():
        with rasterio.open(io.BytesIO(payload)) as source:
            arrays[band] = source.read(1)

    # The 20 m bands arrive at 60x60 for the same ground as the 120x120 10 m
    # bands. Nearest-neighbour onto the 10 m grid, which is what the dataset's
    # own band statistics were computed against.
    from satquery.ingest.copernicus.composites import _resample_to

    target = max((a.shape for a in arrays.values()), key=lambda s: s[0])
    arrays = {name: _resample_to(a, target) for name, a in arrays.items()}

    available: dict[str, str] = {}
    for composite in build_composites(arrays, names=ALL_COMPOSITES):
        relative = Path("composites") / f"{patch_id}_{composite.name}.png"
        path = out_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        composite.to_image().save(path)
        available[composite.name] = relative.as_posix()

    gsd = entry.get("effective_gsd_m", 10.0)
    rows = []
    for index, pair in enumerate(entry.get("qa", [])):
        adapter = pair.get("adapter", entry.get("adapter", "rs_vqa"))
        roles = [n for n in composite_names_for(adapter) if n in available]
        images = [available[n] for n in roles]
        rows.append(
            {
                "sample_id": f"{patch_id}_{index:03d}",
                "adapter": adapter,
                "task": pair.get("task", "single_vqa"),
                "question_type": pair.get("question_type", "other"),
                "images": images,
                "question": pair["question"],
                "answer": pair["answer"],
                "answer_type": "text",
                "image_roles": roles,
                "modality": ["optical"] * len(images),
                "effective_gsd_m": [gsd] * len(images),
                "split": entry.get("split", "train"),
                "source": entry.get("source", SOURCE),
                "licence": entry.get("licence", LICENCE),
                "geometry_source": "BigEarthNet v2.0 archive tile",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="data/chips")
    parser.add_argument(
        "--source",
        default="hf",
        choices=sorted(SOURCES),
        help="hf is the TorchGeo mirror at ~25 MB/s; zenodo is canonical at ~5",
    )
    args = parser.parse_args()

    summary = extract(Path(args.manifest), Path(args.out), SOURCES[args.source])
    print(json.dumps(summary, indent=2))
    if summary["patches_incomplete"]:
        print(
            f"{summary['patches_incomplete']} patch(es) had some but not all bands "
            "and were skipped rather than written partial."
        )
    return 0 if summary["patches_written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
