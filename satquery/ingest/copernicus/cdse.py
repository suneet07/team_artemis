"""Copernicus Data Space client — find a granule, read one patch out of it.

The naive implementation of "fetch 80,000 BigEarthNet patches" downloads each
granule as a SAFE zip. A Sentinel-2 L2A product is roughly a gigabyte, reBEN
draws on hundreds of them, and every one is downloaded to keep 120x120 px of it.
That trades a 600 GB archive for a few hundred GB of transfer -- an improvement
too small to be worth the week it costs.

So this reads **windows**, not products. CDSE exposes the same SAFE trees on S3
with byte-range support, GDAL speaks that through ``/vsis3/``, and rasterio will
decode a single 120x120 window out of a 10,980 px JP2 without fetching the rest.
The whole reBEN manifest then costs on the order of the patches themselves.

Three separate credentials are involved and they are not interchangeable:

* **OIDC username/password** (``CDSE_USERNAME`` / ``CDSE_PASSWORD``) -- a free
  Copernicus Data Space account. Used for the catalogue and for whole-product
  download.
* **S3 access key/secret** (``CDSE_S3_ACCESS_KEY`` / ``CDSE_S3_SECRET_KEY``) --
  generated separately in the CDSE portal under S3 credentials. The OIDC
  password does not work here.
* Nothing at all for a dry run: :func:`search_products` needs no auth on the
  public catalogue.

Nothing in this module falls back to a stub when a credential is missing. A
fetch that silently returns nothing is how a training corpus ends up half the
size nobody notices until the loss curve is already running.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satquery.ingest.copernicus.patch_grid import (
    PatchId,
    assert_grid_convention_verified,
    parse_patch_id,
    patch_window,
)

__all__ = [
    "CDSE_CATALOGUE",
    "CDSE_S3_ENDPOINT",
    "CDSE_TOKEN_URL",
    "GranuleRef",
    "PatchRead",
    "credentials_present",
    "fetch_patch",
    "fetch_patches",
    "resolve_granule",
    "search_products",
]

CDSE_CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

#: Band file suffixes inside a SAFE tree, by resolution directory. Only the 10 m
#: and 20 m sets are listed: section 5.2 drops the 60 m bands when composing
#: model inputs, so fetching B01/B09 would be transfer spent on data the frozen
#: input contract discards.
BANDS_10M = ("B02", "B03", "B04", "B08")
BANDS_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")


@dataclass(frozen=True)
class GranuleRef:
    """One catalogue hit: the product that contains a patch."""

    product_id: str
    name: str
    s3_path: str
    online: bool

    @property
    def baseline(self) -> str:
        """Real processing baseline, which the reBEN patch id does not carry."""
        parts = self.name.split("_")
        return parts[3] if len(parts) > 3 else ""


@dataclass
class PatchRead:
    """Pixels for one patch, per band, plus the georeferencing they came with."""

    patch_id: str
    granule: GranuleRef
    bands: dict[str, Any]
    crs: Any
    transform: Any
    convention: str

    @property
    def shape(self) -> tuple[int, int]:
        first = next(iter(self.bands.values()))
        return first.shape


def credentials_present() -> dict[str, bool]:
    """Which credential sets are configured. Report, never assume.

    Reads the out-of-repo credential file first, if there is one. Values
    already in the environment win, so a one-off shell override still works.
    """
    from satquery.credentials import load_credentials

    load_credentials()
    return {
        "oidc": bool(os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD")),
        "s3": bool(
            os.environ.get("CDSE_S3_ACCESS_KEY") and os.environ.get("CDSE_S3_SECRET_KEY")
        ),
    }


def _require(module: str, extra: str):
    """Import an optional dependency, or say exactly how to install it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{module} is needed to reach Copernicus and is not installed. "
            f'Install it with `pip install -e ".[{extra}]"`. It is an extra, not a '
            "base dependency, because the headless eval path (section 4.11) runs "
            "with no network and must not pull an HTTP stack."
        ) from error


def search_products(patch: PatchId | str, *, limit: int = 10) -> list[GranuleRef]:
    """Find the granule(s) matching a patch id in the public CDSE catalogue.

    Matches on mission, level, sensing time, orbit and tile -- every field of
    the patch id **except** the baseline, which reBEN overwrites with the
    placeholder ``N9999``. Filtering on the id verbatim returns an empty list,
    which reads like a withdrawn product rather than a malformed query.
    """
    requests = _require("requests", "fetch")
    patch = parse_patch_id(patch) if isinstance(patch, str) else patch
    query = patch.product_query

    # Name is the only field carrying tile and orbit together, so the filter is
    # a set of substring tests rather than structured attribute equality.
    clauses = [
        f"startswith(Name,'{query['mission']}_{query['level']}_{query['sensing']}')",
        f"contains(Name,'_{query['orbit']}_')",
        f"contains(Name,'_{query['tile']}_')",
    ]
    response = requests.get(
        CDSE_CATALOGUE,
        params={"$filter": " and ".join(clauses), "$top": limit, "$expand": "Attributes"},
        timeout=60,
    )
    response.raise_for_status()

    granules = []
    for entry in response.json().get("value", []):
        granules.append(
            GranuleRef(
                product_id=entry["Id"],
                name=entry["Name"],
                # Taken from the response, never constructed. The same rule as
                # reading the geotransform out of the granule: the authoritative
                # answer is in the reply, and guessing a bucket layout that
                # changes under us is an avoidable class of bug.
                s3_path=entry.get("S3Path", ""),
                online=bool(entry.get("Online", True)),
            )
        )
    return granules


def resolve_granule(patch: PatchId | str) -> GranuleRef:
    """The single granule for a patch, or an error explaining the ambiguity."""
    patch = parse_patch_id(patch) if isinstance(patch, str) else patch
    granules = search_products(patch)

    if not granules:
        raise LookupError(
            f"No CDSE product matches {patch.raw}. Check the tile is still "
            "published: reBEN was built from an older processing baseline and a "
            "reprocessing campaign can retire the exact product while the same "
            "footprint stays available under a new baseline."
        )
    offline = [g for g in granules if not g.online]
    live = [g for g in granules if g.online]
    if not live:
        raise LookupError(
            f"{len(offline)} product(s) match {patch.raw} but all are offline and "
            "need staging from long-term archive before any read will succeed."
        )
    if len(live) > 1:
        names = ", ".join(g.name for g in live)
        raise LookupError(
            f"{len(live)} online products match {patch.raw} ({names}). They differ "
            "only by processing baseline; pick one deliberately and record which, "
            "because mixing baselines mixes atmospheric-correction versions across "
            "the training corpus."
        )
    return live[0]


def _open_s3(url: str):
    """Open a remote JP2 through GDAL vsis3 with CDSE S3 credentials."""
    rasterio = _require("rasterio", "fetch")

    creds = credentials_present()
    if not creds["s3"]:
        raise RuntimeError(
            "CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY are not set. These are "
            "generated in the CDSE portal under S3 credentials and are NOT the "
            "account password -- the OIDC login will not authenticate S3."
        )
    # Credentials reach GDAL through the process environment, not through
    # rasterio.Env kwargs and not through boto3.
    #
    # rasterio refuses AWS_ACCESS_KEY_ID and friends as Env kwargs outright
    # ("AWS credentials are handled exclusively by boto3"), and its AWSSession
    # imports boto3, which this project does not otherwise need -- it would
    # follow into the Modal image for one S3 endpoint. GDAL's /vsis3/ driver
    # reads the same names straight from the environment, so setting them there
    # is both smaller and closer to what actually consumes them.
    #
    # Restored on exit: leaving credentials in os.environ outlives the read and
    # would leak into any later library that looks for AWS keys.
    @contextmanager
    def _s3_env():
        overrides = {
            "AWS_ACCESS_KEY_ID": os.environ["CDSE_S3_ACCESS_KEY"],
            "AWS_SECRET_ACCESS_KEY": os.environ["CDSE_S3_SECRET_KEY"],
            "AWS_S3_ENDPOINT": CDSE_S3_ENDPOINT.replace("https://", ""),
        }
        previous = {name: os.environ.get(name) for name in overrides}
        os.environ.update(overrides)
        try:
            with rasterio.Env(
                # CDSE serves path-style URLs; virtual hosting would resolve
                # eodata.<endpoint> and 404 on every band.
                AWS_VIRTUAL_HOSTING="FALSE",
                AWS_HTTPS="YES",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            ) as env:
                yield env
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    return _s3_env(), rasterio


def fetch_patch(
    patch_id: str,
    *,
    granule: GranuleRef | None = None,
    bands: tuple[str, ...] = BANDS_10M,
    convention: str | None = None,
    require_verified_convention: bool = True,
) -> PatchRead:
    """Read one reBEN patch out of its granule without downloading the granule.

    ``require_verified_convention`` exists for exactly one caller: the
    verification run that settles the grid convention in the first place, which
    necessarily happens before it is verified. Every other caller leaves it on.
    Fetching a corpus on a guessed convention produces real imagery under the
    wrong labels, and no metric downstream will show it.
    """
    if require_verified_convention:
        assert_grid_convention_verified()

    from satquery.ingest.copernicus.patch_grid import DEFAULT_CONVENTION

    convention = convention or DEFAULT_CONVENTION
    patch = parse_patch_id(patch_id)
    granule = granule or resolve_granule(patch)
    if not granule.s3_path:
        raise LookupError(
            f"{granule.name} carries no S3Path in the catalogue response, so the "
            "windowed read has no address to open. Re-run the search with "
            "$expand=Attributes, or fall back to whole-product download."
        )

    env, rasterio = _open_s3(granule.s3_path)
    window = patch_window(patch, convention).as_rasterio_window()

    out: dict[str, Any] = {}
    crs = transform = None
    with env:
        for band in bands:
            url = _band_url(granule, band)
            with rasterio.open(url) as source:
                scale = source.width / 10980  # 1.0 at 10 m, 0.5 at 20 m
                scaled = rasterio.windows.Window(
                    window.col_off * scale,
                    window.row_off * scale,
                    window.width * scale,
                    window.height * scale,
                )
                out[band] = source.read(1, window=scaled)
                if crs is None:
                    crs = source.crs
                    transform = source.window_transform(scaled)

    return PatchRead(
        patch_id=patch.raw,
        granule=granule,
        bands=out,
        crs=crs,
        transform=transform,
        convention=convention,
    )


#: Resolved band paths per product id. A granule group shares one listing --
#: the whole point of grouping patches by granule is that the per-granule work
#: happens once.
_BAND_PATH_CACHE: dict[str, dict[str, str]] = {}


def _list_nodes(url: str) -> list[str]:
    """Names of the immediate children of a SAFE node, via the OData Nodes API."""
    requests = _require("requests", "fetch")
    response = requests.get(f"{url}/Nodes", timeout=60)
    response.raise_for_status()
    return [node["Name"] for node in response.json().get("result", [])]


def resolve_band_paths(granule: GranuleRef) -> dict[str, str]:
    """Map each band to its exact ``/vsis3/`` path by listing the SAFE tree.

    This used to return a glob -- ``GRANULE/*/IMG_DATA/R10m/*_B04_*.jp2`` --
    on the assumption that GDAL would expand it. **GDAL does not.** ``/vsis3/``
    treats the path as a literal object key, asks S3 for a key containing an
    asterisk, and gets nothing; with ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``
    it cannot list to resolve either, so it retries rather than failing. The
    symptom is an open that hangs with no error, which is why this is worth a
    paragraph.

    The names are genuinely not derivable. The granule subdirectory is
    ``L2A_T33UUP_A010315_20170613T101608`` -- it carries the absolute orbit and
    the datastrip sensing time, neither of which appears in the product name.
    So it is listed, not constructed, the same rule the catalogue query follows.

    The Nodes API needs no authentication, which keeps this on the public
    catalogue path rather than the credentialed one.
    """
    if granule.product_id in _BAND_PATH_CACHE:
        return _BAND_PATH_CACHE[granule.product_id]

    base = (
        f"{CDSE_CATALOGUE}({granule.product_id})/Nodes({granule.name})"
        if granule.product_id
        else ""
    )
    if not base:
        raise LookupError(f"{granule.name} has no product id; cannot list its SAFE tree")

    granule_dirs = _list_nodes(f"{base}/Nodes(GRANULE)")
    if len(granule_dirs) != 1:
        raise LookupError(
            f"{granule.name} has {len(granule_dirs)} granule directories "
            f"({granule_dirs}). A single-tile L2A product has exactly one; more "
            "means the layout changed and picking the first would be a guess."
        )

    root = granule.s3_path.lstrip("/")
    by_resolution: dict[str, dict[str, str]] = {}
    for resolution in ("R10m", "R20m"):
        node = f"{base}/Nodes(GRANULE)/Nodes({granule_dirs[0]})/Nodes(IMG_DATA)/Nodes({resolution})"
        found: dict[str, str] = {}
        for name in _list_nodes(node):
            # T33UUP_20170613T101031_B04_10m.jp2 -> B04
            parts = name.removesuffix(".jp2").split("_")
            if len(parts) >= 3:
                found[parts[-2]] = (
                    f"/vsis3/{root}/GRANULE/{granule_dirs[0]}/IMG_DATA/{resolution}/{name}"
                )
        by_resolution[resolution] = found

    # Resolution is chosen per band, never by listing order. B02/B03/B04 are
    # published at BOTH 10 m and 20 m: a flat merge lets whichever directory was
    # walked last win, which silently resolved the visible bands to 20 m and
    # would have halved the GSD of every true-colour composite with no error
    # anywhere. The native resolution is a property of the band, so read it from
    # the band.
    paths: dict[str, str] = {}
    for band in BANDS_10M:
        if band in by_resolution["R10m"]:
            paths[band] = by_resolution["R10m"][band]
    for band in BANDS_20M:
        if band in by_resolution["R20m"]:
            paths[band] = by_resolution["R20m"][band]
    # Auxiliary layers (SCL, AOT, TCI, WVP) are not part of the frozen input
    # contract but are useful for masking. They have no "native" resolution to
    # honour, so the finest available wins.
    for resolution in ("R20m", "R10m"):
        for band, url in by_resolution[resolution].items():
            if band not in BANDS_10M + BANDS_20M:
                paths[band] = url

    _BAND_PATH_CACHE[granule.product_id] = paths
    return paths


def _band_url(granule: GranuleRef, band: str) -> str:
    """``/vsis3/`` address of one band inside a SAFE tree."""
    paths = resolve_band_paths(granule)
    if band not in paths:
        raise LookupError(
            f"{band} is not in {granule.name}. Present: {sorted(paths)}. A band the "
            "product does not carry is a manifest error, not something to skip."
        )
    return paths[band]



def fetch_patches(
    patch_ids: list[str],
    *,
    granule: GranuleRef,
    bands: tuple[str, ...] = BANDS_10M,
    convention: str | None = None,
    require_verified_convention: bool = True,
    on_error=None,
):
    """Read many patches out of ONE granule, opening each band once.

    :func:`fetch_patch` opens every band's JP2 for every patch. That is correct
    and it is also why a 20-patch fetch took over ten minutes: at 175 patches
    per granule -- which is what stratified sampling produces -- it costs
    6 x 175 = 1,050 network opens of a 10,980 px JP2 where 6 would do. At ~30 s
    per patch a 20,000-patch corpus would take about 167 hours.

    Here each band is opened once and every window is read from that open
    handle, so the per-patch cost drops to the windows themselves. GDAL's block
    cache also starts working for it rather than being discarded per patch.

    Yields ``PatchRead`` as each patch completes, so the caller can write and
    release it instead of holding a granule's worth of arrays in memory. A
    patch that fails is passed to ``on_error`` and skipped -- one bad window
    must not abandon the other 174.
    """
    if require_verified_convention:
        assert_grid_convention_verified()

    from satquery.ingest.copernicus.patch_grid import DEFAULT_CONVENTION

    convention = convention or DEFAULT_CONVENTION
    if not granule.s3_path:
        raise LookupError(
            f"{granule.name} carries no S3Path in the catalogue response, so the "
            "windowed read has no address to open."
        )

    env, rasterio = _open_s3(granule.s3_path)
    parsed = []
    for patch_id in patch_ids:
        try:
            parsed.append(parse_patch_id(patch_id))
        except ValueError as error:
            if on_error is not None:
                on_error(patch_id, error)

    pixels: dict[str, dict[str, Any]] = {p.raw: {} for p in parsed}
    crs: dict[str, Any] = {}
    transform: dict[str, Any] = {}
    failed: set[str] = set()

    with env:
        for band in bands:
            url = _band_url(granule, band)
            with rasterio.open(url) as source:
                scale = source.width / 10980  # 1.0 at 10 m, 0.5 at 20 m
                for patch in parsed:
                    if patch.raw in failed:
                        continue
                    window = patch_window(patch, convention).as_rasterio_window()
                    scaled = rasterio.windows.Window(
                        window.col_off * scale,
                        window.row_off * scale,
                        window.width * scale,
                        window.height * scale,
                    )
                    try:
                        pixels[patch.raw][band] = source.read(1, window=scaled)
                    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                        failed.add(patch.raw)
                        if on_error is not None:
                            on_error(patch.raw, error)
                        continue
                    if patch.raw not in crs:
                        crs[patch.raw] = source.crs
                        transform[patch.raw] = source.window_transform(scaled)

    for patch in parsed:
        if patch.raw in failed:
            continue
        yield PatchRead(
            patch_id=patch.raw,
            granule=granule,
            bands=pixels[patch.raw],
            crs=crs.get(patch.raw),
            transform=transform.get(patch.raw),
            convention=convention,
        )



def write_patch(read: PatchRead, path: Path | str) -> Path:
    """Write a fetched patch as a GeoTIFF carrying its own georeferencing.

    GeoTIFF and not PNG: the whole ingest path (section 4.1) keys off CRS and
    transform, masks are graded as geo-referenced rasters (C18), and a PNG chip
    silently discards exactly the metadata that makes the rest of the system
    work. The micro-split shipped as PNG, which is why nothing downstream of it
    could be co-registered.
    """
    rasterio = _require("rasterio", "fetch")
    numpy = _require("numpy", "fetch")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(read.bands)

    # The 20 m bands come back at half the 10 m bands' side (60x60 against
    # 120x120), because each is read as a window of its own native raster. A
    # GeoTIFF holds one grid and one transform, so they are put on the 10 m grid
    # here -- nearest-neighbour, which replicates measured values rather than
    # interpolating SWIR into numbers no sensor produced.
    #
    # Which bands were upsampled is written into the tags below. Without that
    # the file claims 10 m for every band, and GSD-conditioned prompting
    # (section 5.2) would tell the model a resolution the SWIR bands never had.
    from satquery.ingest.copernicus.composites import _resample_to

    target = max(
        (numpy.asarray(read.bands[name]).shape for name in names),
        key=lambda shape: shape[0],
    )
    upsampled = [
        name
        for name in names
        if numpy.asarray(read.bands[name]).shape != tuple(target)
    ]
    stack = numpy.stack([_resample_to(read.bands[name], target) for name in names])

    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": len(names),
        "dtype": stack.dtype,
        "crs": read.crs,
        "transform": read.transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(stack)
        for index, name in enumerate(names, start=1):
            destination.set_band_description(index, name)
        destination.update_tags(
            patch_id=read.patch_id,
            source_product=read.granule.name,
            processing_baseline=read.granule.baseline,
            grid_convention=read.convention,
            # Native resolution per band, so no reader has to infer it from the
            # raster's own pixel size -- which is now 10 m for every band and
            # would be wrong for six of them.
            grid_gsd_m="10",
            upsampled_bands=",".join(upsampled),
            upsampled_from_gsd_m="20" if upsampled else "",
            resampling="nearest" if upsampled else "none",
        )
        for index, name in enumerate(names, start=1):
            destination.update_tags(
                index,
                band=name,
                native_gsd_m="20" if name in upsampled else "10",
            )
    return path
