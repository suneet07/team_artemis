from satquery.ingest.copernicus import (
    GRID_CONVENTION_VERIFIED,
    PatchId,
    parse_patch_id,
    patch_bounds,
    patch_window,
)
from satquery.ingest.ingest import IngestResult, ingest_raster
from satquery.ingest.reader import (
    RasterMeta,
    build_meta,
    open_raster,
    read_overview,
    read_window,
    resolve_bit_depth,
)

# Only the pure-geometry half of `copernicus` is re-exported here. The network
# half (`satquery.ingest.copernicus.cdse`) needs the `fetch` extra, and the
# headless eval path (section 4.11) imports this package with no HTTP stack
# installed -- so it stays behind an explicit import.
__all__ = [
    "GRID_CONVENTION_VERIFIED",
    "IngestResult",
    "PatchId",
    "RasterMeta",
    "build_meta",
    "ingest_raster",
    "open_raster",
    "parse_patch_id",
    "patch_bounds",
    "patch_window",
    "read_overview",
    "read_window",
    "resolve_bit_depth",
]
