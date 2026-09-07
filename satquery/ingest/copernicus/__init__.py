"""Copernicus acquisition — the one tool that unblocks three deliverables.

``SPECULATIONS.md`` item 2: BigEarthNet.txt ships annotations only. The pixels
live in a monolithic 600 GB+ LMDB that cannot be streamed selectively, and
BEN.txt is the majority source of all four adapters. Nothing about training is
real until that is solved.

Two routes were on the table. Download the whole archive and discard 90% of it,
or fetch the manifest patches from Copernicus by geometry. This package is the
second, and the plan already argues for it: the same machinery is needed for
India holdout v0 (Phase 0 item 13) and for the self-generated bi-temporal change
pairs (C46). One tool, three deliverables.

The correctness risk is not the download. It is **patch geometry** — a fetched
patch whose bounds are off by one grid cell still looks like a valid Sentinel-2
chip, still trains, and carries land-cover labels for the field next door. See
:mod:`satquery.ingest.copernicus.patch_grid`, which refuses to emit a manifest
until the grid convention is verified rather than assumed.
"""

from satquery.ingest.copernicus.patch_grid import (
    GRID_CONVENTION_VERIFIED,
    PATCH_SIDE_PX,
    PatchId,
    assert_grid_convention_verified,
    parse_patch_id,
    patch_bounds,
    patch_window,
    verify_convention,
)

__all__ = [
    "GRID_CONVENTION_VERIFIED",
    "PATCH_SIDE_PX",
    "PatchId",
    "assert_grid_convention_verified",
    "parse_patch_id",
    "patch_bounds",
    "patch_window",
    "verify_convention",
]
