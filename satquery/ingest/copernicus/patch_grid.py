"""reBEN patch identifiers and the tile grid they index into.

A BigEarthNet v2.0 patch id looks like::

    S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57

which is a Sentinel-2 product name with two grid indices appended. Recovering
the pixels means resolving those indices back to a window in the source granule.

WHY THIS MODULE DOES NO MGRS ARITHMETIC
---------------------------------------
The obvious implementation derives the tile's UTM origin from the MGRS code
(``T33UUP`` -> zone 33, band U, 100 km square UP) and computes absolute bounds.
That is a trap. Sentinel-2 tiles are 109,800 m across on a 100,000 m MGRS
lattice, so they overlap their neighbours and their upper-left corners are *not*
a clean function of the 100 km square. Getting it slightly wrong yields patches
that look entirely valid and are shifted by a fixed offset.

So geometry here is expressed **relative to the granule's own geotransform**,
read from the product we downloaded. The authoritative answer ships inside the
file; there is no reason to re-derive it and every reason not to.

WHAT WAS UNVERIFIED, AND HOW IT WAS SETTLED
-------------------------------------------
The order and origin of the two trailing indices. Four readings are plausible
(row-then-column or column-then-row, counted from the north-west or the
south-west corner) and they are indistinguishable from a filename. A wrong
choice does not crash: it returns a real Sentinel-2 chip of the wrong 1.2 km of
ground, carrying CORINE labels for somewhere else. Every downstream number --
loss, accuracy, the D1 agreement rate -- stays plausible while the supervision
is silently scrambled.

Settled 2026-08-29 against patch ``..._T33UUP_26_57``, whose true bounds came
from its own reference map in the reBEN ``Reference_Maps`` archive. The answer
is ``col_row_from_northwest``: **column first**, not row. The placeholder this
module shipped with was ``row_col_from_northwest``, chosen as "the most likely",
and it was the wrong one -- which is the whole argument for the guard rather
than a footnote to it. The arithmetic is recorded beside
:data:`GRID_CONVENTION_VERIFIED` so it can be re-checked rather than trusted.

:func:`assert_grid_convention_verified` blocked manifest emission until then.
:func:`verify_convention` settles it from a single patch of known bounds and
refuses to answer when the evidence does not discriminate -- a diagonal patch,
where the two indices are equal, is consistent with both readings and proves
nothing.

This is the same guard as ``satquery.qgen.boxes.BOX_CONVENTION_VERIFIED``, for
the same reason: a convention that is merely *probable* is not a foundation to
put 60,000 training samples on.
"""

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "GRID_CONVENTIONS",
    "GRID_CONVENTION_VERIFIED",
    "PATCHES_PER_AXIS",
    "PATCH_SIDE_M",
    "PATCH_SIDE_PX",
    "TILE_SIDE_PX",
    "GridConvention",
    "PatchId",
    "PatchWindow",
    "assert_grid_convention_verified",
    "parse_patch_id",
    "patch_bounds",
    "patch_window",
    "verify_convention",
]

#: reBEN patches are 1200 m square (BigEarthNet v2.0 paper), which at the 10 m
#: bands is 120 px. The 20 m and 60 m bands ship at 60 px and 20 px for the same
#: ground footprint -- section 4.1.5 harmonisation upsamples them.
PATCH_SIDE_PX = 120
PATCH_SIDE_M = 1200.0

#: A Sentinel-2 granule is 10,980 px on the 10 m bands (109,800 m).
TILE_SIDE_PX = 10980

#: 10980 / 120 = 91.5. Ninety-one whole patches fit per axis and the trailing
#: 60 px are unusable, so valid indices are 0..90. An index above that is not an
#: edge case to clamp -- it is a parse error, or the grid is not what we think.
PATCHES_PER_AXIS = TILE_SIDE_PX // PATCH_SIDE_PX

GridConvention = Literal[
    "row_col_from_northwest",
    "row_col_from_southwest",
    "col_row_from_northwest",
    "col_row_from_southwest",
]

#: Every reading that a filename alone cannot rule out.
GRID_CONVENTIONS: tuple[GridConvention, ...] = (
    "row_col_from_northwest",
    "row_col_from_southwest",
    "col_row_from_northwest",
    "col_row_from_southwest",
)

#: The reading the rest of the code uses. VERIFIED 2026-08-29, and note that the
#: previous value here was ``row_col_from_northwest`` -- "standard raster order,
#: which is the most likely". It was wrong. Every patch would have been read at
#: its transpose and carried another patch's land-cover labels, with nothing
#: downstream able to see it. This is the entire reason the guard exists.
DEFAULT_CONVENTION: GridConvention = "col_row_from_northwest"

#: Verified, not assumed. The evidence, re-checkable end to end:
#:
#:   patch:  S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
#:   bounds: 331200.0 5330400.0 332400.0 5331600.0  (EPSG:32633)
#:           from the patch's own reference map in reBEN Reference_Maps.tar.zst,
#:           kept at data/reben/reference_sample.tif
#:   tile:   origin (300000, 5400000) @ 10 m, from the granule's geotransform
#:
#:   (331200 - 300000) / 10 = 3120 px = 26 x 120  -> first index is the COLUMN
#:   (5400000 - 5331600) / 10 = 6840 px = 57 x 120 -> second index is the ROW
#:
#: Both exact, no tolerance slack. The patch is off-diagonal (26 != 57), so the
#: two candidate conventions genuinely disagreed and one was discriminated
#: against rather than merely being consistent.
#:
#: Reproduce with:
#:   python scripts/fetch_reben_reference_bounds.py
#:   python scripts/fetch_copernicus_patches.py verify --patch-id ... --bounds ...
GRID_CONVENTION_VERIFIED = True

_PATCH_ID = re.compile(
    r"^(?P<mission>S2[AB])"
    r"_(?P<level>MSIL[12][ABC])"
    r"_(?P<sensing>\d{8}T\d{6})"
    r"_(?P<baseline>N\d{4})"
    r"_(?P<orbit>R\d{3})"
    r"_(?P<tile>T\d{2}[A-Z]{3})"
    r"_(?P<first>\d{1,3})"
    r"_(?P<second>\d{1,3})$"
)


@dataclass(frozen=True)
class PatchId:
    """One reBEN patch identifier, parsed.

    ``first`` and ``second`` are deliberately not named ``row`` and ``col``:
    which is which is exactly the open question, and naming them would bake an
    answer into the type. Use :func:`patch_window` with an explicit convention.
    """

    mission: str
    level: str
    sensing: str
    baseline: str
    orbit: str
    tile: str
    first: int
    second: int
    #: The identifier exactly as the dataset wrote it. Empty only when a PatchId
    #: was constructed by hand rather than parsed.
    original: str = ""

    @property
    def raw(self) -> str:
        """The dataset's own identifier, not a reconstruction of it.

        reBEN **zero-pads** the grid indices: ``..._T29SNB_00_09``. Rebuilding
        the string from the parsed integers produces ``..._T29SNB_0_9``, which
        is an id the dataset does not use -- so every filename and every join
        keyed on it silently diverges for any index below 10. That is about a
        fifth of all patches.
        """
        if self.original:
            return self.original
        return (
            f"{self.mission}_{self.level}_{self.sensing}_{self.baseline}"
            f"_{self.orbit}_{self.tile}_{self.first}_{self.second}"
        )

    @property
    def product_query(self) -> dict[str, str]:
        """Fields that identify the source granule in the Copernicus catalogue.

        The baseline is **excluded on purpose.** reBEN rewrites it to the
        placeholder ``N9999``; the real product carries a processing baseline
        (``N0205``, ``N0500``, ...) and a second timestamp that the patch id
        drops entirely. Searching the catalogue by the patch-id prefix returns
        nothing, which reads as "the product was withdrawn" rather than "the
        query was wrong".
        """
        return {
            "mission": self.mission,
            "level": self.level,
            "sensing": self.sensing,
            "orbit": self.orbit,
            "tile": self.tile,
        }


@dataclass(frozen=True)
class PatchWindow:
    """A patch footprint in granule pixel coordinates, at the 10 m bands."""

    row_off: int
    col_off: int
    height: int = PATCH_SIDE_PX
    width: int = PATCH_SIDE_PX

    def as_rasterio_window(self):
        from rasterio.windows import Window

        return Window(self.col_off, self.row_off, self.width, self.height)


def parse_patch_id(patch_id: str) -> PatchId:
    """Parse a reBEN patch identifier, rejecting anything that is not one."""
    match = _PATCH_ID.match(patch_id.strip())
    if match is None:
        raise ValueError(
            f"not a reBEN patch id: {patch_id!r}. Expected "
            "<mission>_<level>_<sensing>_<baseline>_<orbit>_<tile>_<i>_<j>, e.g. "
            "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
        )
    fields = match.groupdict()
    first, second = int(fields.pop("first")), int(fields.pop("second"))
    for name, value in (("first", first), ("second", second)):
        if not 0 <= value < PATCHES_PER_AXIS:
            raise ValueError(
                f"{name} grid index {value} is outside 0..{PATCHES_PER_AXIS - 1} for "
                f"{patch_id!r}. A 10,980 px granule holds {PATCHES_PER_AXIS} whole "
                f"{PATCH_SIDE_PX} px patches per axis; an index above that means the "
                "id was misparsed or the grid is not what we think it is."
            )
    return PatchId(
        first=first, second=second, original=patch_id.strip(), **fields
    )


def patch_window(
    patch: PatchId, convention: GridConvention = DEFAULT_CONVENTION
) -> PatchWindow:
    """Pixel window of ``patch`` within its granule, under ``convention``.

    Pure integer arithmetic on the 10 m grid. No CRS and no geotransform --
    those come from the granule when the window is turned into bounds.
    """
    if convention not in GRID_CONVENTIONS:
        raise ValueError(
            f"unknown grid convention {convention!r}; one of {GRID_CONVENTIONS}"
        )

    if convention.startswith("row_col"):
        row_index, col_index = patch.first, patch.second
    else:
        col_index, row_index = patch.first, patch.second

    if convention.endswith("from_southwest"):
        # Row 0 at the bottom edge of the usable grid, counting north.
        row_index = PATCHES_PER_AXIS - 1 - row_index

    return PatchWindow(
        row_off=row_index * PATCH_SIDE_PX, col_off=col_index * PATCH_SIDE_PX
    )


def patch_bounds(
    patch: PatchId,
    tile_transform,
    convention: GridConvention = DEFAULT_CONVENTION,
) -> tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` of ``patch`` in the granule own CRS.

    ``tile_transform`` is the granule affine geotransform, read from the
    downloaded product. Deriving it from the MGRS code instead is the mistake
    this module exists to avoid.
    """
    from rasterio.windows import bounds as window_bounds

    window = patch_window(patch, convention)
    return tuple(window_bounds(window.as_rasterio_window(), tile_transform))


def verify_convention(
    patch_id: str,
    expected_bounds: tuple[float, float, float, float],
    tile_transform,
    *,
    tolerance_m: float = 1.0,
) -> GridConvention:
    """Settle the grid convention against one patch of known bounds.

    ``expected_bounds`` must come from the dataset's own metadata. **Not from
    ``metadata.parquet``** -- that file carries labels, split and country and no
    geometry of any kind, verified 2026-08-29. The georeferencing is in
    ``Reference_Maps.tar.zst``, whose members are per-patch GeoTIFFs;
    ``scripts/fetch_reben_reference_bounds.py`` streams one out without
    downloading the 282 MB archive. Not from a guess, and not from a patch this
    code produced.

    Raises when the evidence does not discriminate. Two conventions agreeing is
    not a licence to pick one: on a square grid ``row_col`` and ``col_row``
    coincide wherever ``first == second``, so a diagonal patch proves nothing.
    Re-run with an off-diagonal patch.
    """
    patch = parse_patch_id(patch_id)
    matches = [
        convention
        for convention in GRID_CONVENTIONS
        if max(
            abs(a - b)
            for a, b in zip(
                patch_bounds(patch, tile_transform, convention),
                expected_bounds,
                strict=True,
            )
        )
        <= tolerance_m
    ]

    if not matches:
        raise ValueError(
            f"No grid convention reproduces {expected_bounds} for {patch_id!r}. "
            "Either the expected bounds are in a different CRS from the granule "
            "transform, or the reBEN grid is not a plain 120 px lattice over the "
            "10 m band. Do not proceed on the closest match."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{patch_id!r} is consistent with {len(matches)} conventions ({matches}) "
            f"and settles nothing. first={patch.first} second={patch.second}: pick a "
            "patch where the two indices differ, and one not mirrored about the grid "
            "centre."
        )
    return matches[0]


def assert_grid_convention_verified() -> None:
    """Refuse to build training data on an unverified grid convention.

    Called by manifest and patch-fetch entry points. A wrong convention costs
    the entire training corpus and is invisible in every metric, so this fails
    loudly at build time instead of quietly at accuracy time.
    """
    if not GRID_CONVENTION_VERIFIED:
        raise RuntimeError(
            "The reBEN patch grid convention has not been verified. Run "
            "satquery.ingest.copernicus.verify_convention() against a patch whose "
            "bounds come from the official reBEN metadata, set "
            "GRID_CONVENTION_VERIFIED = True with the settling patch recorded in "
            "CREDITS.md, then re-run. Fetching on a guessed convention yields real "
            "Sentinel-2 imagery labelled with the wrong ground truth."
        )
