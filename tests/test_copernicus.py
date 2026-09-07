"""Copernicus acquisition — patch geometry and the guards around it.

The download is not the risky part. The risky part is that a patch fetched on
the wrong grid convention is a real Sentinel-2 chip of the wrong ground, wearing
the right filename and the wrong labels, and nothing downstream can tell.
"""

import pytest
from rasterio.transform import from_origin

from satquery.ingest.copernicus import cdse, patch_grid
from satquery.ingest.copernicus.patch_grid import (
    PATCH_SIDE_PX,
    PATCHES_PER_AXIS,
    parse_patch_id,
    patch_bounds,
    patch_window,
    verify_convention,
)

PATCH = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"

#: A plausible granule geotransform. Values do not matter to these tests; that
#: the code reads one rather than deriving it from the MGRS code does.
TILE = from_origin(399960.0, 5900040.0, 10.0, 10.0)


# --------------------------------------------------------------- parsing


def test_patch_id_parses_into_its_parts():
    patch = parse_patch_id(PATCH)
    assert (patch.mission, patch.level, patch.tile) == ("S2A", "MSIL2A", "T33UUP")
    assert (patch.first, patch.second) == (26, 57)
    assert patch.raw == PATCH


def test_catalogue_query_drops_the_placeholder_baseline():
    """reBEN rewrites the baseline to N9999; the real product has a real one."""
    query = parse_patch_id(PATCH).product_query
    assert "baseline" not in query
    assert "N9999" not in "".join(query.values())
    assert query["tile"] == "T33UUP"


def test_a_grid_index_past_the_last_whole_patch_is_an_error():
    """10980/120 = 91.5, so 91 is not a patch — it is a misparse."""
    with pytest.raises(ValueError, match="outside 0..90"):
        parse_patch_id(PATCH.replace("_26_57", f"_{PATCHES_PER_AXIS}_57"))


def test_non_patch_strings_are_rejected():
    for bad in ("", "S2A_MSIL2A_20170613T101031", "not-a-patch", f"{PATCH}_3"):
        with pytest.raises(ValueError):
            parse_patch_id(bad)


# --------------------------------------------------------------- geometry


def test_window_is_a_whole_patch_on_the_ten_metre_grid():
    """Column first, row second -- verified 2026-08-29, not assumed.

    This test previously asserted the opposite (``row_off == 26 * side``),
    because it was written against the unverified placeholder convention
    ``row_col_from_northwest``. The verify run against the patch's own reference
    map settled it as ``col_row_from_northwest``, so ``_26_57`` is column 26,
    row 57. The old assertion would have kept the transposed reading in place
    and reported green while every chip carried its transpose's labels.
    """
    window = patch_window(parse_patch_id(PATCH))
    assert (window.height, window.width) == (PATCH_SIDE_PX, PATCH_SIDE_PX)
    assert window.col_off == 26 * PATCH_SIDE_PX
    assert window.row_off == 57 * PATCH_SIDE_PX


def test_bounds_span_exactly_twelve_hundred_metres():
    left, bottom, right, top = patch_bounds(parse_patch_id(PATCH), TILE)
    assert right - left == pytest.approx(patch_grid.PATCH_SIDE_M)
    assert top - bottom == pytest.approx(patch_grid.PATCH_SIDE_M)


def test_the_four_conventions_are_genuinely_different_places():
    """If they coincided there would be nothing to verify."""
    patch = parse_patch_id(PATCH)
    corners = {patch_bounds(patch, TILE, name) for name in patch_grid.GRID_CONVENTIONS}
    assert len(corners) == len(patch_grid.GRID_CONVENTIONS)


def test_row_col_and_col_row_differ_by_a_transpose():
    patch = parse_patch_id(PATCH)
    first = patch_window(patch, "row_col_from_northwest")
    second = patch_window(patch, "col_row_from_northwest")
    assert (first.row_off, first.col_off) == (second.col_off, second.row_off)


def test_unknown_convention_is_refused():
    with pytest.raises(ValueError, match="unknown grid convention"):
        patch_window(parse_patch_id(PATCH), "row_col_from_mars")


# --------------------------------------------------------------- verification


def test_verification_recovers_the_convention_that_produced_the_bounds():
    patch = parse_patch_id(PATCH)
    for expected in patch_grid.GRID_CONVENTIONS:
        bounds = patch_bounds(patch, TILE, expected)
        assert verify_convention(PATCH, bounds, TILE) == expected


def test_a_diagonal_patch_settles_nothing_and_says_so():
    """first == second makes row_col and col_row identical. Refuse, do not pick."""
    diagonal = PATCH.replace("_26_57", "_40_40")
    bounds = patch_bounds(parse_patch_id(diagonal), TILE, "row_col_from_northwest")
    with pytest.raises(ValueError, match="settles nothing"):
        verify_convention(diagonal, bounds, TILE)


def test_bounds_no_convention_reproduces_are_refused_not_approximated():
    with pytest.raises(ValueError, match="No grid convention reproduces"):
        verify_convention(PATCH, (0.0, 0.0, 1200.0, 1200.0), TILE)


def test_manifest_building_is_blocked_until_the_convention_is_verified():
    if patch_grid.GRID_CONVENTION_VERIFIED:
        pytest.skip("convention verified; the guard is meant to be lifted")
    with pytest.raises(RuntimeError, match="has not been verified"):
        patch_grid.assert_grid_convention_verified()


def test_fetching_is_blocked_by_the_same_guard_before_any_network_call():
    """The guard must fire before credentials, not after a download."""
    if patch_grid.GRID_CONVENTION_VERIFIED:
        pytest.skip("convention verified; the guard is meant to be lifted")
    with pytest.raises(RuntimeError, match="has not been verified"):
        cdse.fetch_patch(PATCH)


# --------------------------------------------------------------- credentials


def test_credentials_are_reported_not_assumed(monkeypatch):
    for name in (
        "CDSE_USERNAME",
        "CDSE_PASSWORD",
        "CDSE_S3_ACCESS_KEY",
        "CDSE_S3_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert cdse.credentials_present() == {"oidc": False, "s3": False}

    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "key")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "secret")
    assert cdse.credentials_present()["s3"] is True


def test_missing_s3_credentials_name_the_right_credential(monkeypatch):
    """The OIDC password is not an S3 key, and the error has to say so."""
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NOT the account password"):
        cdse._open_s3("/eodata/whatever")


def test_only_the_bands_the_input_contract_uses_are_fetched():
    """Section 5.2 drops the 60 m bands, so fetching B01/B09 is wasted transfer."""
    assert set(cdse.BANDS_10M) == {"B02", "B03", "B04", "B08"}
    assert "B01" not in cdse.BANDS_10M + cdse.BANDS_20M
    assert "B09" not in cdse.BANDS_10M + cdse.BANDS_20M
