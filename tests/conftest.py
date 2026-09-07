import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture(autouse=True)
def _deterministic_numpy():
    """Reseed numpy's global RNG before every test.

    Several tests build synthetic arrays with bare ``np.random.*`` and then
    assert on a *threshold method* -- that Otsu was accepted rather than the
    fixed fallback. Those assertions are only meaningful for an input that is
    clearly bimodal, and an unseeded draw is not reliably that:
    ``test_execute_sar_backscatter`` measured **37 failures in 400 seeds**, so
    it failed roughly one run in eleven.

    That is worse than a test that simply fails, because it fails elsewhere.
    Twice today a red suite was blamed on an unrelated edit, once mid-deploy --
    which is exactly the wrong conclusion to reach quickly.

    Reseeding per test rather than per session so the value a test sees does
    not depend on how many tests ran before it. Tests that want their own
    generator still use ``np.random.default_rng(seed)`` and are unaffected.
    """
    np.random.seed(20260906)
    yield


@pytest.fixture
def geotiff_4band_cartosat(tmp_path):
    path = tmp_path / "cartosat_proxy.tif"
    transform = from_origin(0, 0, 10, 10)  # 10m resolution
    # Create a 4-band 12-bit raster (stored as uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=4,
        dtype="uint16",
        crs="EPSG:32643",  # UTM zone 43N
        transform=transform,
    ) as dst:
        dst.update_tags(NBITS="12")
        # Write dummy data
        for i in range(1, 5):
            dst.write((np.random.rand(100, 100) * 4095).astype(np.uint16), i)

    yield str(path)


@pytest.fixture
def geotiff_single_pol_sar(tmp_path):
    path = tmp_path / "sar_proxy.tif"
    transform = from_origin(0, 0, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:4326",  # Geographic
        transform=transform,
    ) as dst:
        dst.update_tags(SAR_BAND="C")
        dst.set_band_description(1, "vv")
        # Write dummy SAR data (dB)
        dst.write(np.random.normal(-15, 5, (100, 100)).astype(np.float32), 1)

    yield str(path)


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    """Point the credential loader at a file that does not exist.

    Without this, `satquery.credentials.load_credentials` reads the developer's
    real `~/.satquery/credentials.env` back into the environment *after* a test
    has cleared it with `monkeypatch.delenv`, so the missing-credential paths
    pass on a laptop with no credentials and fail on one that has them. The
    tests that suffered were the two that check credentials are reported rather
    than assumed -- the exact assertions a real file must not be able to bend.
    """
    monkeypatch.setenv("SATQUERY_ENV_FILE", str(tmp_path / "absent.env"))
