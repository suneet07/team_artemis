import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_tif(
    path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32644",
    nodata: float | None = None,
    descriptions: tuple[str, ...] | None = None,
    tags: dict[str, str] | None = None,
    band1_tags: dict[str, str] | None = None,
    pixel_size: float = 10.0,
):
    data = np.asarray(data)
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": from_origin(0.0, height * pixel_size, pixel_size, pixel_size),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for index, desc in enumerate(descriptions, start=1):
                dst.set_band_description(index, desc)
        if tags:
            dst.update_tags(**tags)
        if band1_tags:
            dst.update_tags(1, **band1_tags)
    return path


def optical_multispectral(path) -> str:
    rng = np.random.default_rng(42)
    gradient = np.linspace(400, 3600, 64 * 64).reshape(1, 64, 64)
    noise = rng.integers(-150, 150, size=(4, 64, 64))
    data = np.clip(gradient + noise, 0, 4095).astype(np.uint16)
    data[2] = np.clip(data[2] + 300, 0, 4095)
    data[:, 40:60, 40:60] = 0
    return write_tif(
        path,
        data,
        crs="EPSG:2056",
        nodata=0,
        descriptions=("Blue", "Green", "Red", "NIR"),
        tags={"SENSOR": "CARTOSAT-2S"},
        band1_tags={"NBITS": "12"},
    )


def dark_uint16(path) -> str:
    gradient = np.linspace(0, 3000, 32 * 32).reshape(1, 32, 32)
    return write_tif(path, gradient.astype(np.uint16))


def sar_single_pol(path) -> str:
    rng = np.random.default_rng(7)
    data = rng.exponential(scale=80.0, size=(1, 128, 128)).astype(np.float32)
    return write_tif(path, data, crs="EPSG:32644")


def sar_tagged(path) -> str:
    rng = np.random.default_rng(11)
    data = rng.exponential(scale=50.0, size=(1, 96, 96)).astype(np.float32)
    return write_tif(path, data, tags={"SENSOR": "RISAT-2B", "SAR_BAND": "X"})


def sar_quad_pol(path) -> str:
    """4-band quad-pol amplitude with no band descriptions.

    Band count alone would read this as B,G,R,NIR optical and declare NDVI
    computable, which sends spectral_index at a SAR raster and corrupts D3.
    """
    rng = np.random.default_rng(17)
    data = rng.exponential(scale=60.0, size=(4, 128, 128)).astype(np.float32)
    return write_tif(path, data, crs="EPSG:32644")


def sar_single_pol_db(path) -> str:
    """Sigma-nought already converted to dB, so mostly negative.

    Section 4.2: the ISRO eval pairs arrive pre-corrected. The speckle test
    needs positive amplitudes and cannot see this at all.
    """
    rng = np.random.default_rng(23)
    data = (10.0 * np.log10(rng.exponential(scale=0.05, size=(1, 128, 128)))).astype(np.float32)
    return write_tif(path, data, crs="EPSG:32644")


def sar_db_stack(path) -> str:
    """The 3-band model-input stack P2 itself emits (section 4.2)."""
    rng = np.random.default_rng(29)
    pol1 = 10.0 * np.log10(rng.exponential(scale=0.05, size=(128, 128)))
    pol2 = 10.0 * np.log10(rng.exponential(scale=0.02, size=(128, 128)))
    data = np.stack([pol1, pol2, pol1 - pol2]).astype(np.float32)
    return write_tif(path, data, crs="EPSG:32644")


def multispectral_13band(path) -> str:
    rng = np.random.default_rng(31)
    gradient = np.linspace(200, 4000, 13 * 64 * 64).reshape(13, 64, 64)
    data = np.clip(gradient + rng.integers(-100, 100, (13, 64, 64)), 0, 4095).astype(np.uint16)
    return write_tif(path, data, crs="EPSG:32644")


def ndvi_raster(path) -> str:
    """Float in [-1, 1] and ~40% negative — must not be mistaken for dB SAR."""
    rng = np.random.default_rng(37)
    data = rng.uniform(-1.0, 1.0, size=(1, 64, 64)).astype(np.float32)
    return write_tif(path, data, crs="EPSG:32644")


def partial_descriptions(path, descriptions: tuple[str, ...]) -> str:
    rng = np.random.default_rng(41)
    data = rng.integers(0, 4000, size=(4, 64, 64)).astype(np.uint16)
    return write_tif(path, data, crs="EPSG:32644", descriptions=descriptions)


def pan_optical(path) -> str:
    gradient = np.tile(np.linspace(30, 220, 96), (1, 96, 1))
    data = gradient.astype(np.uint8)
    return write_tif(path, data, crs="EPSG:32644", nodata=None)


def rgb_only(path) -> str:
    rng = np.random.default_rng(3)
    base = rng.integers(60, 200, size=(1, 32, 32))
    data = np.repeat(base, 3, axis=0).astype(np.uint8)
    return write_tif(path, data)


def geographic_crs(path) -> str:
    data = np.full((1, 16, 16), 100, dtype=np.uint8)
    return write_tif(path, data, crs="EPSG:4326")
