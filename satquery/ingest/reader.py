from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterMeta:
    path: str
    width: int
    height: int
    count: int
    crs: str | None
    bounds: tuple[float, float, float, float]
    pixel_size_m: float | None
    native_gsd_m: float | None
    dtype: str
    nodata: float | None
    descriptions: tuple[str | None, ...]
    bit_depth: int
    bit_depth_source: str
    sensor_hint: str | None
    tags: dict[str, str]


def open_raster(path) -> rasterio.DatasetReader:
    return rasterio.open(path)


def read_overview(src: rasterio.DatasetReader, max_side: int = 512) -> np.ma.MaskedArray:
    factor = max(src.width, src.height) / max_side
    if factor <= 1:
        out_shape = (src.count, src.height, src.width)
    else:
        out_shape = (src.count, round(src.height / factor), round(src.width / factor))
    return src.read(masked=True, out_shape=out_shape)


def read_window(
    src: rasterio.DatasetReader, row_off: int, col_off: int, height: int, width: int
) -> np.ma.MaskedArray:
    return src.read(masked=True, window=Window(col_off, row_off, width, height))


def _declared_nbits(src: rasterio.DatasetReader) -> int | None:
    raw = src.tags().get("NBITS")
    if raw is None:
        raw = src.tags(1).get("NBITS")
    if raw is None:
        return None
    try:
        value = int(str(raw))
    except ValueError:
        return None
    return value if value in (8, 12, 16) else None


def resolve_bit_depth(dtype: str, nbits: int | None) -> tuple[int, str]:
    if nbits is not None:
        return nbits, "nbits_metadata"
    info = np.dtype(dtype)
    if info.kind == "f":
        return 32, "float_dtype"
    bits = {1: 8, 2: 16, 4: 32, 8: 64}.get(info.itemsize, 32)
    return bits, "container_dtype"


def _sensor_hint(tags: dict[str, str]) -> str | None:
    for key in ("SENSOR", "sensor", "SATELLITE", "satellite", "PLATFORM", "platform"):
        if key in tags and tags[key]:
            return tags[key]
    return None


def _native_gsd(tags: dict[str, str]) -> float | None:
    for key in ("GSD_M", "gsd_m", "GSD", "gsd", "NATIVE_GSD_M"):
        if key in tags:
            try:
                return float(tags[key])
            except ValueError:
                continue
    return None


def build_meta(
    path, src: rasterio.DatasetReader, overview: np.ma.MaskedArray
) -> tuple[RasterMeta, list[str]]:
    notes: list[str] = []
    tags = {str(k): str(v) for k, v in src.tags().items()}
    valid = overview.compressed()
    max_value = float(valid.max()) if valid.size else 0.0
    if len(set(src.dtypes)) > 1:
        notes.append("mixed band dtypes; bit depth estimated from first band")

    bit_depth, bit_depth_source = resolve_bit_depth(src.dtypes[0], _declared_nbits(src))
    if bit_depth_source == "container_dtype" and bit_depth == 16 and 0 < max_value <= (1 << 12) - 1:
        notes.append(
            "values fit within a 12-bit range but the file declares no NBITS; "
            "reporting the uint16 container depth"
        )

    pixel_size_m: float | None = None
    if src.crs is not None and not src.crs.is_projected:
        notes.append("CRS is geographic; pixel_size_m stored as None (units are degrees)")
    elif src.crs is None:
        notes.append("no CRS defined")
    else:
        pixel_size_m = (abs(src.transform.a) + abs(src.transform.e)) / 2.0

    meta = RasterMeta(
        path=str(path),
        width=src.width,
        height=src.height,
        count=src.count,
        crs=str(src.crs) if src.crs is not None else None,
        bounds=tuple(src.bounds),
        pixel_size_m=pixel_size_m,
        native_gsd_m=_native_gsd(tags),
        dtype=src.dtypes[0],
        nodata=src.nodata,
        descriptions=tuple(src.descriptions),
        bit_depth=bit_depth,
        bit_depth_source=bit_depth_source,
        sensor_hint=_sensor_hint(tags),
        tags=tags,
    )
    if meta.native_gsd_m is None:
        notes.append("native GSD unknown from metadata; differs from pixel size if resampled")
    return meta, notes
