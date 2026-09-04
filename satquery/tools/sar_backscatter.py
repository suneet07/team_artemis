from pathlib import Path
from typing import Any

import numpy as np

from satquery.tools.spectral_index import compute_otsu_threshold


def _generate_synthetic_sar(shape: tuple[int, int] = (100, 100), pol: str = "VV") -> np.ndarray:
    """Generates deterministic calibrated SAR backscatter in dB with distinct zones."""
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((x - shape[1] // 2) ** 2 + (y - shape[0] // 2) ** 2)
    is_water = dist < (min(shape) // 4)

    # Water has low backscatter (<-18 dB), land has higher (-12 to -6 dB)
    base_db = np.where(is_water, -23.0 + 1.0 * np.sin(x / 4.0), -11.0 + 2.0 * np.cos(y / 5.0))
    if pol == "VH":
        base_db -= 6.0  # Cross-pol has lower backscatter
    return base_db.astype(np.float32)


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    pol = params.get("pol", "VV").upper()
    req_method = params.get("threshold_method", "otsu")
    manual_db = params.get("threshold_db")

    pixel_size_m = 10.0
    sar_data: np.ndarray | None = None
    is_synthetic = False
    src_transform = None
    crs_str = "EPSG:32644"

    if context and "bundle" in context:
        bundle = context["bundle"]
        if bundle and getattr(bundle, "images", None):
            sar_imgs = [img for img in bundle.images if img.modality == "sar"]
            img = sar_imgs[0] if sar_imgs else bundle.images[0]
            if img.pixel_size_m is not None:
                pixel_size_m = float(img.pixel_size_m)
            if img.crs:
                crs_str = img.crs
            p = Path(img.path)
            if p.exists() and p.is_file():
                try:
                    import rasterio

                    with rasterio.open(p) as src:
                        sar_data = src.read(1).astype(np.float32)
                        src_transform = src.transform
                        if src.crs:
                            crs_str = str(src.crs)
                except Exception:
                    sar_data = None

    if sar_data is None:
        sar_data = _generate_synthetic_sar((100, 100), pol=pol)
        is_synthetic = True

    # Otsu with bimodality gate on SAR backscatter
    otsu_val, ratio, f0, f1 = compute_otsu_threshold(sar_data)

    if manual_db is not None:
        thresh = float(manual_db)
        chosen_method = "manual"
    elif req_method == "fixed":
        thresh = -18.0
        chosen_method = "fixed_fallback"
    else:
        if ratio >= 0.5 and f0 >= 0.05 and f1 >= 0.05:
            thresh = round(otsu_val, 2)
            chosen_method = "otsu"
        else:
            thresh = -18.0
            chosen_method = "fixed_fallback"

    # Water threshold: pixels with sigma0 <= thresh (or built-up if looking for bright targets)
    mask = (sar_data <= thresh).astype(np.uint8)
    area_px = int(np.sum(mask))
    area_km2 = round(float(area_px * (pixel_size_m ** 2) / 1e6), 4)

    if context is not None:
        mask_cache = context.setdefault("mask_cache", {})
        mask_cache["sar_backscatter"] = mask

    # Write real GeoTIFF mask file to disk
    import tempfile
    mask_dir = Path(tempfile.gettempdir()) / "satquery_assets" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_file = mask_dir / f"sar_{pol.lower()}_mask.tif"

    try:
        import rasterio
        from rasterio.transform import from_origin

        if src_transform is not None:
            transform = src_transform
        else:
            top_y = float(mask.shape[0] * pixel_size_m)
            transform = from_origin(0.0, top_y, pixel_size_m, pixel_size_m)

        with rasterio.open(
            mask_file,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs_str,
            transform=transform,
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        mask_uri = str(mask_file)
    except Exception:
        fallback_file = mask_dir / f"sar_{pol.lower()}_mask.tif"
        fallback_file.write_bytes(mask.tobytes())
        mask_uri = str(fallback_file)

    out_dict: dict[str, Any] = {
        "mask_uri": mask_uri,
        "area_km2": area_km2,
        "threshold_db": thresh,
        "threshold_method": chosen_method,
        "pol": pol,
    }
    if is_synthetic:
        out_dict["synthetic"] = True
        out_dict["warning"] = "Source raster unavailable on disk; synthetic fallback used"
    return out_dict
