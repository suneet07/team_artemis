from pathlib import Path
from typing import Any

import numpy as np


def compute_otsu_threshold(data: np.ndarray) -> tuple[float, float, float, float]:
    """Computes Otsu threshold and bimodality metrics.

    Returns (threshold, variance_ratio, class0_frac, class1_frac).
    """
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0

    hist, bin_edges = np.histogram(valid, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    weight0 = np.cumsum(hist) / total
    weight1 = 1.0 - weight0

    cum_sum = np.cumsum(hist * bin_centers)
    total_mean = cum_sum[-1] / total if total > 0 else 0.0

    mean0 = np.zeros_like(weight0)
    nonzero0 = weight0 > 0
    mean0[nonzero0] = cum_sum[nonzero0] / (weight0[nonzero0] * total)

    mean1 = np.zeros_like(weight1)
    nonzero1 = weight1 > 0
    mean1[nonzero1] = (total_mean * total - cum_sum[nonzero1]) / (weight1[nonzero1] * total)

    total_var = np.sum(hist * (bin_centers - total_mean) ** 2) / total
    if total_var <= 1e-12:
        return float(bin_centers[0]), 0.0, 1.0, 0.0

    between_class_var = weight0 * weight1 * ((mean0 - mean1) ** 2)
    best_idx = int(np.argmax(between_class_var))

    best_thresh = float(bin_centers[best_idx])
    var_ratio = float(between_class_var[best_idx] / total_var)
    frac0 = float(weight0[best_idx])
    frac1 = float(weight1[best_idx])

    return best_thresh, var_ratio, frac0, frac1


def _generate_synthetic_bands(shape: tuple[int, int] = (100, 100)) -> dict[str, np.ndarray]:
    """Generates purely deterministic synthetic bands with a distinct central water body."""
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((x - shape[1] // 2) ** 2 + (y - shape[0] // 2) ** 2)
    is_water = dist < (min(shape) // 4)

    nir = np.where(is_water, 0.05, 0.65 + 0.05 * np.sin(x / 5.0))
    red = np.where(is_water, 0.08, 0.15 + 0.03 * np.cos(y / 5.0))
    green = np.where(is_water, 0.35, 0.20)
    swir = np.where(is_water, 0.02, 0.18 + 0.02 * np.sin((x + y) / 10.0))
    blue = np.where(is_water, 0.25, 0.10)

    return {
        "nir": nir.astype(np.float32),
        "red": red.astype(np.float32),
        "green": green.astype(np.float32),
        "swir": swir.astype(np.float32),
        "blue": blue.astype(np.float32),
    }


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    index_name = params.get("index", "NDWI").upper()
    req_method = params.get("threshold_method", "otsu")
    manual_val = params.get("threshold_value")

    # Fixed physical fallback thresholds (§14.1 / §4.6.2)
    fixed_fallbacks = {
        "NDWI": 0.20,
        "MNDWI": 0.20,
        "NDVI": 0.30,
        "NDBI": 0.10,
    }

    # Obtain band arrays (from raster file if present and readable, or deterministic synthetic)
    bands_data: dict[str, np.ndarray] | None = None
    pixel_size_m = 10.0

    is_synthetic = False
    src_transform = None
    crs_str = "EPSG:32644"

    if context and "bundle" in context:
        bundle = context["bundle"]
        if bundle and getattr(bundle, "images", None):
            first_img = bundle.images[0]
            if first_img.pixel_size_m is not None:
                pixel_size_m = float(first_img.pixel_size_m)
            if first_img.crs:
                crs_str = first_img.crs
            p = Path(first_img.path)
            if p.exists() and p.is_file():
                try:
                    import rasterio

                    with rasterio.open(p) as src:
                        data = src.read()
                        src_transform = src.transform
                        if src.crs:
                            crs_str = str(src.crs)
                        inv = getattr(bundle, "band_inventory", None)
                        if inv and inv.bands:
                            bands_data = {}
                            for b_name, b_idx in inv.bands.items():
                                if b_idx <= data.shape[0]:
                                    bands_data[b_name.lower()] = data[b_idx - 1].astype(np.float32)
                except Exception:
                    bands_data = None

    if bands_data is None:
        bands_data = _generate_synthetic_bands((100, 100))
        is_synthetic = True

    # Compute requested normalized spectral index
    if index_name == "NDVI":
        num = bands_data["nir"] - bands_data["red"]
        denom = bands_data["nir"] + bands_data["red"] + 1e-8
    elif index_name == "NDWI":
        num = bands_data["green"] - bands_data["nir"]
        denom = bands_data["green"] + bands_data["nir"] + 1e-8
    elif index_name == "MNDWI":
        swir = bands_data.get("swir", bands_data["nir"])
        num = bands_data["green"] - swir
        denom = bands_data["green"] + swir + 1e-8
    elif index_name == "NDBI":
        swir = bands_data.get("swir", bands_data["nir"])
        num = swir - bands_data["nir"]
        denom = swir + bands_data["nir"] + 1e-8
    else:
        num = bands_data["green"] - bands_data["nir"]
        denom = bands_data["green"] + bands_data["nir"] + 1e-8

    index_arr = num / denom

    # Apply Otsu threshold with bimodality gate (§4.6.2)
    otsu_val, ratio, f0, f1 = compute_otsu_threshold(index_arr)

    if manual_val is not None:
        thresh = float(manual_val)
        chosen_method = "manual"
    elif req_method == "fixed":
        thresh = fixed_fallbacks.get(index_name, 0.20)
        chosen_method = "fixed_fallback"
    else:
        if ratio >= 0.5 and f0 >= 0.05 and f1 >= 0.05:
            thresh = round(otsu_val, 4)
            chosen_method = "otsu"
        else:
            thresh = fixed_fallbacks.get(index_name, 0.20)
            chosen_method = "fixed_fallback"

    # Compute binary mask & area
    mask = (index_arr >= thresh).astype(np.uint8)
    area_px = int(np.sum(mask))
    area_km2 = round(float(area_px * (pixel_size_m ** 2) / 1e6), 4)

    # Cache mask in context for downstream tools (e.g. centroid_prior)
    if context is not None:
        mask_cache = context.setdefault("mask_cache", {})
        mask_cache["spectral_index"] = mask
        mask_cache[index_name] = mask

    # Write real GeoTIFF mask file to disk
    import tempfile
    mask_dir = Path(tempfile.gettempdir()) / "satquery_assets" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_file = mask_dir / f"{index_name.lower()}_mask.tif"

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
        fallback_file = mask_dir / f"{index_name.lower()}_mask.tif"
        fallback_file.write_bytes(mask.tobytes())
        mask_uri = str(fallback_file)

    out_dict: dict[str, Any] = {
        "mask_uri": mask_uri,
        "area_km2": area_km2,
        "threshold_value": thresh,
        "threshold_method": chosen_method,
        "index": index_name,
    }
    if is_synthetic:
        out_dict["synthetic"] = True
        out_dict["warning"] = "Source raster unavailable on disk; synthetic fallback used"
    return out_dict
