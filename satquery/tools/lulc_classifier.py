from pathlib import Path
from typing import Any

import numpy as np

from satquery.tools.spectral_index import _generate_synthetic_bands


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    threshold = float(params.get("threshold", 0.5))

    bands_data: dict[str, np.ndarray] | None = None

    if context and "bundle" in context:
        bundle = context["bundle"]
        if bundle and getattr(bundle, "images", None):
            first_img = bundle.images[0]
            p = Path(first_img.path)
            if p.exists() and p.is_file():
                try:
                    import rasterio

                    with rasterio.open(p) as src:
                        data = src.read()
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

    nir = bands_data["nir"]
    red = bands_data["red"]
    green = bands_data["green"]
    swir = bands_data.get("swir", nir)

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)

    veg_frac = float(np.mean(ndvi > 0.30))
    water_frac = float(np.mean(ndwi > 0.20))
    urban_frac = float(np.mean(ndbi > 0.10))
    soil_frac = float(np.mean((ndvi <= 0.30) & (ndwi <= 0.20) & (ndbi <= 0.10)))

    # Compute realistic confidences from spectral presence
    veg_conf = round(float(np.clip(0.50 + 0.45 * veg_frac, 0.05, 0.95)), 2)
    water_conf = round(float(np.clip(0.50 + 0.45 * water_frac, 0.05, 0.95)), 2)
    urban_conf = round(float(np.clip(0.30 + 0.50 * urban_frac, 0.05, 0.95)), 2)
    soil_conf = round(float(np.clip(0.20 + 0.60 * soil_frac, 0.05, 0.95)), 2)

    candidate_labels = [
        {"class": "vegetation", "confidence": veg_conf},
        {"class": "water", "confidence": water_conf},
        {"class": "urban", "confidence": urban_conf},
        {"class": "bare_soil", "confidence": soil_conf},
    ]

    candidate_labels.sort(key=lambda x: x["confidence"], reverse=True)
    selected = [lbl for lbl in candidate_labels if lbl["confidence"] >= threshold]
    max_conf = max((lbl["confidence"] for lbl in selected), default=0.5)

    return {
        "labels": selected,
        "confidence": max_conf,
    }
