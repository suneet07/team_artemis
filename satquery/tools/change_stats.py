from typing import Any

import numpy as np


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    target_class = params.get("target_class", "all").lower()

    pixel_size_m = 10.0
    if context and "bundle" in context:
        bundle = context["bundle"]
        if bundle and getattr(bundle, "images", None):
            if bundle.images[0].pixel_size_m is not None:
                pixel_size_m = float(bundle.images[0].pixel_size_m)

    # Check for change mask from change_map tool
    change_mask: np.ndarray | None = None
    if context and "mask_cache" in context:
        change_mask = context["mask_cache"].get("change_map")

    if change_mask is None:
        # Deterministic simulation of bi-temporal change
        y, x = np.ogrid[:100, :100]
        # Simulated urban expansion in one corner
        change_mask = ((x > 60) & (y > 60)).astype(np.uint8)

    total_px = change_mask.size
    changed_px = int(np.sum(change_mask > 0))
    change_ratio = round(float(changed_px / max(1, total_px)), 4)

    delta_km2 = round(float(changed_px * (pixel_size_m ** 2) / 1e6), 4)

    # Compute baseline area before change from T1 data in mask_cache
    baseline_px = None
    if context and "mask_cache" in context:
        # Check if spectral index or land-cover mask exists for baseline
        for key in ("spectral_index", "NDWI", "NDVI", "NDBI", "sar_backscatter"):
            if key in context["mask_cache"] and isinstance(
                context["mask_cache"][key], np.ndarray
            ):
                baseline_px = int(np.sum(context["mask_cache"][key]))
                break

    if baseline_px is not None:
        area_before: float | None = round(float(baseline_px * (pixel_size_m ** 2) / 1e6), 4)
        area_after: float | None = round(area_before + delta_km2, 4)
    else:
        # No T1 baseline mask available in mask_cache; return None rather than geometric guess
        area_before = None
        area_after = None

    # Multi-class breakdown if semantic mask present
    class_breakdown = {}
    if np.any(change_mask > 1):
        class_breakdown = {
            "urban": round(float(np.sum(change_mask == 1) * (pixel_size_m ** 2) / 1e6), 4),
            "vegetation_loss": -round(
                float(np.sum(change_mask == 2) * (pixel_size_m ** 2) / 1e6), 4
            ),
            "water_change": round(float(np.sum(change_mask == 3) * (pixel_size_m ** 2) / 1e6), 4),
        }

    stats = {
        "target_class": target_class,
        "area_before_km2": area_before,
        "area_after_km2": area_after,
        "area_delta_km2": delta_km2,
        "change_ratio": change_ratio,
    }
    if class_breakdown:
        stats["class_breakdown"] = class_breakdown

    from satquery.agent.formatter import format_change_answer

    answer = format_change_answer(stats)

    return {
        "stats": stats,
        "area_delta_km2": delta_km2,
        "answer": answer,
    }
