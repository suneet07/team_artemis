from typing import Any

import numpy as np
import scipy.ndimage


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    min_area_px = int(params.get("min_area_px", 10))

    # Retrieve mask from context if available
    mask: np.ndarray | None = None
    if context and "mask_cache" in context:
        mask_cache = context["mask_cache"]
        # Look for spectral_index or any available mask
        for key in ("spectral_index", "NDWI", "NDVI", "sar_backscatter", "change_map"):
            if key in mask_cache:
                mask = mask_cache[key]
                break

    # Fallback synthetic mask (centered blob) if none in context
    is_synthetic = False
    if mask is None:
        is_synthetic = True
        y, x = np.ogrid[:100, :100]
        mask = (((x - 45) ** 2 + (y - 52) ** 2) < 400).astype(np.uint8)

    height, width = mask.shape
    labeled_mask, num_features = scipy.ndimage.label(mask > 0)

    if num_features == 0:
        out: dict[str, Any] = {
            "centroid": {"x": 0.5, "y": 0.5},
            "area_px": 0,
        }
        if is_synthetic:
            out["synthetic"] = True
        return out

    # Count pixels per component (index 0 is background)
    counts = np.bincount(labeled_mask.ravel())
    counts[0] = 0

    valid_labels = [lbl for lbl, count in enumerate(counts) if count >= min_area_px]
    if not valid_labels:
        # Fallback to the largest component even if smaller than min_area
        largest_label = int(np.argmax(counts))
        largest_area = int(counts[largest_label])
    else:
        # Pick component with maximum area among valid
        largest_label = max(valid_labels, key=lambda lbl: counts[lbl])
        largest_area = int(counts[largest_label])

    if largest_area == 0:
        out = {
            "centroid": {"x": 0.5, "y": 0.5},
            "area_px": 0,
        }
        if is_synthetic:
            out["synthetic"] = True
        return out

    com = scipy.ndimage.center_of_mass(mask, labeled_mask, largest_label)
    cy, cx = com  # row is y, col is x

    norm_x = round(float(cx / max(1, width)), 4)
    norm_y = round(float(cy / max(1, height)), 4)

    out_dict: dict[str, Any] = {
        "centroid": {"x": norm_x, "y": norm_y},
        "area_px": largest_area,
    }
    if is_synthetic:
        out_dict["synthetic"] = True
        out_dict["warning"] = "No upstream mask found; synthetic blob fallback used"
    return out_dict
