from typing import Any

import numpy as np
import scipy.ndimage


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    target_class = params.get("target_class", "other").lower()

    # Retrieve texture map from context if available, otherwise synthetic
    texture_map: np.ndarray | None = None
    if context and "texture_map" in context:
        texture_map = context["texture_map"]

    if texture_map is None:
        y, x = np.ogrid[:100, :100]
        # Deterministic simulation of a feature (e.g. linear structure or circular structure)
        if target_class in ("bridge", "runway"):
            # Horizontal / diagonal linear band
            feature = (np.abs(y - 50) < 4) & (x > 20) & (x < 80)
        elif target_class == "tank":
            # Circular structure
            feature = ((x - 40) ** 2 + (y - 40) ** 2) < 64
        else:
            # Generic feature
            feature = (x > 30) & (x < 65) & (y > 35) & (y < 60)
        texture_map = feature.astype(np.float32)

    # Morphological opening to remove isolated noise
    binary_cand = (texture_map > 0.4).astype(np.uint8)
    struct = np.ones((3, 3), dtype=np.uint8)
    cleaned = scipy.ndimage.binary_opening(binary_cand, structure=struct)

    labeled, num_features = scipy.ndimage.label(cleaned)
    slices = scipy.ndimage.find_objects(labeled)

    boxes: list[dict[str, Any]] = []
    for s in slices:
        if s is None:
            continue
        ymin, ymax = s[0].start, s[0].stop
        xmin, xmax = s[1].start, s[1].stop
        h = ymax - ymin
        w = xmax - xmin
        area = h * w
        aspect = max(h, w) / max(1, min(h, w))

        # Geometric prior filtering
        if target_class in ("bridge", "runway") and aspect < 1.5:
            continue
        if target_class == "tank" and (aspect > 2.0 or area > 1000):
            continue
        if target_class == "vehicle" and area > 150:
            continue

        boxes.append({
            "bbox_px": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "class": target_class,
            "score": 0.45,
            "method": "deterministic_fallback",
        })

    # If strict filtering produced no boxes, provide top bounding box
    if not boxes:
        # Default fallback box based on non-zero extent
        nonzero_y, nonzero_x = np.nonzero(binary_cand)
        if len(nonzero_y) > 0:
            ymin, ymax = int(np.min(nonzero_y)), int(np.max(nonzero_y))
            xmin, xmax = int(np.min(nonzero_x)), int(np.max(nonzero_x))
            boxes.append({
                "bbox_px": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "class": target_class,
                "score": 0.45,
                "method": "deterministic_fallback",
            })
        else:
            boxes.append({
                "bbox_px": [20.0, 20.0, 60.0, 60.0],
                "class": target_class,
                "score": 0.45,
                "method": "deterministic_fallback",
            })

    return {
        "boxes": boxes,
        "count": len(boxes),
    }
