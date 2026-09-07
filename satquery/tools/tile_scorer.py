from typing import Any

import numpy as np


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    top_k = int(params.get("top_k", 4))

    tiles_data: list[dict[str, Any]] = []
    bundle = context.get("bundle") if context else None
    mask_cache = context.get("mask_cache", {}) if context else {}

    # Check if target mask is present (e.g. water or change mask)
    target_mask: np.ndarray | None = None
    for m_key in ("spectral_index", "NDWI", "sar_backscatter", "change_map"):
        if m_key in mask_cache and isinstance(mask_cache[m_key], np.ndarray):
            target_mask = mask_cache[m_key]
            break

    if bundle and getattr(bundle, "tiles", None) and getattr(bundle.tiles, "tiles", None):
        all_tiles = bundle.tiles.tiles
        for t in all_tiles:
            t_id = getattr(t, "tile_id", str(t))
            nodata_frac = float(getattr(t, "nodata_frac", 0.0))
            valid_frac = max(0.0, 1.0 - nodata_frac)

            # Mask overlap or variance score
            mask_score = 0.5
            if target_mask is not None:
                # If tile has bounds or index
                bbox = getattr(t, "bbox_px", None)
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox]
                    sub = target_mask[y0:y1, x0:x1]
                    if sub.size > 0:
                        mask_score = float(np.mean(sub > 0))
            
            # Combine valid fraction and feature presence
            combined_score = round(float(0.40 * valid_frac + 0.60 * mask_score), 4)
            tiles_data.append({
                "tile_id": t_id,
                "score": combined_score,
                "cloud_free_score": round(valid_frac, 2),
            })
    else:
        num_fallback_tiles = max(top_k, 8)
        if target_mask is not None and target_mask.size > 0:
            # Data-driven grid partitioning over available mask
            h, w = target_mask.shape[:2]
            n_rows = 2
            n_cols = max(1, num_fallback_tiles // n_rows)
            tile_h = max(1, h // n_rows)
            tile_w = max(1, w // n_cols)
            for i in range(num_fallback_tiles):
                r = (i // n_cols) % n_rows
                c = i % n_cols
                sub = target_mask[
                    r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w
                ]
                mask_val = float(np.mean(sub > 0)) if sub.size > 0 else 0.5
                score = round(float(0.40 + 0.60 * mask_val), 4)
                tiles_data.append({
                    "tile_id": f"tile_{i}",
                    "score": score,
                    "cloud_free_score": 1.0,
                })
        else:
            # Uniform score when neither discrete tile list nor mask are available
            for i in range(num_fallback_tiles):
                tiles_data.append({
                    "tile_id": f"tile_{i}",
                    "score": 0.80,
                    "cloud_free_score": 1.0,
                })

    tiles_data.sort(key=lambda x: x["score"], reverse=True)
    selected = tiles_data[:top_k]

    return {
        "scores": selected,
    }
