from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage

from satquery.tools import tiling_support

# This tool honours context["current_tile"] (see tiling_support).
SUPPORTS_TILING = True


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    window_size = int(params.get("window_size", 5))

    img_data: np.ndarray | None = None

    if context and "bundle" in context:
        bundle = context["bundle"]
        if bundle and getattr(bundle, "images", None):
            p = Path(bundle.images[0].path)
            if p.exists() and p.is_file():
                try:
                    import rasterio

                    with rasterio.open(p) as src:
                        img_data = src.read(1).astype(np.float32)
                except Exception:
                    img_data = None

    is_synthetic = False
    if img_data is None:
        is_synthetic = True
        # Deterministic synthetic image with textured zones and edges
        y, x = np.ogrid[:100, :100]
        base = 0.5 + 0.3 * np.sin(x / 6.0) * np.cos(y / 6.0)
        # Add high-frequency grid in one quadrant (built-up texture)
        high_freq = 0.2 * ((x % 4 == 0) | (y % 4 == 0)).astype(np.float32)
        img_data = (base + np.where((x > 50) & (y > 50), high_freq, 0.0)).astype(np.float32)

    _scene_shape = img_data.shape[:2]
    _window = tiling_support.window_of(context, _scene_shape)

    # Windowed local variance computation: Var(X) = E[X^2] - (E[X])^2
    mean = scipy.ndimage.uniform_filter(img_data, size=window_size)
    sq_mean = scipy.ndimage.uniform_filter(img_data**2, size=window_size)
    variance = np.maximum(0.0, sq_mean - mean**2)

    # Normalize variance to [0, 1]
    max_v = float(np.max(variance))
    norm_texture = (variance / max_v) if max_v > 1e-8 else variance

    # Crop after filtering and normalisation: the sliding window would see a
    # truncated neighbourhood at a tile edge, and max_v is a scene-level scale,
    # so cropping the input would make tile seams visible in the mosaic.
    norm_texture = tiling_support.crop(norm_texture, _window)

    if context is not None:
        context["texture_map"] = norm_texture

    mean_tex = round(float(np.mean(norm_texture)), 4)
    high_tex_frac = round(float(np.mean(norm_texture > 0.5)), 4)

    if _window is not None:
        # Tiled run: mean_texture and high_texture_frac are scene-level ratios and
        # cannot be recovered by combining per-tile values, so hand the raw
        # response back and let the executor recompute them from the mosaic.
        return {
            "_mask_array": norm_texture,
            "window_size": window_size,
            **tiling_support.tile_report(_window, _scene_shape),
            **({"synthetic": True} if is_synthetic else {}),
        }

    out_dict: dict[str, Any] = {
        "response_map": "/assets/texture_response.tif",
        "window_size": window_size,
        "mean_texture": mean_tex,
        "high_texture_frac": high_tex_frac,
    }
    if is_synthetic:
        out_dict["synthetic"] = True
        out_dict["warning"] = "Source raster unavailable on disk; synthetic fallback used"
    return out_dict
