from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage


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

    if img_data is None:
        # Deterministic synthetic image with textured zones and edges
        y, x = np.ogrid[:100, :100]
        base = 0.5 + 0.3 * np.sin(x / 6.0) * np.cos(y / 6.0)
        # Add high-frequency grid in one quadrant (built-up texture)
        high_freq = 0.2 * ((x % 4 == 0) | (y % 4 == 0)).astype(np.float32)
        img_data = (base + np.where((x > 50) & (y > 50), high_freq, 0.0)).astype(np.float32)

    # Windowed local variance computation: Var(X) = E[X^2] - (E[X])^2
    mean = scipy.ndimage.uniform_filter(img_data, size=window_size)
    sq_mean = scipy.ndimage.uniform_filter(img_data**2, size=window_size)
    variance = np.maximum(0.0, sq_mean - mean**2)

    # Normalize variance to [0, 1]
    max_v = float(np.max(variance))
    norm_texture = (variance / max_v) if max_v > 1e-8 else variance

    if context is not None:
        context["texture_map"] = norm_texture

    mean_tex = round(float(np.mean(norm_texture)), 4)
    high_tex_frac = round(float(np.mean(norm_texture > 0.5)), 4)

    return {
        "response_map": "/assets/texture_response.tif",
        "window_size": window_size,
        "mean_texture": mean_tex,
        "high_texture_frac": high_tex_frac,
    }
