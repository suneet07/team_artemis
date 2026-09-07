"""P8 — evidence overlays for the web viewer (master plan section 4.8).

Overlays are PNGs aligned to the map viewer. They are the *display* artifact, and
they are explicitly not the graded one: the GeoTIFF written by
:mod:`satquery.report.masks` is what a scorer reads. Keeping the two in separate
modules is deliberate — it is how C18 stays true after someone adds a fifth
overlay style.

No matplotlib: an overlay is a colour lookup and an alpha channel, and pulling a
plotting stack into the offline demo container for that would be a poor trade.
"""

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["MASK_COLOURS", "write_overlay", "write_rgb_preview"]

#: Colour-blind-safe, and consistent between the map viewer and the PDF report.
MASK_COLOURS: dict[str, tuple[int, int, int]] = {
    "water": (0, 114, 178),
    "builtup": (213, 94, 0),
    "vegetation": (0, 158, 115),
    "change": (204, 121, 167),
    "agreement": (86, 180, 233),
    "default": (240, 228, 66),
}


def _to_uint8(plane: np.ndarray) -> np.ndarray:
    values = np.asarray(plane, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[valid], [2.0, 98.0])
    if high <= low:
        high = low + 1e-6
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~valid] = 0.0
    return (scaled * 255).astype(np.uint8)


def write_rgb_preview(
    path: Path | str, bands: list[np.ndarray], max_side: int = 1024
) -> Path:
    """Write a stretched RGB (or greyscale) preview of a scene."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    planes = [_to_uint8(band) for band in bands[:3]]
    if len(planes) == 1:
        image = Image.fromarray(planes[0], mode="L").convert("RGB")
    elif len(planes) == 2:
        image = Image.fromarray(np.stack([planes[0], planes[1], planes[0]], axis=-1), "RGB")
    else:
        image = Image.fromarray(np.stack(planes, axis=-1), "RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    image.save(path)
    return path


def write_overlay(
    path: Path | str,
    mask: np.ndarray,
    *,
    base: list[np.ndarray] | None = None,
    target: str = "default",
    alpha: float = 0.55,
    max_side: int = 1024,
) -> Path:
    """Write a mask overlay PNG, optionally composited over a scene preview.

    With no ``base`` the result is a transparent PNG the map viewer can drape
    over its own tiles with an opacity slider, which is what the frontend
    contract expects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask, dtype=bool)
    colour = MASK_COLOURS.get(target, MASK_COLOURS["default"])

    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = colour
    rgba[mask, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
    overlay = Image.fromarray(rgba, mode="RGBA")

    if base:
        planes = [_to_uint8(band) for band in base[:3]]
        while len(planes) < 3:
            planes.append(planes[0])
        scene = Image.fromarray(np.stack(planes[:3], axis=-1), "RGB").convert("RGBA")
        overlay = Image.alpha_composite(scene, overlay)

    if max(overlay.size) > max_side:
        scale = max_side / max(overlay.size)
        overlay = overlay.resize(
            (max(1, int(overlay.width * scale)), max(1, int(overlay.height * scale))),
            Image.Resampling.NEAREST,  # never interpolate a class boundary
        )
    overlay.save(path)
    return path
