"""Shared deterministic texture and morphology primitives.

These are the non-learned building blocks the plan leans on in four places:

* the single-pol SAR model-input stack carries a GLCM-entropy channel instead of
  repeating one band three times (master plan section 4.2);
* ``texture_seg`` segments optical imagery that has no usable spectral bands
  (section 4.6.8);
* ``centroid_prior`` needs connected components and their centroids (4.6.2);
* ``object_box_fallback`` needs the same components plus morphological opening
  and shape filters (4.6.9) — the plan describes it as assembly over this
  module rather than new machinery.

Everything here is deterministic, CPU-only and dependency-light on purpose: the
headless evaluation path (section 4.11) has to run with no GPU and no network.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "Component",
    "components",
    "edge_density",
    "glcm_entropy",
    "largest_component_centroid",
    "local_variance",
    "morphological_open",
    "quantise",
]


def _valid_mask(arr: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    if isinstance(arr, np.ma.MaskedArray):
        return ~np.ma.getmaskarray(arr) & np.isfinite(np.ma.getdata(arr))
    return np.isfinite(arr)


def _filled(arr: np.ndarray | np.ma.MaskedArray, valid: np.ndarray) -> np.ndarray:
    data = np.asarray(np.ma.getdata(arr), dtype=np.float64)
    if valid.all():
        return data
    out = data.copy()
    fill = float(data[valid].mean()) if valid.any() else 0.0
    out[~valid] = fill
    return out


def quantise(arr: np.ndarray | np.ma.MaskedArray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Robustly quantise to ``levels`` grey levels for co-occurrence statistics.

    Returns the integer level image and the validity mask. Quantisation uses the
    2nd-98th percentile of *valid* pixels so a handful of hot pixels cannot
    collapse the whole histogram into one bin.
    """
    if levels < 2:
        raise ValueError("levels must be at least 2")
    valid = _valid_mask(arr)
    data = _filled(arr, valid)
    if not valid.any():
        return np.zeros(data.shape, dtype=np.int32), valid
    lo, hi = np.percentile(data[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-6
    scaled = (data - lo) / (hi - lo)
    levels_img = np.clip(np.floor(scaled * levels), 0, levels - 1).astype(np.int32)
    return levels_img, valid


def _uniform(arr: np.ndarray, window: int) -> np.ndarray:
    return ndimage.uniform_filter(arr, size=window, mode="nearest")


def glcm_entropy(
    arr: np.ndarray | np.ma.MaskedArray,
    window: int = 9,
    levels: int = 16,
    offset: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Windowed grey-level co-occurrence entropy, normalised to [0, 1].

    Computed symbol by symbol rather than by materialising an ``levels**2``
    channel stack, so peak memory stays at a couple of arrays regardless of the
    level count. Horizontal offset by default; SAR speckle is direction-agnostic
    enough that a single offset is the right cost/benefit here, and the caller
    can average two offsets if it ever is not.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd so the response stays grid-aligned")
    levels_img, valid = quantise(arr, levels)
    dy, dx = offset
    shifted = np.roll(np.roll(levels_img, -dy, axis=0), -dx, axis=1)
    pair = levels_img * levels + shifted
    entropy = np.zeros(levels_img.shape, dtype=np.float64)
    total = np.zeros(levels_img.shape, dtype=np.float64)
    for symbol in range(levels * levels):
        indicator = (pair == symbol).astype(np.float64)
        if not indicator.any():
            continue
        p = _uniform(indicator, window)
        total += p
        np.subtract(entropy, np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0.0), out=entropy)
    with np.errstate(invalid="ignore", divide="ignore"):
        entropy = np.where(total > 0, entropy / total, 0.0)
    max_entropy = np.log2(levels * levels)
    out = np.clip(entropy / max_entropy, 0.0, 1.0)
    out[~valid] = 0.0
    return out.astype(np.float32)


def local_variance(arr: np.ndarray | np.ma.MaskedArray, window: int = 9) -> np.ndarray:
    """Windowed variance of the valid pixels, normalised by the scene variance."""
    if window % 2 == 0:
        raise ValueError("window must be odd so the response stays grid-aligned")
    valid = _valid_mask(arr)
    data = _filled(arr, valid)
    mean = _uniform(data, window)
    mean_sq = _uniform(data * data, window)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    scene = float(var[valid].max()) if valid.any() else 0.0
    out = var / scene if scene > 0 else np.zeros_like(var)
    out[~valid] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def edge_density(arr: np.ndarray | np.ma.MaskedArray, window: int = 9) -> np.ndarray:
    """Windowed Sobel gradient magnitude, normalised to [0, 1].

    This is the optical half of ``texture_seg``: built-up land is edge-dense,
    water and bare soil are not, and neither statement needs a spectral band.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd so the response stays grid-aligned")
    valid = _valid_mask(arr)
    data = _filled(arr, valid)
    gy = ndimage.sobel(data, axis=0, mode="nearest")
    gx = ndimage.sobel(data, axis=1, mode="nearest")
    magnitude = np.hypot(gx, gy)
    density = _uniform(magnitude, window)
    peak = float(np.percentile(density[valid], 99.0)) if valid.any() else 0.0
    out = density / peak if peak > 0 else np.zeros_like(density)
    out[~valid] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def morphological_open(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary opening with a disk of ``radius`` — speckle suppression (4.6.9)."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (yy * yy + xx * xx) <= radius * radius
    disk = disk.reshape(size, size)
    return ndimage.binary_opening(mask, structure=disk)


@dataclass(frozen=True)
class Component:
    """One connected component of a binary mask, in pixel coordinates."""

    label: int
    area_px: int
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col), max exclusive
    centroid_rc: tuple[float, float]
    mean_response: float

    @property
    def height_px(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def width_px(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """Long side over short side, so it is always >= 1."""
        short = max(1, min(self.height_px, self.width_px))
        return max(self.height_px, self.width_px) / short

    @property
    def solidity(self) -> float:
        """Filled fraction of the bounding box — a cheap convexity proxy."""
        box = max(1, self.height_px * self.width_px)
        return self.area_px / box

    def normalised_centroid(self, shape: tuple[int, int]) -> tuple[float, float]:
        """Centroid as ``(x, y)`` in [0, 1] — the point prior format of 4.6.2."""
        height, width = shape
        row, col = self.centroid_rc
        return (col / max(1, width - 1), row / max(1, height - 1))


def components(
    mask: np.ndarray,
    response: np.ndarray | None = None,
    min_area_px: int = 1,
) -> list[Component]:
    """Connected components of ``mask``, largest first.

    ``response`` is the continuous map the mask was thresholded from; its mean
    inside each component is what ``object_box_fallback`` ranks proposals by.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    areas = ndimage.sum_labels(np.ones_like(labels, dtype=np.float64), labels, range(1, count + 1))
    slices = ndimage.find_objects(labels)
    centroids = ndimage.center_of_mass(mask, labels, range(1, count + 1))
    if response is not None:
        means = ndimage.mean(
            np.asarray(response, dtype=np.float64), labels, range(1, count + 1)
        )
    else:
        means = [1.0] * count
    out: list[Component] = []
    for index in range(count):
        area = int(areas[index])
        if area < min_area_px:
            continue
        rows, cols = slices[index]
        centroid = centroids[index]
        out.append(
            Component(
                label=index + 1,
                area_px=area,
                bbox=(rows.start, cols.start, rows.stop, cols.stop),
                centroid_rc=(float(centroid[0]), float(centroid[1])),
                mean_response=float(means[index]),
            )
        )
    out.sort(key=lambda c: c.area_px, reverse=True)
    return out


def largest_component_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Normalised ``(x, y)`` centroid of the largest component, or None.

    This is exactly ``centroid_prior`` (section 4.6.2): connected components ->
    largest -> centroid -> normalised point, handed to the grounding adapter
    inside the prompt as a spatial prior.
    """
    found = components(mask)
    if not found:
        return None
    return found[0].normalised_centroid(np.asarray(mask).shape)
