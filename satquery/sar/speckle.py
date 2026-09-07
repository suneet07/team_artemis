"""Refined Lee speckle filter, reimplemented.

Master plan section 4.2 runs the calibrated chain through ESA SNAP as an
external process, and SNAP owns the Refined Lee step there. We still need our
own implementation for two reasons the plan names:

* the **single-pol model-input stack** carries a Refined-Lee-filtered channel
  alongside the raw one, and that stack is built on our side of the chain
  (section 4.2, "Model-input stack — ours");
* SNAP is GPL-3.0 and is invoked strictly as a separate process, never imported
  and never linked (section 4.2 and TEAM_CONTEXT rule 10), so nothing inside
  this package may call into it for a per-array operation.

Method: Lee's adaptive minimum-mean-square-error filter (Lee 1981) with the
directional sub-window refinement that gives it its "refined" name — for each
pixel the local statistics are taken from whichever of eight half-windows best
matches the pixel, so an edge is measured along the edge rather than across it.
Implemented over ``scipy.ndimage`` correlations, so it is vectorised over the
whole array rather than looped per pixel.
"""

import numpy as np
from scipy import ndimage

__all__ = ["DIRECTIONAL_KERNELS", "refined_lee"]


def _directional_kernels(size: int) -> list[np.ndarray]:
    """Eight half-window masks of an ``size`` x ``size`` neighbourhood.

    Four are axis-aligned halves (N, S, W, E) and four are diagonal halves, which
    is the standard eight-direction refinement set. Each is normalised to sum 1
    so a correlation with it is a mean.
    """
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    raw = [
        yy <= 0,
        yy >= 0,
        xx <= 0,
        xx >= 0,
        (yy - xx) <= 0,
        (yy - xx) >= 0,
        (yy + xx) <= 0,
        (yy + xx) >= 0,
    ]
    return [mask.astype(np.float64) / mask.sum() for mask in raw]


DIRECTIONAL_KERNELS = _directional_kernels(7)


def refined_lee(
    intensity: np.ndarray | np.ma.MaskedArray,
    looks: float = 1.0,
    window: int = 7,
) -> np.ndarray:
    """Refined Lee filter over a linear-power (not dB) SAR intensity array.

    ``looks`` is the equivalent number of looks; the speckle coefficient of
    variation for fully developed speckle is ``Cu = 1 / sqrt(looks)``. Speckle is
    multiplicative, so this must run on intensity — running it on dB would filter
    an additive-noise model that does not describe the data.

    Returns a float32 array of the same shape. Invalid pixels (masked or
    non-finite) are excluded from the local statistics and returned as NaN, so a
    nodata region cannot leak a fabricated backscatter value into a mask.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if looks <= 0:
        raise ValueError("looks must be positive")

    data = np.asarray(np.ma.getdata(intensity), dtype=np.float64)
    valid = np.isfinite(data)
    if isinstance(intensity, np.ma.MaskedArray):
        valid &= ~np.ma.getmaskarray(intensity)
    if not valid.any():
        return np.full(data.shape, np.nan, dtype=np.float32)

    filled = np.where(valid, data, 0.0)
    weight = valid.astype(np.float64)
    kernels = _directional_kernels(window)

    # The reference the directional means are compared against. It must be a
    # *smoothed* local estimate, not the raw centre pixel: at one look, speckle
    # has a coefficient of variation of 1, so the raw pixel is roughly as far
    # from the true local mean as the sub-window means are from each other, and
    # selecting on it picks a direction essentially at random. Measured: with
    # the raw pixel this filter preserved edges no better than a boxcar of the
    # same size, which is the one thing "refined" is supposed to buy.
    counts_3 = ndimage.uniform_filter(weight, size=3, mode="nearest")
    sums_3 = ndimage.uniform_filter(filled, size=3, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        reference = np.where(counts_3 > 0, sums_3 / counts_3, filled)

    # Per-direction masked means and variances. The chosen direction is the
    # half-window whose mean best matches the local estimate, i.e. the one that
    # does not straddle an edge.
    means = []
    variances = []
    for kernel in kernels:
        counts = ndimage.correlate(weight, kernel, mode="nearest")
        sums = ndimage.correlate(filled, kernel, mode="nearest")
        sums_sq = ndimage.correlate(filled * filled, kernel, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / counts, 0.0)
            second = np.where(counts > 0, sums_sq / counts, 0.0)
        means.append(mean)
        variances.append(np.maximum(second - mean * mean, 0.0))

    mean_stack = np.stack(means)
    var_stack = np.stack(variances)
    choice = np.argmin(np.abs(mean_stack - reference[None, ...]), axis=0)
    local_mean = np.take_along_axis(mean_stack, choice[None, ...], axis=0)[0]
    local_var = np.take_along_axis(var_stack, choice[None, ...], axis=0)[0]

    cu_sq = 1.0 / looks
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = local_var - cu_sq * local_mean * local_mean
        denominator = local_var * (1.0 + cu_sq)
        gain = np.where(denominator > 0, numerator / denominator, 0.0)
    gain = np.clip(gain, 0.0, 1.0)

    out = local_mean + gain * (filled - local_mean)
    return np.where(valid, out, np.nan).astype(np.float32)
