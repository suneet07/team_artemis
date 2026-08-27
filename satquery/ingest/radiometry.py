import numpy as np

from satquery.config import preprocessing_config


def percentile_stretch(
    arr: np.ndarray,
    nodata: float | None = None,
    low_pct: float | None = None,
    high_pct: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Robust stretch for model input (master plan section 4.1.3).

    Returns the stretched array *and* its bounds. The caller must record the
    bounds in provenance — reproducibility depends on it.
    """
    cfg = preprocessing_config().radiometry
    low_pct = cfg.low_percentile if low_pct is None else low_pct
    high_pct = cfg.high_percentile if high_pct is None else high_pct
    values = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    if not valid.any():
        raise ValueError("percentile stretch has no valid pixels")
    lo, hi = np.percentile(values[valid], [low_pct, high_pct])
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return stretched, (lo, hi)


def nodata_fraction(overview: np.ma.MaskedArray) -> float:
    total = overview.size
    if total == 0:
        return 0.0
    invalid = total - int(np.ma.count(overview))
    return max(0.0, min(1.0, invalid / total))
