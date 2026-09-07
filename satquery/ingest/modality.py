import re

import numpy as np

SENSOR_TAG_KEYS = {"SENSOR", "SATELLITE", "PLATFORM", "SPACECRAFT", "INSTRUMENT", "MISSION"}
SAR_SENSOR_TOKENS = (
    "SENTINEL-1",
    "SENTINEL1",
    "S1",
    "RISAT",
    "EOS-04",
    "CAPELLA",
    "UMBRA",
    "TERRASAR",
    "TANDEM-X",
    "TSX",
    "ALOS-2",
    "PALSAR",
    "ICEYE",
    "SAR",
)
OPTICAL_SENSOR_TOKENS = (
    "SENTINEL-2",
    "SENTINEL2",
    "S2",
    "CARTOSAT",
    "LANDSAT",
    "PLEIADES",
    "WORLDVIEW",
    "QUICKBIRD",
    "NAIP",
    "SPOT-6",
    "SPOT-7",
    "RAPIDEYE",
    "RESOURCESAT",
)


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean())
    if mean == 0:
        return 0.0
    return float(values.std() / mean)


def _skewness(values: np.ndarray) -> float:
    centred = values - values.mean()
    m2 = float((centred**2).mean())
    m3 = float((centred**3).mean())
    if m2 == 0:
        return 0.0
    return m3 / m2**1.5


def _positive_values(band: np.ma.MaskedArray) -> np.ndarray:
    values = np.asarray(band.compressed(), dtype=np.float64)
    return values[np.isfinite(values) & (values > 0)]


def scoped_sensor_text(meta) -> str:
    parts = [str(value) for key, value in meta.tags.items() if key.upper() in SENSOR_TAG_KEYS]
    if meta.sensor_hint:
        parts.append(meta.sensor_hint)
    return " ".join(parts).upper()


def contains_token(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def _looks_like_sigma0_dB(band: np.ma.MaskedArray, mcfg) -> tuple[bool, str]:
    values = np.asarray(band.compressed(), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 16:
        return False, ""
    p_low, p_high = np.percentile(values, mcfg.db_sample_percentiles)
    lo, hi = mcfg.db_plausible_range
    negative_fraction = float((values < 0).mean())
    in_range = lo <= p_low <= hi and lo <= p_high <= hi
    reaches = float(values.min()) <= mcfg.db_must_reach_below
    if in_range and negative_fraction >= mcfg.db_min_negative_fraction and reaches:
        detail = (
            f"sigma-nought dB distribution detected "
            f"(p{mcfg.db_sample_percentiles[0]:g}={p_low:.1f} dB, "
            f"p{mcfg.db_sample_percentiles[1]:g}={p_high:.1f} dB, "
            f"negatives={negative_fraction:.0%})"
        )
        return True, detail
    return False, ""


def detect_modality(meta, overview: np.ma.MaskedArray, mcfg) -> tuple[str, list[str], str]:
    notes: list[str] = []

    text = scoped_sensor_text(meta)
    if text:
        for token in SAR_SENSOR_TOKENS:
            if contains_token(text, token):
                notes.append(f"SAR sensor token '{token}' found in sensor metadata tags")
                return "sar", notes, "sensor_tag"
        for token in OPTICAL_SENSOR_TOKENS:
            if contains_token(text, token):
                notes.append(f"optical sensor token '{token}' found in sensor metadata tags")
                return "optical", notes, "sensor_tag"

    for index in range(meta.count):
        is_dB, detail = _looks_like_sigma0_dB(overview[index], mcfg)
        if is_dB:
            notes.append(detail)
            return "sar", notes, "intensity_signature"

    cvs: list[float] = []
    skews: list[float] = []
    for index in range(meta.count):
        positive = _positive_values(overview[index])
        if positive.size >= 16:
            cvs.append(_cv(positive))
            skews.append(_skewness(positive))

    if cvs:
        cv = float(np.median(cvs))
        skew = float(np.median(skews))
        if cv >= mcfg.speckle_cv_min and skew >= mcfg.speckle_skew_min:
            notes.append(
                f"heavy-tailed speckle signature (cv={cv:.2f}, skew={skew:.2f}); assumed SAR"
            )
            return "sar", notes, "intensity_signature"
        if cv >= mcfg.ambiguous_cv_min and skew >= mcfg.ambiguous_skew_min:
            notes.append(
                f"signature between optical and SAR thresholds "
                f"(cv={cv:.2f}, skew={skew:.2f}); ask the user and record the answer"
            )
            return "unknown", notes, "intensity_signature"

    if meta.count <= 2:
        notes.append("single/two-band without speckle signature; assumed panchromatic optical")
        return "optical", notes, "intensity_signature"

    return "optical", notes, "band_count"
