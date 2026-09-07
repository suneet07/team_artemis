"""P2 — SAR normalisation, our half of the chain (master plan section 4.2).

:class:`~satquery.sar.normalisation.SARNormaliser` runs the same steps against a
file, through SNAP. This module is the same arithmetic over arrays already in
memory, which the file path cannot serve: benchmark chips that arrive already
calibrated, unit tests, the training-time polarisation dropout, and the offline
venue demo on a machine with no SNAP installed. Both paths share
``preprocessing.yaml``, so they cannot drift apart.

The calibrated half (orbit file, sigma-nought calibration, multilooking, terrain
correction) belongs to ESA SNAP and runs as an external process — see
``satquery.sar.snap``. Everything from dB conversion onwards is ours, and this
module is it.

The rule that governs the whole file: **the chain must be byte-identical at
training and inference.** Every constant lives in ``configs/preprocessing.yaml``,
every step that ran or was skipped is recorded in :class:`SarChainResult`, and
that record is what the trace carries. If BigEarthNet's S1 were processed one way
and RISAT another, the domain gap would be manufactured by hand.
"""

from dataclasses import dataclass, field

import numpy as np

from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.ingest.radiometry import percentile_stretch
from satquery.sar.speckle import refined_lee
from satquery.texture import glcm_entropy

__all__ = [
    "DUAL_POL_PAIRS",
    "SarChainResult",
    "apply_pol_dropout",
    "build_model_input_stack",
    "sanity_warnings",
    "to_db",
]

# Ordered so the co-polarised channel is always first in the stack.
DUAL_POL_PAIRS = (("VV", "VH"), ("HH", "HV"))

# Below this linear power a pixel is numerically zero rather than physically
# dark; dB of it is -inf. -50 dB is far below calm water (-25 to -20 dB), so the
# floor cannot swallow a real surface class.
_DB_FLOOR = -50.0


def to_db(sigma0: np.ndarray | np.ma.MaskedArray, floor_db: float = _DB_FLOOR) -> np.ndarray:
    """``10 * log10(sigma0)`` with a physical floor (section 4.2 step 6).

    Non-positive samples are clamped to ``floor_db`` rather than propagating
    ``-inf``: an -inf reaching a percentile stretch collapses the whole scene.
    Masked and non-finite input stays NaN so nodata never becomes backscatter.
    """
    data = np.asarray(np.ma.getdata(sigma0), dtype=np.float64)
    valid = np.isfinite(data)
    if isinstance(sigma0, np.ma.MaskedArray):
        valid &= ~np.ma.getmaskarray(sigma0)
    out = np.full(data.shape, np.nan, dtype=np.float64)
    positive = valid & (data > 0)
    out[positive] = 10.0 * np.log10(data[positive])
    out[valid & ~positive] = floor_db
    return np.maximum(out, floor_db, where=np.isfinite(out), out=out).astype(np.float32)


@dataclass
class SarChainResult:
    """What P2 produced and, just as importantly, what it did not do."""

    stack: np.ndarray  # (3, H, W) float32 in [0, 1]
    stack_kind: str  # "dual_pol" | "single_pol"
    channels: tuple[str, str, str]
    polarisations: tuple[str, ...]
    sar_band: str | None
    stretch_bounds: tuple[tuple[float, float], ...]
    steps_applied: tuple[str, ...]
    steps_skipped: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def as_provenance(self) -> dict:
        """The block a trace step records for this scene."""
        return {
            "stack_kind": self.stack_kind,
            "channels": list(self.channels),
            "polarisations": list(self.polarisations),
            "sar_band": self.sar_band,
            "stretch_bounds": [list(b) for b in self.stretch_bounds],
            "steps_applied": list(self.steps_applied),
            "steps_skipped": list(self.steps_skipped),
        }


def _stretch_all(channels: list[np.ndarray]) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    stretched = []
    bounds = []
    for channel in channels:
        values, channel_bounds = percentile_stretch(channel)
        stretched.append(values)
        bounds.append(channel_bounds)
    return np.stack(stretched).astype(np.float32), tuple(bounds)


def _normalise_pol(name: str) -> str:
    return name.strip().upper()


def build_model_input_stack(
    bands_db: dict[str, np.ndarray],
    sar_band: str | None = None,
    config: PreprocessingConfig | None = None,
    looks: float = 1.0,
    force_single_pol: bool = False,
) -> SarChainResult:
    """Build the 3-channel model input from dB sigma-nought bands (section 4.2).

    ``bands_db`` maps a polarisation ("VV", "VH", "HH", "HV") to its dB array.

    * dual-pol   -> ``[pol1_dB, pol2_dB, ratio_dB]`` where the ratio in dB is the
      difference of the two dB channels.
    * single-pol -> ``[sigma0_dB, refined-Lee sigma0_dB, GLCM-entropy texture]``.

    The single-pol stack is a first-class citizen, not an error path: RISAT-1 and
    EOS-04 FRS modes ship single-pol and RISAT-2B is X-band spotlight. We never
    tile one band three times to fake RGB — the filtered and texture channels
    carry information a repeat does not.
    """
    cfg = config if config is not None else preprocessing_config()
    if not bands_db:
        raise ValueError("no SAR bands supplied")
    bands = {_normalise_pol(k): np.asarray(v) for k, v in bands_db.items()}
    warnings: list[str] = []

    pair = None
    if not force_single_pol:
        for first, second in DUAL_POL_PAIRS:
            if first in bands and second in bands:
                pair = (first, second)
                break

    if pair is not None:
        first, second = pair
        ratio = bands[first] - bands[second]  # a dB difference *is* the ratio in dB
        stack, bounds = _stretch_all([bands[first], bands[second], ratio])
        result = SarChainResult(
            stack=stack,
            stack_kind="dual_pol",
            channels=(f"{first}_dB", f"{second}_dB", f"{first}/{second}_ratio_dB"),
            polarisations=pair,
            sar_band=sar_band,
            stretch_bounds=bounds,
            steps_applied=("db_conversion", "percentile_stretch", "model_input_stack"),
            warnings=warnings,
        )
    else:
        pol = next(iter(bands)) if force_single_pol is False else _preferred_single(bands)
        channel = bands[pol]
        # Refined Lee models multiplicative speckle, so it runs on linear power.
        linear = np.power(10.0, np.asarray(channel, dtype=np.float64) / 10.0)
        filtered_db = to_db(refined_lee(linear, looks=looks))
        entropy = glcm_entropy(
            channel, window=cfg.texture.window_px, levels=cfg.texture.glcm_levels
        )
        stack, bounds = _stretch_all([channel, filtered_db, entropy])
        if len(bands) > 1 and not force_single_pol:
            warnings.append(
                f"polarisations {sorted(bands)} do not form a known dual-pol pair; "
                f"built the single-pol stack on '{pol}'"
            )
        result = SarChainResult(
            stack=stack,
            stack_kind="single_pol",
            channels=(f"{pol}_dB", f"{pol}_refined_lee_dB", "glcm_entropy"),
            polarisations=(pol,),
            sar_band=sar_band,
            stretch_bounds=bounds,
            steps_applied=(
                "db_conversion",
                "speckle_filter_refined_lee",
                "percentile_stretch",
                "model_input_stack",
            ),
            warnings=warnings,
        )

    if sar_band is not None and cfg.sar.inherits_c_band(sar_band):
        result.warnings.append(
            f"{sar_band}-band has no tuned threshold table; inheriting the C-band "
            f"reference and lowering SAR-side confidence (section 4.2)"
        )
    return result


def _preferred_single(bands: dict[str, np.ndarray]) -> str:
    """Co-polarised channel wins when we have to drop to one (4.2)."""
    for pol in ("VV", "HH", "VH", "HV"):
        if pol in bands:
            return pol
    return next(iter(bands))


def apply_pol_dropout(
    bands_db: dict[str, np.ndarray],
    rng: np.random.Generator,
    config: PreprocessingConfig | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Training-time polarisation dropout (section 4.2).

    Converts a dual-pol sample to single-pol with probability
    ``sar.pol_dropout_rate`` so the model has genuinely seen 1-channel SAR before
    the hidden set shows it one. Returns the (possibly reduced) band dict and
    whether the dropout fired.

    Takes an explicit ``Generator`` rather than touching global random state:
    a training run has to be reproducible from its seed.
    """
    cfg = config if config is not None else preprocessing_config()
    bands = {_normalise_pol(k): v for k, v in bands_db.items()}
    if len(bands) < 2 or rng.random() >= cfg.sar.pol_dropout_rate:
        return bands, False
    keep = _preferred_single(bands)
    return {keep: bands[keep]}, True


def sanity_warnings(
    db_array: np.ndarray | np.ma.MaskedArray,
    sar_band: str | None = None,
    polarisation: str | None = None,
    config: PreprocessingConfig | None = None,
) -> list[str]:
    """Warning-only sigma-nought sanity check (section 4.2).

    These asserts **warn, they never block**. A hard assert tuned on C-band would
    refuse a legitimate X-band scene from the graded hidden set — a self-inflicted
    zero. Out-of-range scenes get a trace warning and lowered SAR-side confidence,
    and processing continues.
    """
    cfg = config if config is not None else preprocessing_config()
    data = np.asarray(np.ma.getdata(db_array), dtype=np.float64)
    valid = np.isfinite(data)
    if isinstance(db_array, np.ma.MaskedArray):
        valid &= ~np.ma.getmaskarray(db_array)
    if not valid.any():
        return ["sigma-nought sanity check skipped: no valid pixels"]

    band = (sar_band or "C").upper()
    table = cfg.sar.sanity_ranges_db.get(band) or cfg.sar.sanity_ranges_db.get("C") or {}
    if not table:
        return []
    low = min(bounds[0] for bounds in table.values())
    high = max(bounds[1] for bounds in table.values())
    p2, p98 = np.percentile(data[valid], [2.0, 98.0])
    warnings: list[str] = []
    pol = f" {polarisation.upper()}" if polarisation else ""
    if p2 < low - 10.0 or p98 > high + 10.0:
        warnings.append(
            f"{band}-band{pol} sigma-nought spans {p2:.1f} to {p98:.1f} dB, outside the "
            f"{low:.0f} to {high:.0f} dB surface table; proceeding with lowered SAR "
            f"confidence (section 4.2 — warning only)"
        )
    if cfg.sar.inherits_c_band(sar_band):
        warnings.append(
            f"{band}-band ranges are untuned and inherit the C-band reference table"
        )
    return warnings
