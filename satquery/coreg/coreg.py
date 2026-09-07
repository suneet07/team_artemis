"""P3 — Co-registration (master plan section 4.3).

Misregistration is the #1 cause of bogus change detection: every building
outline becomes a false "change". This module implements the plan's five steps —
compare grids, reproject to a common metric grid, estimate the residual shift,
report ``rmse_px``, and either correct or refuse with an explanation.

Two things the plan is specific about and that are easy to get wrong:

* **Optical<->SAR uses mutual information, not intensity correlation.** The two
  modalities share no visual features, so phase correlation on raw intensity
  finds noise. Mutual information does not care that bright means different
  things in the two images.
* **Nearest neighbour for masks, bilinear/cubic for continuous data.** Never
  interpolate class IDs.

The ISRO eval pairs arrive pre-co-registered. We still run the check because it
is a named deliverable and because live-demo uploads will not be — but on a
pre-co-registered pair the check runs in **verify-only** mode and must not modify
the data.
"""

from dataclasses import dataclass, field

import numpy as np
from rasterio.enums import Resampling
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.coreg.arosics_backend import (
    ArosicsResult,
    arosics_available,
    correct_with_arosics,
)

__all__ = [
    "ArosicsResult",
    "CoregReport",
    "arosics_available",
    "correct_with_arosics",
    "estimate_shift_mutual_information",
    "estimate_shift_phase",
    "mutual_information",
    "resampling_for",
    "register_pair",
]


def resampling_for(is_mask: bool) -> Resampling:
    """Nearest for masks, bilinear for continuous data (section 4.3 step 2).

    Interpolating a class ID invents classes that were never in the source.
    """
    return Resampling.nearest if is_mask else Resampling.bilinear


@dataclass
class CoregReport:
    """Section 4.3 output, plus the fields the trace needs."""

    coregistered: bool
    rmse_px: float | None
    method: str
    correction_applied: bool
    common_crs: str | None
    checks_passed: list[str] = field(default_factory=list)
    shift_px: tuple[float, float] | None = None
    verify_only: bool = False
    refusal: str | None = None
    warnings: list[str] = field(default_factory=list)

    def as_trace_block(self) -> dict:
        return {
            "coregistered": self.coregistered,
            "rmse_px": self.rmse_px,
            "correction_applied": self.correction_applied,
            "method": self.method,
            "common_crs": self.common_crs,
            "checks_passed": list(self.checks_passed),
        }


def _prepare(arr: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    """Reduce to a single finite 2-D plane with zero mean and unit scale."""
    data = np.asarray(np.ma.getdata(arr), dtype=np.float64)
    if data.ndim == 3:
        data = data.mean(axis=0)
    valid = np.isfinite(data)
    if isinstance(arr, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(arr)
        if mask.ndim == 3:
            mask = mask.any(axis=0)
        valid &= ~mask
    if not valid.any():
        raise ValueError("co-registration input has no valid pixels")
    filled = np.where(valid, data, float(data[valid].mean()))
    spread = float(filled.std())
    if spread == 0:
        return np.zeros_like(filled)
    return (filled - float(filled.mean())) / spread


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Mutual information in bits between two equally-shaped arrays."""
    hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    joint = hist / max(hist.sum(), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    denominator = (px @ py)[nonzero]
    return float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / denominator)))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two zero-mean unit-scale planes."""
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator == 0:
        return 0.0
    return float((a * b).sum() / denominator)


def estimate_shift_phase(
    reference: np.ndarray, moving: np.ndarray, upsample_factor: int = 10
) -> tuple[tuple[float, float], float]:
    """Sub-pixel shift by phase cross-correlation (section 4.3 step 3).

    Correct for same-modality pairs, where the two images really do share
    intensity structure. Returns ``((dy, dx), rmse_px)`` where ``(dy, dx)`` is the
    shift that registers ``moving`` onto ``reference``.

    Both spectral normalisations are tried and the winner is decided by the
    cross-correlation the shift actually achieves. Phase normalisation is the
    textbook default and is robust to illumination differences, but it whitens
    the spectrum — on band-limited satellite imagery that amplifies the noise
    floor and it can return a confident zero for a real shift (reproduced on a
    smoothed scene during development). Scoring the candidates costs one
    resampling and removes the failure mode instead of picking a side.
    """
    ref = _prepare(reference)
    mov = _prepare(moving)
    best: tuple[float, float] = (0.0, 0.0)
    best_score = -np.inf
    for normalization in ("phase", None):
        shift, _, _ = phase_cross_correlation(
            ref, mov, upsample_factor=upsample_factor, normalization=normalization
        )
        candidate = (float(shift[0]), float(shift[1]))
        score = _ncc(ref, ndimage.shift(mov, candidate, order=1, mode="nearest"))
        if score > best_score:
            best_score = score
            best = candidate
    return best, float(np.hypot(best[0], best[1]))


def estimate_shift_mutual_information(
    reference: np.ndarray,
    moving: np.ndarray,
    bins: int = 32,
    search_px: int = 8,
    refine_steps: int = 2,
) -> tuple[tuple[float, float], float]:
    """Sub-pixel shift by maximising mutual information (section 4.3 step 3).

    This is the optical<->SAR path. A coarse integer search over
    ``+/- search_px`` is followed by ``refine_steps`` halving passes, which gets
    to sub-pixel without the cost of a dense fine grid.
    """
    ref = _prepare(reference)
    mov = _prepare(moving)
    best = (0.0, 0.0)
    step = 1.0
    centre = (0.0, 0.0)
    radius = float(search_px)
    for _ in range(refine_steps + 1):
        offsets = np.arange(-radius, radius + step / 2, step)
        best_score = -np.inf
        for dy in offsets:
            for dx in offsets:
                candidate = ndimage.shift(
                    mov, (centre[0] + dy, centre[1] + dx), order=1, mode="nearest"
                )
                score = mutual_information(ref, candidate, bins=bins)
                if score > best_score:
                    best_score = score
                    best = (centre[0] + dy, centre[1] + dx)
        centre = best
        radius = step
        step /= 4.0
    dy, dx = best
    return (dy, dx), float(np.hypot(dy, dx))


def register_pair(
    reference: np.ndarray | np.ma.MaskedArray,
    moving: np.ndarray | np.ma.MaskedArray,
    *,
    reference_modality: str = "optical",
    moving_modality: str = "optical",
    reference_crs: str | None = None,
    moving_crs: str | None = None,
    reference_pixel_size_m: float | None = None,
    moving_pixel_size_m: float | None = None,
    pregeoreferenced: bool = False,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray | None, CoregReport]:
    """Run the section 4.3 check over an in-memory pair.

    Returns ``(corrected_moving_or_None, report)``. In verify-only mode — which
    is what ``pregeoreferenced=True`` selects — the corrected array is always
    ``None``: the check reports and never touches the data.
    """
    cfg = config if config is not None else preprocessing_config()
    checks: list[str] = []
    warnings: list[str] = []

    if reference_crs and moving_crs:
        if reference_crs == moving_crs:
            checks.append("crs_match")
        else:
            warnings.append(
                f"CRS mismatch: reference {reference_crs} vs moving {moving_crs}; "
                f"reproject to a common metric grid before registration (4.3 step 2)"
            )

    if reference_pixel_size_m and moving_pixel_size_m:
        ratio = max(reference_pixel_size_m, moving_pixel_size_m) / min(
            reference_pixel_size_m, moving_pixel_size_m
        )
        if ratio <= 4.0:
            checks.append("gsd_ratio_ok")
        else:
            warnings.append(
                f"GSD ratio {ratio:.1f}x between the two inputs; the finer scene "
                f"carries detail the coarser one cannot confirm"
            )

    ref_shape = np.asarray(reference).shape[-2:]
    mov_shape = np.asarray(moving).shape[-2:]
    if ref_shape != mov_shape:
        return None, CoregReport(
            coregistered=False,
            rmse_px=None,
            method="none",
            correction_applied=False,
            common_crs=reference_crs,
            checks_passed=checks,
            refusal=(
                f"inputs cover different pixel grids ({ref_shape} vs {mov_shape}); "
                f"reproject both to a common metric grid before co-registration"
            ),
            warnings=warnings,
        )
    checks.append("extent_overlap")

    cross_modal = reference_modality != moving_modality
    if cross_modal:
        method = "mutual_information"
        shift, rmse = estimate_shift_mutual_information(
            reference, moving, bins=cfg.coreg.mutual_information_bins
        )
    else:
        method = "phase_cross_correlation"
        shift, rmse = estimate_shift_phase(
            reference, moving, upsample_factor=cfg.coreg.upsample_factor
        )

    if rmse > cfg.coreg.refuse_above_rmse_px:
        return None, CoregReport(
            coregistered=False,
            rmse_px=rmse,
            method=method,
            correction_applied=False,
            common_crs=reference_crs,
            checks_passed=checks,
            shift_px=shift,
            refusal=(
                f"residual misregistration of {rmse:.1f} px exceeds the "
                f"{cfg.coreg.refuse_above_rmse_px:.0f} px limit; change detection on this "
                f"pair would report building outlines as change. Supply a co-registered "
                f"pair or enable AROSICS correction."
            ),
            warnings=warnings,
        )

    if pregeoreferenced:
        # Verify-only: the ISRO eval pairs arrive pre-co-registered and
        # re-correcting corrected data degrades it (section 4.3 note).
        if rmse <= cfg.coreg.max_rmse_px:
            checks.append("residual_shift_ok")
        else:
            warnings.append(
                f"pair declared pre-co-registered but the residual shift measures "
                f"{rmse:.2f} px; reporting without modifying the data (verify-only mode)"
            )
        return None, CoregReport(
            coregistered=rmse <= cfg.coreg.max_rmse_px,
            rmse_px=rmse,
            method=method,
            correction_applied=False,
            common_crs=reference_crs,
            checks_passed=checks,
            shift_px=shift,
            verify_only=True,
            warnings=warnings,
        )

    if rmse <= cfg.coreg.max_rmse_px:
        checks.append("residual_shift_ok")
        return None, CoregReport(
            coregistered=True,
            rmse_px=rmse,
            method=method,
            correction_applied=False,
            common_crs=reference_crs,
            checks_passed=checks,
            shift_px=shift,
            warnings=warnings,
        )

    # Translation-only correction. When the residual after it is still above the
    # limit, the misregistration is not a pure translation and the escalation
    # path in section 4.3 step 5 is AROSICS -- see `escalate_to_arosics` and
    # `satquery.coreg.arosics_backend`.
    corrected = ndimage.shift(
        np.asarray(np.ma.getdata(moving), dtype=np.float64),
        (0,) * (np.asarray(moving).ndim - 2) + (shift[0], shift[1]),
        order=1,
        mode="nearest",
    )
    _, residual = (
        estimate_shift_mutual_information(
            reference, corrected, bins=cfg.coreg.mutual_information_bins
        )
        if cross_modal
        else estimate_shift_phase(reference, corrected, upsample_factor=cfg.coreg.upsample_factor)
    )
    checks.append("residual_shift_corrected")
    if residual > cfg.coreg.max_rmse_px:
        warnings.append(
            f"a translation-only correction left {residual:.2f} px of residual, above the "
            f"{cfg.coreg.max_rmse_px:.1f} px limit. The misregistration is not a pure "
            f"translation; escalate to AROSICS (section 4.3 step 5) with "
            f"`satquery.coreg.correct_with_arosics`, which needs the pair on disk."
        )
    return corrected, CoregReport(
        coregistered=residual <= cfg.coreg.max_rmse_px,
        rmse_px=residual,
        method=method,
        correction_applied=True,
        common_crs=reference_crs,
        checks_passed=checks,
        shift_px=shift,
        warnings=warnings
        + [f"applied a {shift[0]:.2f}, {shift[1]:.2f} px shift to the moving image"],
    )
