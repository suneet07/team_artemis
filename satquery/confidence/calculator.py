import os
from typing import Any

from satquery.agent.tiling_policy import TilePlan
from satquery.confidence.calibrator import ConfidenceCalibrator

_CALIBRATOR = ConfidenceCalibrator()


def calculate_confidence(
    tool_results: dict[str, Any],
    agreement: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    tile_plan: TilePlan | None = None,
    used_fallback: bool = False,
    basis: str | None = None,
) -> tuple[float, str]:
    """Computes calibrated/heuristic confidence P(answer correct) on final answer (§10 N8).

    Returns (confidence, confidence_basis).
    """
    chosen_basis = basis or os.environ.get("SATQUERY_CONFIDENCE_BASIS", "calibrated").lower()

    if chosen_basis == "calibrated":
        return _CALIBRATOR.calibrate(
            tool_results=tool_results,
            agreement=agreement,
            warnings=warnings,
            tile_plan=tile_plan,
            used_fallback=used_fallback,
        )

    confidence_basis = "heuristic"
    base_conf = 0.85

    # If fallback proposer contributed, floor the confidence (§10 N8 Rule 4)
    if used_fallback or "object_box_fallback" in tool_results:
        return 0.45, confidence_basis

    # Cross-modal agreement effect
    if agreement is not None:
        iou = agreement.get("iou")
        if agreement.get("verdict") == "disagreement":
            base_conf *= 0.80
        elif iou is not None and iou >= 0.70:
            base_conf = min(0.95, base_conf * 1.10)

    # Threshold method effect (otsu vs fallback)
    for res in tool_results.values():
        if isinstance(res, dict):
            if res.get("threshold_method") == "fixed_fallback":
                base_conf *= 0.90

    # Warning count penalty
    warn_count = len(warnings or [])
    if warn_count > 0:
        base_conf *= max(0.60, 1.0 - (0.05 * warn_count))

    # Tile coverage fraction effect
    if tile_plan is not None and tile_plan.is_tiled:
        # If sampled subset of tiles, scale confidence with coverage
        base_conf *= 0.70 + 0.30 * tile_plan.coverage_frac

    # Clamp strictly in [0.10, 0.99]
    final_conf = max(0.10, min(0.99, round(base_conf, 2)))
    return final_conf, confidence_basis

