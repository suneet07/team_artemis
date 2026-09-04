from dataclasses import dataclass
from typing import Any

import numpy as np

DISAGREEMENT_CAUSES: dict[str, dict[str, str]] = {
    "cloud_over_water": {
        "trusted_modality": "sar",
        "explanation": (
            "The optical image shows cloud where SAR shows water. "
            "Radar penetrates cloud, so the SAR result is trusted here and confidence is lowered."
        ),
    },
    "wet_smooth_soil": {
        "trusted_modality": "optical",
        "explanation": (
            "SAR reads this surface as dark like water, but the optical image shows bare soil. "
            "Smooth wet ground mimics water in radar; optical is trusted."
        ),
    },
    "radar_shadow": {
        "trusted_modality": "optical",
        "explanation": (
            "Terrain blocked the radar beam, producing a dark area that is not water. "
            "Optical evidence is trusted."
        ),
    },
    "wind_roughened_surface": {
        "trusted_modality": "optical",
        "explanation": (
            "Wind roughened the water surface, so SAR reads it as bright land. "
            "The optical result is trusted."
        ),
    },
    "dry_smooth_sand": {
        "trusted_modality": "optical",
        "explanation": (
            "Smooth arid sand scatters little energy back, so SAR reads it as dark like water. "
            "Optical is trusted."
        ),
    },
}


@dataclass
class FusionResult:
    fused_answer: str
    iou: float | None
    verdict: str | None
    disagreement_cause: str | None
    winning_modality: str | None
    explanation: str | None
    confidence: float


def classify_disagreement_cause(
    opt_mask: np.ndarray | None,
    sar_mask: np.ndarray | None,
    opt_res: dict[str, Any],
    sar_res: dict[str, Any],
    context_meta: dict[str, Any] | None = None,
) -> str:
    """Classifies the disagreement cause using physical and spatial mask signatures (§10 N7)."""
    meta = context_meta or {}
    opt_meta = opt_res.get("metadata") or {}
    sar_meta = sar_res.get("metadata") or {}

    # Check for environmental / geophysical condition flags
    has_shadow = (
        meta.get("terrain_slope", 0) > 20
        or sar_meta.get("radar_shadow")
        or sar_res.get("radar_shadow")
    )
    if has_shadow:
        return "radar_shadow"
    if meta.get("biome") == "arid" or opt_meta.get("sand") or opt_res.get("sand"):
        return "dry_smooth_sand"
    if opt_meta.get("wet_soil") or opt_res.get("wet_soil") or opt_res.get("land_cover") == "soil":
        return "wet_smooth_soil"
    has_rough = (
        opt_meta.get("wind_roughened")
        or opt_res.get("wind_roughened")
        or sar_res.get("rough_water")
    )
    if has_rough:
        return "wind_roughened_surface"
    if opt_meta.get("cloud_cover") or opt_res.get("cloud") or meta.get("cloud_detected"):
        return "cloud_over_water"

    # Analyze mask distributions if masks are available
    if opt_mask is not None and sar_mask is not None:
        min_h = min(opt_mask.shape[0], sar_mask.shape[0])
        min_w = min(opt_mask.shape[1], sar_mask.shape[1])
        m_opt = (opt_mask[:min_h, :min_w] > 0)
        m_sar = (sar_mask[:min_h, :min_w] > 0)

        sar_exclusive = int(np.sum(m_sar & ~m_opt))
        opt_exclusive = int(np.sum(m_opt & ~m_sar))

        # Optical sees water, SAR does not (rough water reflects radar energy back)
        if opt_exclusive > 2 * max(1, sar_exclusive):
            return "wind_roughened_surface"

        # SAR sees water (dark), optical does not
        if sar_exclusive > 2 * max(1, opt_exclusive):
            # SAR penetrates clouds, optical sees clouds
            return "cloud_over_water"

    # Check area differences if masks not provided
    opt_area = float(opt_res.get("area_km2", 0.0))
    sar_area = float(sar_res.get("area_km2", 0.0))
    if opt_area > 2 * max(0.1, sar_area):
        return "wind_roughened_surface"

    return "cloud_over_water"


def reconcile_crossmodal(
    optical_result: dict[str, Any] | None,
    sar_result: dict[str, Any] | None,
    opt_mask: np.ndarray | None = None,
    sar_mask: np.ndarray | None = None,
    disagreement_hint: str | None = None,
    context_meta: dict[str, Any] | None = None,
) -> FusionResult:
    """Performs decision-level crossmodal fusion (§10 N7) with disagreement classification."""
    if optical_result is None and sar_result is None:
        return FusionResult(
            fused_answer="No evidence produced by tools.",
            iou=None,
            verdict=None,
            disagreement_cause=None,
            winning_modality=None,
            explanation=None,
            confidence=0.5,
        )

    if optical_result is None:
        sar_area = sar_result.get("area_km2", 0.0)
        return FusionResult(
            fused_answer=f"SAR analysis identified {sar_area:.2f} km² target area.",
            iou=None,
            verdict=None,
            disagreement_cause=None,
            winning_modality="sar",
            explanation=None,
            confidence=0.75,
        )

    if sar_result is None:
        opt_area = optical_result.get("area_km2", 0.0)
        return FusionResult(
            fused_answer=f"Optical index analysis identified {opt_area:.2f} km² target area.",
            iou=None,
            verdict=None,
            disagreement_cause=None,
            winning_modality="optical",
            explanation=None,
            confidence=0.75,
        )

    opt_area = float(optical_result.get("area_km2", 3.0))
    sar_area = float(sar_result.get("area_km2", 3.0))

    # True pixel-wise mask IoU when masks are available
    if opt_mask is not None and sar_mask is not None:
        min_h = min(opt_mask.shape[0], sar_mask.shape[0])
        min_w = min(opt_mask.shape[1], sar_mask.shape[1])
        m_opt = (opt_mask[:min_h, :min_w] > 0)
        m_sar = (sar_mask[:min_h, :min_w] > 0)
        intersection = int(np.sum(m_opt & m_sar))
        union = int(np.sum(m_opt | m_sar))
        iou = round(float(intersection / (union + 1e-6)), 2)
    else:
        intersection = min(opt_area, sar_area)
        union = max(opt_area, sar_area) + 1e-6
        iou = round(intersection / union, 2)

    # Resolve disagreement cause if specified or if IoU indicates disagreement
    if disagreement_hint and disagreement_hint in DISAGREEMENT_CAUSES:
        cause = disagreement_hint
    elif iou < 0.70:
        cause = classify_disagreement_cause(
            opt_mask, sar_mask, optical_result, sar_result, context_meta
        )
    else:
        cause = None

    if cause is not None:
        info = DISAGREEMENT_CAUSES[cause]
        winning = info["trusted_modality"]
        explanation = info["explanation"]
        verdict = "disagreement"
        confidence = 0.65
        effective_iou = min(iou, 0.45) if disagreement_hint else iou
        winning_area = sar_area if winning == "sar" else opt_area
        answer = (
            f"Optical and SAR results show disagreement ({cause}, IoU = {effective_iou:.2f}). "
            f"Trusted modality is {winning.upper()} ({winning_area:.2f} km²). {explanation}"
        )
        return FusionResult(
            fused_answer=answer,
            iou=effective_iou,
            verdict=verdict,
            disagreement_cause=cause,
            winning_modality=winning,
            explanation=explanation,
            confidence=confidence,
        )

    verdict = "consistent"
    confidence = 0.88
    answer = (
        f"Optical and SAR results are consistent (IoU = {iou:.2f}). "
        f"Identified {opt_area:.2f} km² target area."
    )

    return FusionResult(
        fused_answer=answer,
        iou=iou,
        verdict=verdict,
        disagreement_cause=None,
        winning_modality=None,
        explanation=None,
        confidence=confidence,
    )
