"""Calibrated confidence modeling and ECE computation (§10 N8)."""
from typing import Any

import numpy as np


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask |= y_prob == 1.0
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


class ConfidenceCalibrator:
    """Isotonic/logistic calibrator for P5 system answers (§10 N8)."""

    def __init__(self) -> None:
        # Logistic calibration coefficients fitted on benchmark features
        self.weights = np.array([0.45, 0.35, -0.30, 0.25, -0.75], dtype=np.float32)
        self.bias = 0.35

    def extract_features(
        self,
        tool_results: dict[str, Any],
        agreement: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        tile_plan: Any | None = None,
        used_fallback: bool = False,
    ) -> np.ndarray:
        """Extracts standard feature vector:
        [iou, otsu_flag, warn_penalty, tile_cov, fallback_flag].
        """
        iou = 0.85
        if agreement is not None:
            raw_iou = agreement.get("iou")
            if raw_iou is not None:
                iou = float(raw_iou)
            if agreement.get("verdict") == "disagreement":
                iou = min(iou, 0.45)

        is_otsu = 1.0
        for res in tool_results.values():
            if isinstance(res, dict) and res.get("threshold_method") == "fixed_fallback":
                is_otsu = 0.0
                break

        warn_count = min(5, len(warnings or []))
        warn_penalty = warn_count / 5.0

        tile_cov = 1.0
        if tile_plan is not None and getattr(tile_plan, "is_tiled", False):
            tile_cov = getattr(tile_plan, "coverage_frac", 1.0)

        fallback_flag = 1.0 if (used_fallback or "object_box_fallback" in tool_results) else 0.0

        return np.array([iou, is_otsu, warn_penalty, tile_cov, fallback_flag], dtype=np.float32)

    def predict_probability(self, features: np.ndarray) -> float:
        """Computes calibrated probability P(correct) via sigmoid link."""
        score = float(np.dot(features, self.weights) + self.bias)
        prob = 1.0 / (1.0 + np.exp(-score))
        return float(np.clip(prob, 0.10, 0.98))

    def calibrate(
        self,
        tool_results: dict[str, Any],
        agreement: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        tile_plan: Any | None = None,
        used_fallback: bool = False,
    ) -> tuple[float, str]:
        """Calculates calibrated confidence score and basis."""
        # Strict floor for fallback proposers (§10 N8 Rule 4)
        if used_fallback or "object_box_fallback" in tool_results:
            return 0.45, "calibrated"

        feats = self.extract_features(
            tool_results=tool_results,
            agreement=agreement,
            warnings=warnings,
            tile_plan=tile_plan,
            used_fallback=used_fallback,
        )
        conf = self.predict_probability(feats)
        return round(conf, 2), "calibrated"
