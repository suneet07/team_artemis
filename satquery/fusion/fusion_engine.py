from typing import Any

import numpy as np

from satquery.config import preprocessing_config

DISAGREEMENT_RULES = [
    {"condition": "sar_water_optical_not", "cause": "cloud_over_water", "winner": "sar"},
    {"condition": "sar_dark_optical_soil", "cause": "wet_smooth_soil", "winner": "optical"},
    {"condition": "sar_dark_terrain_slope", "cause": "radar_shadow", "winner": "optical"},
    {"condition": "sar_bright_over_water", "cause": "wind_roughened_surface", "winner": "optical"},
    {"condition": "sar_dark_arid_region", "cause": "dry_smooth_sand", "winner": "optical"},
]


class DecisionFusionEngine:
    """
    D1: Decision-level cross-modal fusion with physically-explained disagreement.
    """

    def __init__(self):
        self.rules = DISAGREEMENT_RULES

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection / union)

    def _classify_disagreement(
        self, optical_mask: np.ndarray, sar_mask: np.ndarray, target: str
    ) -> dict[str, Any]:
        """
        Classifies the physical cause of disagreement using heuristic rules.
        For a production system, this would analyze the underlying spectral and
        backscatter distributions in the regions of disagreement.
        """
        # Calculate regions where they disagree
        sar_only = np.logical_and(sar_mask, ~optical_mask)
        optical_only = np.logical_and(optical_mask, ~sar_mask)

        sar_only_ratio = sar_only.sum() / sar_mask.size
        optical_only_ratio = optical_only.sum() / optical_mask.size

        # Simplified heuristics based on the target class
        if target == "water":
            if sar_only_ratio > optical_only_ratio:
                # SAR sees water, optical doesn't -> likely cloud cover obscuring water
                return {
                    "condition": "sar_water_optical_not",
                    "cause": "cloud_over_water",
                    "winner": "sar",
                }
            else:
                # Optical sees water, SAR doesn't (bright SAR) -> wind roughened surface
                return {
                    "condition": "sar_bright_over_water",
                    "cause": "wind_roughened_surface",
                    "winner": "optical",
                }
        # The section 4.7.2 table is written for water. There is no row for a
        # built-up disagreement, and the previous code returned the *water* row
        # `wet_smooth_soil` for it -- a physical explanation that does not
        # describe what happened. Disagreement-cause accuracy is a reported
        # metric, so a confidently wrong cause costs more than no cause.
        #
        # Returning no winner is a real answer: "these disagree and we cannot
        # explain why". The plan is explicit -- **never silently pick one**.
        return {"condition": "unknown", "cause": None, "winner": None}

    def reconcile(
        self, optical_mask: np.ndarray, sar_mask: np.ndarray, target: str
    ) -> dict[str, Any]:
        """
        Reconciles independent optical and SAR decisions.
        """
        iou = self._compute_iou(optical_mask, sar_mask)

        # Threshold comes from `configs/preprocessing.yaml`, not from a literal:
        # it decides which mask ships as the answer, so it is a contract value.
        if iou >= preprocessing_config().fusion.iou_consistent:
            reconciled_mask = np.logical_or(optical_mask, sar_mask)
            return {
                "reconciled_mask": reconciled_mask,
                "iou": iou,
                "verdict": "high_agreement",
                "disagreement_cause": None,
                "winning_modality": "both",
                "confidence_penalty": 0.0,
                "confidence_multiplier": 1.0,
            }

        # If agreement is low, we classify the disagreement
        disagreement = self._classify_disagreement(optical_mask, sar_mask, target)

        cfg = preprocessing_config().fusion
        winner = disagreement["winner"]
        if winner is None:
            # No rule fired. Report the extent both modalities agree on, say so,
            # and take the penalty -- rather than nominating a modality by
            # default and presenting a guess as a resolution.
            return {
                "reconciled_mask": np.logical_and(optical_mask, sar_mask),
                "iou": iou,
                "verdict": "unexplained_disagreement",
                "disagreement_cause": None,
                "winning_modality": None,
                "confidence_penalty": 1.0 - cfg.disagreement_penalty,
                "confidence_multiplier": cfg.disagreement_penalty,
            }

        return {
            "reconciled_mask": sar_mask if winner == "sar" else optical_mask,
            "iou": iou,
            "verdict": "disagreement_resolved",
            "disagreement_cause": disagreement["cause"],
            "winning_modality": winner,
            "confidence_penalty": 1.0 - cfg.disagreement_penalty,
            "confidence_multiplier": cfg.disagreement_penalty,
        }
