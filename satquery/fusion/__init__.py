"""P7 — decision-level fusion and disagreement (master plan section 4.7)."""

from satquery.fusion.fusion import (
    DISAGREEMENT_RULES,
    DisagreementRule,
    FusionOutcome,
    classify_disagreement,
    fuse_masks,
    iou,
    slope_mask_from_dem,
)
from satquery.fusion.fusion_engine import DecisionFusionEngine

__all__ = [
    "DISAGREEMENT_RULES",
    "DecisionFusionEngine",
    "DisagreementRule",
    "FusionOutcome",
    "classify_disagreement",
    "fuse_masks",
    "iou",
    "slope_mask_from_dem",
]
