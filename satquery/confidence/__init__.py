"""P7 — confidence and calibration (master plan section 4.7.3).

``ConfidenceCalibrator`` is the one-dimensional isotonic mapping; ``Calibrator``
is the feature-vector version section 4.7.3 actually specifies, since the claim
is a calibrated P(answer correct) over mask statistics, cross-modal agreement,
the threshold-method flag, the router path, the warning count and the answer
log-probability -- not over a single scalar.
"""

from satquery.confidence.calibration import (
    FEATURE_NAMES,
    Calibrator,
    ConfidenceFeatures,
    expected_calibration_error,
    heuristic_confidence,
    reliability_diagram,
)
from satquery.confidence.calibrator import ConfidenceCalibrator

__all__ = [
    "FEATURE_NAMES",
    "Calibrator",
    "ConfidenceCalibrator",
    "ConfidenceFeatures",
    "expected_calibration_error",
    "heuristic_confidence",
    "reliability_diagram",
]
