"""P7 — confidence and calibration (master plan section 4.7.3).

**What we actually claim:** a calibrated ``P(answer correct)`` on the final
system answer. Not per-component calibration stapled together — that claim would
be untrue, and calibration is only worth reporting if the number is honest.

* **Component signals are features, not the claim:** thresholded-mask statistics,
  cross-modal agreement IoU, the ``threshold_method`` flag, classifier scores,
  the VLM answer log-probability (a weak signal for free-form text — a feature,
  never alone), the router path, and the warning count.
* **Calibrator:** isotonic regression mapping the feature vector to P(correct),
  with a logistic fallback when the data is thin.
* **Fit data:** ~500 labelled end-to-end system outputs, collected in Phase 2 by
  running the system over the 300-query routing set plus benchmark validation
  queries. One labelled set, used twice.
* **Before the calibrator exists** (Phases 1-2), aggregation is a conservative
  min/product heuristic, labelled ``"confidence_basis": "heuristic"`` in the
  trace until the fitted calibrator replaces it with ``"calibrated"``.

The plan specifies scikit-learn. This module uses ``scipy.optimize`` for the
pool-adjacent-violators fit and implements the logistic fallback directly, so the
CPU-only headless container (section 4.11) does not carry a second large
dependency for two estimators. The estimators themselves are the ones the plan
names, and :meth:`Calibrator.to_dict` round-trips a fitted model to JSON so the
fit is reproducible and auditable rather than a pickle nobody can read.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "Calibrator",
    "ConfidenceFeatures",
    "expected_calibration_error",
    "heuristic_confidence",
    "reliability_diagram",
]

#: Fixed order. Changing it invalidates any fitted calibrator, which is why the
#: fitted model records the names it was trained on.
FEATURE_NAMES: tuple[str, ...] = (
    "tool_confidence_min",
    "tool_confidence_mean",
    "agreement_iou",
    "threshold_fallback_fraction",
    "router_is_rules",
    "warning_count",
    "answer_logprob",
    "deterministic_fallback_used",
)


@dataclass
class ConfidenceFeatures:
    """The feature vector, with defaults that mean "signal absent"."""

    tool_confidence_min: float = 0.5
    tool_confidence_mean: float = 0.5
    agreement_iou: float = 1.0
    threshold_fallback_fraction: float = 0.0
    router_is_rules: float = 1.0
    warning_count: float = 0.0
    answer_logprob: float = 0.0
    deterministic_fallback_used: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


def heuristic_confidence(features: ConfidenceFeatures) -> float:
    """The interim aggregation used until the calibrator is fitted (4.7.3).

    Conservative by construction: it starts from the *weakest* component, not the
    average, then applies the penalties the plan names. A system that reports its
    best component's confidence as the system's confidence is reporting a number
    that cannot be true.
    """
    value = min(features.tool_confidence_min, features.tool_confidence_mean)
    value *= 0.6 + 0.4 * float(np.clip(features.agreement_iou, 0.0, 1.0))
    value *= 1.0 - 0.25 * float(np.clip(features.threshold_fallback_fraction, 0.0, 1.0))
    if features.router_is_rules < 1.0:
        value *= 0.95  # an LLM tie-break is one more thing that can be wrong
    value *= max(0.6, 1.0 - 0.05 * features.warning_count)
    if features.deterministic_fallback_used >= 1.0:
        value = min(value, 0.35)
    return float(np.clip(value, 0.0, 1.0))


def _pav(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the monotone least-squares fit."""
    y = np.asarray(y, dtype=np.float64).copy()
    weights = np.asarray(weights, dtype=np.float64).copy()
    n = y.size
    level_values = y.copy()
    level_weights = weights.copy()
    level_sizes = np.ones(n, dtype=np.int64)
    index = 0
    active = 0
    while index < n:
        active += 1
        level_values[active - 1] = y[index]
        level_weights[active - 1] = weights[index]
        level_sizes[active - 1] = 1
        index += 1
        while active > 1 and level_values[active - 2] > level_values[active - 1]:
            total = level_weights[active - 2] + level_weights[active - 1]
            merged = (
                level_weights[active - 2] * level_values[active - 2]
                + level_weights[active - 1] * level_values[active - 1]
            ) / total
            level_values[active - 2] = merged
            level_weights[active - 2] = total
            level_sizes[active - 2] += level_sizes[active - 1]
            active -= 1
    out = np.empty(n, dtype=np.float64)
    position = 0
    for level in range(active):
        size = int(level_sizes[level])
        out[position : position + size] = level_values[level]
        position += size
    return out


@dataclass
class Calibrator:
    """Feature vector -> P(answer correct).

    The features are reduced to a single score by a logistic fit, and that score
    is mapped to a probability by isotonic regression. Two stages because a
    one-dimensional isotonic fit cannot use eight features and a logistic fit
    alone is not calibrated — the composition is the standard recipe and is what
    "isotonic (or logistic if data is thin)" means once there is more than one
    feature.
    """

    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    #: (score, probability) breakpoints of the isotonic step function.
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    n_samples: int = 0
    method: str = "unfitted"

    @property
    def fitted(self) -> bool:
        return self.method != "unfitted"

    def _score(self, features: np.ndarray) -> np.ndarray:
        return features @ np.asarray(self.weights) + self.bias

    def fit(
        self,
        features: list[ConfidenceFeatures] | np.ndarray,
        correct: list[bool] | np.ndarray,
        *,
        min_samples_for_isotonic: int = 100,
        iterations: int = 400,
        learning_rate: float = 0.25,
    ) -> "Calibrator":
        matrix = (
            np.stack([f.as_array() for f in features])
            if isinstance(features, list)
            else np.asarray(features, dtype=np.float64)
        )
        labels = np.asarray(correct, dtype=np.float64)
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same length")
        if matrix.shape[0] < 10:
            raise ValueError(
                f"a calibrator fitted on {matrix.shape[0]} samples would be a fiction; "
                f"section 4.7.3 budgets ~500 labelled end-to-end outputs"
            )

        # Standardise so no feature dominates the logistic fit by scale alone.
        mean = matrix.mean(axis=0)
        scale = np.where(matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0)
        standardised = (matrix - mean) / scale

        weights = np.zeros(standardised.shape[1], dtype=np.float64)
        bias = 0.0
        for _ in range(iterations):
            logits = standardised @ weights + bias
            predictions = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            error = predictions - labels
            weights -= learning_rate * (standardised.T @ error) / len(labels)
            bias -= learning_rate * float(error.mean())

        # Fold the standardisation into the stored weights so `predict` needs
        # nothing but the vector.
        self.weights = (weights / scale).tolist()
        self.bias = float(bias - float(np.sum(weights * mean / scale)))
        self.feature_names = FEATURE_NAMES
        self.n_samples = int(matrix.shape[0])

        scores = self._score(matrix)
        order = np.argsort(scores)
        if matrix.shape[0] >= min_samples_for_isotonic:
            isotonic = _pav(labels[order], np.ones_like(labels))
            self.knots_x = scores[order].tolist()
            self.knots_y = isotonic.tolist()
            self.method = "isotonic"
        else:
            self.knots_x = []
            self.knots_y = []
            self.method = "logistic"
        return self

    def predict(self, features: ConfidenceFeatures | np.ndarray) -> float:
        if not self.fitted:
            raise RuntimeError("calibrator is unfitted; use heuristic_confidence() instead")
        vector = (
            features.as_array()
            if isinstance(features, ConfidenceFeatures)
            else np.asarray(features)
        )
        score = float(np.asarray(vector) @ np.asarray(self.weights) + self.bias)
        if self.method == "logistic" or not self.knots_x:
            return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score)))))
        return float(np.clip(np.interp(score, self.knots_x, self.knots_y), 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights,
            "bias": self.bias,
            "knots_x": self.knots_x,
            "knots_y": self.knots_y,
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "Calibrator":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        calibrator = cls(
            weights=raw["weights"],
            bias=raw["bias"],
            knots_x=raw.get("knots_x", []),
            knots_y=raw.get("knots_y", []),
            feature_names=tuple(raw.get("feature_names", FEATURE_NAMES)),
            n_samples=raw.get("n_samples", 0),
            method=raw.get("method", "logistic"),
        )
        if calibrator.feature_names != FEATURE_NAMES:
            raise ValueError(
                "this calibrator was fitted on a different feature set; refit it rather "
                "than applying it to features it has never seen"
            )
        return calibrator


def reliability_diagram(
    probabilities: np.ndarray, correct: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    """Bin-by-bin reliability data (section 4.7.3), ready to plot or table."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (probabilities > low) & (probabilities <= high)
        if low == 0.0:
            in_bin |= probabilities == 0.0
        count = int(in_bin.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "mean_confidence": float(probabilities[in_bin].mean()) if count else 0.0,
                "accuracy": float(correct[in_bin].mean()) if count else 0.0,
            }
        )
    return rows


def expected_calibration_error(
    probabilities: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """ECE: the sample-weighted gap between confidence and accuracy."""
    rows = reliability_diagram(probabilities, correct, bins=bins)
    total = sum(row["count"] for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            row["count"] / total * abs(row["accuracy"] - row["mean_confidence"])
            for row in rows
            if row["count"]
        )
    )
