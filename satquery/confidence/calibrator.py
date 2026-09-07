import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None


class ConfidenceCalibrator:
    """
    §4.7.3: End-to-end calibrator.
    Uses Isotonic Regression to map heuristic or raw model confidences
    to true empirical probabilities based on a holdout set.
    """

    @property
    def is_fitted(self) -> bool:
        """Whether `predict` returns a calibrated number or a passthrough."""
        return self.calibrator is not None and hasattr(self.calibrator, "X_min_")

    def __init__(self):
        self.calibrator = None
        if IsotonicRegression is not None:
            # y_min=0, y_max=1 ensures output is a valid probability
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10
    ) -> list[dict[str, float]]:
        """Per-bin confidence vs accuracy — the plot section 4.7.3 asks for.

        ECE is one number and hides the shape of the error; the reliability
        diagram is what goes in the deck, and no system in the landscape survey
        reports one.
        """
        confidences = np.asarray(confidences, dtype=float)
        accuracies = np.asarray(accuracies, dtype=float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        rows: list[dict[str, float]] = []
        for index in range(n_bins):
            low, high = edges[index], edges[index + 1]
            if index == n_bins - 1:
                in_bin = (confidences >= low) & (confidences <= high)
            else:
                in_bin = (confidences >= low) & (confidences < high)
            count = int(in_bin.sum())
            rows.append(
                {
                    "bin_low": float(low),
                    "bin_high": float(high),
                    "count": count,
                    "mean_confidence": float(confidences[in_bin].mean()) if count else 0.0,
                    "accuracy": float(accuracies[in_bin].mean()) if count else 0.0,
                }
            )
        return rows

    def fit(self, raw_confidences: np.ndarray, ground_truth: np.ndarray):
        """
        Fits the isotonic regression model.
        raw_confidences: 1D array of uncalibrated scores [0, 1]
        ground_truth: 1D boolean array (1 if correct, 0 if incorrect)
        """
        if self.calibrator is None:
            raise ImportError("scikit-learn is required to fit the calibrator.")

        self.calibrator.fit(raw_confidences, ground_truth)

    def predict(self, raw_confidence: float) -> float:
        """
        Calibrates a single prediction.
        """
        if self.calibrator is None:
            # Uncalibrated passthrough. The caller must keep reporting
            # `confidence_basis: "heuristic"` in this state: a trace that says
            # "calibrated" over an unfitted passthrough is a false claim about
            # the one number section 4.7.3 says has to be honest.
            return float(raw_confidence)

        # IsotonicRegression predict expects 1D array
        calibrated = self.calibrator.predict([raw_confidence])[0]
        return float(calibrated)

    def predict_batch(self, raw_confidences: np.ndarray) -> np.ndarray:
        """
        Calibrates a batch of predictions.
        """
        if self.calibrator is None:
            return raw_confidences
        return self.calibrator.predict(raw_confidences)

    @staticmethod
    def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes the Expected Calibration Error (ECE).
        confidences: 1D array of predicted probabilities
        accuracies: 1D boolean array of whether the prediction was correct
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        for index, (bin_lower, bin_upper) in enumerate(
            zip(bin_lowers, bin_uppers, strict=False)
        ):
            # The last bin must include 1.0. With a half-open `< upper` on every
            # bin, a perfectly confident prediction fell into no bin at all and
            # was silently dropped from ECE -- and those are exactly the samples
            # a calibration number is most often wrong about.
            if index == n_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = in_bin.astype(float).mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)
