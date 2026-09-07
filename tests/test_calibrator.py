import numpy as np

from satquery.confidence.calibrator import ConfidenceCalibrator


def test_ece_perfect_calibration():
    # If confidence exactly matches accuracy, ECE should be 0

    # We construct accuracy array such that bin accuracy matches confidence
    # Actually simpler: ECE is computed by binning.
    # Let's just create 10 predictions of 0.1, with 1 correct (10%)
    confs_01 = np.full(10, 0.1)
    accs_01 = np.zeros(10, dtype=bool)
    accs_01[0] = True

    # 10 predictions of 0.9, with 9 correct (90%)
    confs_09 = np.full(10, 0.9)
    accs_09 = np.ones(10, dtype=bool)
    accs_09[0] = False

    confs = np.concatenate([confs_01, confs_09])
    accs = np.concatenate([accs_01, accs_09])

    ece = ConfidenceCalibrator.compute_ece(confs, accs, n_bins=10)
    assert np.isclose(ece, 0.0)


def test_ece_poor_calibration():
    # Overconfident: 90% confidence, but only 10% correct
    confs_09 = np.full(10, 0.9)
    accs_09 = np.zeros(10, dtype=bool)
    accs_09[0] = True

    ece = ConfidenceCalibrator.compute_ece(confs_09, accs_09, n_bins=10)
    # The gap is |0.9 - 0.1| = 0.8
    assert np.isclose(ece, 0.8)


if __name__ == "__main__":
    test_ece_perfect_calibration()
    test_ece_poor_calibration()
    print("Calibrator tests passed!")
