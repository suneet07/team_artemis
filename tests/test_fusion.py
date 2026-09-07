import numpy as np

from satquery.fusion.fusion_engine import DecisionFusionEngine


def test_fusion_high_agreement():
    engine = DecisionFusionEngine()

    # 10x10 mask, both agree on a 5x5 region
    opt_mask = np.zeros((10, 10), dtype=bool)
    sar_mask = np.zeros((10, 10), dtype=bool)
    opt_mask[2:7, 2:7] = True
    sar_mask[2:7, 2:7] = True

    res = engine.reconcile(opt_mask, sar_mask, "water")

    assert res["iou"] == 1.0
    assert res["verdict"] == "high_agreement"
    assert res["winning_modality"] == "both"
    assert np.array_equal(res["reconciled_mask"], opt_mask)


def test_fusion_sar_wins_cloud():
    engine = DecisionFusionEngine()

    # SAR sees 5x5 water, optical sees nothing (cloud)
    opt_mask = np.zeros((10, 10), dtype=bool)
    sar_mask = np.zeros((10, 10), dtype=bool)
    sar_mask[2:7, 2:7] = True

    res = engine.reconcile(opt_mask, sar_mask, "water")

    assert res["iou"] == 0.0
    assert res["verdict"] == "disagreement_resolved"
    assert res["disagreement_cause"] == "cloud_over_water"
    assert res["winning_modality"] == "sar"
    assert np.array_equal(res["reconciled_mask"], sar_mask)


def test_fusion_optical_wins_wind():
    engine = DecisionFusionEngine()

    # Optical sees 5x5 water, SAR sees nothing (wind roughened surface)
    opt_mask = np.zeros((10, 10), dtype=bool)
    sar_mask = np.zeros((10, 10), dtype=bool)
    opt_mask[2:7, 2:7] = True

    res = engine.reconcile(opt_mask, sar_mask, "water")

    assert res["iou"] == 0.0
    assert res["verdict"] == "disagreement_resolved"
    assert res["disagreement_cause"] == "wind_roughened_surface"
    assert res["winning_modality"] == "optical"
    assert np.array_equal(res["reconciled_mask"], opt_mask)


if __name__ == "__main__":
    test_fusion_high_agreement()
    test_fusion_sar_wins_cloud()
    test_fusion_optical_wins_wind()
    print("Fusion tests passed!")
