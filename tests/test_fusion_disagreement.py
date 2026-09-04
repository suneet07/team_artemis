import json
from pathlib import Path

from satquery.fusion.reconcile import reconcile_crossmodal


def test_fusion_consistent_case():
    opt = {"area_km2": 3.0}
    sar = {"area_km2": 3.2}

    res = reconcile_crossmodal(opt, sar)
    assert res.verdict == "consistent"
    assert res.iou is not None and res.iou >= 0.70
    assert res.disagreement_cause is None
    assert res.confidence > 0.80


def test_fusion_five_disagreement_rules():
    fixture_path = (
        Path(__file__).parent.parent
        / "frontend"
        / "mocks"
        / "fixtures"
        / "disagreement_causes.json"
    )
    fixture_causes = json.loads(fixture_path.read_text(encoding="utf-8"))["causes"]
    expected_causes = {c["code"]: c for c in fixture_causes}

    opt = {"area_km2": 4.5}
    sar = {"area_km2": 1.2}

    for cause_code, cause_data in expected_causes.items():
        res = reconcile_crossmodal(opt, sar, disagreement_hint=cause_code)
        assert res.verdict == "disagreement"
        assert res.disagreement_cause == cause_code
        assert res.winning_modality == cause_data["trusted_modality"]
        assert res.explanation == cause_data["explanation"]
        # Confidence must be strictly lower than consistent case
        assert res.confidence < 0.80


def test_fusion_pixel_wise_iou():
    import numpy as np

    # 100x100 masks with known 50% overlap
    opt_mask = np.zeros((100, 100), dtype=np.uint8)
    sar_mask = np.zeros((100, 100), dtype=np.uint8)

    # Opt water in [20:60, 20:60] -> 1600 px
    opt_mask[20:60, 20:60] = 1
    # SAR water in [20:60, 20:60] -> identical -> IoU = 1.0
    sar_mask[20:60, 20:60] = 1

    opt = {"area_km2": 1.6}
    sar = {"area_km2": 1.6}

    res = reconcile_crossmodal(opt, sar, opt_mask=opt_mask, sar_mask=sar_mask)
    assert res.verdict == "consistent"
    assert res.iou == 1.0


def test_fusion_data_driven_all_five_causes_without_hint():
    import numpy as np

    # 1. cloud_over_water: SAR sees water, optical has cloud (or 0 water)
    sar_mask = np.zeros((100, 100), dtype=np.uint8)
    sar_mask[20:70, 20:70] = 1
    opt_mask = np.zeros((100, 100), dtype=np.uint8)
    opt = {"area_km2": 0.0, "cloud": True}
    sar = {"area_km2": 2.5}
    r1 = reconcile_crossmodal(opt, sar, opt_mask=opt_mask, sar_mask=sar_mask)
    assert r1.verdict == "disagreement"
    assert r1.disagreement_cause == "cloud_over_water"
    assert r1.winning_modality == "sar"
    assert r1.confidence < 0.80

    # 2. wind_roughened_surface: Optical sees water, SAR sees bright/no water
    opt_mask2 = np.zeros((100, 100), dtype=np.uint8)
    opt_mask2[20:70, 20:70] = 1
    sar_mask2 = np.zeros((100, 100), dtype=np.uint8)
    opt2 = {"area_km2": 2.5}
    sar2 = {"area_km2": 0.0}
    r2 = reconcile_crossmodal(opt2, sar2, opt_mask=opt_mask2, sar_mask=sar_mask2)
    assert r2.verdict == "disagreement"
    assert r2.disagreement_cause == "wind_roughened_surface"
    assert r2.winning_modality == "optical"
    assert r2.confidence < 0.80

    # 3. wet_smooth_soil: SAR sees dark specular, optical detects bare soil
    opt3 = {"area_km2": 0.5, "wet_soil": True}
    sar3 = {"area_km2": 3.0}
    r3 = reconcile_crossmodal(opt3, sar3)
    assert r3.verdict == "disagreement"
    assert r3.disagreement_cause == "wet_smooth_soil"
    assert r3.winning_modality == "optical"
    assert r3.confidence < 0.80

    # 4. radar_shadow: SAR has shadow in steep terrain
    opt4 = {"area_km2": 0.2}
    sar4 = {"area_km2": 3.5, "radar_shadow": True}
    r4 = reconcile_crossmodal(opt4, sar4)
    assert r4.verdict == "disagreement"
    assert r4.disagreement_cause == "radar_shadow"
    assert r4.winning_modality == "optical"
    assert r4.confidence < 0.80

    # 5. dry_smooth_sand: SAR sees dark like water, optical detects sand
    opt5 = {"area_km2": 0.3, "sand": True}
    sar5 = {"area_km2": 2.8}
    r5 = reconcile_crossmodal(opt5, sar5)
    assert r5.verdict == "disagreement"
    assert r5.disagreement_cause == "dry_smooth_sand"
    assert r5.winning_modality == "optical"
    assert r5.confidence < 0.80

