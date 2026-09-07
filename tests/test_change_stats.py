import numpy as np

from satquery.tools.deterministic import execute_change_stats


def test_change_stats_increase():
    mask_t1 = np.zeros((10, 10), dtype=bool)
    mask_t2 = np.zeros((10, 10), dtype=bool)

    mask_t1[0:2, 0:2] = True  # 4 pixels
    mask_t2[0:4, 0:4] = True  # 16 pixels

    # 1 pixel = 100 m^2
    params = {"mask_t1": mask_t1, "mask_t2": mask_t2, "pixel_area_m2": 100.0}

    res = execute_change_stats(params)
    assert res["area_t1_m2"] == 400.0
    assert res["area_t2_m2"] == 1600.0
    assert res["delta_m2"] == 1200.0
    assert res["ratio"] == 4.0
    assert "increased" in res["answer"]
    assert "1200.00" in res["answer"]


def test_change_stats_decrease():
    mask_t1 = np.zeros((10, 10), dtype=bool)
    mask_t2 = np.zeros((10, 10), dtype=bool)

    mask_t1[0:4, 0:4] = True  # 16 pixels
    mask_t2[0:2, 0:2] = True  # 4 pixels

    params = {"mask_t1": mask_t1, "mask_t2": mask_t2, "pixel_area_m2": 10.0}

    res = execute_change_stats(params)
    assert res["area_t1_m2"] == 160.0
    assert res["area_t2_m2"] == 40.0
    assert res["delta_m2"] == -120.0
    assert res["ratio"] == 0.25
    assert "decreased" in res["answer"]
    assert "120.00" in res["answer"]


if __name__ == "__main__":
    test_change_stats_increase()
    test_change_stats_decrease()
    print("Change stats tests passed!")
