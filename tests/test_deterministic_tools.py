import numpy as np

from satquery.config import preprocessing_config
from satquery.tools.cv_utils import get_largest_component_centroid, otsu_with_gate
from satquery.tools.deterministic import (
    execute_object_box_fallback,
    execute_sar_backscatter,
    execute_spectral_index,
    execute_texture_seg,
)


def test_otsu_with_gate_bimodal():
    # Create bimodal synthetic data
    data = np.concatenate([np.random.normal(0.1, 0.05, 500), np.random.normal(0.8, 0.05, 500)])
    method, thresh = otsu_with_gate(data, 0.3)
    assert method == "otsu"
    assert 0.1 < thresh < 0.8


def test_otsu_with_gate_unimodal():
    # Create unimodal synthetic data (e.g. mostly background)
    data = np.concatenate([np.random.normal(0.1, 0.05, 990), np.random.normal(0.8, 0.05, 10)])
    method, thresh = otsu_with_gate(data, 0.3)
    assert method == "fixed_fallback"
    assert thresh == 0.3


def test_get_largest_component_centroid():
    mask = np.zeros((10, 10), dtype=bool)
    # Blob 1 (small)
    mask[1:3, 1:3] = True
    # Blob 2 (large)
    mask[5:9, 5:9] = True

    centroid = get_largest_component_centroid(mask)
    assert centroid is not None
    x, y = centroid
    # Center of [5:9, 5:9] is at index 6.5, width is 10, so ~0.65
    assert 0.6 < x < 0.7
    assert 0.6 < y < 0.7


def test_execute_spectral_index():
    index_array = np.concatenate(
        [np.random.normal(0.1, 0.05, 500), np.random.normal(0.8, 0.05, 500)]
    ).reshape(10, 100)
    params = {"index_array": index_array, "index": "NDVI"}
    result = execute_spectral_index(params)
    assert result["threshold_method"] == "otsu"
    assert "centroid_prior" in result


def test_execute_sar_backscatter():
    # Seeded. Unseeded, this drew two Gaussians afresh every run and asserted
    # that Otsu was accepted -- but the bimodality gate rejects about **9% of
    # those draws** (37 of 400 seeds), so the test failed roughly one run in
    # eleven, on a schedule that looked like anything except its actual cause.
    # It cost two false "the deploy is broken" diagnoses before the pattern was
    # visible.
    #
    # The seed is not chosen to dodge the gate: the point of the test is that a
    # clearly bimodal input gets an adaptive threshold, so it needs an input
    # that is clearly bimodal. A draw that lands inside the gate's margin tests
    # the gate, not the tool, and does it non-reproducibly.
    rng = np.random.default_rng(0)
    sigma0_array = np.concatenate(
        [rng.normal(-25, 2, 500), rng.normal(-5, 2, 500)]
    ).reshape(10, 100)
    params = {"sigma0_array": sigma0_array, "target": "water"}
    result = execute_sar_backscatter(params)
    assert result["threshold_method"] == "otsu"
    assert "centroid_prior" in result


def test_execute_texture_seg():
    # Seeded for the same reason as the test above: it asserts a threshold
    # method, and an unseeded draw makes that a coin toss with good odds.
    rng = np.random.default_rng(1)
    image_array = rng.random((20, 20))
    # Add a high variance region
    image_array[5:15, 5:15] = rng.random((10, 10)) * 10
    params = {"image_array": image_array}
    result = execute_texture_seg(params)
    assert result["threshold_method"] == "otsu"
    assert "centroid_prior" in result


def test_execute_object_box_fallback():
    rng = np.random.default_rng(2)
    image_array = np.zeros((50, 50))
    # Add an object
    image_array[10:20, 10:20] = rng.random((10, 10)) * 10
    params = {"image_array": image_array}
    result = execute_object_box_fallback(params)
    assert result["method"] == "deterministic_fallback"
    assert len(result["boxes"]) >= 1
    # Section 4.6.9 declares this a low-confidence proposer: it must never
    # outrank learned grounding. The ceiling is a config value, not a literal --
    # asserting a magic 0.1 pinned a number the plan never specified.
    ceiling = preprocessing_config().fusion.deterministic_fallback_ceiling
    assert 0.0 < result["confidence"] <= ceiling


def test_eight_bit_sar_is_refused_rather_than_silently_thresholded():
    """A pre-stretched 8-bit SAR array must not be cut with a decibel threshold.

    OpenEarthMap-SAR ships "radiometrically normalized backscatter intensity
    standardized to an 8-bit format", so its pixels run 0..255. The configured
    water cut is -18 dB, which no 0..255 value can fall below: before this
    guard the tool returned an **empty** water mask and reported it as a
    finding. Section 4.7 forbids exactly that -- the system must never quietly
    pick an answer it cannot support.
    """
    stretched = np.linspace(0, 255, 1000).reshape(10, 100)
    result = execute_sar_backscatter({"sigma0_array": stretched, "target": "water"})
    assert result["confidence"] == 0.0
    assert "not calibrated decibels" in result["answer"]


def test_calibrated_decibel_sar_still_thresholds():
    """The guard keys on units, not on the array being difficult.

    A genuine dB scene spans roughly -25 dB over calm water to +5 dB over
    urban, per the C-band reference table, so it always carries negatives.
    """
    water = np.random.normal(-22.0, 1.0, 500)
    urban = np.random.normal(-2.0, 1.0, 500)
    sigma0 = np.concatenate([water, urban]).reshape(10, 100)
    result = execute_sar_backscatter({"sigma0_array": sigma0, "target": "water"})
    assert result["confidence"] > 0.0
    assert "centroid_prior" in result
