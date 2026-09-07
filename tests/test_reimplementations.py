"""Every reimplemented method, checked against an independent reference.

The plan names the sources it expects us to reimplement rather than import
(section 2.4, "Methods we read and reimplement"), and TEAM_CONTEXT's work
discipline says to reimplement rather than spend a day on a stranger's
``requirements.txt``. The risk that creates is a method that *looks* like the
paper and does not *behave* like it — which unit tests written against our own
expectations cannot catch, because they encode the same misunderstanding.

So each one is pinned to something independent: scikit-image for Otsu, a direct
co-occurrence matrix for GLCM entropy, SciPy's isotonic regression for the
pool-adjacent-violators fit, closed-form values for the index formulas, and
physical properties (equivalent number of looks, edge transition width, mean
preservation) for the speckle filter, which has no reference implementation we
are licensed to link against.

Two real defects were found by exactly this file and by nothing else: the
Refined Lee direction selection was noise-driven at one look and preserved edges
no better than a boxcar of the same size, and the two co-registration estimators
had to be pinned to a single sign convention or the cross-modal path would have
doubled a misregistration instead of removing it.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from satquery.confidence.calibration import _pav
from satquery.coreg import (
    estimate_shift_mutual_information,
    estimate_shift_phase,
    mutual_information,
)
from satquery.fusion import iou
from satquery.ingest.band_inventory import BandInventory
from satquery.sar.chain import to_db
from satquery.sar.speckle import refined_lee
from satquery.texture import components, glcm_entropy, largest_component_centroid, quantise
from satquery.tools.base import Scene
from satquery.tools.deterministic import change_statistics, compute_index
from satquery.tools.thresholds import otsu_threshold


def enl(array: np.ndarray) -> float:
    """Equivalent number of looks: mean^2 / variance. Higher = less speckle."""
    values = array[np.isfinite(array)]
    return float(values.mean() ** 2 / values.var())


def rise_width(image: np.ndarray) -> float:
    """10-90% transition width across the step edge, in pixels. Lower = sharper.

    Measuring the step *height* across a span wider than the kernel proves
    nothing — both a boxcar and an edge-preserving filter pass that. The width of
    the transition is what separates them.
    """
    profile = np.nanmean(image[:, 110:146], axis=0)
    low, high = profile[:5].mean(), profile[-5:].mean()
    xs = np.arange(profile.size)
    return float(
        np.interp(low + 0.9 * (high - low), profile, xs)
        - np.interp(low + 0.1 * (high - low), profile, xs)
    )


def test_otsu_matches_skimage_and_exposes_a_usable_criterion():
    """We reimplement Otsu only to get the criterion the gate needs (4.6.2)."""
    rng = np.random.default_rng(0)
    for data in (
        np.concatenate([rng.normal(0.1, 0.05, 5000), rng.normal(0.8, 0.05, 5000)]),
        np.concatenate([rng.normal(-20, 2, 5000), rng.normal(-4, 3, 5000)]),
        rng.uniform(0, 1, 10000),
    ):
        mine, _ = otsu_threshold(data)
        assert abs(mine - float(threshold_otsu(data))) < (data.max() - data.min()) / 128

    _, bimodal = otsu_threshold(np.concatenate([np.zeros(500), np.ones(500)]))
    _, uniform = otsu_threshold(rng.uniform(0, 1, 10000))
    assert bimodal > 0.9 > uniform, (
        f"the criterion must separate a bimodal histogram ({bimodal:.3f}) from a "
        f"uniform one ({uniform:.3f}); it is the whole basis of the gate"
    )


def test_glcm_entropy_matches_a_direct_co_occurrence_matrix():
    rng = np.random.default_rng(0)
    image = ndimage.gaussian_filter(rng.random((64, 64)), 2)
    levels = 16

    windowed = glcm_entropy(image, window=63, levels=levels, offset=(0, 1))

    quantised, _ = quantise(image, levels)
    shifted = np.roll(quantised, -1, axis=1)
    hist = np.histogram2d(
        quantised.ravel(), shifted.ravel(), bins=levels, range=[[0, levels], [0, levels]]
    )[0]
    probability = hist / hist.sum()
    nonzero = probability > 0
    direct = -np.sum(probability[nonzero] * np.log2(probability[nonzero])) / np.log2(
        levels * levels
    )
    assert abs(float(windowed[32, 32]) - direct) < 0.06

    assert glcm_entropy(np.zeros((32, 32)), window=9, levels=8).mean() < glcm_entropy(
        rng.random((32, 32)), window=9, levels=8
    ).mean()


def test_refined_lee_denoises_preserves_edges_and_stays_unbiased():
    rng = np.random.default_rng(0)
    truth = np.full((256, 256), 1.0)
    truth[:, 128:] = 4.0
    speckled = truth * rng.exponential(1.0, truth.shape)  # fully developed, 1 look
    filtered = refined_lee(speckled, looks=1.0)
    boxcar = ndimage.uniform_filter(speckled, 7)

    assert enl(filtered[:, :120]) > 4 * enl(speckled[:, :120]), "speckle must be suppressed"
    assert abs(float(np.nanmean(filtered[:, :120])) - float(speckled[:, :120].mean())) < 0.05, (
        "an unbiased filter must preserve the local mean, or every sigma-nought "
        "threshold downstream is applied to a shifted distribution"
    )
    assert rise_width(filtered) < rise_width(boxcar), (
        f"refined Lee must beat a boxcar of the same window on edge sharpness "
        f"({rise_width(filtered):.2f} px vs {rise_width(boxcar):.2f} px), otherwise "
        f"the directional refinement is not doing anything"
    )


def test_mutual_information_separates_dependent_from_independent():
    rng = np.random.default_rng(0)
    x = rng.random(20000)
    assert mutual_information(x, x, bins=32) > 3.0
    assert mutual_information(x, rng.random(20000), bins=32) < 0.2


def test_both_shift_estimators_return_the_correction_in_the_same_sign():
    """A sign disagreement would double a cross-modal misregistration."""
    rng = np.random.default_rng(0)
    base = ndimage.gaussian_filter(rng.random((128, 128)), 2)

    for dy, dx in [(3.0, -2.0), (-5.0, 4.0), (0.0, 0.0)]:
        moved = ndimage.shift(base, (dy, dx), order=1, mode="nearest")
        shift, _ = estimate_shift_phase(base, moved)
        corrected = ndimage.shift(moved, shift, order=1, mode="nearest")
        residual = np.abs(corrected[20:-20, 20:-20] - base[20:-20, 20:-20]).mean()
        raw = np.abs(moved[20:-20, 20:-20] - base[20:-20, 20:-20]).mean()
        assert residual <= raw * 0.35 + 1e-9, f"applying the returned shift must register {dy},{dx}"

    # Cross-modal: monotone but radiometrically unrelated, where intensity
    # correlation fails and mutual information is the plan's answer (4.3 step 3).
    sar_like = np.exp(base * 3)
    displaced = ndimage.shift(sar_like, (2.0, -1.0), order=1, mode="nearest")
    (mdy, mdx), _ = estimate_shift_mutual_information(base, displaced, search_px=4)
    assert abs(mdy + 2.0) < 1.0 and abs(mdx - 1.0) < 1.0

    same = ndimage.shift(base, (2.0, -1.0), order=1, mode="nearest")
    (pdy, pdx), _ = estimate_shift_phase(base, same)
    (qdy, qdx), _ = estimate_shift_mutual_information(base, same, search_px=4)
    assert np.sign(pdy) == np.sign(qdy) and np.sign(pdx) == np.sign(qdx)


def test_pav_matches_scipy_isotonic_regression():
    from scipy.optimize import isotonic_regression

    rng = np.random.default_rng(0)
    y = rng.random(200)
    assert np.allclose(_pav(y, np.ones_like(y)), isotonic_regression(y).x, atol=1e-9)
    assert np.all(np.diff(_pav(rng.random(500), np.ones(500))) >= -1e-12)


def test_connected_components_areas_boxes_and_centroids():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:30] = True  # 200 px, 10 rows x 20 cols
    mask[60:64, 60:64] = True  # 16 px

    found = components(mask)
    assert [component.area_px for component in found] == [200, 16]
    biggest = found[0]
    assert biggest.bbox == (10, 10, 20, 30)
    assert abs(biggest.aspect_ratio - 2.0) < 1e-9
    assert abs(biggest.solidity - 1.0) < 1e-9

    x, y = largest_component_centroid(mask)
    assert abs(x - 19.5 / 99) < 0.02 and abs(y - 14.5 / 99) < 0.02


def test_spectral_index_formulas_match_their_definitions():
    green, red, nir, swir = (np.full((4, 4), v) for v in (0.30, 0.10, 0.50, 0.20))
    scene = Scene(
        name="probe",
        modality="optical",
        bands={"green": green, "red": red, "nir": nir, "swir": swir},
        inventory=BandInventory(
            bands={"green": 1, "red": 2, "nir": 3, "swir": 4}, has_swir=True, has_nir=True
        ),
    )
    expected = {
        "NDVI": (0.5 - 0.1) / (0.5 + 0.1),
        "NDWI": (0.3 - 0.5) / (0.3 + 0.5),
        "MNDWI": (0.3 - 0.2) / (0.3 + 0.2),
        "NDBI": (0.2 - 0.5) / (0.2 + 0.5),
    }
    for name, value in expected.items():
        assert abs(float(compute_index(scene, name)[0, 0]) - value) < 1e-12


def test_db_conversion_floors_zero_instead_of_returning_negative_infinity():
    assert abs(float(to_db(np.array([[0.01]]))[0, 0]) + 20.0) < 1e-9
    assert np.isfinite(to_db(np.array([[0.0]]))).all(), (
        "one -inf reaching a percentile stretch collapses the whole scene"
    )


def test_iou_is_intersection_over_union():
    a = np.zeros((10, 10), dtype=bool)
    a[:5, :] = True
    b = np.zeros((10, 10), dtype=bool)
    b[2:7, :] = True
    assert abs(iou(a, b) - 30 / 70) < 1e-12


def test_change_statistics_are_exact():
    before = np.zeros((10, 10), dtype=int)
    before[:3, :] = 1
    after = np.zeros((10, 10), dtype=int)
    after[:6, :] = 1

    stats = change_statistics(
        before, after, class_names={0: "bg", 1: "built"}, pixel_size_m=10.0
    )
    built = next(entry for entry in stats["per_class"] if entry["class"] == "built")
    assert built["delta_px"] == 30
    assert abs(built["class_change_ratio"] - 1.0) < 1e-12
    assert abs(stats["change_ratio"] - 0.30) < 1e-12
    assert abs(built["delta_km2"] - 30 * 100 / 1e6) < 1e-12
    assert stats["largest_change_class"] and stats["smallest_change_class"]


def test_change_ratio_is_undefined_not_infinite_for_new_construction():
    """float('inf') is not valid JSON, and this lands in a graded artifact."""
    before = np.zeros((8, 8), dtype=int)
    after = np.zeros((8, 8), dtype=int)
    after[:2, :] = 1
    stats = change_statistics(before, after, class_names={0: "bg", 1: "built"})
    built = next(entry for entry in stats["per_class"] if entry["class"] == "built")
    assert built["class_change_ratio"] is None
