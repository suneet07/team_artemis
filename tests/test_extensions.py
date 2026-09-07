"""The six gaps closed after the completeness audit.

Each of these was a documented shortfall against the master plan rather than a
bug: AROSICS declared but never imported, tiling built but never in the query
path, two coexisting box conventions, polarisation dropout configured but never
applied, a CLIP scorer specified but only approximated, and one disagreement rule
that could not fire.

They are tested together because they share a failure mode. Every one of them is
the kind of gap that a wrapper module can *appear* to close — an import guard, a
fallback stub, an extra field — while the real path still does the old thing.
These tests assert the behaviour reaches the pipeline, not that a helper exists.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.agent.tiled import tiled_answer_query
from satquery.coreg import arosics_available, correct_with_arosics
from satquery.fusion import classify_disagreement, slope_mask_from_dem
from satquery.qgen import boxes
from satquery.qgen.dropout import DropoutStats, with_pol_dropout
from satquery.tiling import clip_available, score_tiles


@pytest.fixture
def big_scene(tmp_path):
    """A scene several tiles across, so tiling is not a no-op."""
    path = tmp_path / "scene.tif"
    rng = np.random.default_rng(0)
    side = 1200
    gradient = np.linspace(400, 3600, 4 * side * side).reshape(4, side, side)
    data = np.clip(
        gradient + rng.integers(-150, 150, (4, side, side)), 0, 4095
    ).astype("uint16")
    profile = {
        "driver": "GTiff",
        "height": side,
        "width": side,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32644",
        "transform": from_origin(0.0, side * 10.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(("Blue", "Green", "Red", "NIR"), start=1):
            dst.set_band_description(index, name)
    return path


# --------------------------------------------------------------------- AROSICS


def test_arosics_absence_is_reported_not_swallowed():
    """An unavailable escalation path must say so, never look like success."""
    result = correct_with_arosics("ref.tif", "tgt.tif", "out.tif")
    if arosics_available():
        pytest.skip("AROSICS installed; this asserts the unavailable path")
    assert result.succeeded is False
    assert result.method == "arosics_unavailable"
    assert any("not installed" in message for message in result.warnings)
    assert result.corrected_path is None


def test_arosics_probe_is_cheap():
    """A capability probe must not import GDAL to answer a yes/no question."""
    import time

    start = time.perf_counter()
    arosics_available()
    assert time.perf_counter() - start < 1.0


# ------------------------------------------------------------------- box convention


def test_one_box_convention_reaches_both_producers():
    box = boxes.from_pixels([10, 20, 30, 60], width=100, height=100, order="yxyx")
    assert box.as_list == [0.2, 0.1, 0.6, 0.3]
    assert boxes.to_yxyx_1000(box) == [100, 200, 300, 600]
    assert boxes.to_prompt_box(box) == [200, 100, 600, 300]  # xyxy at 0-1000


def test_box_round_trips_through_the_legacy_serialisation():
    original = boxes.BoxXYXY(0.2, 0.1, 0.6, 0.3)
    assert boxes.from_yxyx_1000(boxes.to_yxyx_1000(original)).as_list == original.as_list


def test_inverted_box_is_rejected_not_silently_accepted():
    """An inverted box scores zero IoU and reads as a model failure."""
    with pytest.raises(ValueError):
        boxes.BoxXYXY(0.8, 0.1, 0.2, 0.3)


def test_training_targets_refuse_an_unverified_convention():
    """TEAM_CONTEXT section 10: verify against the model card before generating."""
    if boxes.BOX_CONVENTION_VERIFIED:
        pytest.skip("convention has been verified; the guard is meant to be lifted")
    with pytest.raises(RuntimeError, match="not been verified"):
        boxes.assert_convention_verified()


def test_object_box_fallback_emits_the_canonical_fields():
    from satquery.tools.deterministic import execute_object_box_fallback

    rng = np.random.default_rng(0)
    image = np.zeros((50, 50))
    image[10:20, 10:20] = rng.random((10, 10)) * 10
    result = execute_object_box_fallback({"image_array": image, "target": "tanks"})
    assert result["boxes"], "the fixture contains one obvious blob"
    box = result["boxes"][0]
    assert "bbox_xyxy_normalised" in box and "bbox_prompt" in box
    assert box["bbox_prompt_format"] == f"{boxes.PROMPT_BOX_FORMAT}@{boxes.PROMPT_BOX_SCALE}"


# ----------------------------------------------------------------- pol dropout


def test_pol_dropout_fires_at_the_configured_rate():
    from satquery.config import preprocessing_config

    vv, vh = np.zeros((4, 4)), np.ones((4, 4))
    samples = [{"id": i, "bands_db": {"VV": vv, "VH": vh}} for i in range(2000)]
    stats = DropoutStats()
    list(with_pol_dropout(samples, seed=7, stats=stats))

    target = preprocessing_config().sar.pol_dropout_rate
    assert abs(stats.realised_rate - target) < 0.03, (
        f"realised {stats.realised_rate:.1%} against a configured {target:.0%}"
    )


def test_pol_dropout_is_reproducible_from_its_seed():
    vv, vh = np.zeros((4, 4)), np.ones((4, 4))
    samples = [{"id": i, "bands_db": {"VV": vv, "VH": vh}} for i in range(200)]
    first = [s["pol_dropout_applied"] for s in with_pol_dropout(samples, seed=11)]
    second = [s["pol_dropout_applied"] for s in with_pol_dropout(samples, seed=11)]
    third = [s["pol_dropout_applied"] for s in with_pol_dropout(samples, seed=12)]
    assert first == second, "a run must be reproducible from its seed"
    assert first != third, "a different seed must give a different draw"


def test_pol_dropout_records_what_it_did_in_the_sample():
    vv, vh = np.zeros((4, 4)), np.ones((4, 4))
    converted = [
        s
        for s in with_pol_dropout(
            [{"id": i, "bands_db": {"VV": vv, "VH": vh}} for i in range(100)], seed=3
        )
        if s["pol_dropout_applied"]
    ]
    assert converted, "some samples must convert at a 25% rate over 100 draws"
    assert converted[0]["polarisations"] == ["VV"], "co-pol survives the drop"


def test_optical_samples_pass_through_untouched():
    samples = [{"id": "optical-only"}]
    assert list(with_pol_dropout(samples, seed=1)) == samples


# ------------------------------------------------------------------ CLIP scorer


def test_clip_probe_is_cheap_and_honest():
    import time

    start = time.perf_counter()
    available = clip_available()
    assert time.perf_counter() - start < 1.0, "the probe must not import torch"
    assert isinstance(available, bool)


def test_score_tiles_uses_a_batch_scorer_when_offered():
    """One forward pass for the whole scene, not one per tile."""

    class BatchScorer:
        __name__ = "clip:test-double"

        def __init__(self):
            self.batch_calls = 0
            self.single_calls = 0

        def score_batch(self, query, tiles):
            self.batch_calls += 1
            return [float(index) / len(tiles) for index in range(len(tiles))]

        def __call__(self, query, tile):
            self.single_calls += 1
            return 0.5

    scorer = BatchScorer()
    tiles = [(index, np.zeros((8, 8))) for index in range(6)]
    ranked = score_tiles("water", tiles, scorer=scorer)

    assert scorer.batch_calls == 1
    assert scorer.single_calls == 0
    assert [entry.index for entry in ranked] == [5, 4, 3, 2, 1, 0]
    assert ranked[0].basis == "clip:test-double"


# ------------------------------------------------------------------ tiled query


def test_tiled_query_selects_top_k_and_mosaics_to_full_scene(big_scene, tmp_path):
    """C18 survives tiling: the deliverable is one full-scene mask."""
    out = tiled_answer_query(
        "Where is the water body?", big_scene, top_k=3, output_dir=tmp_path / "evidence"
    )
    assert len(out.selected) == 3
    assert len(out.outcomes) == 3
    assert out.scene_shape == (1200, 1200)
    assert out.mask_uris, "a mosaicked mask must be written"

    with rasterio.open(out.mask_uris[0]) as mask:
        assert (mask.height, mask.width) == out.scene_shape
        assert mask.crs is not None


def test_tiled_query_records_its_tile_selection(big_scene, tmp_path):
    out = tiled_answer_query("Is there water?", big_scene, top_k=2, output_dir=tmp_path)
    block = out.as_trace_block()
    assert block["scene_shape"] == [1200, 1200]
    for entry in block["tiles_selected"]:
        assert {"index", "row_off", "col_off", "score", "scoring_basis"} <= set(entry)


def test_top_k_one_is_the_week_six_cut_list_fallback(big_scene, tmp_path):
    """"Global downsample plus one detail crop" is a parameter, not a rewrite."""
    out = tiled_answer_query("Is there water?", big_scene, top_k=1, output_dir=tmp_path)
    assert len(out.outcomes) == 1


# ---------------------------------------------------------------- radar shadow


def test_radar_shadow_follows_the_look_direction():
    """The row needs terrain and a look direction; both change the answer."""
    _, x = np.mgrid[0:64, 0:64]
    ridge = np.where(x < 32, x * 20.0, (64 - x) * 20.0)

    east = slope_mask_from_dem(ridge, 10.0, look_azimuth_deg=90.0, incidence_deg=35.0)
    west = slope_mask_from_dem(ridge, 10.0, look_azimuth_deg=270.0, incidence_deg=35.0)

    assert east[:, 40:].mean() > east[:, :24].mean()
    assert west[:, :24].mean() > west[:, 40:].mean()


def test_flat_terrain_casts_no_radar_shadow():
    assert not slope_mask_from_dem(
        np.zeros((32, 32)), 10.0, look_azimuth_deg=90.0
    ).any()


def test_radar_shadow_rule_fires_once_a_dem_is_supplied():
    _, x = np.mgrid[0:64, 0:64]
    ridge = np.where(x < 32, x * 20.0, (64 - x) * 20.0)
    shadow = slope_mask_from_dem(ridge, 10.0, look_azimuth_deg=90.0)

    optical = np.zeros((64, 64), dtype=bool)
    sar = np.zeros((64, 64), dtype=bool)
    sar[:, 40:] = True

    rule = classify_disagreement("water", optical, sar, slope_mask=shadow)
    assert rule.cause == "radar_shadow"
    assert rule.winner == "optical"


def test_without_a_dem_the_cause_is_not_fabricated():
    """No DEM means no shadow claim — a different rule or none, never a guess."""
    optical = np.zeros((64, 64), dtype=bool)
    sar = np.zeros((64, 64), dtype=bool)
    sar[:, 40:] = True
    rule = classify_disagreement("water", optical, sar)
    assert rule is None or rule.cause != "radar_shadow"
