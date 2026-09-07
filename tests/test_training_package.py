"""The shared training library — the thing whose absence made the ML side an island.

Every test here pins a property that a previous version of this code got wrong
in a way no metric revealed:

* configs restating ``max_pixels`` instead of reading the frozen contract, so
  the timing sweep priced one config and the trainer ran another;
* a manifest loader that skipped malformed rows, shrinking a corpus by an amount
  nobody measured;
* GSD conditioning on the *native* resolution of a source that enters training
  downsampled;
* average accuracy computed as overall accuracy, which is a different number
  whenever the question-type distribution is skewed, which it always is;
* composites built from whichever bands were present.

No test here imports torch. The package is arranged so that it does not have to.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from satquery.config import preprocessing_config
from satquery.ingest.copernicus.composites import (
    COMPOSITES,
    REQUIRED_BANDS,
    build_composites,
    composite_names_for,
    stretch,
)
from satquery.training.config import ADAPTERS, TrainingConfig, adapter_composites
from satquery.training.dataset import (
    CanonicalSample,
    RealChipDataset,
    gsd_prompt_prefix,
    load_canonical_manifest,
    resolve_sample_images,
)
from satquery.training.metrics import (
    ScoreReport,
    box_iou,
    grounding_accuracy,
    normalise_answer,
    score_predictions,
)


def _row(**overrides):
    row = {
        "sample_id": "s0",
        "adapter": "rs_vqa",
        "task": "single_vqa",
        "images": ["a.png"],
        "question": "Is there water?",
        "answer": "yes",
        "split": "train",
    }
    row.update(overrides)
    return row


def _write(tmp_path, rows, name="corpus.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_max_pixels_comes_from_the_frozen_contract_not_a_default():
    cfg = TrainingConfig.from_frozen()
    assert cfg.max_pixels == preprocessing_config().tiling.max_pixels
    assert cfg.frozen_max_pixels
    assert "configs/preprocessing.yaml" in cfg.provenance()


def test_an_override_is_recorded_and_never_reads_as_frozen():
    cfg = TrainingConfig.from_frozen(max_pixels=65536)
    assert cfg.max_pixels == 65536
    assert not cfg.frozen_max_pixels
    assert cfg.overrides["max_pixels"]["frozen"] == preprocessing_config().tiling.max_pixels
    assert "OVERRIDDEN" in cfg.provenance()


def test_view_counts_are_per_sample_not_per_date():
    """The count is views per *sample*, and the adapters do not agree on it.

    ``rs_ground_caption`` is two because its sub-metre sources are RGB and the
    deployment sensor has no SWIR. ``change_vqa`` is six -- three composites at
    each of two dates -- and that is the value the whole task depends on: at
    three, the loader would hand the model a single date and it would learn the
    answer prior instead of the change. Section 5.5 prices it at 1,536 vision
    tokens per sample, which is six views, not three.
    """
    assert adapter_composites("rs_ground_caption") == 2
    assert adapter_composites("change_vqa") == 6
    assert adapter_composites("rs_vqa") == 3
    assert adapter_composites("optsar_fusion") == 3
    # Every adapter must have an explicit entry rather than inheriting a default.
    assert all(adapter_composites(a) > 0 for a in ADAPTERS)


def test_unknown_adapter_raises_rather_than_defaulting_to_three():
    with pytest.raises(ValueError, match="unknown adapter"):
        adapter_composites("rs_vqa_v2")


def test_max_pixels_is_a_cap_not_a_target():
    # Qwen3-VL never upsamples: a 120 px BEN patch costs 16 tokens whatever the
    # cap says. Pricing it at max_pixels/1024 is what overstated the budget.
    cfg = TrainingConfig.from_frozen(max_pixels=262144)
    assert cfg.vision_tokens_per_image == 256
    assert cfg.vision_tokens_for(120) == 14
    assert cfg.vision_tokens_for(2048) == 256


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def test_manifest_row_missing_a_required_field_raises():
    with pytest.raises(ValueError, match="missing required field"):
        CanonicalSample.from_row({"sample_id": "s0", "images": ["a.png"]})


def test_a_malformed_line_is_an_error_not_a_silent_skip(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_row()) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_canonical_manifest(path)


def test_an_empty_filter_result_is_an_error_not_an_empty_loader(tmp_path):
    path = _write(tmp_path, [_row()])
    with pytest.raises(ValueError, match="no samples"):
        load_canonical_manifest(path, adapter="change_vqa")


def test_filters_select_by_adapter_and_split(tmp_path):
    path = _write(
        tmp_path,
        [
            _row(sample_id="a"),
            _row(sample_id="b", adapter="change_vqa"),
            _row(sample_id="c", split="test"),
        ],
    )
    assert [s.sample_id for s in load_canonical_manifest(path, adapter="rs_vqa")] == ["a", "c"]
    assert [s.sample_id for s in load_canonical_manifest(path, split="test")] == ["c"]


def test_windows_backslash_paths_resolve(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.png").write_bytes(b"")
    sample = CanonicalSample.from_row(_row(images=["sub\\a.png"]))
    assert resolve_sample_images(sample, tmp_path) == [tmp_path / "sub" / "a.png"]


def test_a_missing_image_names_the_drift_rather_than_returning_a_path(tmp_path):
    sample = CanonicalSample.from_row(_row())
    with pytest.raises(FileNotFoundError, match="drifted"):
        resolve_sample_images(sample, tmp_path)


def test_gsd_prefix_uses_the_effective_gsd_and_is_empty_when_absent():
    # RSVQA-HR is 0.15 m natively and enters training at 0.30 m. Conditioning on
    # 0.15 would teach the model a scale its pixels do not have.
    downsampled = CanonicalSample.from_row(_row(effective_gsd_m=[0.30]))
    assert gsd_prompt_prefix(downsampled) == "[ground sample distance: 0.3 m] "
    assert gsd_prompt_prefix(CanonicalSample.from_row(_row())) == ""


def test_gsd_prefix_lists_every_view_when_they_differ():
    sample = CanonicalSample.from_row(_row(effective_gsd_m=[0.5, 10.0]))
    assert gsd_prompt_prefix(sample) == "[ground sample distance per view: 0.5 m, 10 m] "


def test_dataset_refuses_an_empty_sample_list(tmp_path):
    with pytest.raises(ValueError, match="at least one sample"):
        RealChipDataset([], root=tmp_path)


def test_composite_count_repeats_rather_than_shortening_the_sequence(tmp_path):
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    dataset = RealChipDataset(
        [CanonicalSample.from_row(_row())], root=tmp_path, composites=3
    )
    assert len(dataset[0]["images"]) == 3


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_normalisation_strips_articles_and_punctuation_but_not_meaning():
    assert normalise_answer("The Yes.") == "yes"
    assert normalise_answer("  no,  ") == "no"
    # Numbers keep their decimal point: 12.5 must not become 12 5.
    assert normalise_answer("12.5") == "12.5"


def test_average_accuracy_is_the_mean_of_per_type_not_the_overall():
    # 9 of 10 "presence" right, 0 of 2 "count" right.
    predictions = ["yes"] * 9 + ["no"] + ["1", "1"]
    references = ["yes"] * 10 + ["7", "8"]
    types = ["presence"] * 10 + ["count"] * 2
    report = score_predictions(predictions, references, types)
    assert report.overall_accuracy == pytest.approx(9 / 12)
    assert report.average_accuracy == pytest.approx((0.9 + 0.0) / 2)


def test_misaligned_prediction_and_reference_lists_raise():
    with pytest.raises(ValueError, match="not aligned"):
        score_predictions(["a"], ["a", "b"])


def test_empty_report_does_not_divide_by_zero():
    assert ScoreReport().average_accuracy == 0.0
    assert ScoreReport().overall_accuracy == 0.0


def test_box_iou_is_zero_for_disjoint_boxes_and_one_for_identical():
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == pytest.approx(1.0)
    assert box_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_an_unparsable_box_counts_as_a_miss_rather_than_being_dropped():
    # Dropping them is how a grounding number ends up describing only the
    # samples the model found easy.
    result = grounding_accuracy([None, (0, 0, 2, 2)], [(0, 0, 2, 2), (0, 0, 2, 2)])
    assert result["acc@0.5"] == pytest.approx(0.5)
    assert result["unparsable"] == pytest.approx(0.5)


def test_every_report_carries_the_official_scorer_caveat():
    assert "NOT the official" in score_predictions(["a"], ["a"]).as_dict()["caveat"]


# --------------------------------------------------------------------------
# composites
# --------------------------------------------------------------------------


def test_the_three_composites_are_the_ones_section_4_2_names():
    assert COMPOSITES["true_colour"] == ("B04", "B03", "B02")
    assert COMPOSITES["false_colour"] == ("B08", "B04", "B03")
    assert COMPOSITES["short_wave"] == ("B12", "B11", "B08")


def test_dropping_the_third_composite_drops_the_short_wave_one_specifically():
    assert composite_names_for("rs_ground_caption") == ("true_colour", "false_colour")
    assert "short_wave" in composite_names_for("rs_vqa")


def _bands(size=120):
    rng = np.random.default_rng(0)
    bands = {b: rng.integers(0, 3000, (size, size)) for b in ("B02", "B03", "B04", "B08")}
    bands.update({b: rng.integers(0, 3000, (size // 2, size // 2)) for b in ("B11", "B12")})
    return bands


def test_swir_bands_are_upsampled_onto_the_ten_metre_grid():
    built = {c.name: c for c in build_composites(_bands())}
    assert built["short_wave"].array.shape == (120, 120, 3)
    assert built["short_wave"].array.dtype == np.uint8


def test_a_missing_band_raises_and_names_what_to_fetch():
    bands = _bands()
    del bands["B12"]
    with pytest.raises(ValueError, match="B12"):
        build_composites(bands)
    assert "B12" in REQUIRED_BANDS


def test_a_degenerate_band_maps_to_mid_grey_instead_of_dividing_by_zero():
    flat, cuts = stretch(np.full((4, 4), 500.0))
    assert (flat == 128).all()
    assert cuts == (500.0, 500.0)


def test_the_stretch_records_the_cut_values_it_used():
    # A stretch is lossy and irreversible; one that is not written down cannot
    # be reproduced or compared across runs.
    composite = build_composites(_bands(), names=("true_colour",))[0]
    assert composite.percentiles == (2.0, 98.0)
    assert len(composite.cut_values) == 3
    assert all(low < high for low, high in composite.cut_values)


def _view_manifest(tmp_path, n_views: int, adapter: str = "change_vqa") -> str:
    from PIL import Image

    for i in range(n_views):
        Image.new("RGB", (16, 16)).save(tmp_path / f"v{i}.png")
    manifest = tmp_path / f"{adapter}_{n_views}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": f"{adapter}_{n_views}",
                "adapter": adapter,
                "task": "single_vqa",
                "images": [f"v{i}.png" for i in range(n_views)],
                "question": "did anything change?",
                "answer": "yes",
                "effective_gsd_m": [10.0] * n_views,
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return str(manifest)


def test_a_bitemporal_row_keeps_both_dates(tmp_path):
    """Six views in, six views out -- the second date must survive the loader.

    Truncating to three kept only date 0. Training ran without error, the loss
    fell, and the model answered change questions from the answer prior, because
    it had never been shown a second image.
    """
    from satquery.training.dataset import RealChipDataset, load_canonical_manifest

    manifest = _view_manifest(tmp_path, 6)
    dataset = RealChipDataset(
        load_canonical_manifest(manifest),
        root=tmp_path,
        composites=adapter_composites("change_vqa"),
    )
    assert len(dataset[0]["images"]) == 6


def test_a_real_multi_view_row_passes_through_with_its_own_count(tmp_path):
    """A row carrying several real views is its source's shape, not a bug.

    ``change_vqa`` draws from two sources with different view counts: SpaceNet 7
    is RGB-only Planet, so one composite per date and two views a sample, while
    the self-generated Sentinel tiles carry true-colour, false-colour and
    short-wave at each of two dates, so six. Forcing both to the adapter's
    number would either truncate Sentinel to a single date or pad SpaceNet 7
    with duplicates, so ``composites`` stays the sequence-length budget it has
    always been and these rows pass through intact.
    """
    from satquery.training.dataset import RealChipDataset, load_canonical_manifest

    manifest = _view_manifest(tmp_path, 4)
    dataset = RealChipDataset(
        load_canonical_manifest(manifest),
        root=tmp_path,
        composites=adapter_composites("change_vqa"),
    )
    assert len(dataset[0]["images"]) == 4


def test_views_beyond_the_ceiling_still_raise(tmp_path):
    """The budgeting ceiling is the one view count that is genuinely an error.

    Truncating is the dangerous direction and is still never silent: past
    ``MAX_VIEWS`` a single sample's image tokens dominate the sequence and the
    batch size measured for the adapter no longer holds, so this raises rather
    than quietly dropping the tail.
    """
    from satquery.training.dataset import (
        MAX_VIEWS,
        RealChipDataset,
        load_canonical_manifest,
    )

    manifest = _view_manifest(tmp_path, MAX_VIEWS + 1)
    dataset = RealChipDataset(
        load_canonical_manifest(manifest),
        root=tmp_path,
        composites=adapter_composites("change_vqa"),
    )
    with pytest.raises(ValueError, match=f"carries {MAX_VIEWS + 1} view"):
        dataset[0]


def test_a_single_view_is_still_repeated_because_the_corpus_does_that(tmp_path):
    """RSVQA rows carry one image and were trained repeated to three."""
    from satquery.training.dataset import RealChipDataset, load_canonical_manifest

    manifest = _view_manifest(tmp_path, 1, adapter="rs_vqa")
    dataset = RealChipDataset(
        load_canonical_manifest(manifest),
        root=tmp_path,
        composites=adapter_composites("rs_vqa"),
    )
    assert len(dataset[0]["images"]) == 3


def test_resize_views_is_an_explicit_opt_in_for_measurement(tmp_path):
    """The timing sweep and the composites ablation vary the count on purpose."""
    from satquery.training.dataset import RealChipDataset, load_canonical_manifest

    manifest = _view_manifest(tmp_path, 6, adapter="rs_vqa")
    samples = load_canonical_manifest(manifest)
    assert (
        len(
            RealChipDataset(
                samples, root=tmp_path, composites=3, resize_views=True
            )[0]["images"]
        )
        == 3
    )


def test_the_demo_and_training_build_the_same_prompt(tmp_path):
    """One prompt assembler, or the demo measures something training never saw.

    The demo endpoint had its own copy of the scale prefix, including its own
    rule for which size thresholds apply. The two agreed on the day they were
    written and nothing enforced it afterwards -- so a change to the training
    thresholds would have left the demo showing a prompt the model was never
    trained on, which is indistinguishable from the model being bad.
    """
    import sys

    from PIL import Image

    from satquery.training.dataset import CanonicalSample, gsd_prompt_prefix

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ask_server import _source_for

    image = Image.new("RGB", (512, 512))
    question = "Is a small building present?"
    gsd = 0.3048

    trained = gsd_prompt_prefix(
        CanonicalSample(
            sample_id="s",
            adapter="rs_vqa",
            task="single_vqa",
            images=["a.png"],
            question=question,
            answer="yes",
            effective_gsd_m=[gsd],
            source="RSVQA-HR (Zenodo 6344367)",
        ),
        [image],
    )

    from satquery.training.dataset import scale_prefix

    served = scale_prefix([gsd], question, _source_for(gsd), [image])

    assert trained == served
    assert "small <100 m2" in served
