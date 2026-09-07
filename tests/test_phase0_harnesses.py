"""The Phase 0 evaluation harnesses, and the two scripts that feed them.

These replaced a scaffold that printed "Result: Pending ML Weights" for all four
ablations and could not fail. The tests here pin the properties that make the
replacements capable of failing:

* a harness refuses to run rather than reporting a result it did not compute;
* an arm that was never exercised is reported as unexercised, not as a pass;
* scoring numbers use a tolerance where the answer is numeric and exact match
  where it is not;
* a patch identifier that does not parse stops the manifest build, because the
  fetcher derives all its geometry from that string.

None of these tests loads a model. Every one of them exercises a code path that
runs before a model would be loaded, which is where the harnesses' decisions
actually live.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_ben_manifest import build as build_ben_manifest  # noqa: E402
from scripts.stage_benchmarks import SPECS  # noqa: E402
from training.eval.router_zero_shot import (  # noqa: E402
    ARMS,
    UNAMBIGUOUS_GATE,
    LLMTaskClassifier,
    _score,
    _verdict,
    parse_task,
    run_hybrid,
    run_rules,
)
from training.eval.run_ablations import (  # noqa: E402
    EXPERIMENTS,
    HANDLERS,
    first_number,
    numeric_accuracy,
)
from training.eval.score_routing import load_dataset  # noqa: E402
from training.eval.zero_shot import (  # noqa: E402
    CANDIDATE_BASES,
    _d2_framing,
    _parse_box,
)


@pytest.fixture(scope="module")
def routing_rows():
    return load_dataset()


# --------------------------------------------------------------------------
# item 9 / item 10 — the bake-off and D2
# --------------------------------------------------------------------------


def test_the_base_model_is_committed_and_the_2b_is_gone():
    """Committed 2026-08-29. The measured budget depends on this exact model."""
    assert CANDIDATE_BASES == ("Qwen/Qwen3-VL-4B-Instruct",)
    assert not any("3.5-2B" in name for name in CANDIDATE_BASES)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[10, 20, 30, 40]", (10.0, 20.0, 30.0, 40.0)),
        ("the box is 0.1 0.2 0.5 0.9 roughly", (0.1, 0.2, 0.5, 0.9)),
        ("I cannot tell", None),
        ("1 2 3", None),
    ],
)
def test_box_parsing_returns_none_rather_than_guessing(text, expected):
    assert _parse_box(text) == expected


def test_a_degenerate_box_is_rejected_rather_than_scored():
    # x2 <= x1 is not a box. Scoring it would produce a real IoU number from a
    # shape that does not exist.
    assert _parse_box("30 40 10 20") is None
    assert _parse_box("10 10 10 10") is None


def test_d2_framing_says_explainability_when_the_prior_is_neutral():
    neutral = _d2_framing(0.001)
    assert "explainability differentiator" in neutral
    assert "must not claim an accuracy gain" in neutral


def test_d2_framing_says_evidence_only_when_the_prior_hurts():
    assert "HURTS" in _d2_framing(-0.05)
    assert "do not inject it into prompts" in _d2_framing(-0.05)


def test_d2_framing_says_differentiator_when_the_prior_helps():
    assert "accuracy differentiator" in _d2_framing(0.05)


# --------------------------------------------------------------------------
# item 11 — the router arms
# --------------------------------------------------------------------------


def test_the_task_parser_prefers_the_longest_match():
    # "crossmodal_vqa" contains "vqa"; a shortest-match scan would mislabel it.
    assert parse_task("crossmodal_vqa") == "crossmodal_vqa"
    assert parse_task("  Change_VQA\n") == "change_vqa"
    assert parse_task("single_grounding please") == "single_grounding"


def test_an_unrecognised_answer_is_none_rather_than_the_commonest_class():
    # Defaulting to single_vqa would score as correct on the majority class and
    # make the LLM arm look better than it is.
    assert parse_task("I am not sure") is None
    assert parse_task("") is None


def test_the_rules_arm_still_clears_the_risk_register_trigger(routing_rows):
    report = run_rules(routing_rows)
    assert report["unambiguous_accuracy"] >= UNAMBIGUOUS_GATE
    assert report["n"] == len(routing_rows)


def test_the_rules_arm_reports_how_often_it_defers(routing_rows):
    report = run_rules(routing_rows)
    assert 0.0 <= report["defer_rate"] <= 1.0
    assert report["deferred_to_llm"] == round(report["defer_rate"] * report["n"])


def test_hybrid_without_an_llm_equals_rules_and_says_the_stage_was_untested(routing_rows):
    rules = run_rules(routing_rows)
    hybrid = run_hybrid(routing_rows, rules, None)
    assert hybrid["task_accuracy"] == rules["task_accuracy"]
    assert hybrid["llm_substitutions"] == 0

    verdict = _verdict({"arms": {"rules": rules, "hybrid": hybrid}})
    assert "never consulted" in verdict


def test_an_llm_that_returns_nothing_does_not_erase_the_rules_answer(routing_rows):
    rules = run_rules(routing_rows)
    blind = {"predictions": [None] * len(routing_rows)}
    hybrid = run_hybrid(routing_rows, rules, blind)
    assert hybrid["predictions"] == rules["predictions"]
    assert hybrid["llm_substitutions"] == 0


def test_the_verdict_calls_for_deleting_stage_two_when_it_hurts():
    rows_scored = {"task_accuracy": 1.0, "unambiguous_accuracy": 1.0, "unambiguous_n": 285}
    hybrid = {**rows_scored, "unambiguous_accuracy": 0.9, "llm_substitutions": 15}
    verdict = _verdict({"arms": {"rules": rows_scored, "hybrid": hybrid}})
    assert "Delete stage 2" in verdict


def test_a_failing_rules_arm_names_the_week_one_trigger():
    failing = {"task_accuracy": 0.5, "unambiguous_accuracy": 0.5, "unambiguous_n": 285}
    assert "Fix the rules now" in _verdict({"arms": {"rules": failing}})


def test_the_llm_prompt_carries_the_context_the_router_actually_has(routing_rows):
    prompt = LLMTaskClassifier.prompt_for(routing_rows[0])
    assert routing_rows[0]["query_text"] in prompt
    assert "Images available:" in prompt
    # Routing happens before ingestion, so the prompt must be text-only.
    assert "<image>" not in prompt


def test_scoring_counts_unparsable_predictions_as_wrong(routing_rows):
    rows = routing_rows[:10]
    report = _score(rows, [None] * len(rows))
    assert report["task_accuracy"] == 0.0
    assert report["unparsable"] == len(rows)


def test_every_named_arm_has_a_runner():
    assert set(ARMS) == {"rules", "llm", "hybrid"}


# --------------------------------------------------------------------------
# the ablation dispatcher
# --------------------------------------------------------------------------


def test_every_experiment_dispatches_to_a_real_handler():
    assert set(EXPERIMENTS) == set(HANDLERS)
    assert all(callable(handler) for handler in HANDLERS.values())


def test_numeric_answers_are_scored_with_a_tolerance_not_exact_match():
    result = numeric_accuracy(["12.0 km2"], ["12 km2"])
    assert result["accuracy"] == 1.0
    assert result["median_relative_error"] == 0.0


def test_a_number_outside_tolerance_is_wrong():
    assert numeric_accuracy(["20"], ["12"])["accuracy"] == 0.0


def test_an_answer_with_no_number_counts_as_wrong_and_is_reported_separately():
    result = numeric_accuracy(["I cannot tell", "12"], ["12", "12"])
    assert result["accuracy"] == 0.5
    assert result["unparsable"] == 1


def test_a_reference_with_no_number_raises_rather_than_scoring_zero():
    with pytest.raises(ValueError, match="cannot be scored numerically"):
        numeric_accuracy(["12"], ["quite a lot"])


def test_first_number_handles_negatives_and_decimals():
    assert first_number("delta -3.5 m2") == -3.5
    assert first_number("none") is None


def test_tool_vs_vlm_math_refuses_a_manifest_with_no_masks(tmp_path):
    import argparse

    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "s0",
                "adapter": "change_vqa",
                "task": "change_vqa",
                "images": ["a.png"],
                "question": "how much changed?",
                "answer": "12 m2",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=str(manifest),
        split="test",
        limit=0,
        image_root=str(tmp_path),
        model="unused",
        adapter=None,
        batch_size=1,
        max_pixels=0,
    )
    with pytest.raises(SystemExit, match="mask_t1"):
        HANDLERS["tool_vs_vlm_math"](args)


# --------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------


VALID_PATCH = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"


def test_the_ben_manifest_groups_questions_by_patch():
    rows = [
        {"patch_id": VALID_PATCH, "question": "q1", "answer": "yes"},
        {"patch_id": VALID_PATCH, "question": "q2", "answer": "no"},
    ]
    entries, stats = build_ben_manifest(rows, adapter="rs_vqa", split="test")
    assert stats == {
        **stats,
        "patches": 1,
        "qa_pairs": 2,
        "tiles": 1,
        "granule_fetches": 1,
    }
    assert entries[0]["tile"] == "T33UUP"
    assert entries[0]["effective_gsd_m"] == 10.0


def test_an_unparsable_patch_id_stops_the_manifest_build():
    rows = [{"patch_id": "not_a_patch", "question": "q", "answer": "a"}]
    with pytest.raises(SystemExit, match="do not parse"):
        build_ben_manifest(rows, adapter="rs_vqa", split="test")


def test_the_manifest_carries_no_image_paths():
    # Canonical rows carry images; this manifest cannot, because no image has
    # been fetched yet. Emitting paths here is how a corpus goes half-real.
    entries, _ = build_ben_manifest(
        [{"patch_id": VALID_PATCH, "question": "q", "answer": "a"}],
        adapter="rs_vqa",
        split="test",
    )
    assert "images" not in entries[0]
    assert entries[0]["bands_required"]


def test_rsvqa_hr_is_staged_at_the_downsampled_gsd():
    spec = SPECS["rsvqa_hr"]
    assert spec.native_gsd_m == 0.15
    assert spec.downsample == 2
    assert spec.effective_gsd_m == 0.30
    assert "0.15 m -> 0.3 m" in spec.note()


def test_rsvqa_lr_is_staged_as_shipped():
    assert SPECS["rsvqa_lr"].downsample == 1
    assert SPECS["rsvqa_lr"].note() == "staged as shipped"


def test_eval_sets_are_never_resampled():
    # The policy is explicit: eval preprocessing must match training per-source,
    # and a set someone has resampled is no longer comparable to published
    # numbers.
    for key in ("cdvqa", "vrsbench"):
        assert SPECS[key].eval_only
        assert SPECS[key].downsample == 1


def test_vrsbench_is_forbidden_from_training():
    assert SPECS["vrsbench"].train_forbidden
    assert "EVALUATION ONLY" in SPECS["vrsbench"].licence


# --------------------------------------------------------------------------
# item 7 — serving
# --------------------------------------------------------------------------


def test_the_serving_smoke_reports_a_missing_vllm_as_a_failure_not_a_pass():
    pytest.importorskip  # noqa: B018 - documents the intent below
    from satquery.serving.vllm_smoke import run_smoke

    try:
        import vllm  # noqa: F401
    except ImportError:
        report = run_smoke("Qwen/Qwen3-VL-4B-Instruct", {})
        assert not report.passed
        assert "not installed" in report.failure
    else:  # pragma: no cover - only on a machine with the serving stack
        pytest.skip("vLLM is installed; this asserts the absent-dependency path")


def test_the_smoke_prompt_is_gsd_conditioned():
    from satquery.serving.vllm_smoke import SMOKE_PROMPT

    assert "ground sample distance" in SMOKE_PROMPT


# --------------------------------------------------------------------------
# the clubbed sweep — one model load for every experiment
# --------------------------------------------------------------------------


def test_the_sweep_is_offered_as_an_experiment_choice():
    import argparse as _argparse
    import contextlib
    import io

    from training.eval import run_ablations

    # --list exits 0 without touching a model; the parser is what we want to see.
    parser_choices = None
    original = _argparse.ArgumentParser.add_argument

    def capture(self, *args, **kwargs):
        nonlocal parser_choices
        if args and args[0] == "--experiment":
            parser_choices = kwargs.get("choices")
        return original(self, *args, **kwargs)

    _argparse.ArgumentParser.add_argument = capture
    try:
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            run_ablations.main()
    finally:
        _argparse.ArgumentParser.add_argument = original

    assert parser_choices is not None
    assert "all" in parser_choices
    assert set(EXPERIMENTS) <= set(parser_choices)


def test_the_model_is_loaded_once_and_reused_across_experiments():
    """Four handlers sharing one model is the entire point of the sweep."""
    import argparse

    from training.eval.run_ablations import _runner

    sentinel = object()
    args = argparse.Namespace(
        model="unused", adapter=None, max_pixels=0, _shared_runner=sentinel
    )
    assert _runner(args) is sentinel
    assert _runner(args) is sentinel


def test_d2_accepts_a_runner_handed_to_it_rather_than_loading_its_own():
    # point_prior delegates to run_d2 across a module boundary; without this the
    # sweep pays a second 8 GB load for the experiment it shares a model with.
    import inspect

    from training.eval import zero_shot

    source = inspect.getsource(zero_shot.run_d2)
    assert "_shared_runner" in source


def test_point_prior_forwards_the_shared_runner():
    import inspect

    from training.eval.run_ablations import run_point_prior

    assert "_shared_runner" in inspect.getsource(run_point_prior)


def test_a_train_manifest_of_reben_test_patches_is_refused():
    """The split label on a derived subset is not evidence.

    `ben-micro-split/train_metadata.jsonl` presents 21 patches as training data.
    All 21 are in reBEN's *test* split. Building a train manifest from them
    yields a corpus that trains on the benchmark's own evaluation set, and every
    BEN.txt number reported afterwards would be inflated with no crash, no
    warning and no symptom. It has to be a hard stop.
    """
    from scripts.build_ben_manifest import official_splits

    if official_splits().get(VALID_PATCH) != "test":
        pytest.skip("reBEN metadata.parquet not staged; nothing to check against")

    rows = [{"patch_id": VALID_PATCH, "question": "q", "answer": "a"}]
    with pytest.raises(SystemExit, match="not in reBEN's 'train' split"):
        build_ben_manifest(rows, adapter="rs_vqa", split="train")


def test_the_override_exists_but_must_be_asked_for():
    rows = [{"patch_id": VALID_PATCH, "question": "q", "answer": "a"}]
    entries, _ = build_ben_manifest(
        rows, adapter="rs_vqa", split="train", enforce_official_split=False
    )
    assert entries and entries[0]["split"] == "train"


def test_a_composites_comparison_between_two_floor_scores_is_refused():
    """Two unusable numbers do not make a result.

    The first real run scored 0.006 and 0.004 average accuracy on yes/no
    questions, where guessing scores ~0.5, and the harness reported a confident
    "drop to two composites" from the -0.002 difference. Zero-shot Qwen answers
    a yes/no question with a sentence, so neither arm was answering the question
    set at all.
    """
    import argparse

    from training.eval import run_ablations

    calls = {"n": 0}

    def fake_arm(runner, dataset, batch_size):
        calls["n"] += 1
        return {"average_accuracy": 0.006 if calls["n"] == 1 else 0.004, "samples": 500}

    original_arm = run_ablations._score_arm
    original_runner = run_ablations._runner
    original_dataset = run_ablations._dataset
    run_ablations._score_arm = fake_arm
    run_ablations._runner = lambda args: object()
    run_ablations._dataset = lambda args, composites=None: []
    try:
        report = run_ablations.run_optical_composites(
            argparse.Namespace(
                manifest="unused", image_root=None, split="test", limit=0,
                model="unused", adapter=None, batch_size=1, max_pixels=0,
            )
        )
    finally:
        run_ablations._score_arm = original_arm
        run_ablations._runner = original_runner
        run_ablations._dataset = original_dataset

    assert "NO VERDICT" in report["verdict"]
    assert "drop to two" not in report["verdict"].lower()
