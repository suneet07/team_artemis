"""The answer formatter that the headless mode — and therefore a judge — sees.

Master plan section 6.4 requires this layer be unit-tested rather than trusted.
It had no tests, which is how three of the five RSVQA-HR area class strings sat
wrong (``between 0m2 and 10m2`` for ``between 1m2 and 10m2``, and the same
off-by-one at the 100 and 1000 boundaries) in the path a judge grades. Under
exact match that scored a correct answer as wrong on the majority of the
vocabulary.

Every vocabulary asserted here was read off the staged eval manifests: BEN.txt
binary is yes/no and its MCQ answers are bare letters a-d, RSVQA-LR adds
rural/urban, and RSVQA-HR's area answers are the paper's five classes.

The abstention cases matter as much as the recoveries. Serving must always emit
something scoreable, so an unrecoverable answer falls back to a class *and*
reports ``matched=False`` — the flag is what keeps a guess from being counted as
instruction-following.
"""

import pytest

from satquery.evalcli.formatter import (
    BENCHMARK_FORMATS,
    contract_for,
    format_answer,
    format_mcq,
)

MCQ_QUESTION = (
    "Which class covers the most area? "
    "a) water b) industrial or commercial units c) forest d) arable land"
)


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("yes", "yes"),
        ("no", "no"),
        ("Yes.", "yes"),
        ("Yes, there is a small building in the lower left.", "yes"),
        ("No, none is visible.", "no"),
        # Negation is checked first: this sentence contains "there are".
        ("There are no buildings in this image.", "no"),
    ],
)
def test_binary_recovers_the_class(prediction, expected):
    result = format_answer(prediction, "ben_binary_vqa")
    assert result.matched
    assert result.text == expected


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("b", "b"),
        ("B", "b"),
        ("(c)", "c"),
        ("d.", "d"),
        ("b) industrial or commercial units", "b"),
        # Named the option rather than lettering it.
        ("Industrial or commercial units.", "b"),
        ("arable land", "d"),
    ],
)
def test_mcq_recovers_the_letter(prediction, expected):
    result = format_answer(prediction, "ben_mcq", question=MCQ_QUESTION)
    assert result.matched
    assert result.text == expected


def test_mcq_does_not_read_a_leading_article_as_option_a():
    """"a large building" is prose, not option (a) — the classic false positive."""
    result = format_answer(
        "a large building is visible", "ben_mcq", question=MCQ_QUESTION
    )
    assert not result.matched


def test_mcq_option_text_needs_the_whole_phrase():
    """One word shared between two options must not decide which was meant."""
    question = "Pick one. a) broad-leaved forest b) mixed forest"
    assert not format_mcq("forest", question).matched


def test_mcq_abstains_rather_than_emitting_nothing():
    """Serving must always produce a scoreable answer, flagged as a guess."""
    result = format_answer("I cannot tell", "ben_mcq", question=MCQ_QUESTION)
    assert result.text in BENCHMARK_FORMATS["ben_mcq"].vocabulary
    assert not result.matched


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("0m2", "0m2"),
        ("more than 1000m2", "more than 1000m2"),
        ("between 11m2 and 100m2", "between 11m2 and 100m2"),
        ("between 101m2 and 1000m2", "between 101m2 and 1000m2"),
        # A raw magnitude is an answer in the wrong wording, not a wrong answer.
        ("10631m2", "more than 1000m2"),
        ("The area is 55 m2.", "between 11m2 and 100m2"),
    ],
)
def test_area_lands_in_a_real_class(prediction, expected):
    result = format_answer(prediction, "rsvqa_hr_area")
    assert result.matched
    assert result.text == expected
    assert result.text in BENCHMARK_FORMATS["rsvqa_hr_area"].vocabulary


def test_area_vocabulary_matches_the_paper():
    """The boundaries are 1/11/101, not 0/10/100 — the bug this file exists for."""
    assert BENCHMARK_FORMATS["rsvqa_hr_area"].vocabulary == (
        "0m2",
        "between 1m2 and 10m2",
        "between 11m2 and 100m2",
        "between 101m2 and 1000m2",
        "more than 1000m2",
    )


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [("0", "0"), ("14", "14"), ("There are 14 buildings.", "14"), ("2392", "2392")],
)
def test_hr_count_stays_exact(prediction, expected):
    """RSVQA-HR counts are not quantised: the paper keeps them exact (max 89)."""
    result = format_answer(prediction, "rsvqa_hr_count")
    assert result.matched
    assert result.text == expected


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("0", "0"),
        ("7", "between 1 and 10"),
        ("47", "between 11 and 100"),
        ("403", "between 101 and 1000"),
        ("2392", "more than 1000"),
        ("There are 257 buildings.", "between 101 and 1000"),
        ("more than 1000", "more than 1000"),
    ],
)
def test_lr_count_is_quantised(prediction, expected):
    """RSVQA-LR counts are five classes, per Lobry et al. 2020.

    The two splits differing is the point: LR tiles cover 6.55 km2 and can hold
    thousands of objects, HR tiles top out at 89. Applying one rule to both
    would score one of them against a task the benchmark never set.
    """
    result = format_answer(prediction, "rsvqa_lr_count")
    assert result.matched
    assert result.text == expected


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [("urban", "urban"), ("This is a rural scene.", "rural")],
)
def test_rural_urban(prediction, expected):
    assert format_answer(prediction, "rsvqa_lr_rural_urban").text == expected


@pytest.mark.parametrize(
    ("benchmark", "question_type", "expected"),
    [
        ("ben", "binary_presence", "ben_binary_vqa"),
        ("ben", "mcq_area", "ben_mcq"),
        ("ben", "captioning_caption", "ben_caption"),
        ("rsvqa_lr", "presence", "rsvqa_lr_presence"),
        ("rsvqa_lr", "comp", "rsvqa_lr_comparison"),
        ("rsvqa_lr", "rural_urban", "rsvqa_lr_rural_urban"),
        ("rsvqa_hr", "comp", "rsvqa_hr_comparison"),
        ("rsvqa_hr", "area", "rsvqa_hr_area"),
        ("rsvqa_hr", "count", "rsvqa_hr_count"),
    ],
)
def test_every_scored_question_type_has_a_contract(
    benchmark, question_type, expected
):
    """A type with no contract is scored on raw model prose, which is the bug."""
    name = contract_for(benchmark, question_type)
    assert name == expected
    assert name in BENCHMARK_FORMATS


def test_unknown_benchmark_is_refused_not_guessed():
    with pytest.raises(KeyError):
        format_answer("yes", "not_a_benchmark")
