"""CDVQA answer contracts, and the normalisation that makes them usable.

CDVQA is the organiser-nominated change benchmark and its answers are a closed
vocabulary of nineteen values. Scoring them wrong does not raise -- it reports a
capable model as incapable, which is the most expensive kind of quiet failure
this project has: the record already names it "the CDVQA vocabulary problem ...
format compliance, not capability".

Three bugs these pin, all of which were live:

1. The vocabularies were written from the paper's prose. ``change_ratio`` was
   scored against ("no", "very small", "small", ...) while the data answers
   "0_to_10" and "80_to_90"; ``largest_change`` against "low vegetation" while
   the data says "low_vegetation" and "NVG_surface". Every correct answer would
   have been marked wrong.
2. ``_CONTRACTS`` had no ``cdvqa`` rows at all, so ``contract_for`` returned
   None and none of the formats were reachable.
3. ``_match_vocabulary`` lowercased the model's text but compared it against the
   raw entries, so "NVG_surface" could never match -- and the miss *abstained*
   to the first vocabulary entry, manufacturing a wrong answer rather than
   recording an unparsable one.
"""

import pytest

from satquery.evalcli.formatter import (
    CDVQA_CLASSES,
    CDVQA_RATIO_BUCKETS,
    contract_for,
    format_answer,
)

CDVQA_TYPES = (
    "change_or_not",
    "change_ratio",
    "change_ratio_types",
    "change_to_what",
    "increase_or_not",
    "decrease_or_not",
    "largest_change",
    "smallest_change",
)


@pytest.mark.parametrize("question_type", CDVQA_TYPES)
def test_every_question_type_has_a_reachable_contract(question_type):
    contract = contract_for("cdvqa", question_type)
    assert contract is not None, f"{question_type} has no answer contract"
    assert contract.startswith("cdvqa_")


def test_ratio_types_is_not_swallowed_by_the_ratio_prefix():
    """``_CONTRACTS`` returns the first prefix match, and one is a prefix of the other.

    Listed the other way round, every ``change_ratio_types`` row would be scored
    against the plain ratio contract. They share a vocabulary today, so it would
    look fine and break the day they diverge.
    """
    assert contract_for("cdvqa", "change_ratio_types") == "cdvqa_change_ratio_types"
    assert contract_for("cdvqa", "change_ratio") == "cdvqa_change_ratio"


@pytest.mark.parametrize(
    ("raw", "question_type", "expected"),
    [
        ("Yes, the trees changed.", "change_or_not", "yes"),
        ("no", "change_or_not", "no"),
        # Spoken form of a bucket. The bare "0" is also a valid answer meaning
        # no change at all, so the longer band must win.
        ("about 0 to 10 percent", "change_ratio", "0_to_10"),
        ("0", "change_ratio", "0"),
        ("80_to_90", "change_ratio", "80_to_90"),
        # Underscores are presentation. A model writing them out has answered.
        ("low vegetation", "change_to_what", "low_vegetation"),
        ("NVG surface", "smallest_change", "NVG_surface"),
        # And the exact token, mixed case, inside a sentence.
        ("It changed to NVG_surface.", "largest_change", "NVG_surface"),
        ("playgrounds", "largest_change", "playgrounds"),
    ],
)
def test_answers_normalise_onto_the_released_vocabulary(raw, question_type, expected):
    formatted = format_answer(raw, contract_for("cdvqa", question_type))
    assert formatted.text == expected
    assert formatted.matched


def test_an_answer_outside_the_vocabulary_is_marked_unmatched():
    """Abstaining is fine; abstaining *silently* is not.

    The fallback returns the first vocabulary entry so downstream code has a
    string, but ``matched`` must be False or a refusal is scored as a wrong
    answer to a question the model declined.
    """
    formatted = format_answer("a helicopter landing pad", contract_for("cdvqa", "largest_change"))
    assert formatted.matched is False
    assert "outside the closed vocabulary" in formatted.basis


def test_vocabularies_match_the_released_annotations():
    """Read off the data, not the paper. "0" is distinct from "0_to_10"."""
    assert CDVQA_CLASSES == (
        "NVG_surface", "buildings", "low_vegetation", "trees", "water", "playgrounds",
    )
    assert CDVQA_RATIO_BUCKETS[0] == "0"
    assert "0_to_10" in CDVQA_RATIO_BUCKETS
    assert len(CDVQA_RATIO_BUCKETS) == 11
