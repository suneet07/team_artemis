"""Per-benchmark answer formatting (master plan section 6.4).

*"The water body covers approximately 3.4 km²" scores zero where the scorer wants
"yes". Format compliance is worth more points than a better model, and it costs
two days.*

The three helpers here are the thin API the rest of the agent uses. The closed
vocabularies, the CDVQA ratio buckets and the RSVQA area buckets live in
:mod:`satquery.evalcli.formatter`, which is where the unit tests pin them
against each benchmark's published examples.

The substring bug this replaces was not cosmetic: ``format_yes_no`` matched
``"no"`` anywhere in the string, so "there is a road to the **no**rth", "**no**ne
of the fields", "we can**no**t tell" and "we k**no**w" all scored ``no``. On a
binary VQA benchmark that is a silent, systematic wrong answer on exactly the
descriptive phrasings a captioning-tuned model produces.
"""

import re

from satquery.evalcli.formatter import format_ratio
from satquery.evalcli.formatter import format_yes_no as _yes_no

__all__ = ["format_class_name", "format_number", "format_ratio", "format_yes_no"]


def format_yes_no(answer: str) -> str:
    """Constrain an arbitrary string to a canonical 'yes' or 'no'.

    Negation is tested before affirmation and on word boundaries: "there are no
    buildings" contains "there are", and a first-match scan gets it backwards.
    """
    return _yes_no(answer).text


def format_number(answer: str) -> str:
    """Extract the first number, preserving decimals."""
    match = re.search(r"[-+]?\d*\.\d+|\d+", answer)
    return match.group(0) if match else "0"


def format_class_name(answer: str, allowed_classes: list[str]) -> str:
    """Map to the nearest canonical class name.

    Longest candidate first, so "no change" wins over "no" -- the CDVQA class
    list contains both, and matching in declaration order returned the wrong one.
    """
    lowered = answer.lower()
    for candidate in sorted(allowed_classes, key=len, reverse=True):
        if candidate.lower() in lowered:
            return candidate
    return answer.strip()
