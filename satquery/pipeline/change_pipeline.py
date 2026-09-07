"""Answer change questions by measurement instead of inference.

The trained VQA adapter scores 43.5% average accuracy on the held-out split and
gets **1.2 points** of its counting accuracy from actually reading the images --
measured by shuffling each question against the wrong image pair. It is not
counting; it is reproducing the answer distribution.

This module does the same job the other way round. A rule-based router reads
the question and decides what to *compute*; ``change_stats`` computes it from
building polygons; the answer is a number, not a guess. The design is the
plan's own sections 4.5.2 (rules-first routing) and 4.6.4 (change_stats), which
were specified before any adapter was trained and never wired together.

**Two things this separates that the VLM conflates.** Whether the arithmetic is
right, and whether the perception is good enough. Run against ground-truth
footprints it reports the ceiling -- what the architecture achieves if
detection were perfect. Run against a detector's output it reports what that
detector is worth. The VLM offers no such decomposition: one number, and no way
to tell which half is failing.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["route", "answer", "OPERATIONS"]

#: The operations change_stats can perform. Each maps to arithmetic on two sets
#: of building IDs, nothing more.
OPERATIONS = (
    "presence_appeared",
    "presence_demolished",
    "count_appeared",
    "count_demolished",
    "compare",
    "direction",
    "magnitude",
    "where",
)

#: Bands must match gen_change.MAGNITUDE_BANDS exactly. They are the corpus's
#: contract, and a mismatch here scores a correct measurement as wrong.
_BANDS: tuple[tuple[int, str], ...] = ((0, "none"), (10, "a few"), (50, "dozens"), (10**9, "many"))

#: Ordered, and the order matters. The magnitude question contains the exact
#: words "how many new buildings appeared", so a naive scan matches it as a
#: count and returns an integer where a band was asked for. Longest and most
#: specific patterns first.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("magnitude", re.compile(r"none,\s*a few,\s*dozens,?\s*or many", re.I)),
    ("where", re.compile(r"north,\s*south,\s*east,?\s*or west", re.I)),
    ("direction", re.compile(r"increase,\s*decrease,?\s*or stay unchanged", re.I)),
    ("compare", re.compile(r"which image contains more", re.I)),
    ("count_demolished", re.compile(r"how many buildings were demolished", re.I)),
    ("count_appeared", re.compile(r"how many new buildings appeared", re.I)),
    ("presence_demolished", re.compile(r"were any buildings demolished", re.I)),
    ("presence_appeared", re.compile(r"did any new buildings appear", re.I)),
)


def route(question: str) -> str | None:
    """Which measurement answers this question, or None if nothing does.

    None is a real answer and must stay one: a router that guesses on an
    unrecognised question produces a confident wrong number, which is worse
    than deferring to the VLM. Every abstention here is a row the language
    model handles instead.
    """
    for name, pattern in _RULES:
        if pattern.search(question):
            return name
    return None


def _band(count: int) -> str:
    for ceiling, label in _BANDS:
        if count <= ceiling:
            return label
    raise AssertionError("_BANDS must end with a catch-all")


def _compass(points: list[tuple[float, float]], frame: list[tuple[float, float]]) -> str | None:
    """Which half the new buildings sit in, against the scene's own extent.

    The frame is every building at the later date, never the appeared subset:
    measuring a cluster against its own mean puts half the points on each side
    by construction. That was a real bug in the question generator and it would
    be the same bug here.
    """
    points = [p for p in points if p]
    scene = [p for p in frame if p]
    if len(points) < 6 or len(scene) < 6:
        return None
    xs = [p[0] for p in scene]
    ys = [p[1] for p in scene]
    mid_x = (min(xs) + max(xs)) / 2
    mid_y = (min(ys) + max(ys)) / 2
    tally = {"north": 0, "south": 0, "east": 0, "west": 0}
    for x, y in points:
        tally["east" if x >= mid_x else "west"] += 1
        tally["north" if y >= mid_y else "south"] += 1
    best = None
    for axis in (("north", "south"), ("east", "west")):
        label = max(axis, key=lambda k: tally[k])
        share = tally[label] / len(points)
        if share >= 0.65 and (best is None or share > best[1]):
            best = (label, share)
    return best[0] if best else None


def answer(operation: str, before: dict[Any, Any], after: dict[Any, Any]) -> str | None:
    """Compute the answer from two ID->centroid maps.

    ``before`` and ``after`` are whatever the perception stage produced: the
    ground-truth footprints, or a detector's tracked polygons. The arithmetic
    is identical either way, which is the point -- swapping the detector
    changes the accuracy and not a line of this function.
    """
    appeared = set(after) - set(before)
    demolished = set(before) - set(after)

    if operation == "presence_appeared":
        return "yes" if appeared else "no"
    if operation == "presence_demolished":
        return "yes" if demolished else "no"
    if operation == "count_appeared":
        return str(len(appeared))
    if operation == "count_demolished":
        return str(len(demolished))
    if operation == "compare":
        if len(before) == len(after):
            # The corpus only asks this when the counts differ, so a tie means
            # the perception disagrees with the reference. Abstain rather than
            # coin-flip.
            return None
        return "second" if len(after) > len(before) else "first"
    if operation == "direction":
        if len(after) == len(before):
            return "unchanged"
        return "increased" if len(after) > len(before) else "decreased"
    if operation == "magnitude":
        return _band(len(appeared))
    if operation == "where":
        return _compass([after[i] for i in appeared], list(after.values()))
    raise ValueError(f"unknown operation {operation!r}; expected one of {OPERATIONS}")
