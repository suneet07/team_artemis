"""P4 — semantic segmentation masks into VQA rows. OEM-SAR, HRSCD, BEN CLC.

Feeds ``optsar_fusion`` and ``rs_vqa``. A mask says more than a label vector:
it knows *how much* of each class is present, so this is where area-fraction and
dominant-class questions legitimately come from.

**The mask is read, not assumed.** ``classes_present`` on the primitive is a
convenience field; the fractions come from counting pixels. A generator that
trusted the field would emit "which class covers most of the image" without
ever knowing which one does.

**Fractions are binned, not quoted to the pixel.** "38.4% grassland" is a
precision the model cannot see and the scorer cannot fairly grade; the bins are
the coarse buckets a human would use. Exact percentages read like rigour and
train the model to hallucinate decimals.
"""

from collections.abc import Iterator
from typing import Any

from satquery.qgen.common import canonical_row, numeric_answer, rng_for, sample_negatives
from satquery.qgen.primitives import P4SegMask

__all__ = ["COVERAGE_BINS", "coverage_bin", "gen_segmask_qa", "class_fractions"]

#: Upper bound (exclusive) -> phrase. A model can see these distinctions; it
#: cannot see the difference between 38% and 41%.
COVERAGE_BINS: tuple[tuple[float, str], ...] = (
    (0.01, "none"),
    (0.10, "a small part"),
    (0.35, "some"),
    (0.65, "about half"),
    (0.90, "most"),
    (1.01, "nearly all"),
)


def coverage_bin(fraction: float) -> str:
    for upper, phrase in COVERAGE_BINS:
        if fraction < upper:
            return phrase
    return "nearly all"


def class_fractions(mask, class_values: dict[int, str]) -> dict[str, float]:
    """Fraction of pixels per class, read from the mask array itself."""
    import numpy as np

    array = np.asarray(mask)
    total = array.size
    if not total:
        return {}
    fractions: dict[str, float] = {}
    for value, name in class_values.items():
        share = float((array == value).sum()) / total
        if share > 0:
            fractions[name] = share
    return fractions


def gen_segmask_qa(
    primitive: P4SegMask,
    fractions: dict[str, float],
    vocabulary: list[str],
    *,
    adapter: str = "optsar_fusion",
    max_qa: int = 6,
) -> Iterator[dict[str, Any]]:
    """Emit coverage, dominant-class and presence questions from mask statistics.

    ``fractions`` comes from :func:`class_fractions` -- passed in rather than
    computed here so the caller loads each mask once for however many question
    shapes it feeds.
    """
    present = [name for name, share in fractions.items() if share >= 0.01]
    if not present:
        return

    rng = rng_for(primitive.sample_id, "p4")
    dominant = max(fractions.items(), key=lambda item: item[1])[0]
    rows: list[tuple[str, str, str, str]] = [
        (
            "dominant_class",
            "Which land-cover class covers the largest area in this image?",
            dominant,
            "text",
        )
    ]

    for name in sorted(present, key=lambda n: -fractions[n])[:2]:
        rows.append(
            (
                "coverage",
                f"How much of this image is covered by {name.lower()}?",
                coverage_bin(fractions[name]),
                "text",
            )
        )

    for name in sample_negatives(present, vocabulary, 1, rng):
        rows.append(
            (
                "presence",
                f"Is {name.lower()} present in this image?",
                "no",
                "text",
            )
        )
    rows.append(
        (
            "presence",
            f"Is {present[0].lower()} present in this image?",
            "yes",
            "text",
        )
    )
    rows.append(
        (
            "class_count",
            "How many distinct land-cover classes are present in this image?",
            numeric_answer(len(present)),
            "number",
        )
    )

    for index, (question_type, question, answer, answer_type) in enumerate(
        rows[:max_qa]
    ):
        yield canonical_row(
            primitive,
            index=index,
            adapter=adapter,
            task="single_vqa",
            question=question,
            answer=answer,
            answer_type=answer_type,
            question_type=question_type,
            extra={"class_fractions": {k: round(v, 4) for k, v in fractions.items()}},
        )
