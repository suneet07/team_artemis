"""P1 — multi-label class vectors into VQA rows. BEN CLC maps, OEM-SAR classes.

Feeds ``rs_vqa``. The annotation is a set of land-cover classes present in the
patch, which supports three question shapes and no more:

* **presence** -- "is X present", answerable yes or no from the label set;
* **co-occurrence** -- "do X and Y both appear", the same fact over two classes;
* **count of classes** -- how many distinct classes the patch carries.

Deliberately *not* generated: area, adjacency or "how much of the image".
A multi-label vector says which classes are present, not where or how much of
each. RSVQA-style area questions come from a pixel map, and inventing them from
a label set produces confident questions with unknowable answers -- which is
worse than having fewer question types, because the model learns to guess.

Negatives come from the class vocabulary, so the corpus is not all "yes".
"""

from collections.abc import Iterator
from typing import Any

from satquery.qgen.common import (
    MAX_QA_PER_IMAGE,
    canonical_row,
    numeric_answer,
    rng_for,
    sample_negatives,
    take,
)
from satquery.qgen.primitives import P1MultiLabel

__all__ = ["gen_multilabel_qa"]


def gen_multilabel_qa(
    primitive: P1MultiLabel,
    vocabulary: list[str],
    *,
    adapter: str = "rs_vqa",
    max_qa: int = MAX_QA_PER_IMAGE,
) -> Iterator[dict[str, Any]]:
    """Emit presence, co-occurrence and class-count questions.

    ``vocabulary`` is the full class list of the source dataset, needed to ask
    about classes that are *absent*. Without it every presence answer is "yes".
    """
    present = [label for label in primitive.labels if label]
    if not present:
        # A patch with no labels supports no question this generator can answer.
        return
    if not vocabulary:
        raise ValueError(
            f"{primitive.sample_id}: a class vocabulary is required. Without it "
            "every presence question is drawn from the classes that are present "
            "and the answer is always 'yes'."
        )

    rng = rng_for(primitive.sample_id, "p1")
    budget = max(1, max_qa)
    rows: list[dict[str, Any]] = []

    # Half the presence questions positive, half negative.
    half = max(1, budget // 3)
    positives = take(present, half, rng)
    negatives = sample_negatives(present, vocabulary, len(positives), rng)

    for label in positives:
        rows.append(("presence", f"Is {label.lower()} present in this image?", "yes"))
    for label in negatives:
        rows.append(("presence", f"Is {label.lower()} present in this image?", "no"))

    # Co-occurrence: one true pair and one false pair, when both are available.
    if len(present) >= 2:
        pair = take(present, 2, rng)
        rows.append(
            (
                "co_occurrence",
                f"Do both {pair[0].lower()} and {pair[1].lower()} appear in this image?",
                "yes",
            )
        )
    if present and negatives:
        rows.append(
            (
                "co_occurrence",
                f"Do both {present[0].lower()} and {negatives[0].lower()} appear "
                "in this image?",
                "no",
            )
        )

    rows.append(
        (
            "class_count",
            "How many distinct land-cover classes are present in this image?",
            numeric_answer(len(present)),
        )
    )

    for index, (question_type, question, answer) in enumerate(rows[:budget]):
        yield canonical_row(
            primitive,
            index=index,
            adapter=adapter,
            task="single_vqa",
            question=question,
            answer=answer,
            question_type=question_type,
            extra={"labels": present},
        )
