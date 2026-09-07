"""Shared machinery for every question generator.

The generators differ in what they read -- labels, polygons, masks, change
deltas -- and agree on everything else. That agreement lives here, because the
failure modes below are corpus-wide and a rule enforced in four of five
generators is not enforced at all.

**Balanced answers.** The obvious way to write a presence question is to ask
about a class that is present, which yields a corpus where the answer is always
"yes". A model trained on it learns to answer "yes" and scores well until it
meets a balanced evaluation set. Every yes/no generator here draws negatives
from the classes the sample does *not* contain, and :func:`balance_report` makes
the resulting skew visible rather than implicit.

**A cap per image.** Left alone, a 200-building scene generates hundreds of
questions and a 2-building scene generates three. The corpus then reflects scene
complexity rather than the task, one image can dominate a batch, and the
effective dataset size is far smaller than the row count suggests.
:data:`MAX_QA_PER_IMAGE` caps it and the selection is deterministic.

**Deterministic sampling.** Seeded from the sample id, so regenerating the
corpus on another machine produces the same rows. A corpus that differs between
teammates cannot be compared between teammates -- and with four people training
four adapters on four accounts, that matters more here than usual.

**No unanswerable questions.** Every generator computes its answer from the
annotation it was handed. If it cannot, it emits nothing rather than a
plausible-looking guess.
"""

import hashlib
import random
from typing import Any

__all__ = [
    "MAX_QA_PER_IMAGE",
    "balance_report",
    "canonical_row",
    "numeric_answer",
    "rng_for",
    "sample_negatives",
    "take",
]

#: Ceiling on questions generated from one image. 12 keeps a complex scene from
#: crowding out a simple one while still using the annotation properly.
MAX_QA_PER_IMAGE = 12

#: Yes/no generators aim for this share of "no" answers. Not enforced per image
#: -- a sample with every class present has no negatives to draw -- but tracked
#: across the corpus by :func:`balance_report`.
TARGET_NEGATIVE_SHARE = 0.5


def rng_for(sample_id: str, salt: str = "") -> random.Random:
    """A generator seeded from the sample id, so output is reproducible.

    Hashed rather than using ``hash()``: Python salts string hashing per
    process, so ``hash()`` would give a different corpus on every run and
    between teammates.
    """
    digest = hashlib.sha256(f"{sample_id}|{salt}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def take(items: list[Any], limit: int, rng: random.Random) -> list[Any]:
    """At most ``limit`` items, chosen deterministically.

    Shuffles before truncating so the cap does not systematically keep whatever
    the annotation happened to list first -- in a footprint file that is often
    spatial order, which would bias every capped sample toward one corner of
    the image.
    """
    if len(items) <= limit:
        return list(items)
    pool = list(items)
    rng.shuffle(pool)
    return pool[:limit]


def sample_negatives(
    present: list[str], vocabulary: list[str], count: int, rng: random.Random
) -> list[str]:
    """Classes that are *absent*, for questions whose answer is "no"."""
    absent = [c for c in vocabulary if c not in set(present)]
    return take(absent, count, rng)


def numeric_answer(value: float, unit: str = "") -> str:
    """Format a number the way the scorer expects to read it back.

    One decimal place and no thousands separators. ``metrics.numeric_accuracy``
    reads the first number out of the string, and "1,250" parses as 1 -- a
    formatting choice silently becoming a 99.9% error.
    """
    if value == int(value):
        text = str(int(value))
    else:
        text = f"{value:.1f}"
    return f"{text} {unit}".strip()


def canonical_row(
    primitive: Any,
    *,
    index: int,
    adapter: str,
    task: str,
    question: str,
    answer: str,
    answer_type: str = "text",
    question_type: str = "",
    image_roles: list[str] | None = None,
    modality: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical row, carrying provenance from the primitive.

    ``licence`` and ``provenance_chain`` are copied rather than left to the
    caller: C45 makes the provenance chain a hard gate on manifest build, and a
    row that loses it cannot be cleared for training later without re-deriving
    where it came from.

    ``question_type`` is not optional in practice. Average accuracy is the mean
    of per-type accuracies, so a row without a type silently collapses into a
    single bucket and changes the headline metric.
    """
    if not question or not answer:
        raise ValueError(
            f"{primitive.sample_id}: refusing to emit a row with an empty "
            f"question ({question!r}) or answer ({answer!r})."
        )
    images = list(primitive.image_paths)
    return {
        "sample_id": f"{primitive.sample_id}_{index:04d}",
        "adapter": adapter,
        "task": task,
        "images": images,
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "question_type": question_type or task,
        "image_roles": image_roles or ["t0"] * len(images),
        "modality": modality or ["optical"] * len(images),
        "effective_gsd_m": list(primitive.effective_gsd_m),
        "split": primitive.split,
        "source": primitive.source,
        "licence": primitive.licence,
        "provenance_chain": list(primitive.provenance_chain),
        "source_ann_id": primitive.source_ann_id,
        **(extra or {}),
    }


def balance_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer distribution, per question type. Report, never assume.

    A generated corpus can look healthy by row count while being 95% "yes", or
    while three images supply a fifth of the rows. Both are invisible in a
    manifest and obvious in this summary.
    """
    from collections import Counter

    by_type: dict[str, Counter] = {}
    per_image: Counter = Counter()
    for row in rows:
        by_type.setdefault(row.get("question_type", "?"), Counter())[row["answer"]] += 1
        per_image[tuple(row["images"])] += 1

    yes_no = Counter()
    for counter in by_type.values():
        for answer, count in counter.items():
            if answer.lower() in {"yes", "no"}:
                yes_no[answer.lower()] += count

    total_yes_no = sum(yes_no.values())
    images = len(per_image)
    return {
        "rows": len(rows),
        "images": images,
        "rows_per_image_mean": round(len(rows) / images, 2) if images else 0.0,
        "rows_per_image_max": max(per_image.values()) if per_image else 0,
        "question_types": {k: sum(v.values()) for k, v in by_type.items()},
        "yes_share": round(yes_no["yes"] / total_yes_no, 3) if total_yes_no else None,
        "distinct_answers_per_type": {k: len(v) for k, v in by_type.items()},
    }
