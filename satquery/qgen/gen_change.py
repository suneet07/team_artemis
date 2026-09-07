"""P5 — bi-temporal pairs into change questions. SpaceNet 7, HRSCD, OSCD, C46.

Feeds ``change_vqa``, the adapter with no training data and the largest measured
training cost (15.7 h at three composites). Everything here is computed from the
change mask or the tracking-ID delta, never from the imagery.

**Both dates, in order, always.** Rows carry two images with roles ``t0`` and
``t1``. A change row that loses its ordering is a row whose answer is reversed --
"were buildings added" becomes "were buildings removed" -- and nothing
downstream can detect it, because both readings are grammatical and plausible.

**The no-change case is generated deliberately.** Most bi-temporal pairs over a
real AOI show nothing happening, and a corpus built only from pairs that changed
teaches a model that something always changed. Those rows are what stop it.

**Counts come from tracking IDs where the source has them.** SpaceNet 7's
persistent building IDs give added/removed/persisted directly (C59), which is a
real count rather than a mask-derived estimate. HRSCD and OSCD have masks only,
so they get area questions and not counting questions -- a count inferred from
connected components in a change mask is an artefact of the labelling
resolution, and asking a model to reproduce it teaches it to reproduce noise.
"""

from collections.abc import Iterator
from typing import Any

from satquery.qgen.common import canonical_row, numeric_answer, rng_for
from satquery.qgen.primitives import P5Change

__all__ = ["gen_change_qa", "changed_area_m2"]


def changed_area_m2(mask, gsd_m: float) -> float:
    """Area of the changed region, from a binary change mask."""
    import numpy as np

    array = np.asarray(mask)
    return float((array > 0).sum()) * (gsd_m**2)


def gen_change_qa(
    primitive: P5Change,
    *,
    changed_area: float | None = None,
    class_name: str = "building",
    adapter: str = "change_vqa",
    max_qa: int = 6,
) -> Iterator[dict[str, Any]]:
    """Emit change questions from an ID delta and/or a changed-area figure.

    ``changed_area`` is square metres from :func:`changed_area_m2`; pass None
    when the source has no mask. With neither a delta nor an area, nothing is
    emitted -- there is no question this generator can answer honestly.
    """
    if len(primitive.image_paths) != 2:
        raise ValueError(
            f"{primitive.sample_id}: a change sample needs exactly two images, got "
            f"{len(primitive.image_paths)}. The pair's order carries the direction "
            "of every answer in this generator."
        )

    delta = primitive.id_delta or {}
    added = delta.get("added")
    removed = delta.get("removed")
    persisted = delta.get("persisted")
    has_counts = added is not None and removed is not None

    if not has_counts and changed_area is None:
        return

    rng = rng_for(primitive.sample_id, "p5")
    _ = rng
    rows: list[tuple[str, str, str, str]] = []

    if has_counts:
        total_change = int(added) + int(removed)
        rows.append(
            (
                "change_presence",
                f"Did the number of {class_name}s change between the first and "
                "second image?",
                "yes" if total_change else "no",
                "text",
            )
        )
        rows.append(
            (
                "change_count",
                f"How many {class_name}s were added between the first and second "
                "image?",
                numeric_answer(int(added)),
                "number",
            )
        )
        rows.append(
            (
                "change_count",
                f"How many {class_name}s were removed between the first and "
                "second image?",
                numeric_answer(int(removed)),
                "number",
            )
        )
        rows.append(
            (
                "change_direction",
                f"Were more {class_name}s built or demolished between the two "
                "images?",
                "built" if added > removed else ("demolished" if removed > added else "neither"),
                "text",
            )
        )
        if persisted is not None:
            rows.append(
                (
                    "change_count",
                    f"How many {class_name}s are present in both images?",
                    numeric_answer(int(persisted)),
                    "number",
                )
            )

    if changed_area is not None:
        rows.append(
            (
                "change_presence",
                "Did any part of this area change between the two images?",
                "yes" if changed_area > 0 else "no",
                "text",
            )
        )
        if changed_area > 0:
            rows.append(
                (
                    "change_area",
                    "What area changed between the two images, in square metres?",
                    numeric_answer(changed_area, "m2"),
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
            task="change_vqa",
            question=question,
            answer=answer,
            answer_type=answer_type,
            question_type=question_type,
            image_roles=["t0", "t1"],
            extra={
                "id_delta": delta or None,
                "changed_area_m2": (
                    round(changed_area, 1) if changed_area is not None else None
                ),
                # The tool-vs-VLM ablation needs these to score deterministic
                # arithmetic against generation; without them it refuses to run.
                "pixel_area_m2": (
                    primitive.effective_gsd_m[0] ** 2
                    if primitive.effective_gsd_m
                    else None
                ),
            },
        )
