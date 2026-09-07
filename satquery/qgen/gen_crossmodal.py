"""P6 — co-registered optical/SAR pairs. BEN S1+S2, SpaceNet 6, OEM-SAR.

Feeds ``optsar_fusion``, and supplies the SAR-only samples ``rs_vqa`` needs
under C26.

**Three arms from one annotation.** The same labelled scene emits an optical-only
row, a SAR-only row, and a fused row carrying both. That is what makes the
corpus able to teach modality dropout rather than merely tolerate it: the model
sees the same question answered from either modality alone and from both, so it
cannot learn that the answer lives in one fixed input slot.

This is also the corpus the D1/G5 story rests on. The PS input scope permits
"one optical/multispectral **or** SAR image", and BEN.txt is always co-registered
S1+S2 -- so without deliberately withheld-modality rows, a SAR-only input at
inference is out of distribution for every adapter (C26).

**Modality is recorded per image, not per row.** ``modality`` and
``effective_gsd_m`` are parallel to ``images``, because a fused row genuinely
carries two different sensors at two different resolutions, and GSD-conditioned
prompting reads them per view.
"""

from collections.abc import Iterator
from typing import Any

from satquery.qgen.common import canonical_row
from satquery.qgen.primitives import P6CrossModal

__all__ = ["ARMS", "gen_crossmodal_qa"]

#: The three views generated from one co-registered pair.
ARMS = ("optical", "sar", "fused")


def gen_crossmodal_qa(
    primitive: P6CrossModal,
    question: str,
    answer: str,
    *,
    optical_index: int = 0,
    sar_index: int = 1,
    question_type: str = "cross_modal",
    adapter: str = "optsar_fusion",
    arms: tuple[str, ...] = ARMS,
    answer_type: str = "text",
) -> Iterator[dict[str, Any]]:
    """Emit the same question against optical alone, SAR alone, and both.

    The question and answer are supplied by whichever P1-P4 generator read the
    underlying annotation; this generator's job is the modality arms, not the
    semantics. Splitting it that way keeps one place deciding what is askable
    about a scene and one place deciding which sensors it is asked over.
    """
    paths = list(primitive.image_paths)
    gsds = list(primitive.effective_gsd_m)
    if len(paths) < 2:
        raise ValueError(
            f"{primitive.sample_id}: a cross-modal sample needs an optical and a "
            f"SAR view, got {len(paths)} image(s)."
        )

    views = {
        "optical": ([optical_index], ["optical"]),
        "sar": ([sar_index], ["sar"]),
        "fused": ([optical_index, sar_index], ["optical", "sar"]),
    }

    index = 0
    for arm in arms:
        if arm not in views:
            raise ValueError(f"unknown arm {arm!r}; known: {sorted(views)}")
        indices, modality = views[arm]
        subset = type(primitive)(
            **{
                **primitive.__dict__,
                "image_paths": [paths[i] for i in indices],
                "effective_gsd_m": [
                    gsds[i] if i < len(gsds) else (gsds[0] if gsds else 10.0)
                    for i in indices
                ],
            }
        )
        yield canonical_row(
            subset,
            index=index,
            adapter=adapter,
            task="crossmodal_vqa" if arm == "fused" else "single_vqa",
            question=question,
            answer=answer,
            answer_type=answer_type,
            question_type=f"{question_type}_{arm}",
            image_roles=[arm] if arm != "fused" else ["optical", "sar"],
            modality=modality,
            extra={"modality_arm": arm},
        )
        index += 1
