from collections.abc import Iterator
from typing import Any

from satquery.qgen.primitives import Primitive


def gen_passthrough(
    primitive: Primitive, qa_pairs: list[dict[str, str]]
) -> Iterator[dict[str, Any]]:
    """
    Pass-through generator for datasets that already have Q&A pairs (e.g., BEN.txt, RSVQA).
    Applies filtering and translates into the Canonical JSONL schema.
    
    qa_pairs format: [{"question": "...", "answer": "..."}]
    """
    for idx, qa in enumerate(qa_pairs):
        # We assume answer_type is freeform text for pass-through unless specified
        yield {
            "sample_id": f"{primitive.sample_id}_qa_{idx}",
            "adapter": "rs_vqa",
            "task": "vqa",
            "images": primitive.image_paths,
            "image_roles": ["t0"] * len(primitive.image_paths),
            # Default; the caller overrides it when the source is SAR.
            "modality": ["optical"] * len(primitive.image_paths),
            "effective_gsd_m": primitive.effective_gsd_m,
            "question": qa["question"],
            "answer": qa["answer"],
            "answer_type": "text",
            "answer_vocab": None,
            "template_id": "PASSTHROUGH",
            "paraphrase_id": 0,
            "source": primitive.source,
            "source_ann_ids": [primitive.source_ann_id],
            "licence": primitive.licence,
            "provenance_chain": primitive.provenance_chain,
            "generator_version": "qgen-1.0.0",
            "split": primitive.split,
            "balance_key": "passthrough|text"
        }
