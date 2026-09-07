import random
from collections.abc import Iterator
from typing import Any

from satquery.qgen.boxes import assert_convention_verified, from_pixels, to_prompt_box
from satquery.qgen.primitives import P2BoundingBox


def normalize_bbox(bbox: list[float], img_width: int, img_height: int) -> list[int]:
    """Serialise a pixel box ``[ymin, xmin, ymax, xmax]`` for a training target.

    Delegates to :mod:`satquery.qgen.boxes`, which owns the axis order and the
    scale. It used to hardcode y-first at 0-1000 here, while
    ``object_box_fallback`` emitted x-first normalised floats -- two conventions
    in one system, and the wrong one silently teaches the model transposed
    geometry (TEAM_CONTEXT section 10).

    Also rounds rather than truncates: ``int(0.9995 * 1000)`` is 999, and a
    systematic half-pixel bias toward the top-left across an entire corpus is
    exactly the kind of error that shows up only as a slightly disappointing
    mIoU.
    """
    assert_convention_verified()
    return to_prompt_box(from_pixels(bbox, img_width, img_height, order="yxyx"))


def gen_bbox_qa(
    primitive: P2BoundingBox, img_width: int = 512, img_height: int = 512
) -> Iterator[dict[str, Any]]:
    """
    Generates Grounding Q&A for P2 primitives (bounding boxes).
    Filters out bboxes below the legibility threshold (e.g., 12px at effective GSD).
    """
    valid_bboxes = []
    
    for b in primitive.bboxes:
        ymin, xmin, ymax, xmax = b["bbox"]
        # Legibility floor: reject if longest side < 12 pixels
        if max(ymax - ymin, xmax - xmin) >= 12:
            valid_bboxes.append(b)

    # 1. Generate Refer Box (Single object)
    if len(valid_bboxes) > 0:
        # Pick one random object for a refer_box question
        target = random.choice(valid_bboxes)
        n_box = normalize_bbox(target["bbox"], img_width, img_height)
        cls_name = target["class"]
        
        yield {
            "sample_id": f"{primitive.sample_id}_refer_1",
            "adapter": "rs_ground_caption",
            "task": "refer_box",
            "images": primitive.image_paths,
            "image_roles": ["t0"] * len(primitive.image_paths),
            "modality": ["optical"] * len(primitive.image_paths),
            "effective_gsd_m": primitive.effective_gsd_m,
            "question": f"Find the {cls_name} in this image and provide its bounding box.",
            "answer": f"There is an {cls_name} located at {n_box}.",
            "answer_type": "box",
            "answer_vocab": None,
            "template_id": "GRD.REFER.SINGLE.v1",
            "paraphrase_id": 1,
            "source": primitive.source,
            "source_ann_ids": [primitive.source_ann_id],
            "licence": primitive.licence,
            "provenance_chain": primitive.provenance_chain,
            "generator_version": "qgen-1.0.0",
            "split": primitive.split,
            "balance_key": f"refer_box|{cls_name}"
        }

    # 2. Generate Counting
    count = len(valid_bboxes)
    # Determine log bucket for balance_key
    # Count buckets, low to high. A chain of nested conditionals on one line
    # is a table pretending not to be one.
    bucket = next(
        label
        for limit, label in (
            (0, "0"),
            (1, "1"),
            (3, "2-3"),
            (10, "4-10"),
            (30, "11-30"),
            (float("inf"), "31+"),
        )
        if count <= limit
    )
    
    # Just picking the first class if present, else default to 'object' or infer from source
    target_class = valid_bboxes[0]["class"] if count > 0 else "aircraft" 

    yield {
        "sample_id": f"{primitive.sample_id}_count_1",
        "adapter": "rs_ground_caption",
        "task": "count",
        "images": primitive.image_paths,
        "image_roles": ["t0"] * len(primitive.image_paths),
        "modality": ["optical"] * len(primitive.image_paths),
        "effective_gsd_m": primitive.effective_gsd_m,
        "question": f"How many {target_class}s are visible in the image?",
        "answer": str(count),
        "answer_type": "count",
        "answer_vocab": None,
        "template_id": "GRD.COUNT.v1",
        "paraphrase_id": 1,
        "source": primitive.source,
        "source_ann_ids": [primitive.source_ann_id],
        "licence": primitive.licence,
        "provenance_chain": primitive.provenance_chain,
        "generator_version": "qgen-1.0.0",
        "split": primitive.split,
        "balance_key": f"count|{target_class}|{bucket}"
    }
