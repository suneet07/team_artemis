"""P3 — polygon footprints into VQA rows. SpaceNet 6, SpaceNet 7.

Feeds ``optsar_fusion`` (SpaceNet 6 buildings) and supplies the per-date half of
``change_vqa`` (SpaceNet 7). Three question shapes, all computable from the
polygon list plus the pixel size:

* **count** -- how many footprints;
* **presence** -- whether any footprint is present, including the empty case;
* **area** -- total footprint area in square metres.

**The empty scene is generated on purpose.** SpaceNet 7 encodes "no buildings"
as ``POLYGON EMPTY`` rather than an absent record, and a corpus built only from
scenes that contain something teaches a model that the answer is never zero.
Those samples are the ones that catch a model counting confidently on an empty
field.

Area is computed with the shoelace formula in pixel space and scaled by the
GSD, rather than by calling into a geometry stack: the polygons here are small
and simple, and it keeps the generator free of a GEOS dependency that the
headless eval path (section 4.11) must not pull in.
"""

import re
from collections.abc import Iterator
from typing import Any

from satquery.qgen.common import canonical_row, numeric_answer, rng_for
from satquery.qgen.primitives import P3Polygon

__all__ = ["gen_polygon_qa", "polygon_area_px", "parse_wkt_polygon"]

_COORD = re.compile(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    """Exterior ring of a WKT polygon as (x, y) pairs. Empty when degenerate.

    ``POLYGON EMPTY`` is a valid, meaningful value in SpaceNet 7 -- it marks a
    scene with no buildings -- so it returns an empty ring rather than raising.
    """
    if not wkt or "EMPTY" in wkt.upper():
        return []
    exterior = wkt.split("(")[-1].split(")")[0] if "(" in wkt else ""
    return [(float(x), float(y)) for x, y in _COORD.findall(exterior)]


def polygon_area_px(ring: list[tuple[float, float]]) -> float:
    """Shoelace area of a ring, in square pixels. Zero for degenerate rings."""
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def gen_polygon_qa(
    primitive: P3Polygon,
    *,
    adapter: str = "optsar_fusion",
    class_name: str = "building",
    max_qa: int = 4,
) -> Iterator[dict[str, Any]]:
    """Emit count, presence and area questions from a footprint list."""
    gsd = primitive.effective_gsd_m[0] if primitive.effective_gsd_m else 1.0
    rings = [
        parse_wkt_polygon(str(polygon.get("geometry", "")))
        for polygon in primitive.polygons
    ]
    real = [ring for ring in rings if len(ring) >= 3]
    count = len(real)
    area_m2 = sum(polygon_area_px(ring) for ring in real) * (gsd**2)

    rng = rng_for(primitive.sample_id, "p3")
    rows: list[tuple[str, str, str, str]] = [
        (
            "count",
            f"How many {class_name}s are visible in this image?",
            numeric_answer(count),
            "number",
        ),
        (
            "presence",
            f"Are there any {class_name}s in this image?",
            "yes" if count else "no",
            "text",
        ),
    ]
    # Area only when there is area to speak of. "0 m2" as an area answer teaches
    # a numeric format for a question the presence row already answers better.
    if count:
        rows.append(
            (
                "area",
                f"What is the total {class_name} footprint area in this image, "
                "in square metres?",
                numeric_answer(area_m2, "m2"),
                "number",
            )
        )
    _ = rng  # ordering here is fixed; the seed is kept for future variants

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
            extra={"object_count": count, "footprint_area_m2": round(area_m2, 1)},
        )
