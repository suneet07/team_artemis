"""Shared tile-window helpers for deterministic raster tools (§12, Rule 7).

P4 has not landed, so no `TileIndex` producer exists yet and the tile dict shape
is not frozen. This module is the single place that interprets a tile entry, so
that when P4 does land there is exactly one function to change.

A tile is a mapping that locates a pixel window in the source raster grid. Two
spellings are accepted:

    {"bbox_px": [x0, y0, x1, y1]}              # origin top-left, x right, y down
    {"col_off": x0, "row_off": y0, "width": w, "height": h}

Anything else -- including a tile carrying only an id and a score -- resolves to
"no window", and the caller runs whole-scene. That is the safe default: a tool
that cannot locate its tile must not pretend it processed a subset, because the
executor mosaics on the offsets this module returns.
"""

from typing import Any

import numpy as np

# Tool modules set this to True once they honour context["current_tile"].
# The executor only runs its per-tile loop for tools that declare it -- see
# satquery/agent/executor.py. Without the flag a tool is called once on the
# whole scene, which is slower than tiling but never wrong.
SUPPORTS_TILING = True


def resolve_window(
    tile: Any,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Resolve a tile entry to (row0, col0, row1, col1), clipped to `shape`.

    Returns None when the tile does not locate a window, or when the window is
    empty or entirely outside the array.
    """
    if not isinstance(tile, dict):
        return None

    height, width = int(shape[0]), int(shape[1])

    bbox = tile.get("bbox_px")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        col0, row0, col1, row1 = int(x0), int(y0), int(x1), int(y1)
    elif all(k in tile for k in ("col_off", "row_off", "width", "height")):
        col0 = int(tile["col_off"])
        row0 = int(tile["row_off"])
        col1 = col0 + int(tile["width"])
        row1 = row0 + int(tile["height"])
    else:
        return None

    # Normalise inverted bounds, then clip to the array.
    if row1 < row0:
        row0, row1 = row1, row0
    if col1 < col0:
        col0, col1 = col1, col0
    row0 = max(0, min(row0, height))
    col0 = max(0, min(col0, width))
    row1 = max(0, min(row1, height))
    col1 = max(0, min(col1, width))

    if row1 <= row0 or col1 <= col0:
        return None
    return row0, col0, row1, col1


def window_of(
    context: dict[str, Any] | None,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Resolve the window for `context["current_tile"]`, or None when untiled."""
    if not context:
        return None
    return resolve_window(context.get("current_tile"), shape)


def crop(arr: np.ndarray, window: tuple[int, int, int, int] | None) -> np.ndarray:
    """Return the sub-array for `window`, or `arr` unchanged when window is None."""
    if window is None:
        return arr
    row0, col0, row1, col1 = window
    return arr[row0:row1, col0:col1]


def tile_report(
    window: tuple[int, int, int, int] | None,
    scene_shape: tuple[int, int],
) -> dict[str, Any]:
    """Private output keys the executor uses to mosaic. Stripped before tracing.

    Keys are underscore-prefixed so that a tool's public output contract -- the
    one declared in its manifest and recorded in the trace -- is unchanged.
    """
    if window is None:
        return {"_scene_shape": [int(scene_shape[0]), int(scene_shape[1])]}
    row0, col0, _row1, _col1 = window
    return {
        "_tile_offset": [row0, col0],
        "_scene_shape": [int(scene_shape[0]), int(scene_shape[1])],
    }


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x0, y0, x1, y1] boxes in the same coordinate space."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    """Suppress duplicate detections across tile seams (§12.4).

    An object straddling two overlapping tiles is proposed twice, once per tile,
    and both proposals are in scene coordinates by the time they reach here. The
    higher-scoring one wins.
    """
    ordered = sorted(
        (b for b in boxes if isinstance(b, dict) and "bbox_px" in b),
        key=lambda b: float(b.get("score") or 0.0),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for box in ordered:
        if all(iou(box["bbox_px"], k["bbox_px"]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def aggregate_tile_answers(per_tile: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-tile learned-tool answers by the §12.4 table.

    | Count    | sum across tiles, de-duplicated in the overlap region |
    | Presence | any tile yes -> yes                                   |
    | Free text| answer from the single highest-scoring tile; never concatenated |
    | Boxes    | union, scene coordinates, NMS across tile seams       |
    """
    if not per_tile:
        return {}
    if len(per_tile) == 1:
        return dict(per_tile[0])

    ranked = sorted(per_tile, key=lambda r: float(r.get("tile_score") or 0.0), reverse=True)
    best = dict(ranked[0])

    # Free text: the best tile's answer stands alone. Concatenating answers from
    # four tiles produces a paragraph no scorer will match.
    texts = [str(r.get("answer") or "") for r in ranked if r.get("answer")]
    if texts:
        best["answer"] = texts[0]

    # Presence: any yes wins.
    if any(_is_yes(r.get("answer")) for r in ranked):
        best["presence"] = True
    elif texts:
        best["presence"] = False

    # Boxes: union in scene space, then suppress seam duplicates.
    boxes: list[dict[str, Any]] = []
    for r in ranked:
        if isinstance(r.get("boxes"), list):
            boxes.extend(r["boxes"])
    if boxes:
        deduped = nms(boxes)
        best["boxes"] = deduped
        best["count"] = len(deduped)

    best["confidence"] = max(
        (float(r["confidence"]) for r in ranked if r.get("confidence") is not None),
        default=best.get("confidence"),
    )
    return best


def _is_yes(answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    head = answer.strip().lower()
    return head.startswith("yes") or " yes," in head
