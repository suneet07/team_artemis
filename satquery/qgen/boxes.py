"""One box convention, converted at the edges (TEAM_CONTEXT section 10).

The problem this exists to end: two live specs disagreed. The RarePlanes brief
emits ``[ymin, xmin, ymax, xmax]`` scaled 0-1000; QGP v1 specifies
``box_xyxy_normalised``; and ``object_box_fallback`` emitted a third thing.
Adding a fourth field to each box does not reconcile anything — it makes the
disagreement permanent and moves the decision to whoever reads the JSON.

So: **one internal representation, converted at the boundaries.** Internally a
box is ``BoxXYXY`` in normalised [0, 1] floats — the form the geometry code
already produces and the only one that survives a resize. Everything else is a
serialisation, chosen once by
``configs/preprocessing.yaml``-independent constant :data:`PROMPT_BOX_FORMAT`.

**The axis order is not settled and must not be guessed.** TEAM_CONTEXT is
explicit that it "must match what the base model was trained to emit — verify
against Qwen3-VL's own documented format rather than assuming". Published Qwen-VL
generations have used ``(x1,y1),(x2,y2)`` normalised to 0-1000, and Qwen2.5-VL
moved to absolute-pixel ``bbox_2d: [x1,y1,x2,y2]`` — both **x-first**, whereas
the RarePlanes brief is y-first. That is a real conflict, not a formatting
detail: y-first boxes fed to an x-first model teach it transposed geometry, and
the error is invisible on square crops.

:data:`PROMPT_BOX_FORMAT` is therefore a single named switch with the open
question recorded next to it. Flip it once, in one place, when someone has
checked the model card — and :func:`assert_convention_verified` fails loudly in
the training path until they have.
"""

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "PROMPT_BOX_FORMAT",
    "PROMPT_BOX_SCALE",
    "BoxXYXY",
    "assert_convention_verified",
    "from_pixels",
    "from_yxyx_1000",
    "normalise_box_answer",
    "to_prompt_box",
    "to_yxyx_1000",
]

BoxFormat = Literal["xyxy", "yxyx"]

#: Axis order for a box in a prompt or a training target. **Verified** -- see
#: BOX_CONVENTION_VERIFIED below for the evidence.
PROMPT_BOX_FORMAT: BoxFormat = "xyxy"

#: Integer scale for prompt boxes. Qwen-VL family conventions use 0-1000.
PROMPT_BOX_SCALE: int = 1000

#: VERIFIED 2026-08-30 from two independent sources that agree.
#:
#: 1. **The base model.** Qwen3-VL's own spatial-understanding cookbook
#:    de-serialises its output as ``abs_x = point[0]/1000 * width`` and
#:    ``abs_y = point[1]/1000 * height`` -- index 0 is x, and the scale is
#:    0-1000. (The GitHub issue asking this has no maintainer reply; the
#:    cookbook is the authority.)
#:
#: 2. **The training data.** BEN.txt referring questions name a land-cover class
#:    and answer with a box. Checking which reading of the box actually lands on
#:    that class in the patch's own reference map gave **80 of 86 boxes (93%)
#:    for x-first across 20 patches, 0 ties**, with several y-first readings at
#:    exactly 0.00 coverage. The six exceptions are near-square boxes where
#:    either reading overlaps by chance.
#:
#: Note what does NOT discriminate: point-in-box containment. Transposing both
#: the box and the point leaves the arithmetic identical, so an earlier test
#: "passed" 91/91 while proving nothing -- the same trap as a diagonal patch in
#: the grid convention. The test above is asymmetric on purpose.
BOX_CONVENTION_VERIFIED: bool = True


@dataclass(frozen=True)
class BoxXYXY:
    """A box in normalised [0, 1] coordinates, x first.

    Normalised because it is the only form that stays correct through a resize,
    a crop remap or a tile mosaic; x first because the geometry code, the mask
    code and every array-to-image convention in this package already are.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise ValueError(
                f"box corners are inverted: ({self.xmin}, {self.ymin}) to "
                f"({self.xmax}, {self.ymax}). A silently inverted box has zero IoU with "
                f"the truth and looks like a model failure."
            )

    @property
    def as_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def clamped(self) -> "BoxXYXY":
        clamp = lambda v: max(0.0, min(1.0, float(v)))  # noqa: E731
        return BoxXYXY(clamp(self.xmin), clamp(self.ymin), clamp(self.xmax), clamp(self.ymax))


def from_pixels(
    bbox_pixels: tuple[int, int, int, int] | list[int],
    width: int,
    height: int,
    order: BoxFormat = "yxyx",
) -> BoxXYXY:
    """Normalise a pixel box. ``order`` says how the input is laid out.

    Defaults to ``yxyx`` because that is what ``scipy.ndimage.find_objects``
    slices and the RarePlanes annotations both give, so the default matches the
    most common *input* rather than the internal representation.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive to normalise a box")
    if order == "yxyx":
        ymin, xmin, ymax, xmax = bbox_pixels
    else:
        xmin, ymin, xmax, ymax = bbox_pixels
    return BoxXYXY(xmin / width, ymin / height, xmax / width, ymax / height).clamped()


def to_yxyx_1000(box: BoxXYXY) -> list[int]:
    """Serialise as ``[ymin, xmin, ymax, xmax]`` scaled 0-1000.

    The RarePlanes brief's convention, kept so already-generated data can be
    read back without re-deriving it.
    """
    clamped = box.clamped()
    return [
        int(round(clamped.ymin * 1000)),
        int(round(clamped.xmin * 1000)),
        int(round(clamped.ymax * 1000)),
        int(round(clamped.xmax * 1000)),
    ]


def from_yxyx_1000(values: list[int] | tuple[int, int, int, int]) -> BoxXYXY:
    """Parse ``[ymin, xmin, ymax, xmax]`` at 0-1000 back to the internal form."""
    ymin, xmin, ymax, xmax = (v / 1000.0 for v in values)
    return BoxXYXY(xmin, ymin, xmax, ymax).clamped()


def to_prompt_box(box: BoxXYXY, *, scale: int | None = None) -> list[int]:
    """Serialise for a prompt or a training target, per :data:`PROMPT_BOX_FORMAT`."""
    clamped = box.clamped()
    scale = PROMPT_BOX_SCALE if scale is None else scale
    if PROMPT_BOX_FORMAT == "yxyx":
        ordered = (clamped.ymin, clamped.xmin, clamped.ymax, clamped.xmax)
    else:
        ordered = (clamped.xmin, clamped.ymin, clamped.xmax, clamped.ymax)
    return [int(round(value * scale)) for value in ordered]


def assert_convention_verified() -> None:
    """Refuse to generate training targets on an unverified box convention.

    Called by the data-generation path, not by inference: emitting a box in the
    wrong axis order at *inference* is a bug someone will see, while baking it
    into 80,000 training samples is a bug that trains the model to be wrong and
    shows up only as a mediocre grounding score nobody can explain.
    """
    if not BOX_CONVENTION_VERIFIED:
        raise RuntimeError(
            "The prompt box convention has not been verified against the base model's "
            "documented output format (TEAM_CONTEXT section 10). Qwen-VL family "
            f"conventions are x-first; the RarePlanes brief is y-first; this build is "
            f"set to '{PROMPT_BOX_FORMAT}' at scale {PROMPT_BOX_SCALE}. Check the model "
            "card, then set BOX_CONVENTION_VERIFIED = True in satquery/qgen/boxes.py. "
            "Do not generate a training corpus before that."
        )


def normalise_box_payload(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Attach the canonical fields to a tool's box dict, in place.

    Emits exactly two serialisations — the internal normalised one and the
    prompt one — and no more. Three coexisting formats is the state this module
    exists to leave behind.
    """
    if "bbox_pixels" in payload:
        box = from_pixels(payload["bbox_pixels"], width, height, order="yxyx")
    elif "bbox_xyxy_normalised" in payload:
        xmin, ymin, xmax, ymax = payload["bbox_xyxy_normalised"]
        box = BoxXYXY(xmin, ymin, xmax, ymax).clamped()
    else:
        raise KeyError("box payload carries neither bbox_pixels nor bbox_xyxy_normalised")

    payload["bbox_xyxy_normalised"] = [round(v, 5) for v in box.as_list]
    payload["bbox_prompt"] = to_prompt_box(box)
    payload["bbox_prompt_format"] = f"{PROMPT_BOX_FORMAT}@{PROMPT_BOX_SCALE}"
    return payload

def normalise_box_answer(text: str, *, scale: int = PROMPT_BOX_SCALE) -> str:
    """Rewrite a box answer onto the prompt scale, leaving other text alone.

    BEN.txt writes boxes as normalised floats -- ``[0.64 0.0, 1.0 0.71]`` --
    while Qwen3-VL was pretrained on 0-1000 integers. Training on both means the
    model must learn two coordinate systems for one task, and the one it already
    knows is the one we would be teaching it to abandon.

    Idempotent: a box already on the target scale (any coordinate above 1) is
    returned unchanged, so re-running over a manifest cannot multiply it twice.
    Text with no parsable box is returned untouched -- this is applied per row
    and a caption must survive it.
    """
    import re

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if len(numbers) < 4:
        return text
    values = [float(v) for v in numbers[:4]]
    if any(v > 1.0 for v in values):
        return text  # already scaled
    scaled = [int(round(v * scale)) for v in values]
    return f"[{scaled[0]} {scaled[1]}, {scaled[2]} {scaled[3]}]"
