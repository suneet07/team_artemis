"""The prompts that produced the measured grounding and captioning numbers.

``rs_ground_caption`` ships the **base** model -- there is no adapter and there
is not meant to be one. That makes the prompt load-bearing in a way it is not
for a trained adapter: ``rs_vqa`` and ``change_vqa`` were fine-tuned on
``scale_prefix(...) + question`` and reproduce their scores from it, but the
base model reproduces nothing unless it is asked the way the benchmark asked.

Both numbers this project quotes for that tool came from a specific prompt:

* **grounding, 62.7% acc@0.5** (above published fine-tuned GeoChat at 60.6%)
  came from the ``qwen_precise`` arm, which is :data:`PRECISE_PROMPT`.
* **captioning, ROUGE-L 0.252** against a 0.222 blind floor came from the
  ``strong`` arm, which is :data:`STRONG_PROMPT`.

Serving passed the user's raw question instead. For captioning that costs the
register the metric rewards; for grounding it is worse than that, because the
box parser expects ``[{"bbox_2d": [...]}]`` on a 0-1000 coordinate grid, and
that instruction lives only in the prompt. Without it the model answers in
prose or in pixel coordinates and the reply is unparsable -- the same failure
that once made 75% of grounding replies unreadable until the 0-1000 scale was
pinned.

They live here rather than in ``scripts/`` because a serving path must not
import an evaluation script, and the evaluation scripts now import these, so
the two cannot drift.
"""

from __future__ import annotations

__all__ = ["PRECISE_PROMPT", "STRONG_PROMPT", "served_prompt"]

#: Grounding. The smallest-box rules matter as much as the format: the model
#: over-pads by default, and acc@0.5 is unforgiving about it.
PRECISE_PROMPT = (
    "Find this object in the satellite image: {phrase}\n\n"
    "Rules for the box:\n"
    "- Give the SMALLEST axis-aligned rectangle that fully contains the object.\n"
    "- If the object lies diagonally, the rectangle must still cover its full "
    "diagonal extent, corner to corner.\n"
    "- Exclude shadows, wake, surrounding pavement and neighbouring objects.\n"
    "- Targets are usually small, often under 3% of the image. Do not pad the box.\n"
    "- Coordinates run 0-1000, left to right and top to bottom.\n\n"
    'Respond with only: [{{"bbox_2d": [x1, y1, x2, y2]}}]'
)

#: Captioning. Derived from the reference captions, not invented: telling the
#: model the house style measures sight rather than register, because a caption
#: in the wrong register is marked down for register alone.
#:
#: The provenance clause is GONE, deliberately. It read
#: "The image, sourced from GoogleEarth," because that literal token
#: appears in 86% of VRSBench references, so stating it bought the n-gram.
#: It carries zero information about the picture, and the graded hidden set
#: is Cartosat-2S -- opening 'sourced from GoogleEarth' on a Cartosat scene
#: is factually false, and a system whose selling point is auditable output
#: cannot state a false provenance to farm a metric.
#:
#: The cost is real and belongs beside the number: ROUGE-L 0.252 was
#: measured WITH that opener, against a 0.222 blind floor. Removing it will
#: lower the VRSBench score by an amount nobody has measured yet;
#: scripts/eval_captioning.py is what settles it. Everything the register
#: actually needs -- 47 words, three sentences, counts as words, positional
#: phrasing, surroundings -- is untouched.
STRONG_PROMPT = (
    "Write one caption for this aerial satellite image.\n\n"
    "Format, followed exactly:\n"
    "- Open with what the image shows, features or captures.\n"
    "- About 47 words across 3 sentences.\n"
    "- Name each object type and how many, spelled as words: one, two, three, "
    "several.\n"
    "- Call objects small or large. Most are small.\n"
    "- Place each one using: in the, on the, at the, towards the -- with top, "
    "bottom, left, right, middle, corner, side, edge, or a pair such as "
    "bottom-right.\n"
    "- Close by mentioning surroundings: roads, green areas, water, buildings, "
    "parking, bare ground.\n"
    "- Plain prose. No bullet points, no bold, no headings.\n"
    "- State only what is visible. No guessing at location, purpose or season."
)


def served_prompt(mode: str, question: str) -> str:
    """The benchmarked prompt for ``mode``, or the question unchanged.

    ``mode`` is what the router planned, not what the tool guessed: the same
    manifest serves both capabilities, and only the planner knows which task
    the question was routed to. An unknown mode returns the question untouched
    rather than picking one, because silently captioning a grounding request
    would score as a bad model instead of a misrouted one.
    """
    if mode == "grounding":
        # `.rstrip(". ")` matches how the benchmark fed the phrase: the
        # reference questions end in a full stop and the prompt reads as an
        # object name, not a sentence.
        return PRECISE_PROMPT.format(phrase=question.rstrip(". "))
    if mode == "caption":
        return STRONG_PROMPT
    return question
