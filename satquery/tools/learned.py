"""The trained adapters, behind the manifest names the planner already uses.

``satquery.agent.executor`` decides a tool is unavailable when
``implementation(name)`` returns ``None``, records it, and carries on -- which
is why every answer the running system produces ends "Learned components not
yet available in this build". The adapters exist; nothing registered them.

Registering one here is the whole difference between a deterministic
remote-sensing toolkit and the system the problem statement asks for: *"A
generic LLM or VLM without remote-sensing adaptation will not satisfy the
requirements."*

**The prompt is the training prompt, and that is not a detail.** Views are
composed to three, the scale prefix comes from
:func:`satquery.training.dataset.scale_prefix`, and the question text is
whatever the corpus carried. A prompt the model was never trained on does not
fail loudly -- it answers fluently and worse, and no metric can point at the
cause. That is why the composition below repeats a view rather than padding
with blanks: repetition is what the corpus did for single-image sources, so it
is the in-distribution choice.

**Weights are loaded once per process, lazily.** The catalogue is imported by
the parameter gate and by the headless CPU path, neither of which should pull
4B of weights onto a machine that is only validating a plan.
"""

from __future__ import annotations

import os
from typing import Any

from satquery.ingest.copernicus.composites import REQUIRED_BANDS, build_composites
from satquery.tools.base import ToolContext, ToolResult

__all__ = ["LearnedVqaTool", "register_learned_tools", "BASE_MODEL"]

BASE_MODEL = os.environ.get("SATQUERY_BASE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")

#: Fallback only. The authority is ``adapter_composites(name)`` in
#: ``satquery.training.config`` -- rs_vqa 3, change_vqa 6, rs_ground_caption 2
#: -- and a single hardcoded 3 for every adapter was padding a duplicate view
#: onto the two-image tasks that were benchmarked with two.
_VIEWS = 3

#: One runner per BASE, not per adapter. Keying this by (base, adapter) loaded a
#: separate full 4B model for every registered tool -- three of them on a 22 GB
#: L4, which OOMs mid-query. The executor logs that as a tool failure and
#: answers from whatever else ran, so it surfaced as "change_vqa never runs"
#: rather than as an allocation bug.
#:
#: C1 commits to one shared base with LoRAs swapped over it, which is also what
#: makes multi-adapter serving affordable. Adapter weights are tens of MB each.
_RUNNERS: dict[str, Any] = {}


def _runner(base: str, adapter: str | None):
    """The shared runner for ``base``, with ``adapter`` made active on it."""
    if base not in _RUNNERS:
        from satquery.training.generate import VLMRunner

        # Built bare, then adapters attach on demand -- so the first tool to run
        # does not decide which adapter the base is permanently married to.
        _RUNNERS[base] = VLMRunner(base, adapter_path=None)
    runner = _RUNNERS[base]
    runner.select_adapter(adapter)
    return runner


def _views_for(name: str) -> int:
    """The view budget the adapter was trained and scored with."""
    from satquery.training.config import adapter_composites

    try:
        return adapter_composites(name)
    except ValueError:
        return _VIEWS


def _composite_views(scene, names: tuple[str, ...] | None = None) -> list | None:
    """The three training composites for one multispectral scene, or ``None``.

    ``None`` whenever the scene cannot produce them -- SAR, panchromatic, a
    plain RGB upload, a file that names no bands -- and the caller falls back.
    """
    codes = {
        designation.strip().upper(): index
        for index, designation in enumerate(scene.designations or ())
        if designation
    }
    if not REQUIRED_BANDS or any(code not in codes for code in REQUIRED_BANDS):
        return None

    planes = list(scene.bands.values())
    try:
        bands = {code: planes[codes[code]] for code in REQUIRED_BANDS}
    except IndexError:  # designations longer than the band stack
        return None
    return [
        composite.to_image()
        for composite in build_composites(bands, names=names or None)
    ]


def _views(scenes, adapter: str | None = None) -> list:
    """The views the corpus fed the model, rebuilt from the scene.

    This used to take the first three bands in file order and stretch them
    together, and the docstring claimed that was what the corpus did. It was
    not, and the gap was doing real damage on multispectral input:

    * **Channels were reversed.** Sentinel-2 stores B02, B03, B04, so file
      order is blue, green, red -- fed straight into the R, G, B slots. The
      model saw every true-colour scene with red and blue swapped.
    * **Two thirds of the input was a duplicate.** One scene produced one view,
      repeated to fill three slots. The corpus fills those slots with three
      *different* composites -- ``true_colour`` (B04/B03/B02), ``false_colour``
      (B08/B04/B03) and ``short_wave`` (B12/B11/B08) -- so NIR and SWIR reached
      the adapter in training and never at serving time.

    Both together mean questions whose answer lives outside visible light --
    water, built-up, moisture, bare soil -- were being asked of a model that
    could not see the evidence, and it guessed. ``_29_57`` is the case that
    exposed it: a chip that is 100% coniferous forest drew a confident "yes" to
    both "is a water area present" and "is a residential area present".

    ``build_composites`` is the same renderer that wrote the corpus PNGs,
    including its per-channel 2/98 stretch and its 20 m upsample, so the served
    view matches the trained one rather than merely resembling it. It raises
    rather than substituting a neighbouring band, which is why the fallback is
    guarded by an explicit band check instead of a bare ``except``.
    """
    import numpy as np
    from PIL import Image

    from satquery.ingest.copernicus.composites import composite_names_for

    budget = _views_for(adapter) if adapter else _VIEWS

    # One multispectral scene expands into the composite set the corpus used;
    # anything else keeps the original first-three-bands behaviour, which is
    # honest for SAR and for an ordinary RGB upload that has nothing more to
    # give.
    if len(scenes) == 1:
        # Which composites, not just how many: rs_ground_caption trains on two
        # and it must be the *first* two, because its sub-metre sources are RGB
        # and Cartosat-2S has no SWIR at all. Dropping whichever happened to be
        # last would be a different pair.
        names = composite_names_for(adapter) if adapter else None
        composites = _composite_views(scenes[0], names)
        if composites:
            return composites

    views = []
    for scene in scenes:
        bands = list(scene.bands)[:3] or list(scene.bands)[:1]
        planes = [np.asarray(scene.band(band), dtype="float32") for band in bands]
        while len(planes) < 3:
            planes.append(planes[-1])
        stack = np.stack(planes[:3], axis=-1)
        finite = stack[np.isfinite(stack)]
        if finite.size:
            low, high = np.percentile(finite, (2, 98))
            if high > low:
                stack = np.clip((stack - low) / (high - low), 0, 1)
        views.append(Image.fromarray((np.nan_to_num(stack) * 255).astype("uint8"), "RGB"))

    if not views:
        raise ValueError("no scene to show the model")

    # The corpus rule, from ``satquery.training.dataset``: a single view is
    # repeated up to the budget, and a row that carries several *real* views
    # passes through with the views it has. Padding a two-image change pair up
    # to three fed the adapter a duplicate date it was never scored with, and
    # truncating is the dangerous direction -- keeping the first three views of
    # a six-view row hands the model one date and it converges on the answer
    # prior.
    if len(views) == 1:
        return views * budget
    return views


def _parse_boxes(reply: str) -> list[list[float]]:
    """Boxes from a grounding reply, normalised to 0-1, or an empty list.

    `PRECISE_PROMPT` asks for `[{"bbox_2d": [x1, y1, x2, y2]}]` on a 0-1000
    grid, which is Qwen's native convention. The same three traps that cost the
    grounding benchmark a day apply here and are handled by the shared
    formatter rather than re-implemented: the `2` in `bbox_2d` must not be read
    as a coordinate, the scale must not be guessed, and a degenerate box is
    dropped rather than widened into a rectangle nothing measured.
    """
    import re

    from satquery.evalcli.formatter import format_box

    # `scale=1000` is Qwen's convention and the one PRECISE_PROMPT states. It is
    # passed explicitly rather than inferred: a `60` is 0.60 on a 0-100 grid and
    # 0.06 on a 0-1000 one, so magnitude alone cannot separate them, and
    # guessing once buried every reference box in the top-left corner.
    formatted = format_box(reply, scale=1000.0)
    if not formatted.matched:
        return []
    numbers = [float(v) for v in re.findall(r"-?\d+\.?\d*", formatted.text)]
    if len(numbers) != 4:
        return []
    x1, y1, x2, y2 = numbers
    # A degenerate box is dropped, not widened. VRSBench ships a few of these
    # itself and inventing a rectangle around one reports a detection nothing
    # measured.
    if x2 <= x1 or y2 <= y1:
        return []
    return [[round(v, 6) for v in (x1, y1, x2, y2)]]


class LearnedVqaTool:
    """One trained adapter, answering in the register its corpus taught it."""

    def __init__(self, name: str, adapter_path: str | None, base: str = BASE_MODEL):
        self.name = name
        self.adapter_path = adapter_path
        self.base = base

    def run(self, context: ToolContext) -> ToolResult:
        from satquery.agent.prompt import assemble_prompt
        from satquery.agent.served_prompts import served_prompt

        scenes = list(context.scenes)
        if not scenes:
            raise ValueError(f"{self.name} needs at least one scene")

        views = _views(scenes, self.name)

        # rs_ground_caption serves the BASE model, so its prompt is what
        # reproduces the measured numbers -- 62.7% acc@0.5 grounding and 0.252
        # ROUGE-L captioning both came from a specific prompt, not from the
        # user's raw question. The trained adapters take the opposite rule:
        # they were fine-tuned on scale_prefix + question and any other wording
        # is a format they never saw.
        question = context.query_text
        if self.name == "rs_ground_caption":
            # The planner's extracted target when it has one, so the prompt gets
            # the referring phrase the benchmark used rather than the user's
            # whole sentence.
            phrase = str(context.params.get("phrase") or question)
            question = served_prompt(
                str(context.params.get("mode", "caption")), phrase
            )

        prompt = assemble_prompt(
            [],
            [],
            [scene.modality for scene in scenes],
            # `pixel_size_m`, which is what Scene actually carries. After
            # preprocessing this IS the effective GSD -- the size of a pixel in
            # the array handed to the model -- whereas `native_gsd_m` describes
            # the file before any resampling. Reading a field Scene does not
            # have made every rs_vqa step fail with an AttributeError that the
            # executor recorded as a warning and swallowed, so the adapter
            # never ran and the answer quietly fell back to deterministic
            # evidence.
            [s.pixel_size_m for s in scenes if s.pixel_size_m],
            question,
            source=getattr(scenes[0], "source", "") or "",
            loaded_images=views,
        )

        runner = _runner(self.base, self.adapter_path)
        answer = runner.answer_all(
            [{"images": views, "question": prompt}],
            batch_size=1,
            max_new_tokens=int(context.params.get("max_new_tokens", 128)),
        )[0]

        outputs: dict[str, Any] = {"answer": answer.strip()}
        if str(context.params.get("mode", "")) == "grounding":
            # The manifest declares `boxes: bbox_list` and nothing ever filled
            # it: the tool returned only the raw text, so even a perfectly
            # formed `[{"bbox_2d": [...]}]` never became a box the console could
            # draw. The answer read as "[]" and the map stayed empty, which
            # looks like the model finding nothing rather than the box being
            # dropped on the floor.
            #
            # Parsed with the same formatter the grounding benchmark scores
            # with, so what the UI draws is what acc@0.5 measured.
            outputs["boxes"] = _parse_boxes(answer)

        return ToolResult(
            outputs=outputs,
            # No calibrated number yet: the end-to-end calibrator (4.7.3) is what
            # turns a generation into a confidence, and inventing one here would
            # put a figure in the trace that nothing measured.
            confidence=None,
            confidence_basis="learned_logprob",
            param_provenance={
                "base_model": self.base,
                "adapter": self.adapter_path or "none (base model)",
                "views": len(views),
                "prompt_prefix": prompt[: len(prompt) - len(context.query_text)],
            },
        )


def register_learned_tools(adapters: dict[str, str | None], base: str = BASE_MODEL) -> list[str]:
    """Register one implementation per available adapter.

    ``adapters`` maps a manifest name to an adapter directory, or to ``None`` to
    serve that tool from the base model. Returns the names registered, so a
    caller can report them in ``/meta/health`` -- a system claiming an adapter
    it did not load is worse than one that says it has none.
    """
    from satquery.tools.catalog import register_implementation

    registered: list[str] = []
    for name, path in adapters.items():
        register_implementation(LearnedVqaTool(name, path, base))
        registered.append(f"{name}@{'base' if path is None else 'adapter'}")
    return registered
