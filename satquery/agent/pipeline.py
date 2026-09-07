"""The end-to-end query pipeline — P1 through P8, in order.

This is the single place the whole system is wired together, and the order is the
plan's:

    ingest (P1) -> co-registration check (P3) -> rules router (4.5.2)
      -> validator gate (4.5.3) -> parameter gate (4.5.4) -> execute (4.5.6)
      -> decision-level fusion (4.7.1) -> disagreement (4.7.2)
      -> confidence (4.7.3) -> masks and evidence (4.8) -> trace

Everything above is deterministic and runs CPU-only, which is what makes the
headless mode of section 4.11 possible at all: a container with no GPU, no
network, no Redis and no queue still produces an answer and a schema-valid trace.
Learned tools upgrade the answer when their adapters are present; nothing here
depends on them existing, because the scheduling rule is that training is never
on the demo's critical path.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from satquery.agent.executor import check_plan, execute_plan
from satquery.agent.router import QueryContext, route
from satquery.agent.task_enum import RouterPath, Task
from satquery.agent.trace import TraceBuilder
from satquery.agent.validator import validate
from satquery.confidence import ConfidenceFeatures, heuristic_confidence
from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.fusion import fuse_masks
from satquery.ingest.scene import LoadedScene, load_scene
from satquery.report.masks import write_mask
from satquery.report.overlays import write_overlay
from satquery.tools.base import ToolContext
from satquery.tools.deterministic import compute_index
from satquery.tools.registry import ToolRegistry

__all__ = ["QueryOutcome", "answer_query"]

def disagreement_hints(scene, optical_response=None) -> dict:
    """Derive the evidence the section 4.7.2 rule table needs to fire.

    Three of the five rows key on something other than the two masks:
    ``wet_smooth_soil`` needs to know the optical scene reads as bare soil there,
    ``dry_smooth_sand`` needs to know the scene is arid, and ``radar_shadow``
    needs terrain. Without them only two rows could ever fire, so the
    "physically-explained disagreement" centrepiece was explaining half the cases
    it claims to — and `disagreement-cause accuracy` is a reported metric
    (section 6.2), so an unreachable row is a silently capped score.

    Soil and aridity are derivable from bands we already have. **Radar shadow is
    not**: it needs a DEM and a look direction, and inventing a slope proxy from
    the optical image would be a fabricated cause, which is worse than none. That
    row stays unreachable until a DEM is wired in, and this docstring is the
    record of why.
    """
    hints: dict = {}
    if scene is None or scene.modality != "optical":
        return hints
    try:
        ndvi = compute_index(scene, "NDVI") if scene.has("nir") and scene.has("red") else None
    except KeyError:
        ndvi = None
    if ndvi is None:
        return hints

    valid = np.isfinite(ndvi)
    if not valid.any():
        return hints

    # Bare soil: little vegetation but bright in the visible. Deliberately
    # conservative -- a rule that fires on everything explains nothing.
    brightness = np.nanmean(
        np.stack([scene.band(b) for b in ("red", "green") if scene.has(b)]), axis=0
    )
    bright_cut = np.nanpercentile(brightness[valid], 60)
    hints["optical_soil_mask"] = valid & (ndvi < 0.15) & (brightness > bright_cut)

    # Arid scene: the whole scene is unvegetated, not just the disputed patch.
    hints["arid_hint"] = bool(np.nanmedian(ndvi[valid]) < 0.1)
    return hints


_MASK_TOOLS = ("spectral_index", "sar_backscatter", "texture_seg", "object_box_fallback")


@dataclass
class QueryOutcome:
    trace: dict[str, Any]
    answer: str
    confidence: float
    refused: bool = False
    evidence: list[Path] = field(default_factory=list)
    #: Tool name -> full-resolution boolean mask, in memory. Callers that need
    #: to mosaic (the tiled path) need the arrays, not the written files.
    masks: dict[str, Any] = field(default_factory=dict)


#: Classes `texture_seg` emits. They are morphological, not material: the tool
#: separates structure from smoothness and cannot tell what a surface is made
#: of. Naming them without that qualifier let "Smooth covers 69.8%" sit beside
#: "are there any trees?" as though it had measured trees.
_TEXTURE_CLASSES = {"smooth", "builtup", "textured"}


#: Which sensor each mask tool speaks for. Named in the sentence, because on a
#: cross-modal query two tools describe the *same* class from different physics
#: and the answer read as a contradiction:
#:
#:     "Builtup covers 19.0% of the valid pixels ... Builtup covers 0.3% ..."
#:
#: Both were true -- 19.0% from optical texture, 0.3% from radar backscatter --
#: and with neither attributed, the reader has no way to tell that, or which
#: number to believe. Attribution is also the honest framing: these are two
#: opinions, which is the entire premise of the fusion panel beneath them.
_TOOL_SENSOR = {
    "spectral_index": "optical",
    "texture_seg": "optical",
    "sar_backscatter": "radar",
}


def _describe_target(outputs: dict[str, Any], tool: str = "") -> str:
    target = outputs.get("target", "the requested class")
    coverage = outputs.get("coverage_fraction")
    area = outputs.get("area_km2")
    sensor = _TOOL_SENSOR.get(tool)
    lead = f"{target} covers" if sensor is None else f"by {sensor}, {target} covers"
    parts = [f"{lead} {coverage:.1%} of the valid pixels" if coverage is not None else ""]
    if area is not None:
        parts.append(f"about {area:.3g} km²")
    described = ", ".join(part for part in parts if part)
    if tool == "texture_seg" and str(target).lower() in _TEXTURE_CLASSES:
        # Section 4.6.8: morphological evidence, not spectral. Saying so in the
        # answer rather than only in the warnings, because the warning is a
        # panel the reader may never open and this sentence is the answer.
        described += " (surface texture, not a material class)"
    return described


#: BIFOLD's measured operating point (74.95% on 6,000 held-out reBEN rows,
#: 50.1% floor). Multi-label sigmoid, so this is per class, not a softmax.
_LULC_THRESHOLD = 0.5


def _compose_answer(
    task: Task,
    steps: list,
    fusion_outcome,
    unavailable: list[str],
) -> str:
    """A plain-language answer from deterministic evidence only.

    Deliberately conservative in what it claims. Where a learned adapter would
    normally speak, this states what was measured and names the adapter that is
    missing, rather than dressing a threshold up as a caption.
    """
    # When a learned adapter answered, its answer stands alone.
    #
    # The benchmark numbers -- RSVQA-HR 85.06, RSVQA-LR 83.08, CDVQA AA 68.0 --
    # were all measured with the adapter's own text as the whole answer. Adding
    # a deterministic sentence after it does not annotate that answer, it
    # changes it: "yes" scored, "yes. Smooth covers 67.6% of the valid pixels."
    # is a different string, and a reader cannot tell which half the model said.
    # Worse, the two halves can disagree -- a threshold reporting water on a
    # scene the adapter correctly called dry -- and the appended half always
    # reads as the model's own confirmation of itself.
    #
    # The measurements are not lost. Every tool's outputs stay in the trace
    # steps and the evidence panel, where they are attributed to the tool that
    # produced them. This changes which of the two is *the answer*.
    learned: list[str] = []
    for step in steps:
        answer = str(step.result.outputs.get("answer") or "").strip()
        if answer:
            # Terminated. The adapter is trained to answer in one word -- "yes",
            # "mixed", "buildings" -- and two such answers would otherwise run
            # together into one string.
            learned.append(answer if answer.endswith((".", "!", "?")) else f"{answer}.")
    if learned:
        return " ".join(learned)

    sentences: list[str] = []
    if fusion_outcome is not None:
        verdict = fusion_outcome.verdict
        if verdict == "consistent":
            sentences.append(
                f"Optical and SAR agree on {fusion_outcome.target} "
                f"(IoU {fusion_outcome.iou:.2f})."
            )
        elif fusion_outcome.explanation:
            sentences.append(
                f"Optical and SAR disagree on {fusion_outcome.target} "
                f"(IoU {fusion_outcome.iou:.2f}): {fusion_outcome.explanation}."
            )
        else:
            sentences.append(
                f"Optical and SAR only partly agree on {fusion_outcome.target} "
                f"(IoU {fusion_outcome.iou:.2f}); reporting the agreed extent."
            )

    for step in steps:
        outputs = step.result.outputs
        if step.tool in _MASK_TOOLS and outputs.get("coverage_fraction") is not None:
            sentences.append(_describe_target(outputs, step.tool).capitalize() + ".")
        if step.tool == "change_stats" and outputs.get("change_ratio") is not None:
            sentences.append(
                f"{outputs['change_ratio']:.1%} of the valid area changed"
                + (
                    f", with the largest change in {outputs['largest_change_class']}."
                    if outputs.get("largest_change_class")
                    else "."
                )
            )
        if step.tool == "lulc_classifier" and outputs.get("labels"):
            # The 19-class radar inventory, which nothing else in the stack can
            # produce. Rendering only the classes the model is actually
            # confident about: a multi-label sigmoid emits all nineteen every
            # time, and listing the 0.002 ones as findings would bury the
            # signal in its own tail.
            # 0.5, because that is the operating point the tool was measured
            # at: 74.95% on 6,000 held-out reBEN rows against a 50.1% floor.
            # A lower display threshold would show classes the reported number
            # never covered, so the answer and the benchmark would describe
            # different behaviour.
            strong = [
                label
                for label in outputs["labels"]
                if label.get("score", 0) >= _LULC_THRESHOLD
            ]
            shown = strong or outputs["labels"][:1]
            named = ", ".join(
                f"{label['class']} ({label['score']:.0%})" for label in shown
            )
            sentences.append(
                f"Radar land cover: {named}"
                + ("" if strong else ", though no class is confidently present")
                + "."
            )
        if step.tool == "object_box_fallback":
            count = len(outputs.get("boxes", []))
            sentences.append(
                f"{count} deterministic box proposal{'s' if count != 1 else ''} for "
                f"'{outputs.get('target')}' — classical vision, not learned detection."
            )
        if step.tool == "coreg_check" and outputs.get("rmse_px") is not None:
            sentences.append(
                f"The pair is co-registered to {outputs['rmse_px']:.2f} px "
                f"({outputs.get('method')})."
            )

    if unavailable:
        sentences.append(
            "Learned components not yet available in this build: "
            + ", ".join(sorted(set(unavailable)))
            + ". The answer above rests on deterministic evidence only."
        )
    if not sentences:
        sentences.append(
            "No deterministic tool produced a measurement for this question, and no "
            "learned adapter is loaded, so there is nothing to report honestly."
        )
    return " ".join(sentences)


def _compatibility(loaded: list[LoadedScene]) -> dict[str, Any]:
    """The compatibility report the trace schema and the UI panel expect.

    ``graph.py`` sets this from the bundle; ``answer_query`` never did, so every
    trace from the pipeline path -- which is the one the API serves -- shipped
    without it. Compatibility checking is a named deliverable in the problem
    statement, and a panel with nothing in it reads as a system that does not
    check rather than one that forgot to say so.

    Assembled from what ingest already measured, and shaped by
    ``configs/trace_schema.json`` rather than by what seemed reasonable: the
    schema types ``coregistered`` as a plain boolean, so a single scene reports
    ``True`` -- one input is trivially co-registered with the empty set of
    others -- while ``checks_passed`` omits ``crs_match``, which only means
    something across two or more scenes. That keeps the honest distinction in
    the field that can carry it instead of in one that cannot.

    ``rmse_px`` stays ``None`` here. The real figure comes from ``coreg_check``
    on the bi-temporal and cross-modal paths, which the executor records as its
    own step, and inventing a number would put a measurement in the trace that
    nothing measured.
    """
    reports = [entry.ingest.compatibility for entry in loaded]
    crs_list = [entry.scene.crs for entry in loaded if entry.scene.crs]
    common_crs = crs_list[0] if crs_list and len(set(crs_list)) == 1 else None

    checks: list[str] = []
    if reports and all(r.format_ok for r in reports):
        checks.append("format_ok")
    if reports and all(r.crs_valid for r in reports):
        checks.append("crs_valid")
    if common_crs and len(loaded) > 1:
        checks.append("crs_match")

    primary = reports[0] if reports else None
    return {
        "coregistered": True if len(loaded) < 2 else bool(common_crs),
        "rmse_px": None,
        "correction_applied": False,
        "method": None,
        "common_crs": common_crs,
        "checks_passed": checks,
        "ingest": {
            "format_ok": bool(primary and primary.format_ok),
            "crs_valid": bool(primary and primary.crs_valid),
            "georeferenced_required_for_masks": True,
            "modality": primary.modality if primary else None,
            "modality_source": primary.modality_source if primary else None,
            "bands_present": list(primary.bands_present) if primary else [],
            "computable_indices": list(primary.computable_indices) if primary else [],
            "nodata_frac": round(primary.nodata_frac, 6) if primary else 0.0,
            "bit_depth": primary.bit_depth if primary else None,
            "bit_depth_source": primary.bit_depth_source if primary else None,
            "pixel_size_m": primary.pixel_size_m if primary else None,
            "native_gsd_m": primary.native_gsd_m if primary else None,
            "warnings": list(primary.warnings) if primary else [],
        },
    }


def answer_query(
    question: str,
    image_paths: list[str | Path],
    *,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
    output_dir: Path | str | None = None,
    config: PreprocessingConfig | None = None,
    registry: ToolRegistry | None = None,
    write_evidence: bool = True,
) -> QueryOutcome:
    """Run one query end to end and return the answer plus its graded trace."""
    cfg = config if config is not None else preprocessing_config()
    registry = registry or ToolRegistry.default()
    output_dir = Path(output_dir) if output_dir else None

    loaded: list[LoadedScene] = []
    for index, path in enumerate(image_paths):
        override = modalities[index] if modalities and index < len(modalities) else None
        date = dates[index] if dates and index < len(dates) else None
        loaded.append(load_scene(path, modality_override=override, date=date, config=cfg))

    scenes = [entry.scene for entry in loaded]
    builder = TraceBuilder(question)
    builder.set_inputs([entry.as_trace_input() for entry in loaded])
    builder.set_compatibility(_compatibility(loaded))

    context = QueryContext(
        query_text=question,
        modalities=[scene.modality for scene in scenes],
        inventories=[scene.inventory for scene in scenes],
        image_count=len(scenes),
        dates=[scene.date for scene in scenes if scene.date],
    )
    decision = route(context)
    builder.set_routing(decision.task, decision.router_path)

    validation = validate(
        context,
        decision,
        nodata_fractions=[entry.ingest.compatibility.nodata_frac for entry in loaded],
        nodata_warn_fraction=cfg.ingest.nodata_warn_fraction,
        crs_list=[scene.crs for scene in scenes],
    )
    for note in validation.routing_notes:
        builder.add_routing_note(note)
    for entry in loaded:
        for warning in entry.warnings:
            builder.add_warning(warning)
    for warning in validation.warnings:
        builder.add_warning(warning)

    if not validation.passed:
        refusal = validation.refusal
        assert refusal is not None
        builder.set_parameter_check(True, [])
        builder.set_tools_invoked([])
        builder.set_outputs(answer=refusal.reason, confidence=0.0, refusal=refusal.as_dict())
        trace = builder.build()
        return QueryOutcome(trace=trace, answer=refusal.reason, confidence=0.0, refused=True)

    band_inventory = scenes[0].inventory
    gate = check_plan(
        validation.plan,
        registry,
        band_inventory=band_inventory,
        modalities=[scene.modality for scene in scenes],
    )
    for record in gate.checked:
        builder.add_planned_step(
            record["tool"],
            record["params"],
            within_manifest=record["within_manifest"],
            defaults_applied=record.get("defaults_applied"),
        )
    builder.set_parameter_check(gate.passed, gate.rejected)

    if not gate.passed:
        reason = (
            "The plan was rejected before execution because its parameters are not "
            "permitted by the tool manifests: " + "; ".join(gate.rejected)
        )
        builder.add_warning(reason)
        builder.set_tools_invoked([])
        builder.set_outputs(
            answer=reason,
            confidence=0.0,
            refusal={"reason": "; ".join(gate.rejected), "category": "parameter_gate"},
        )
        return QueryOutcome(trace=builder.build(), answer=reason, confidence=0.0, refused=True)

    for name in gate.unavailable:
        note = (
            f"'{name}' is planned but its adapter is not loaded in this build; the "
            f"deterministic path answers without it (scheduling rule, Part 8)"
        )
        builder.add_routing_note(note)
        # And as a warning, because routing notes are not in the emitted trace.
        # A tool that was planned and then silently dropped is indistinguishable
        # from one that was never planned, and that ambiguity cost real time:
        # `lulc_classifier` -- the component holding the 74.95% radar number --
        # was missing from a served SAR answer with nothing anywhere saying why.
        builder.add_warning(note)

    tool_context = ToolContext(query_text=question, scenes=scenes, params={}, config=cfg)
    execution = execute_plan(gate.runnable, tool_context)
    for warning in execution.warnings:
        builder.add_warning(warning)

    # --- D1: decision-level fusion when both modalities produced a mask ---
    masks = tool_context.artifacts.get("masks", {})
    fusion_outcome = None
    optical_step = next(
        (s for s in execution.steps if s.tool in ("spectral_index", "texture_seg")), None
    )
    sar_step = next((s for s in execution.steps if s.tool == "sar_backscatter"), None)
    if optical_step is not None and sar_step is not None:
        optical_mask = optical_step.result.mask
        sar_mask = sar_step.result.mask
        if (
            optical_mask is not None
            and sar_mask is not None
            and optical_mask.shape == sar_mask.shape
        ):
            fusion_outcome = fuse_masks(
                str(optical_step.result.outputs.get("target", "water")),
                optical_mask,
                sar_mask,
                optical_confidence=optical_step.result.confidence or 0.5,
                sar_confidence=sar_step.result.confidence or 0.5,
                config=cfg,
                # `tool_context`, not `context`. Both are in scope here and only
                # one of them has scenes: `context` is the router's
                # QueryContext, which carries modalities and band inventories,
                # while ToolContext carries the loaded rasters. Reading the
                # wrong one raised AttributeError inside the fusion branch, so
                # every crossmodal query -- the one path where an optical and a
                # SAR mask both exist and get reconciled -- failed outright with
                # "'QueryContext' object has no attribute 'scene_of'".
                **disagreement_hints(tool_context.scene_of("optical")),
            )
            # A texture mask is not a spectral opinion, and the D1 table is
            # written for spectral-vs-radar.
            #
            # `texture_seg` answers "smooth or textured"; `sar_backscatter`
            # answers "bright or dark". On a scene whose bands cannot compute an
            # index -- every raster that arrives without band descriptions,
            # which includes all of reBEN -- the two masks measure different
            # quantities, so a low IoU is the expected result and not evidence
            # that the sensors conflict. Reported as a conflict with no
            # explanation, it reads as the system catching a real disagreement.
            #
            # Not suppressed: the agreed extent is still the honest answer, and
            # the masks are still worth showing. What changes is that the
            # warning says why the comparison is weak instead of leaving the
            # reader to assume the sensors disagreed about the ground.
            if optical_step.tool == "texture_seg" and fusion_outcome.verdict != "consistent":
                builder.add_warning(
                    "the optical side of this comparison is a texture proxy, not a "
                    "spectral index: this scene's bands do not name themselves, so "
                    "no index could be computed. Texture and radar backscatter "
                    "measure different quantities, so a low agreement here is "
                    "expected and is not evidence that the two sensors disagree "
                    "about the surface"
                )
            builder.set_agreement(
                fusion_outcome.iou, fusion_outcome.verdict, fusion_outcome.disagreement_cause
            )
            for warning in fusion_outcome.warnings:
                builder.add_warning(warning)
            masks["fused"] = fusion_outcome.mask
        else:
            builder.add_warning(
                "optical and SAR masks are on different grids, so decision-level fusion "
                "was skipped; co-register the pair first (P3)"
            )

    # --- steps into the trace ---
    for step in execution.steps:
        params = dict(step.params)
        params.update(step.result.param_provenance)
        builder.add_step(
            step.tool,
            params,
            step.result.outputs,
            confidence=step.result.confidence,
            latency_ms=step.latency_ms,
            param_source="manifest_validated",
        )

    # --- masks and evidence (C18) ---
    evidence: list[Path] = []
    mask_uris: list[str] = []
    if write_evidence and output_dir is not None and masks:
        primary = scenes[0]
        for name, mask in masks.items():
            if mask is None or np.asarray(mask).shape != primary.shape:
                continue
            try:
                record = write_mask(
                    output_dir / f"{name}_mask.tif",
                    mask,
                    crs=primary.crs,
                    transform=primary.transform,
                    scene_shape=primary.shape,
                    description=name,
                )
            except ValueError as error:
                builder.add_warning(f"mask '{name}' not exported: {error}")
                continue
            mask_uris.append(record.uri)
            builder.add_evidence(record.uri)
            overlay = write_overlay(
                output_dir / f"{name}_overlay.png",
                mask,
                base=[primary.band(band) for band in list(primary.bands)[:3]],
                target=str(name),
            )
            evidence.append(overlay)
            builder.add_evidence(str(overlay))

    # --- confidence (4.7.3) ---
    # A step that produced nothing (an unmet dependency, a refused
    # co-registration) reports confidence 0. Feeding that into the minimum makes
    # the system claim certainty that the answer is wrong, which is a different
    # and equally false statement. Such steps become a penalty and a warning
    # instead, and only steps that actually measured something set the floor.
    measured = [
        step.result.confidence
        for step in execution.steps
        if step.result.confidence is not None and step.result.confidence > 0.0
    ]
    no_ops = sum(
        1
        for step in execution.steps
        if step.result.confidence is not None and step.result.confidence == 0.0
    )
    confidences = measured
    fallbacks = [
        1.0
        for step in execution.steps
        if str(step.result.param_provenance.get("threshold_method", "")).endswith("fallback")
    ]
    features = ConfidenceFeatures(
        tool_confidence_min=min(confidences) if confidences else 0.4,
        tool_confidence_mean=float(np.mean(confidences)) if confidences else 0.4,
        agreement_iou=fusion_outcome.iou if fusion_outcome else 1.0,
        threshold_fallback_fraction=len(fallbacks) / max(1, len(execution.steps)),
        router_is_rules=1.0 if decision.router_path is RouterPath.RULES else 0.0,
        warning_count=float(len(builder.warnings)),
        deterministic_fallback_used=1.0
        if any(step.tool == "object_box_fallback" for step in execution.steps)
        else 0.0,
    )
    confidence = heuristic_confidence(features)
    if no_ops:
        confidence *= 0.6**no_ops

    answer = _compose_answer(decision.task, execution.steps, fusion_outcome, gate.unavailable)
    builder.set_tools_invoked(execution.tools_invoked)
    area = next(
        (
            step.result.outputs["area_km2"]
            for step in execution.steps
            if "area_km2" in step.result.outputs
        ),
        None,
    )
    builder.set_outputs(
        answer=answer,
        masks=mask_uris or None,
        area_km2=area,
        confidence=round(confidence, 4),
        extra={"confidence_basis": "heuristic"},
    )
    trace = builder.build()
    return QueryOutcome(
        trace=trace,
        answer=answer,
        confidence=confidence,
        evidence=evidence,
        masks={name: mask for name, mask in masks.items() if mask is not None},
    )
