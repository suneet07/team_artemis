"""P5 — the validator gate (master plan section 4.5.3).

Non-LLM, hard rules, and it runs **before any tool executes**. The rule table is
the plan's, verbatim:

===================================================  ==========================
Condition                                            Action
===================================================  ==========================
change query + 1 image                               refuse, request the second
mismatched CRS / extent                              register, or refuse w/ RMSE
SAR-only + colour or spectral question                explain the limitation
requested index needs an absent band                 reroute (D3), log it
pan-only / RGB-only optical + spectral question      reroute, else explain
single-pol SAR + polarimetric question               explain, answer what holds
nodata fraction over threshold                       warn, mask the statistics
===================================================  ==========================

The framing matters as much as the rules: *a graceful, explained refusal scores
better than a confident hallucination*, and "compatibility checking" is a named
deliverable of the problem statement. A refusal here is a product feature with a
schema-valid category, not an error path.
"""

from dataclasses import dataclass, field
from typing import Any

from satquery.agent.router import (
    POLARIMETRIC_TERMS,
    SPECTRAL_TERMS,
    QueryContext,
    RoutingDecision,
    contains,
)
from satquery.agent.task_enum import Task
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.manifest import ToolManifest
from satquery.tools.registry import check_parameters

__all__ = ["Refusal", "ToolValidator", "ValidationResult", "validate"]

_CHANGE_TASKS = (Task.CHANGE_VQA, Task.CHANGE_MAP, Task.CHANGE_DESCRIPTION)
_COLOUR_TERMS = ("colour", "color", "rgb", "how green", "what shade", "hue")

#: Spectral questions that texture cannot stand in for. Edge density and local
#: variance can tell you where vegetation *is*; they cannot tell you whether it
#: is healthy, how much chlorophyll it carries, or how moist the soil is. Those
#: are radiometric quantities, and a texture answer to them would be a confident
#: fabrication -- which the plan is explicit scores worse than a refusal.
_IRREDUCIBLY_SPECTRAL_TERMS = (
    "healthy",
    "health",
    "ndvi",
    "ndwi",
    "ndbi",
    "mndwi",
    "chlorophyll",
    "moisture",
    "stress",
    "vigour",
    "vigor",
    "greenness",
    "spectral",
    "reflectance",
)


@dataclass(frozen=True)
class Refusal:
    """A refusal the trace can carry: category is from the frozen schema enum."""

    reason: str
    category: str

    def as_dict(self) -> dict:
        return {"reason": self.reason, "category": self.category}


@dataclass
class ValidationResult:
    passed: bool
    refusal: Refusal | None = None
    routing_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Steps the gate removed from the plan, with the reason, so the trace can
    #: show what was *not* run and why (D3 substitutions land here).
    dropped_steps: list[dict] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.passed and bool(self.plan)


def validate(
    context: QueryContext,
    decision: RoutingDecision,
    *,
    nodata_fractions: list[float] | None = None,
    nodata_warn_fraction: float = 0.5,
    crs_list: list[str | None] | None = None,
) -> ValidationResult:
    """Run the section 4.5.3 table over a routing decision."""
    text = context.lowered
    notes: list[str] = list(decision.notes)
    warnings: list[str] = []
    plan = [dict(step) for step in decision.plan]
    dropped: list[dict] = []

    # --- change query with one image ------------------------------------
    if decision.task in _CHANGE_TASKS and context.image_count < 2:
        return ValidationResult(
            passed=False,
            refusal=Refusal(
                reason=(
                    "This is a change question, and change needs two images of the same "
                    "area at two times. Only one was supplied. Upload the second "
                    "acquisition and this becomes answerable."
                ),
                category="missing_input",
            ),
            routing_notes=notes,
            warnings=warnings,
        )

    # --- cross-modal task with one modality -----------------------------
    cross_modal_task = decision.task in (
        Task.CROSSMODAL_EXTRACTION,
        Task.CROSSMODAL_VQA,
    )
    if cross_modal_task and not context.is_cross_modal:
        return ValidationResult(
            passed=False,
            refusal=Refusal(
                reason=(
                    "This question compares optical and SAR evidence, but only "
                    f"{', '.join(context.modalities) or 'one modality'} was supplied."
                ),
                category="missing_input",
            ),
            routing_notes=notes,
            warnings=warnings,
        )

    # --- mismatched CRS -------------------------------------------------
    if crs_list:
        declared = [crs for crs in crs_list if crs]
        if len(set(declared)) > 1:
            warnings.append(
                f"inputs are in different coordinate systems ({sorted(set(declared))}); "
                f"co-registration reprojects to a common metric grid before comparison"
            )
        if len(declared) < len(crs_list):
            warnings.append(
                "an input has no CRS: masks cannot be geo-referenced for it, and a mask "
                "without a CRS is not a scoreable artifact (C18)"
            )

    # --- SAR-only + colour or spectral question -------------------------
    sar_only = context.has_sar and not context.has_optical
    if sar_only and contains(text, _COLOUR_TERMS):
        return ValidationResult(
            passed=False,
            refusal=Refusal(
                reason=(
                    "This is a SAR scene. Radar measures backscatter, not reflected "
                    "sunlight, so it carries no colour information at all. It can answer "
                    "questions about structure, roughness, water extent and built-up "
                    "areas — ask one of those and this scene will answer it."
                ),
                category="modality_limitation",
            ),
            routing_notes=notes,
            warnings=warnings,
        )
    if sar_only and contains(text, SPECTRAL_TERMS):
        notes.append(
            "spectral question on a SAR-only input; answered from backscatter evidence "
            "with the limitation stated rather than refused outright"
        )
        warnings.append(
            "SAR cannot measure vegetation health or spectral indices; the answer rests "
            "on backscatter structure alone and its confidence is lowered accordingly"
        )

    # --- pan-only / RGB-only optical + spectral question ----------------
    if context.optical_is_spectrally_blind and contains(text, SPECTRAL_TERMS):
        if contains(text, _IRREDUCIBLY_SPECTRAL_TERMS) and not context.has_sar:
            return ValidationResult(
                passed=False,
                refusal=Refusal(
                    reason=(
                        "This sensor cannot answer this question. It carries no "
                        "near-infrared or shortwave-infrared band, and vegetation health, "
                        "chlorophyll, moisture and every normalised index are radiometric "
                        "quantities that need one. Texture can tell you where vegetation "
                        "is, not how it is doing. What this scene can answer is extent, "
                        "structure and built-up area -- or supply a multispectral or SAR "
                        "acquisition of the same place."
                    ),
                    category="modality_limitation",
                ),
                routing_notes=notes,
                warnings=warnings,
            )
        if context.has_sar:
            notes.append(
                "optical source has no NIR or SWIR, so no spectral index is computable; "
                "SAR takes primary for this question (D3)"
            )
        elif any(step["tool"] == "texture_seg" for step in plan):
            notes.append(
                "optical source has no NIR or SWIR; rerouted to texture and morphology "
                "evidence at reduced confidence (sections 4.1.4 and 4.6.8)"
            )
        else:
            return ValidationResult(
                passed=False,
                refusal=Refusal(
                    reason=(
                        "This sensor cannot answer spectral questions: the source carries "
                        "no near-infrared or shortwave-infrared band, and every vegetation "
                        "and moisture index needs one. What it can answer is structure, "
                        "extent and built-up area — or supply a multispectral or SAR "
                        "acquisition of the same scene."
                    ),
                    category="modality_limitation",
                ),
                routing_notes=notes,
                warnings=warnings,
            )

    # --- single-pol SAR + polarimetric question -------------------------
    if context.is_single_pol and contains(text, POLARIMETRIC_TERMS):
        notes.append(
            "single-polarisation source: polarimetric decomposition is unavailable, so "
            "the answer is limited to what one channel supports (section 4.5.3)"
        )
        warnings.append(
            "the question asks about polarimetry but the scene carries one polarisation; "
            "answering from backscatter magnitude only"
        )

    # --- D3: drop steps whose bands are absent --------------------------
    indices = set(context.computable_indices())
    kept: list[dict] = []
    for step in plan:
        if step["tool"] == "spectral_index":
            index = str(step["params"].get("index", "")).upper()
            if index and index not in indices:
                dropped.append({"step": step, "reason": f"{index} needs a band this source lacks"})
                notes.append(
                    f"{index} unavailable: source lacks the required band; "
                    f"substituted per D3 band-availability routing"
                )
                continue
        kept.append(step)
    plan = kept

    if not plan and decision.task not in (Task.SINGLE_CAPTION,):
        return ValidationResult(
            passed=False,
            refusal=Refusal(
                reason=(
                    "No tool in the registry can answer this question with the bands and "
                    "modalities supplied. "
                    + (
                        "The source lacks the spectral bands every applicable index needs."
                        if dropped
                        else "The question does not map onto a supported task."
                    )
                ),
                category="modality_limitation",
            ),
            routing_notes=notes,
            warnings=warnings,
            dropped_steps=dropped,
        )

    # --- nodata ---------------------------------------------------------
    for index, fraction in enumerate(nodata_fractions or []):
        if fraction > nodata_warn_fraction:
            warnings.append(
                f"input {index} is {fraction:.0%} nodata; statistics are computed over the "
                f"valid pixels only and areas are correspondingly partial"
            )

    return ValidationResult(
        passed=True,
        routing_notes=notes,
        warnings=warnings,
        dropped_steps=dropped,
        plan=plan,
    )


class ToolValidator:
    """The section 4.5.4 parameter gate.

    This used to be a second, weaker implementation of parameter checking: it
    read ``required`` and ``choices``, which are not keys the manifest schema
    has (it uses ``optional``, ``values`` and ``range``), and it performed **no
    range check at all** — so the plan's own worked example, a ``spectral_index``
    threshold of 47, passed straight through the gate whose entire purpose is to
    stop it.

    It now delegates to :func:`satquery.tools.registry.check_parameters`, which
    is the single implementation of the rule: unknown names rejected, values
    outside their declared range or enum **rejected and never silently clamped**,
    and band prerequisites checked against the BandInventory.
    """

    @classmethod
    def validate_parameters(
        cls,
        tool_name: str,
        params: dict[str, Any],
        manifest: "ToolManifest | dict[str, Any]",
        band_inventory: BandInventory | None = None,
        modalities: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate ``params`` for ``tool_name`` against its manifest.

        Accepts either a :class:`~satquery.tools.manifest.ToolManifest` or the
        raw mapping a manifest was parsed from; callers held both.
        """
        if isinstance(manifest, dict):
            manifest = ToolManifest.from_dict(manifest)
        if manifest.name != tool_name:
            return {
                "passed": False,
                "rejected": [
                    f"manifest '{manifest.name}' does not describe tool '{tool_name}'"
                ],
            }
        outcome = check_parameters(
            manifest, params, band_inventory=band_inventory, modalities=modalities
        )
        return {"passed": outcome.passed, "rejected": list(outcome.rejected)}
