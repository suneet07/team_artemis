import re
from dataclasses import dataclass, field
from typing import Any

from satquery.agent.bundle import ImageBundle
from satquery.agent.refusals import (
    ACTION_ADD_2ND_IMAGE,
    create_refusal,
)
from satquery.agent.task_enum import Task

COLOR_SPECTRAL_KEYWORDS = [
    "color",
    "colour",
    "red",
    "green",
    "blue",
    "rgb",
    "spectral",
    "ndvi",
    "vegetation index",
    "chlorophyll",
    "true color",
    "false color",
]

POLARIMETRIC_KEYWORDS = [
    "polarimetric decomposition",
    "pauli",
    "cloude-pottier",
    "h-alpha",
    "quad-pol",
    "full polarimetric",
    "polarimetric scattering",
]

TRAINED_GROUNDING_VOCABULARY = {
    "land cover",
    "land-cover",
    "building",
    "buildings",
    "aircraft",
    "airplane",
    "ship",
    "ships",
    "vessel",
    "urban",
    "water",
    "forest",
}


@dataclass
class ValidationResult:
    passed: bool
    refusal: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    routing_notes: list[str] = field(default_factory=list)


def validate_query_compatibility(
    bundle: ImageBundle,
    question: str,
    task: Task,
) -> ValidationResult:
    """Executes hard, non-LLM compatibility rules (V1–V9) before any tool runs.

    Collects warnings and determines whether execution must be refused.
    Never raises an exception.
    """
    warnings: list[str] = []
    notes: list[str] = []
    failures: list[dict[str, Any]] = []
    ql = question.lower()
    modalities = [img.modality for img in bundle.images]
    is_sar_only = bool(modalities and all(m == "sar" for m in modalities))

    # V1: change_* but only 1 image
    if task in (Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA, Task.CHANGE_MAP):
        if len(bundle.images) < 2:
            refusal = create_refusal(
                category="missing_input",
                reason=(
                    f"Task '{task.value}' requires a temporal pair, "
                    f"but only {len(bundle.images)} image was provided."
                ),
                action=ACTION_ADD_2ND_IMAGE,
                label="Add the second acquisition",
                suggested_questions=[
                    "What land cover types are present?",
                    "Where are the structures in this image?",
                ],
            )
            failures.append(refusal)

    # V2: CRS / extent mismatch or coregistration failure
    if bundle.coreg is not None:
        rmse = bundle.coreg.rmse_px
        if not bundle.coreg.coregistered or (rmse is not None and rmse > 1.5):
            rmse_str = (
                f"with RMSE {rmse:.2f} px (threshold: 1.50 px)"
                if rmse is not None
                else "failed"
            )
            refusal = create_refusal(
                category="validator",
                reason=(
                    f"Coregistration check failed {rmse_str}. "
                    "Images are not sufficiently aligned for analysis."
                ),
                action="reupload",
                label="Re-upload co-registered images",
            )
            failures.append(refusal)

    # V3: SAR-only + colour/spectral question
    has_color_kw = any(
        re.search(r"\b" + re.escape(kw) + r"\b", ql) for kw in COLOR_SPECTRAL_KEYWORDS
    )
    if is_sar_only and has_color_kw:
        refusal = create_refusal(
            category="modality_limitation",
            reason=(
                "Question asks about color or optical spectral properties, "
                "but the scene contains only SAR (radar) data. "
                "SAR measures surface roughness and dielectric properties (backscatter σ⁰), "
                "not optical reflectance."
            ),
            action="add_optical",
            label="Add an optical acquisition",
            suggested_questions=[
                "What is the radar backscatter level?",
                "Are high-backscatter structures present?",
                "Identify water bodies based on specular reflection.",
            ],
        )
        failures.append(refusal)

    # V8: crs_valid False / un-georeferenced input
    for img in bundle.images:
        if img.crs is None or img.crs == "":
            if task == Task.CHANGE_MAP:
                refusal = create_refusal(
                    category="validator",
                    reason=(
                        f"Input raster '{img.scene_id}' lacks a valid Coordinate "
                        "Reference System (CRS). Cannot produce georeferenced change map."
                    ),
                    action="reupload",
                    label="Re-upload with georeferenced CRS",
                )
                failures.append(refusal)
            else:
                warnings.append(
                    f"Image '{img.scene_id}' has no valid CRS; outputs will be pixel-space only."
                )

    # V4: Index needs absent band -> reroute, note in routing_notes
    if bundle.band_inventory:
        if not bundle.band_inventory.has_swir:
            notes.append(
                "SWIR band unavailable: MNDWI and NDBI indices dropped from computable set (V4)."
            )

    # V5: Pan-only/RGB-only + spectral question
    if bundle.band_inventory and bundle.band_inventory.is_pan_only:
        if any(kw in ql for kw in ("vegetation", "ndvi", "water index", "mndwi", "ndwi")):
            notes.append(
                "Panchromatic source: rerouting spectral query to texture segmentation (V5)."
            )

    # V6: Single-pol SAR + polarimetric question -> degrade & warn
    if is_sar_only and any(p in ql for p in POLARIMETRIC_KEYWORDS):
        pols = [p for img in bundle.images for p in img.polarisations]
        if len(set(pols)) <= 1:
            warnings.append(
                "Single-polarisation SAR cannot perform quad-pol decomposition; "
                "degrading to single-pol backscatter (V6)."
            )
            notes.append("Single-pol SAR: polarimetric analysis degraded to σ⁰ thresholding.")

    # V7: nodata_frac over threshold
    for img in bundle.images:
        if img.nodata_frac > 0.20:
            warnings.append(
                f"High nodata fraction ({img.nodata_frac * 100:.1f}%) in image "
                f"'{img.scene_id}'; statistics will be masked (V7)."
            )

    # V9: Grounding noun outside trained vocabulary
    if task == Task.SINGLE_GROUNDING:
        has_in_vocab = any(vocab in ql for vocab in TRAINED_GROUNDING_VOCABULARY)
        if not has_in_vocab:
            notes.append(
                "Grounding noun outside trained vocabulary: "
                "routing to deterministic object_box_fallback (V9)."
            )

    if failures:
        primary = failures[0]
        return ValidationResult(
            passed=False,
            refusal=primary,
            failures=failures,
            warnings=warnings,
            routing_notes=notes,
        )

    return ValidationResult(
        passed=True,
        failures=[],
        warnings=warnings,
        routing_notes=notes,
    )
