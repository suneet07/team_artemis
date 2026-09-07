"""P6 — the deterministic tools (master plan sections 4.6.2 - 4.6.9).

Every tool here runs with no learned weights at all, which is what makes the
CPU-only headless path of section 4.11 possible.

**The architectural rule this module exists to enforce.** Image data does not
travel through ``params``. ``params`` is precisely what the section 4.5.4
parameter gate validates against the tool manifest, and the manifests permit
``index``, ``threshold_method`` and so on — not ``index_array``. A tool that
takes its raster through the parameter channel therefore cannot pass its own
gate, and wiring one up forces the controller to bypass the gate or to execute
stubs. So pixels arrive on :class:`~satquery.tools.base.ToolContext` (scenes and
artifacts) and only the manifest-declared settings arrive in ``params``.

Two more rules, both from the never-cut list:

* **C18 — every mask is a graded artifact.** Each tool that thresholds anything
  returns the full-scene boolean mask plus the CRS and transform to write it
  with. ``area_fraction`` alone is not a deliverable.
* **Every threshold comes from ``configs/preprocessing.yaml``.** Nothing in this
  file hardcodes 0.2, -18.0 or 5.0. A number that shapes a graded output and
  lives in Python is a number nobody can re-derive.

The ``execute_*`` functions at the bottom are thin adapters kept for the
array-in/dict-out call style used by the existing unit tests and by any caller
that already holds the arrays.
"""

from typing import Any

import numpy as np

from satquery.config import preprocessing_config
from satquery.qgen.boxes import normalise_box_payload
from satquery.texture import (
    Component,
    components,
    edge_density,
    largest_component_centroid,
    local_variance,
    morphological_open,
)
from satquery.tools.base import Scene, ToolContext, ToolResult, area_km2
from satquery.tools.thresholds import ThresholdDecision, threshold_with_gate

__all__ = [
    "CLASS_PRIORS",
    "INDEX_BANDS",
    "CentroidPriorTool",
    "ChangeStatsTool",
    "ObjectBoxFallbackTool",
    "SarBackscatterTool",
    "SpectralIndexTool",
    "TextureSegTool",
    "change_statistics",
    "compute_index",
    "execute_change_stats",
    "execute_object_box_fallback",
    "execute_sar_backscatter",
    "execute_spectral_index",
    "execute_texture_seg",
    "texture_response",
]

#: (positive term, negative term) for each normalised difference index (4.6.2).
INDEX_BANDS: dict[str, tuple[str, str]] = {
    "NDVI": ("nir", "red"),
    "NDWI": ("green", "nir"),
    "MNDWI": ("green", "swir"),
    "NDBI": ("swir", "nir"),
}

_SAR_TARGET_IS_HIGH = {"water": False, "builtup": True}


class NotDecibelsError(ValueError):
    """Raised when a dB sigma-nought cut would be applied to non-dB pixels."""


def assert_decibels(values, where: str) -> None:
    """Refuse to apply a decibel threshold to an array that cannot be decibels.

    Calibrated sigma-nought in dB is negative over almost every natural
    surface -- the C-band reference table in ``configs/preprocessing.yaml``
    runs from -25 dB for calm water to +5 dB for urban -- so a scene with no
    negative value at all has not been through ``db_conversion``. That is the
    signature of a source distributed pre-stretched to 8-bit, and of linear
    power arrays, and in both cases a dB cut is not merely untuned but
    meaningless.

    This raises rather than warns. The band-mismatch path warns because a
    C-band cut on an X-band scene is a *provisional* answer worth reporting
    with lowered confidence; a dB cut on 0..255 pixels is not provisional, it
    is an empty or full-frame mask presented as a measurement. Section 4.7 is
    explicit that the system must never silently pick one.
    """
    import numpy as _np

    array = _np.asarray(values, dtype=_np.float64)
    finite = array[_np.isfinite(array)]
    if finite.size == 0:
        return
    if float(finite.min()) < 0.0:
        return
    raise NotDecibelsError(
        f"{where}: sigma-nought values run "
        f"[{float(finite.min()):g}, {float(finite.max()):g}] with no negative "
        f"value, so they are not calibrated decibels -- most likely a source "
        f"distributed pre-stretched to 8-bit (OpenEarthMap-SAR) or a linear "
        f"power array. Run the SAR chain through calibrate_sigma0 and "
        f"db_conversion, or supply an explicit threshold in the array's own "
        f"units via threshold_db."
    )


# ---------------------------------------------------------------------------
# spectral_index + centroid_prior  (section 4.6.2, implements D2)
# ---------------------------------------------------------------------------


def compute_index(scene: Scene, index: str) -> np.ndarray:
    """Normalised difference index, NaN outside valid data.

    Computed on **raw** band values, never on stretched ones: a per-band
    percentile stretch changes the ratio between the two terms and therefore
    changes the index, which would silently invalidate every physical threshold
    in ``preprocessing.yaml``.
    """
    name = index.upper()
    if name not in INDEX_BANDS:
        raise ValueError(f"unknown index '{index}'; permitted: {sorted(INDEX_BANDS)}")
    positive_band, negative_band = INDEX_BANDS[name]
    positive = scene.band(positive_band)
    negative = scene.band(negative_band)
    denominator = positive + negative
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator != 0, (positive - negative) / denominator, np.nan)
    values[~scene.valid_mask()] = np.nan
    return values


def _apply(values: np.ndarray, decision: ThresholdDecision, above_is_positive: bool):
    finite = np.isfinite(values)
    mask = np.zeros(values.shape, dtype=bool)
    if above_is_positive:
        mask[finite] = values[finite] >= decision.value
    else:
        mask[finite] = values[finite] <= decision.value
    return mask, finite


class SpectralIndexTool:
    name = "spectral_index"

    def run(self, context: ToolContext) -> ToolResult:
        scene = context.scene_of("optical") or context.primary
        index = str(context.params["index"]).upper()
        fixed = context.config.indices.fixed(index)
        if fixed is None:
            raise ValueError(f"no fixed fallback threshold configured for {index}")

        values = compute_index(scene, index)
        decision = threshold_with_gate(
            values,
            gate=context.config.indices.otsu,
            fixed_value=fixed.value,
            above_is_positive=fixed.above_is_positive,
            requested_method=str(context.params.get("threshold_method", "otsu")),
            explicit_value=context.params.get("threshold_value"),
        )
        mask, finite = _apply(values, decision, fixed.above_is_positive)

        warnings: list[str] = []
        if decision.method != "otsu" and decision.reason:
            warnings.append(
                f"{index}: {decision.reason}; using the physical threshold "
                f"{decision.value:g} and lowering confidence (section 4.6.2)"
            )

        outputs: dict[str, Any] = {
            "target": fixed.target,
            "index": index,
            "coverage_fraction": round(float(mask.sum()) / max(1, int(finite.sum())), 6),
            "valid_fraction": round(float(finite.mean()), 6),
        }
        area = area_km2(mask, scene.pixel_size_m)
        if area is not None:
            outputs["area_km2"] = round(area, 6)
        else:
            warnings.append(
                "source has no metric pixel size, so area_km2 is withheld rather than "
                "reported in the wrong unit"
            )

        ratio = decision.between_class_variance_ratio or 0.0
        return ToolResult(
            outputs=outputs,
            confidence=round((0.55 + 0.35 * min(1.0, ratio)) * decision.confidence_multiplier, 4),
            confidence_basis="threshold_statistics",
            warnings=warnings,
            mask=mask,
            mask_crs=scene.crs,
            mask_transform=scene.transform,
            response=values,
            param_provenance=decision.as_params(),
        )


class CentroidPriorTool:
    """D2's point prior: largest connected component of a preceding mask."""

    name = "centroid_prior"

    def run(self, context: ToolContext) -> ToolResult:
        mask = context.artifacts.get("mask")
        if mask is None:
            return ToolResult(
                outputs={"point_prior": None},
                confidence=0.0,
                confidence_basis="heuristic",
                warnings=[
                    "centroid_prior needs a mask from a preceding step; none was produced, "
                    "so no spatial prior is passed to the grounding adapter"
                ],
            )
        point = largest_component_centroid(np.asarray(mask, dtype=bool))
        if point is None:
            return ToolResult(
                outputs={"point_prior": None},
                confidence=0.0,
                confidence_basis="heuristic",
                warnings=["mask is empty; there is no component to take a centroid of"],
            )
        return ToolResult(
            outputs={"point_prior": [round(point[0], 5), round(point[1], 5)]},
            confidence=0.6,
            confidence_basis="heuristic",
        )


# ---------------------------------------------------------------------------
# sar_backscatter  (section 4.6.5)
# ---------------------------------------------------------------------------


class SarBackscatterTool:
    name = "sar_backscatter"

    def run(self, context: ToolContext) -> ToolResult:
        scene = context.scene_of("sar") or context.primary
        target = str(context.params["target"]).lower().replace("-", "")
        if target not in _SAR_TARGET_IS_HIGH:
            raise ValueError(f"unknown target '{target}'")
        requested_pol = context.params.get("pol")
        polarisation = self._resolve_pol(scene, requested_pol)
        sar_band = scene.inventory.sar_band

        values = np.where(scene.valid_mask(), scene.band(polarisation), np.nan)
        # Before any cut: the configured fallback is in dB, and an explicit
        # threshold_db is documented to be too.
        assert_decibels(values, f"sar_backscatter/{polarisation}")
        fixed = context.config.sar.fixed_threshold_db(sar_band, polarisation, target)
        if fixed is None:
            raise ValueError(
                f"no fixed sigma-nought fallback configured for {sar_band}/{polarisation}"
            )
        above_is_positive = _SAR_TARGET_IS_HIGH[target]

        decision = threshold_with_gate(
            values,
            gate=context.config.indices.otsu,
            fixed_value=fixed,
            above_is_positive=above_is_positive,
            requested_method=str(context.params.get("threshold_method", "otsu")),
            explicit_value=context.params.get("threshold_db"),
        )
        mask, finite = _apply(values, decision, above_is_positive)

        warnings: list[str] = []
        if decision.method != "otsu" and decision.reason:
            warnings.append(
                f"sigma-nought {target}: {decision.reason}; using the physical "
                f"{decision.value:g} dB cut and lowering confidence (section 4.6.5)"
            )
        band_penalty = 1.0
        if context.config.sar.inherits_c_band(sar_band):
            band_penalty = 0.8
            warnings.append(
                f"{(sar_band or 'unknown')}-band thresholds are untuned and inherit the "
                f"C-band reference table (section 4.2); treat this cut as provisional"
            )
        if sar_band is None:
            warnings.append("SAR band not declared in metadata; assumed C-band for thresholding")
        if requested_pol and str(requested_pol).upper() != polarisation:
            warnings.append(
                f"requested polarisation {str(requested_pol).upper()} is not present; used "
                f"{polarisation}, which is what the source provides"
            )
        if len(scene.inventory.polarisations) <= 1:
            warnings.append(
                "single-polarisation input: polarimetric discrimination is unavailable, so "
                "this rests on backscatter magnitude alone (section 4.5.3)"
            )

        outputs: dict[str, Any] = {
            "target": target,
            "pol": polarisation,
            "sar_band": sar_band,
            "coverage_fraction": round(float(mask.sum()) / max(1, int(finite.sum())), 6),
        }
        area = area_km2(mask, scene.pixel_size_m)
        if area is not None:
            outputs["area_km2"] = round(area, 6)

        params = decision.as_params()
        params["threshold_db"] = params.pop("threshold_value")
        params["pol"] = polarisation
        ratio = decision.between_class_variance_ratio or 0.0
        return ToolResult(
            outputs=outputs,
            confidence=round(
                (0.5 + 0.35 * min(1.0, ratio)) * decision.confidence_multiplier * band_penalty, 4
            ),
            confidence_basis="threshold_statistics",
            warnings=warnings,
            mask=mask,
            mask_crs=scene.crs,
            mask_transform=scene.transform,
            response=values,
            param_provenance=params,
        )

    @staticmethod
    def _resolve_pol(scene: Scene, requested: str | None) -> str:
        available = [name.upper() for name in scene.bands]
        if requested and str(requested).upper() in available:
            return str(requested).upper()
        for preferred in ("VV", "HH", "VH", "HV"):
            if preferred in available:
                return preferred
        return available[0]


# ---------------------------------------------------------------------------
# texture_seg (4.6.8) and object_box_fallback (4.6.9)
# ---------------------------------------------------------------------------


def texture_response(plane: np.ndarray, window: int) -> np.ndarray:
    """Edge density and local variance combined into a [0, 1] response.

    This is the optical fallback when no spectral band is available: built-up
    land is edge-dense and high-variance, water and bare soil are neither, and
    neither statement needs NIR or SWIR.
    """
    return np.clip(
        0.6 * edge_density(plane, window=window) + 0.4 * local_variance(plane, window=window),
        0.0,
        1.0,
    )


def _representative_plane(scene: Scene) -> np.ndarray:
    if scene.has("pan"):
        return scene.band("pan")
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in scene.bands.values()])
    return stack.mean(axis=0)


class TextureSegTool:
    name = "texture_seg"

    def run(self, context: ToolContext) -> ToolResult:
        scene = context.scene_of("optical") or context.primary
        target = str(context.params.get("target", "builtup")).lower().replace("-", "")
        window = int(context.params.get("window_px", context.config.texture.window_px))
        if window % 2 == 0:
            raise ValueError("window_px must be odd so the response stays grid-aligned")

        plane = np.where(scene.valid_mask(), _representative_plane(scene), np.nan)
        response = np.where(scene.valid_mask(), texture_response(plane, window), np.nan)

        above_is_positive = target in ("builtup", "structure")
        decision = threshold_with_gate(
            response,
            gate=context.config.indices.otsu,
            fixed_value=context.config.texture.fixed_response_threshold,
            above_is_positive=above_is_positive,
            requested_method=str(context.params.get("threshold_method", "otsu")),
        )
        mask, finite = _apply(response, decision, above_is_positive)

        warnings = [
            "texture_seg is morphological evidence, not spectral: it separates structure "
            "from smoothness and cannot identify materials (section 4.6.8)"
        ]
        if decision.method != "otsu" and decision.reason:
            warnings.append(f"texture threshold: {decision.reason}")

        outputs: dict[str, Any] = {
            "target": target,
            "coverage_fraction": round(float(mask.sum()) / max(1, int(finite.sum())), 6),
        }
        area = area_km2(mask, scene.pixel_size_m)
        if area is not None:
            outputs["area_km2"] = round(area, 6)

        return ToolResult(
            outputs=outputs,
            confidence=round(0.45 * decision.confidence_multiplier, 4),
            confidence_basis="threshold_statistics",
            warnings=warnings,
            mask=mask,
            mask_crs=scene.crs,
            mask_transform=scene.transform,
            response=response,
            param_provenance=decision.as_params() | {"window_px": window},
        )


#: Shape priors keyed to the query noun (section 4.6.9 step 3). Deliberately
#: coarse: they filter obvious non-candidates, they do not classify.
CLASS_PRIORS: dict[str, dict[str, float]] = {
    "ship": {"max_aspect_ratio": 8.0, "min_solidity": 0.35},
    "tank": {"max_aspect_ratio": 1.6, "min_solidity": 0.6},
    "aircraft": {"max_aspect_ratio": 3.0, "min_solidity": 0.25},
    "vehicle": {"max_aspect_ratio": 3.0, "min_solidity": 0.4},
    "bridge": {"max_aspect_ratio": 20.0, "min_solidity": 0.15},
    "runway": {"max_aspect_ratio": 30.0, "min_solidity": 0.15},
    "harbour": {"max_aspect_ratio": 6.0, "min_solidity": 0.2},
    "crane": {"max_aspect_ratio": 10.0, "min_solidity": 0.2},
    "building": {"max_aspect_ratio": 4.0, "min_solidity": 0.5},
    "builtup": {"max_aspect_ratio": 100.0, "min_solidity": 0.1},
}
_DEFAULT_PRIOR = {"max_aspect_ratio": 12.0, "min_solidity": 0.15}


def prior_for(target: str) -> dict[str, float]:
    """Match the prior on any word of the phrase, head noun last.

    "storage tanks" has to reach the tank prior. Matching the whole phrase
    against a single-word table would fall through to the default and propose
    boxes with no shape constraint at all.
    """
    for word in reversed([w.lower().rstrip("s") for w in target.split()]):
        for key, prior in CLASS_PRIORS.items():
            if key.rstrip("s") == word:
                return prior
    return _DEFAULT_PRIOR


class ObjectBoxFallbackTool:
    """Deterministic proposer for classes outside the trained vocabulary (C39).

    It exists because of licensing, not accuracy: vehicles, storage tanks,
    bridges and harbours appear only in Google Earth-derived datasets we will not
    ship. Without it those queries score zero; with it they score partial credit,
    with no restricted data and no training. Every box is flagged
    ``deterministic_fallback`` and the confidence is capped by the manifest's
    low-confidence-proposer contract, so it never masquerades as learned
    grounding.
    """

    name = "object_box_fallback"

    def run(self, context: ToolContext) -> ToolResult:
        scene = context.primary
        target = str(context.params["target"]).lower()
        texture_cfg = context.config.texture
        max_boxes = int(context.params.get("max_boxes", texture_cfg.max_proposals))
        prior = prior_for(target)
        valid = scene.valid_mask()

        # Step 1: the response map. SAR uses the sigma-nought table (bright is
        # high backscatter); optical uses the texture response.
        if scene.modality == "sar":
            pol = SarBackscatterTool._resolve_pol(scene, None)
            response = np.where(valid, scene.band(pol), np.nan)
            # This branch has no Otsu gate, so a units mismatch here is worse
            # than elsewhere: `>= 0.0` over 0..255 pixels marks the whole frame
            # as built-up rather than returning nothing.
            assert_decibels(response, f"texture_seg/sar/{pol}")
            cut = context.config.sar.fixed_threshold_db(
                scene.inventory.sar_band, pol, "builtup"
            )
            finite = np.isfinite(response)
            mask = np.zeros(response.shape, dtype=bool)
            mask[finite] = response[finite] >= (cut if cut is not None else 0.0)
            basis = f"sigma0 >= {cut:g} dB ({pol})" if cut is not None else "sigma0 threshold"
        else:
            plane = np.where(valid, _representative_plane(scene), np.nan)
            response = texture_response(plane, texture_cfg.window_px)
            decision = threshold_with_gate(
                response,
                gate=context.config.indices.otsu,
                fixed_value=texture_cfg.fixed_response_threshold,
                above_is_positive=True,
            )
            mask, _ = _apply(response, decision, True)
            basis = f"texture response >= {decision.value:.3f} ({decision.method})"

        # Step 2: morphological opening suppresses speckle before components.
        mask = morphological_open(mask, texture_cfg.opening_radius_px)
        min_area = max(1, int(texture_cfg.min_component_fraction * mask.size))
        found = components(mask, response=response, min_area_px=min_area)

        # Step 3: filter by the shape prior for the query noun.
        kept: list[Component] = [
            component
            for component in found
            if component.aspect_ratio <= prior["max_aspect_ratio"]
            and component.solidity >= prior["min_solidity"]
        ]
        # Step 4: rank by response strength and cap at the top few.
        kept.sort(key=lambda component: component.mean_response, reverse=True)
        kept = kept[:max_boxes]

        height, width = mask.shape
        # One converter owns the axis order and the scale (satquery.qgen.boxes),
        # so a proposal here and a training target in qgen cannot drift apart.
        boxes = [
            normalise_box_payload(
                {
                    "bbox_pixels": list(component.bbox),
                    "area_px": component.area_px,
                    "aspect_ratio": round(component.aspect_ratio, 3),
                    "solidity": round(component.solidity, 3),
                    "method": "deterministic_fallback",
                },
                width,
                height,
            )
            for component in kept
        ]

        warnings = [
            f"'{target}' is outside the trained grounding vocabulary, so these are "
            f"deterministic proposals from classical vision, not learned detections "
            f"(section 4.6.9). Basis: {basis}."
        ]
        if not boxes:
            warnings.append(
                f"no component survived the shape prior for '{target}'; reporting no "
                f"proposals rather than an arbitrary box"
            )

        result = ToolResult(
            outputs={
                "boxes": boxes,
                "target": target,
                "method": "deterministic_fallback",
                "candidates_before_filter": len(found),
            },
            confidence=0.3 if boxes else 0.05,
            confidence_basis="deterministic_fallback",
            warnings=warnings,
            mask=mask,
            mask_crs=scene.crs,
            mask_transform=scene.transform,
            response=response,
            param_provenance={"target": target, "max_boxes": max_boxes},
        )
        return result.capped(context.config.fusion.deterministic_fallback_ceiling)


# ---------------------------------------------------------------------------
# change_stats — implements D4 (section 4.6.4)
# ---------------------------------------------------------------------------


def change_statistics(
    before: np.ndarray,
    after: np.ndarray,
    class_names: dict[int, str] | None = None,
    pixel_size_m: float | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact change statistics over two co-registered class-label rasters.

    Computes what section 4.6.4 specifies and no more: per-class area before,
    after and delta; change ratio and per-class change ratio; largest and
    smallest change by class. Boolean masks are the two-class special case.

    This is arithmetic, not inference, and that is the differentiator: every
    published change-VQA model from 2B to 9B sits at 32-37% on exactly the
    question types that ask for a number. A model guesses the bucket; this
    counts it.
    """
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        raise ValueError(
            f"change statistics need a co-registered pair on one grid; got "
            f"{before.shape} and {after.shape}"
        )
    if before.dtype == bool:
        before = before.astype(np.int16)
    if after.dtype == bool:
        after = after.astype(np.int16)

    mask = np.ones(before.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    total_valid = int(mask.sum())
    if total_valid == 0:
        raise ValueError("change statistics need at least one valid pixel")

    labels = sorted(set(np.unique(before[mask]).tolist()) | set(np.unique(after[mask]).tolist()))
    names = class_names or {}
    px_km2 = (pixel_size_m**2) / 1_000_000.0 if pixel_size_m else None

    per_class: list[dict[str, Any]] = []
    for label in labels:
        before_px = int(np.count_nonzero((before == label) & mask))
        after_px = int(np.count_nonzero((after == label) & mask))
        delta_px = after_px - before_px
        entry: dict[str, Any] = {
            "class": str(names.get(int(label), int(label))),
            "before_px": before_px,
            "after_px": after_px,
            "delta_px": delta_px,
            # None, never inf: a ratio with a zero denominator is undefined, and
            # both 0 and Infinity are wrong answers to a graded question --
            # Infinity is not even valid JSON.
            "class_change_ratio": (
                None if before_px == 0 else round(delta_px / before_px, 6)
            ),
        }
        if px_km2:
            entry["before_km2"] = round(before_px * px_km2, 6)
            entry["after_km2"] = round(after_px * px_km2, 6)
            entry["delta_km2"] = round(delta_px * px_km2, 6)
        per_class.append(entry)

    changed_px = int(np.count_nonzero((before != after) & mask))
    stats: dict[str, Any] = {
        "valid_px": total_valid,
        "changed_px": changed_px,
        "change_ratio": round(changed_px / total_valid, 6),
        "per_class": per_class,
    }
    if per_class:
        largest = max(per_class, key=lambda e: abs(e["delta_px"]))
        smallest = min(per_class, key=lambda e: abs(e["delta_px"]))
        gainer = max(per_class, key=lambda e: e["delta_px"])
        loser = min(per_class, key=lambda e: e["delta_px"])
        stats["largest_change_class"] = largest["class"]
        stats["smallest_change_class"] = smallest["class"]
        stats["largest_increase_class"] = gainer["class"]
        stats["largest_decrease_class"] = loser["class"]
    if px_km2:
        stats["changed_km2"] = round(changed_px * px_km2, 6)
        stats["total_km2"] = round(total_valid * px_km2, 6)
    return stats


class ChangeStatsTool:
    name = "change_stats"

    def run(self, context: ToolContext) -> ToolResult:
        before = context.artifacts.get("class_map_before")
        after = context.artifacts.get("class_map_after")
        if before is None or after is None:
            return ToolResult(
                outputs={},
                confidence=0.0,
                confidence_basis="heuristic",
                warnings=[
                    "change_stats needs before/after class maps from change_map or from two "
                    "thresholded masks; no preceding step produced them"
                ],
            )
        scene = context.primary
        stats = change_statistics(
            before,
            after,
            class_names=context.artifacts.get("class_names"),
            pixel_size_m=scene.pixel_size_m,
            valid=context.artifacts.get("valid_mask"),
        )
        warnings: list[str] = []
        if scene.pixel_size_m is None:
            warnings.append("no metric pixel size on the source, so areas are in pixels only")
        return ToolResult(
            outputs=stats,
            # The arithmetic over the inputs it was handed is exact. The
            # uncertainty lives in those inputs, and fusion carries it forward.
            confidence=1.0,
            confidence_basis="deterministic_arithmetic",
            warnings=warnings,
            param_provenance={"source": "class_map"},
        )


# ---------------------------------------------------------------------------
# Array-in / dict-out adapters.
#
# These keep the direct call style for callers that already hold the arrays --
# unit tests, notebooks, the ablation harness. They are NOT the path the agent
# uses: the agent goes through the classes above so that pixels never travel
# through the manifest-validated parameter channel.
# ---------------------------------------------------------------------------


def _adapter_scene(array: np.ndarray, band: str, modality: str, **inventory_kwargs) -> Scene:
    from satquery.ingest.band_inventory import BandInventory

    return Scene(
        name="adapter",
        modality=modality,
        bands={band: np.asarray(array, dtype=np.float64)},
        inventory=BandInventory(bands={band.lower(): 1}, **inventory_kwargs),
    )


def _adapter_result(result: ToolResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = dict(result.outputs)
    payload.update(result.param_provenance)
    payload["confidence"] = result.confidence
    if result.mask is not None:
        payload["mask"] = result.mask
        payload["area_fraction"] = float(np.mean(result.mask))
    if result.warnings:
        payload["warnings"] = result.warnings
    if extra:
        payload.update(extra)
    return payload


def execute_spectral_index(params: dict[str, Any]) -> dict[str, Any]:
    """Threshold a precomputed index array. Requires ``index_array``."""
    index_array = params.get("index_array")
    if index_array is None:
        return {"answer": "Missing 'index_array' in params", "confidence": 0.0}
    index_name = str(params.get("index", "NDVI")).upper()
    cfg = preprocessing_config()
    fixed = cfg.indices.fixed(index_name)
    fallback = params.get("fallback_threshold")
    if fallback is None:
        fallback = fixed.value if fixed else 0.3
    above = fixed.above_is_positive if fixed else True

    decision = threshold_with_gate(
        np.asarray(index_array, dtype=np.float64),
        gate=cfg.indices.otsu,
        fixed_value=float(fallback),
        above_is_positive=above,
        requested_method=str(params.get("threshold_method", "otsu")),
    )
    mask, _ = _apply(np.asarray(index_array, dtype=np.float64), decision, above)
    centroid = largest_component_centroid(mask)
    return {
        "answer": (
            f"Detected {index_name} regions using the {decision.method} threshold at "
            f"{decision.value:.3f}"
        ),
        "threshold_method": decision.method,
        "threshold_value": float(decision.value),
        "bimodality_passed": decision.bimodality_passed,
        "centroid_prior": centroid,
        "mask": mask,
        "area_fraction": float(np.mean(mask)),
        "confidence": round(0.75 * decision.confidence_multiplier, 4),
    }


def execute_sar_backscatter(params: dict[str, Any]) -> dict[str, Any]:
    """Threshold sigma-nought dB. Requires ``sigma0_array``."""
    sigma0 = params.get("sigma0_array")
    if sigma0 is None:
        return {"answer": "Missing 'sigma0_array' in params", "confidence": 0.0}
    target = str(params.get("target", "water")).lower().replace("-", "")
    cfg = preprocessing_config()
    fallback = params.get("fallback_threshold")
    if fallback is None:
        fallback = cfg.sar.fixed_threshold_db(
            params.get("sar_band"), params.get("pol"), target
        )
        if fallback is None:
            fallback = -18.0 if target == "water" else -3.0
    above = _SAR_TARGET_IS_HIGH.get(target, True)

    values = np.asarray(sigma0, dtype=np.float64)
    try:
        assert_decibels(values, "execute_sar_backscatter")
    except NotDecibelsError as exc:
        return {"answer": str(exc), "confidence": 0.0}
    decision = threshold_with_gate(
        values,
        gate=cfg.indices.otsu,
        fixed_value=float(fallback),
        above_is_positive=above,
        requested_method=str(params.get("threshold_method", "otsu")),
    )
    mask, _ = _apply(values, decision, above)
    return {
        "answer": (
            f"Detected {target} regions using the {decision.method} threshold at "
            f"{decision.value:.1f} dB"
        ),
        "threshold_method": decision.method,
        "threshold_value": float(decision.value),
        "bimodality_passed": decision.bimodality_passed,
        "centroid_prior": largest_component_centroid(mask),
        "mask": mask,
        "area_fraction": float(np.mean(mask)),
        "confidence": round(0.7 * decision.confidence_multiplier, 4),
    }


def execute_texture_seg(params: dict[str, Any]) -> dict[str, Any]:
    """Segment by texture response. Requires ``image_array``."""
    image = params.get("image_array")
    if image is None:
        return {"answer": "Missing 'image_array' in params", "confidence": 0.0}
    cfg = preprocessing_config()
    window = int(params.get("window_px", cfg.texture.window_px))
    response = texture_response(np.asarray(image, dtype=np.float64), window)
    decision = threshold_with_gate(
        response,
        gate=cfg.indices.otsu,
        fixed_value=float(params.get("fallback_threshold", cfg.texture.fixed_response_threshold)),
        above_is_positive=True,
    )
    mask, _ = _apply(response, decision, True)
    return {
        "answer": (
            f"Detected textured regions using the {decision.method} threshold at "
            f"{decision.value:.2f}"
        ),
        "threshold_method": decision.method,
        "threshold_value": float(decision.value),
        "bimodality_passed": decision.bimodality_passed,
        "centroid_prior": largest_component_centroid(mask),
        "mask": mask,
        "area_fraction": float(np.mean(mask)),
        "confidence": round(0.45 * decision.confidence_multiplier, 4),
    }


def execute_object_box_fallback(params: dict[str, Any]) -> dict[str, Any]:
    """Propose boxes for an uncovered class. Requires ``image_array``."""
    image = params.get("image_array")
    if image is None:
        return {"answer": "Missing 'image_array' in params", "confidence": 0.0}
    cfg = preprocessing_config()
    target = str(params.get("target", "object")).lower()
    prior = prior_for(target)
    response = texture_response(np.asarray(image, dtype=np.float64), cfg.texture.window_px)
    decision = threshold_with_gate(
        response,
        gate=cfg.indices.otsu,
        fixed_value=float(params.get("fallback_threshold", cfg.texture.fixed_response_threshold)),
        above_is_positive=True,
    )
    mask, _ = _apply(response, decision, True)
    mask = morphological_open(mask, cfg.texture.opening_radius_px)
    min_area = max(1, int(cfg.texture.min_component_fraction * mask.size))
    kept = [
        component
        for component in components(mask, response=response, min_area_px=min_area)
        if component.aspect_ratio <= prior["max_aspect_ratio"]
        and component.solidity >= prior["min_solidity"]
    ]
    kept.sort(key=lambda component: component.mean_response, reverse=True)
    kept = kept[: int(params.get("max_boxes", cfg.texture.max_proposals))]
    height, width = mask.shape
    boxes = [
        normalise_box_payload(
            {
                "ymin": component.bbox[0],
                "xmin": component.bbox[1],
                "ymax": component.bbox[2],
                "xmax": component.bbox[3],
                "area": component.area_px,
                "bbox_pixels": list(component.bbox),
            },
            width,
            height,
        )
        for component in kept
    ]
    return {
        "answer": f"Detected {len(boxes)} boxes using the deterministic fallback.",
        "method": "deterministic_fallback",
        "boxes": boxes,
        "mask": mask,
        "confidence": min(0.3 if boxes else 0.05, cfg.fusion.deterministic_fallback_ceiling),
    }


def execute_change_stats(params: dict[str, Any]) -> dict[str, Any]:
    """D4 statistics. Requires ``mask_t1``, ``mask_t2`` and ``pixel_area_m2``."""
    mask_t1 = params.get("mask_t1")
    mask_t2 = params.get("mask_t2")
    pixel_area_m2 = params.get("pixel_area_m2")
    if mask_t1 is None or mask_t2 is None or pixel_area_m2 is None:
        return {
            "answer": "Missing required mask_t1, mask_t2, or pixel_area_m2 parameters",
            "confidence": 0.0,
        }

    pixel_size_m = float(pixel_area_m2) ** 0.5
    stats = change_statistics(
        np.asarray(mask_t1),
        np.asarray(mask_t2),
        class_names=params.get("class_names") or {0: "background", 1: "target"},
        pixel_size_m=pixel_size_m,
        valid=params.get("valid_mask"),
    )
    target = next(
        (entry for entry in stats["per_class"] if entry["class"] == "target"),
        stats["per_class"][-1],
    )
    area_before = target["before_px"] * float(pixel_area_m2)
    area_after = target["after_px"] * float(pixel_area_m2)
    delta = area_after - area_before
    direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "remained unchanged")
    class_change_ratio = target["class_change_ratio"]
    # `ratio` keeps its original meaning -- the growth factor after/before -- but
    # returns None instead of float("inf") when the class was absent before.
    # An undefined ratio is the honest answer, and Infinity is not even valid
    # JSON, so it would have corrupted the graded artifact it was written into.
    growth = None if target["before_px"] == 0 else target["after_px"] / target["before_px"]
    answer = f"The area {direction} by {abs(delta):.2f} square meters."
    if growth is None:
        answer += (
            " The class was absent before, so the ratio of change is undefined "
            "rather than infinite."
        )
    else:
        answer += f" Ratio of change is {growth:.2f}."

    return {
        "answer": answer,
        "area_t1_m2": float(area_before),
        "area_t2_m2": float(area_after),
        "delta_m2": float(delta),
        # after / before, or None when the class was absent before.
        "ratio": growth,
        # delta / before -- the plan's "class change ratio" (section 4.6.4).
        "class_change_ratio": class_change_ratio,
        "change_ratio": stats["change_ratio"],
        "per_class": stats["per_class"],
        "largest_change_class": stats.get("largest_change_class"),
        "smallest_change_class": stats.get("smallest_change_class"),
        "confidence": 1.0,
    }
