"""P7 — decision-level fusion and disagreement classification (section 4.7).

**D1, the centrepiece.** Two independent decisions, then reconcile. *Not*
feature-level: EarthMind's own ablation showed naive token concatenation fails on
optical-SAR heterogeneity, and decision-level fusion is more robust to residual
misregistration and gives interpretable confidence — which is the only kind of
confidence worth reporting.

============  =================================  ==========================
Target        Optical evidence                   SAR evidence
============  =================================  ==========================
Water         MNDWI (or NDWI, or texture_seg)    sigma-nought below threshold
Built-up      NDBI (or texture_seg)              sigma-nought high (double bounce)
============  =================================  ==========================

The disagreement table is the part nobody else in the landscape survey has. When
the two modalities conflict we name a *physical* cause, name which modality wins
and why, and lower confidence. **Never silently pick one.** A system that says
"the radar sees water here and the optical does not, because there is cloud over
it, so trust the radar" is doing something a concatenation of tokens cannot.
"""

from dataclasses import dataclass, field

import numpy as np

from satquery.config import PreprocessingConfig, preprocessing_config

__all__ = [
    "DISAGREEMENT_RULES",
    "slope_mask_from_dem",
    "DisagreementRule",
    "FusionOutcome",
    "classify_disagreement",
    "fuse_masks",
    "iou",
]


def slope_mask_from_dem(
    dem: np.ndarray,
    pixel_size_m: float,
    *,
    look_azimuth_deg: float,
    incidence_deg: float = 35.0,
) -> np.ndarray:
    """Terrain facing away from the sensor steeply enough to be in radar shadow.

    This is the evidence the ``radar_shadow`` row of the section 4.7.2 table
    needs, and it is the reason that row cannot fire without a DEM. Radar shadow
    is a *geometry* fact: a slope is shadowed when it tilts away from the sensor
    more steeply than the beam's grazing angle, so it needs the terrain **and**
    the look direction. Neither is derivable from an optical image, and proxying
    slope from optical texture would put a fabricated physical cause into a
    graded trace — worse than reporting no cause at all.

    ``look_azimuth_deg`` is the direction the sensor looks *towards*, clockwise
    from north. ``incidence_deg`` is the beam incidence angle; a slope is
    shadowed when the terrain slope away from the sensor exceeds ``90 -
    incidence``.
    """
    dem = np.asarray(dem, dtype=np.float64)
    if pixel_size_m <= 0:
        raise ValueError("a DEM needs a positive metric pixel size to give a slope")

    # Gradient in metres per metre. Row index increases southward, so the
    # north-south component is negated to point north.
    d_dy, d_dx = np.gradient(dem, pixel_size_m)
    north = -d_dy
    east = d_dx

    # Component of the terrain gradient along the look direction. Positive means
    # the ground rises away from the sensor; negative means it falls away, which
    # is the shadowing case.
    azimuth = np.deg2rad(look_azimuth_deg)
    along_look = east * np.sin(azimuth) + north * np.cos(azimuth)

    grazing = np.deg2rad(90.0 - incidence_deg)
    return np.arctan(-along_look) > grazing


@dataclass(frozen=True)
class DisagreementRule:
    """One row of the section 4.7.2 table."""

    key: str
    cause: str
    winner: str
    explanation: str


#: Section 4.7.2, with the physical explanation each row rests on. The order is
#: the evaluation order: the first rule whose condition holds wins.
DISAGREEMENT_RULES: tuple[DisagreementRule, ...] = (
    DisagreementRule(
        key="sar_water_optical_not",
        cause="cloud_over_water",
        winner="sar",
        explanation=(
            "radar sees water where the optical sensor does not; cloud is opaque to "
            "optical wavelengths and transparent to C- and X-band radar, so the radar "
            "observation is the reliable one here"
        ),
    ),
    DisagreementRule(
        key="sar_dark_optical_soil",
        cause="wet_smooth_soil",
        winner="optical",
        explanation=(
            "radar reads dark but the optical spectrum says bare soil; wet or very "
            "smooth soil is specular at radar wavelengths and mimics water, while the "
            "spectral signature separates the two"
        ),
    ),
    DisagreementRule(
        key="sar_dark_terrain_slope",
        cause="radar_shadow",
        winner="optical",
        explanation=(
            "the dark radar return follows terrain facing away from the sensor: this is "
            "radar shadow, a geometry artifact rather than a surface property"
        ),
    ),
    DisagreementRule(
        key="sar_bright_over_water",
        cause="wind_roughened_surface",
        winner="optical",
        explanation=(
            "radar reads bright over what the optical sensor resolves as water; wind "
            "roughens the surface and raises backscatter, so the water is still water"
        ),
    ),
    DisagreementRule(
        key="sar_dark_arid_region",
        cause="dry_smooth_sand",
        winner="optical",
        explanation=(
            "dry smooth sand is specular at radar wavelengths and reads as dark as "
            "water; the optical spectrum distinguishes sand from water unambiguously"
        ),
    ),
)

_RULES_BY_KEY = {rule.key: rule for rule in DISAGREEMENT_RULES}


def iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Intersection over union of two boolean masks over the valid pixels."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"masks must share a grid to be compared: {a.shape} vs {b.shape}")
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        a = a & mask
        b = b & mask
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0  # both empty: they agree that nothing is there
    return float(np.count_nonzero(a & b) / union)


@dataclass
class FusionOutcome:
    """What D1 produced, and how much of it to believe."""

    target: str
    mask: np.ndarray
    iou: float
    verdict: str  # "consistent" | "partial" | "conflict" | "single_modality"
    disagreement_cause: str | None = None
    winning_modality: str | None = None
    explanation: str | None = None
    confidence: float = 0.0
    confidence_basis: str = "heuristic"
    warnings: list[str] = field(default_factory=list)
    agreement_px: int = 0
    optical_only_px: int = 0
    sar_only_px: int = 0

    def as_agreement_block(self) -> dict:
        return {
            "iou": round(self.iou, 4),
            "verdict": self.verdict,
            "disagreement_cause": self.disagreement_cause,
        }


def classify_disagreement(
    target: str,
    optical_mask: np.ndarray,
    sar_mask: np.ndarray,
    *,
    optical_soil_mask: np.ndarray | None = None,
    slope_mask: np.ndarray | None = None,
    arid_hint: bool = False,
) -> DisagreementRule | None:
    """Name the physical cause of a conflict, or None if the table does not fire.

    Returning None is a real answer: it means "these two disagree and we do not
    have a physical explanation for it", which is more honest than forcing a
    label. The caller lowers confidence either way.
    """
    optical = np.asarray(optical_mask, dtype=bool)
    sar = np.asarray(sar_mask, dtype=bool)
    sar_only = sar & ~optical
    optical_only = optical & ~sar

    if target == "water":
        if sar_only.sum() > optical_only.sum():
            # Radar sees water the optical does not. Which explanation depends on
            # what the optical scene says is there instead.
            soil_overlap = (
                (sar_only & optical_soil_mask).sum()
                if optical_soil_mask is not None
                else 0
            )
            if soil_overlap > 0.5 * sar_only.sum():
                return _RULES_BY_KEY["sar_dark_optical_soil"]
            if slope_mask is not None and (sar_only & slope_mask).sum() > 0.5 * sar_only.sum():
                return _RULES_BY_KEY["sar_dark_terrain_slope"]
            if arid_hint:
                return _RULES_BY_KEY["sar_dark_arid_region"]
            return _RULES_BY_KEY["sar_water_optical_not"]
        if optical_only.sum() > 0:
            # Optical resolves water where radar reads bright.
            return _RULES_BY_KEY["sar_bright_over_water"]
    return None


def fuse_masks(
    target: str,
    optical_mask: np.ndarray | None,
    sar_mask: np.ndarray | None,
    *,
    optical_confidence: float = 0.5,
    sar_confidence: float = 0.5,
    valid: np.ndarray | None = None,
    config: PreprocessingConfig | None = None,
    **disagreement_hints,
) -> FusionOutcome:
    """Reconcile the optical and SAR decisions for one target (section 4.7.1)."""
    cfg = config if config is not None else preprocessing_config()

    if optical_mask is None and sar_mask is None:
        raise ValueError("fusion needs at least one modality's decision")
    if optical_mask is None or sar_mask is None:
        only = optical_mask if optical_mask is not None else sar_mask
        modality = "optical" if optical_mask is not None else "sar"
        return FusionOutcome(
            target=target,
            mask=np.asarray(only, dtype=bool),
            iou=1.0,
            verdict="single_modality",
            winning_modality=modality,
            confidence=round(
                (optical_confidence if modality == "optical" else sar_confidence) * 0.9, 4
            ),
            warnings=[
                f"only the {modality} decision is available, so there is no cross-modal "
                f"corroboration; confidence is reduced accordingly"
            ],
            agreement_px=int(np.count_nonzero(only)),
        )

    optical = np.asarray(optical_mask, dtype=bool)
    sar = np.asarray(sar_mask, dtype=bool)
    if optical.shape != sar.shape:
        raise ValueError(
            f"decision-level fusion needs both masks on the same grid: "
            f"{optical.shape} vs {sar.shape}. Co-register first (P3)."
        )
    if valid is not None:
        keep = np.asarray(valid, dtype=bool)
        optical = optical & keep
        sar = sar & keep

    overlap = iou(optical, sar)
    both = optical & sar
    warnings: list[str] = []

    if overlap >= cfg.fusion.iou_consistent:
        return FusionOutcome(
            target=target,
            mask=optical | sar,
            iou=overlap,
            verdict="consistent",
            confidence=round(
                min(0.97, max(optical_confidence, sar_confidence) + 0.1 * overlap), 4
            ),
            agreement_px=int(both.sum()),
            optical_only_px=int((optical & ~sar).sum()),
            sar_only_px=int((sar & ~optical).sum()),
        )

    rule = classify_disagreement(target, optical, sar, **disagreement_hints)
    verdict = "conflict" if overlap < cfg.fusion.iou_conflict else "partial"

    if rule is None:
        warnings.append(
            f"optical and SAR disagree on {target} (IoU {overlap:.2f}) and no physical "
            f"rule in the section 4.7.2 table explains it. Reporting the agreed extent "
            f"only, with reduced confidence, rather than choosing a modality silently."
        )
        return FusionOutcome(
            target=target,
            mask=both,
            iou=overlap,
            verdict=verdict,
            confidence=round(
                min(optical_confidence, sar_confidence) * cfg.fusion.disagreement_penalty, 4
            ),
            warnings=warnings,
            agreement_px=int(both.sum()),
            optical_only_px=int((optical & ~sar).sum()),
            sar_only_px=int((sar & ~optical).sum()),
        )

    winner = optical if rule.winner == "optical" else sar
    winner_confidence = optical_confidence if rule.winner == "optical" else sar_confidence
    warnings.append(
        f"optical and SAR disagree on {target} (IoU {overlap:.2f}): {rule.explanation}. "
        f"Taking the {rule.winner} decision and lowering confidence."
    )
    return FusionOutcome(
        target=target,
        mask=winner,
        iou=overlap,
        verdict=verdict,
        disagreement_cause=rule.cause,
        winning_modality=rule.winner,
        explanation=rule.explanation,
        confidence=round(winner_confidence * cfg.fusion.disagreement_penalty, 4),
        warnings=warnings,
        agreement_px=int(both.sum()),
        optical_only_px=int((optical & ~sar).sum()),
        sar_only_px=int((sar & ~optical).sum()),
    )
