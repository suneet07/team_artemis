"""AROSICS as the escalation path for P3 (master plan section 4.3 step 5).

*"If shift exceeds threshold -> AROSICS correction, or refuse with explanation."*

:func:`satquery.coreg.register_pair` corrects with a single sub-pixel
translation, which is right for the common case and wrong when the residual is
not a pure translation. AROSICS is the escalation, and this module is the only
place it is imported.

**What AROSICS actually does, stated precisely.** ``COREG`` computes one global
X/Y shift — a translation, at sub-pixel precision, and nothing else.
``COREG_LOCAL`` computes a *grid* of local shifts and warps the image through
them, which absorbs local distortion (and therefore, in effect, mild
scale/rotation) without ever solving a global affine. Neither estimates a
rotation angle or a scale factor. Claiming otherwise would set up exactly the
failure the plan warns about — a co-registration everyone believes is handling
geometry it never touched, feeding change detection that then reports building
outlines as change.

**Licence.** AROSICS must be pinned ``>=1.0.0``: pre-1.0 releases were GPL-3.0
(C58). ``tests/test_license_blocklist.py`` enforces the pin.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AROSICS_AVAILABLE", "ArosicsResult", "arosics_available", "correct_with_arosics"]


def arosics_available() -> bool:
    """Whether AROSICS is installed.

    A function rather than an import-time stub swap: the headless container
    (section 4.11) installs no extras, and a module-level ``try/except ImportError``
    that quietly substitutes a no-op is how a system ends up reporting a
    correction it never performed. ``find_spec`` keeps the probe cheap — AROSICS
    pulls in GDAL, so importing it just to ask whether it exists is expensive.
    """
    return importlib.util.find_spec("arosics") is not None


#: Snapshot at import. Prefer calling :func:`arosics_available` — an extra can be
#: installed into a running environment, and a stale constant would deny it.
AROSICS_AVAILABLE = arosics_available()


@dataclass
class ArosicsResult:
    """Outcome of an AROSICS run, including the reason it did not run."""

    corrected_path: Path | None
    shift_px: tuple[float, float] | None
    reliable: bool
    method: str
    warnings: list[str]

    @property
    def succeeded(self) -> bool:
        return self.corrected_path is not None and self.reliable


def correct_with_arosics(
    reference_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    *,
    local: bool = False,
    grid_res: int = 200,
    window_size: tuple[int, int] = (256, 256),
    max_shift_px: int = 20,
) -> ArosicsResult:
    """Co-register ``target_path`` onto ``reference_path`` with AROSICS.

    ``local=True`` selects ``COREG_LOCAL``, which fits a grid of local shifts —
    use it when the residual varies across the scene, which is the case a single
    global translation cannot fix.

    Output is GeoTIFF, not AROSICS' ENVI default: every other artifact in this
    system is a GeoTIFF, and a mask or scene in a second format is one more thing
    to convert at the venue.

    Never raises for an absent or failing AROSICS. It returns a result carrying
    ``reliable=False`` and the reason, because the caller's fallback — refuse with
    the measured RMSE — is a better outcome than a crash and a *much* better
    outcome than silently returning the uncorrected image as if it were fixed.
    """
    messages: list[str] = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not arosics_available():
        return ArosicsResult(
            corrected_path=None,
            shift_px=None,
            reliable=False,
            method="arosics_unavailable",
            warnings=[
                "AROSICS is not installed, so the non-translation escalation path is "
                "unavailable. Install it with `pip install -e \".[coreg]\"` (the >=1.0.0 "
                "pin is a licence control, C58). Falling back to the translation-only "
                "correction."
            ],
        )

    from arosics import COREG, COREG_LOCAL

    common = {
        "path_out": str(output_path),
        "fmt_out": "GTiff",
        "max_shift": max_shift_px,
        "q": True,
    }
    try:
        if local:
            engine = COREG_LOCAL(
                str(reference_path),
                str(target_path),
                grid_res=grid_res,
                window_size=window_size,
                **common,
            )
            engine.correct_shifts()
            shift = None  # a local fit has no single global shift to report
            reliable = True
            method = "arosics_local"
        else:
            engine = COREG(
                str(reference_path), str(target_path), ws=window_size, **common
            )
            engine.calculate_spatial_shifts()
            # AROSICS itself reports whether the match was reliable. Ignoring
            # that flag and using the shift anyway is how a bad match becomes a
            # confident wrong answer.
            reliable = bool(getattr(engine, "success", True))
            shift = (
                float(getattr(engine, "y_shift_px", 0.0)),
                float(getattr(engine, "x_shift_px", 0.0)),
            )
            if reliable:
                engine.correct_shifts()
            else:
                messages.append(
                    "AROSICS could not find a reliable match (low correlation or too "
                    "little overlap); the shift it reports must not be applied"
                )
            method = "arosics_global"
    except Exception as error:  # noqa: BLE001 - a failed correction must not lose the query
        return ArosicsResult(
            corrected_path=None,
            shift_px=None,
            reliable=False,
            method="arosics_failed",
            warnings=[f"AROSICS correction failed: {type(error).__name__}: {error}"],
        )

    if reliable and shift is not None:
        messages.append(
            f"AROSICS applied a global translation of {shift[0]:.2f}, {shift[1]:.2f} px. "
            f"This is a translation only — AROSICS estimates no rotation or scale."
        )
    elif reliable:
        messages.append(
            f"AROSICS applied a local shift grid at {grid_res} px spacing, which absorbs "
            f"spatially varying residuals. It still solves no global rotation or scale."
        )

    return ArosicsResult(
        corrected_path=output_path if reliable and output_path.exists() else None,
        shift_px=shift,
        reliable=reliable,
        method=method,
        warnings=messages,
    )
