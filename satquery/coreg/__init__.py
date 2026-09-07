"""P3 — Co-registration (master plan section 4.3)."""

from satquery.coreg.arosics_backend import (
    AROSICS_AVAILABLE,
    ArosicsResult,
    arosics_available,
    correct_with_arosics,
)
from satquery.coreg.coreg import (
    CoregReport,
    estimate_shift_mutual_information,
    estimate_shift_phase,
    mutual_information,
    register_pair,
    resampling_for,
)

__all__ = [
    "AROSICS_AVAILABLE",
    "ArosicsResult",
    "CoregReport",
    "arosics_available",
    "correct_with_arosics",
    "estimate_shift_mutual_information",
    "estimate_shift_phase",
    "mutual_information",
    "register_pair",
    "resampling_for",
]
