"""Training-time polarisation dropout (master plan section 4.2).

*"During ``optsar_fusion`` and SAR-facing training, apply polarisation dropout:
convert ~25% of dual-pol training samples to the single-pol stack so the model
has genuinely seen 1-channel SAR before the hidden set shows it one."*

This is the mitigation for a **Critical** risk row — RISAT-1 and EOS-04 FRS ship
single-pol, RISAT-2B is X-band spotlight — and until something calls it, the
mitigation is inert: the rate sits in ``preprocessing.yaml`` describing behaviour
nothing performs.

Two properties this has to have, both of which a naive wrapper misses.

**Reproducible.** Dropout that draws from the global ``random`` module cannot be
reproduced from a run's seed, so two runs of the "same" configuration train on
different data and nobody can tell which difference explains a score change.
Every draw here comes from an explicit ``numpy.random.Generator``.

**Rate from the contract.** The rate is ``sar.pol_dropout_rate``. Hardcoding
``0.25`` as a default parameter creates a second source of truth that silently
wins wherever the caller forgets to pass it.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from satquery.config import PreprocessingConfig, preprocessing_config
from satquery.sar.chain import apply_pol_dropout

__all__ = ["DropoutStats", "drop_polarisations", "with_pol_dropout"]


@dataclass
class DropoutStats:
    """What actually happened, so the realised rate can be checked against the plan."""

    total: int = 0
    dropped: int = 0
    eligible: int = 0

    @property
    def realised_rate(self) -> float:
        """Dropped over *eligible*, not over total.

        Single-pol samples cannot be dropped out, so dividing by the total
        understates the rate on any mixed corpus and makes a correctly
        configured run look under-target.
        """
        return self.dropped / self.eligible if self.eligible else 0.0

    def as_dict(self) -> dict:
        return {
            "samples": self.total,
            "dual_pol_eligible": self.eligible,
            "converted_to_single_pol": self.dropped,
            "realised_rate": round(self.realised_rate, 4),
        }


def drop_polarisations(
    bands_db: dict[str, np.ndarray],
    rng: np.random.Generator,
    config: PreprocessingConfig | None = None,
    stats: DropoutStats | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Apply dropout to one sample's polarisation bands, recording the outcome."""
    cfg = config if config is not None else preprocessing_config()
    eligible = len(bands_db) >= 2
    bands, dropped = apply_pol_dropout(bands_db, rng, cfg)
    if stats is not None:
        stats.total += 1
        stats.eligible += int(eligible)
        stats.dropped += int(dropped)
    return bands, dropped


def with_pol_dropout(
    samples: Iterable[dict],
    *,
    seed: int,
    bands_key: str = "bands_db",
    config: PreprocessingConfig | None = None,
    stats: DropoutStats | None = None,
) -> Iterator[dict]:
    """Stream training samples, converting a share of dual-pol ones to single-pol.

    Each yielded sample gains ``pol_dropout_applied`` and its resulting
    ``polarisations``, so the manifest records which samples were converted. A
    dropout you cannot see in the data is one nobody can audit against the
    realised rate.

    Samples with no ``bands_key`` pass through untouched — an optical-only
    corpus should not have to know this function exists.
    """
    cfg = config if config is not None else preprocessing_config()
    rng = np.random.default_rng(seed)
    for sample in samples:
        bands = sample.get(bands_key)
        if not isinstance(bands, dict) or not bands:
            yield sample
            continue
        reduced, dropped = drop_polarisations(bands, rng, cfg, stats)
        updated = dict(sample)
        updated[bands_key] = reduced
        updated["pol_dropout_applied"] = dropped
        updated["polarisations"] = sorted(reduced)
        yield updated
