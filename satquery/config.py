"""Loader for the frozen preprocessing contract (configs/preprocessing.yaml).

Every constant that shapes what the model sees lives in the YAML, not in code
(master plan section 4.2). This module only reads it, and reads it strictly: an
unknown or missing key raises, so a config that has silently drifted away from
the frozen contract fails at load rather than at inference time.
"""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from satquery.paths import PREPROCESSING_CONFIG_PATH


class ConfigError(ValueError):
    """The preprocessing contract is missing, malformed, or has drifted."""


def _mapping(raw: Any, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    missing = sorted(set(keys) - set(raw))
    unknown = sorted(set(raw) - set(keys))
    if missing or unknown:
        detail = "; ".join(
            part
            for part in (
                f"missing {missing}" if missing else "",
                f"unknown {unknown}" if unknown else "",
            )
            if part
        )
        raise ConfigError(f"{where}: frozen config drift: {detail}")
    return raw


def _number(raw: dict[str, Any], key: str, where: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key}: expected a number, got {value!r}")
    return float(value)


def _pair(raw: dict[str, Any], key: str, where: str) -> tuple[float, float]:
    value = raw[key]
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{where}.{key}: expected a [low, high] pair, got {value!r}")
    low, high = (float(v) for v in value)
    if low >= high:
        raise ConfigError(f"{where}.{key}: low {low} must be below high {high}")
    return low, high


@dataclass(frozen=True)
class IngestConfig:
    overview_max_side: int
    nodata_warn_fraction: float

    @classmethod
    def from_dict(cls, raw: Any) -> "IngestConfig":
        where = "ingest"
        raw = _mapping(raw, ("overview_max_side", "nodata_warn_fraction"), where)
        side = raw["overview_max_side"]
        if not isinstance(side, int) or isinstance(side, bool) or side < 1:
            raise ConfigError(f"{where}.overview_max_side: expected a positive int, got {side!r}")
        return cls(
            overview_max_side=side,
            nodata_warn_fraction=_number(raw, "nodata_warn_fraction", where),
        )


@dataclass(frozen=True)
class RadiometryConfig:
    low_percentile: float
    high_percentile: float

    @classmethod
    def from_dict(cls, raw: Any) -> "RadiometryConfig":
        where = "radiometry"
        raw = _mapping(raw, ("low_percentile", "high_percentile"), where)
        low = _number(raw, "low_percentile", where)
        high = _number(raw, "high_percentile", where)
        if not 0.0 <= low < high <= 100.0:
            raise ConfigError(f"{where}: percentiles must satisfy 0 <= {low} < {high} <= 100")
        return cls(low_percentile=low, high_percentile=high)


@dataclass(frozen=True)
class ModalityConfig:
    speckle_cv_min: float
    speckle_skew_min: float
    ambiguous_cv_min: float
    ambiguous_skew_min: float
    db_min_negative_fraction: float
    db_plausible_range: tuple[float, float]
    db_sample_percentiles: tuple[float, float]
    db_must_reach_below: float

    @classmethod
    def from_dict(cls, raw: Any) -> "ModalityConfig":
        where = "modality"
        keys = (
            "speckle_cv_min",
            "speckle_skew_min",
            "ambiguous_cv_min",
            "ambiguous_skew_min",
            "db_min_negative_fraction",
            "db_plausible_range",
            "db_sample_percentiles",
            "db_must_reach_below",
        )
        raw = _mapping(raw, keys, where)
        cfg = cls(
            speckle_cv_min=_number(raw, "speckle_cv_min", where),
            speckle_skew_min=_number(raw, "speckle_skew_min", where),
            ambiguous_cv_min=_number(raw, "ambiguous_cv_min", where),
            ambiguous_skew_min=_number(raw, "ambiguous_skew_min", where),
            db_min_negative_fraction=_number(raw, "db_min_negative_fraction", where),
            db_plausible_range=_pair(raw, "db_plausible_range", where),
            db_sample_percentiles=_pair(raw, "db_sample_percentiles", where),
            db_must_reach_below=_number(raw, "db_must_reach_below", where),
        )
        if not 0.0 <= cfg.db_sample_percentiles[0] < cfg.db_sample_percentiles[1] <= 100.0:
            raise ConfigError(f"{where}.db_sample_percentiles: must lie within [0, 100]")
        if cfg.ambiguous_cv_min > cfg.speckle_cv_min:
            raise ConfigError(f"{where}: ambiguous_cv_min must not exceed speckle_cv_min")
        if cfg.ambiguous_skew_min > cfg.speckle_skew_min:
            raise ConfigError(f"{where}: ambiguous_skew_min must not exceed speckle_skew_min")
        low, high = cfg.db_plausible_range
        if not low <= cfg.db_must_reach_below <= high:
            raise ConfigError(f"{where}: db_must_reach_below must sit inside db_plausible_range")
        return cfg


@dataclass(frozen=True)
class BandsConfig:
    assumed_orders: dict[int, tuple[str, ...]]

    @classmethod
    def from_dict(cls, raw: Any) -> "BandsConfig":
        where = "bands"
        raw = _mapping(raw, ("assumed_orders",), where)
        orders: dict[int, tuple[str, ...]] = {}
        for count, names in (raw["assumed_orders"] or {}).items():
            if not isinstance(count, int) or isinstance(count, bool):
                raise ConfigError(f"{where}.assumed_orders: band count {count!r} is not an int")
            if not isinstance(names, list) or len(names) != count:
                raise ConfigError(
                    f"{where}.assumed_orders.{count}: expected {count} band names, got {names!r}"
                )
            orders[count] = tuple(str(name).lower() for name in names)
        return cls(assumed_orders=orders)


@dataclass(frozen=True)
class SarConfig:
    """Contract for P2 (section 4.2). Frozen now; consumed when P2 lands."""

    chain: tuple[str, ...]
    skip_terrain_correction_if_pregeoreferenced: bool
    pol_dropout_rate: float
    sanity_ranges_db: dict[str, dict[str, tuple[float, float]] | None]

    @classmethod
    def from_dict(cls, raw: Any) -> "SarConfig":
        where = "sar"
        keys = (
            "chain",
            "skip_terrain_correction_if_pregeoreferenced",
            "pol_dropout_rate",
            "sanity_ranges_db",
        )
        raw = _mapping(raw, keys, where)
        rate = _number(raw, "pol_dropout_rate", where)
        if not 0.0 <= rate <= 1.0:
            raise ConfigError(f"{where}.pol_dropout_rate: {rate} outside [0, 1]")
        ranges: dict[str, dict[str, tuple[float, float]] | None] = {}
        for band, table in (raw["sanity_ranges_db"] or {}).items():
            if table is None:
                ranges[str(band)] = None
                continue
            ranges[str(band)] = {
                str(surface): _pair(table, surface, f"{where}.sanity_ranges_db.{band}")
                for surface in table
            }
        return cls(
            chain=tuple(str(step) for step in raw["chain"]),
            skip_terrain_correction_if_pregeoreferenced=bool(
                raw["skip_terrain_correction_if_pregeoreferenced"]
            ),
            pol_dropout_rate=rate,
            sanity_ranges_db=ranges,
        )

    def sanity_range(self, sar_band: str | None, surface: str) -> tuple[float, float] | None:
        """C-band is the calibrated reference; unturned bands inherit it (4.2)."""
        table = self.sanity_ranges_db.get((sar_band or "C").upper())
        if table is None:
            table = self.sanity_ranges_db.get("C") or {}
        return table.get(surface)


@dataclass(frozen=True)
class TilingConfig:
    """Contract for P4 (section 4.4). Frozen now; consumed when P4 lands."""

    tile_overlap_fraction: float
    max_pixels: int | None

    @classmethod
    def from_dict(cls, raw: Any) -> "TilingConfig":
        where = "tiling"
        raw = _mapping(raw, ("tile_overlap_fraction", "max_pixels"), where)
        overlap = _number(raw, "tile_overlap_fraction", where)
        if not 0.0 <= overlap < 1.0:
            raise ConfigError(f"{where}.tile_overlap_fraction: {overlap} outside [0, 1)")
        max_pixels = raw["max_pixels"]
        if max_pixels is not None and (not isinstance(max_pixels, int) or max_pixels < 1):
            raise ConfigError(f"{where}.max_pixels: expected a positive int or null")
        return cls(tile_overlap_fraction=overlap, max_pixels=max_pixels)


@dataclass(frozen=True)
class AgentConfig:
    """Agent thresholds and tile budget (§10, §12, §14)."""

    learned_tool_tile_budget: int
    spectral_thresholds: dict[str, float]
    sar_threshold_db: float
    rmse_threshold_px: float

    @classmethod
    def from_dict(cls, raw: Any) -> "AgentConfig":
        where = "agent"
        keys = (
            "learned_tool_tile_budget",
            "spectral_thresholds",
            "sar_threshold_db",
            "rmse_threshold_px",
        )
        raw = _mapping(raw, keys, where)
        budget = raw["learned_tool_tile_budget"]
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ConfigError(f"{where}.learned_tool_tile_budget: expected positive int")
        thresh_raw = raw["spectral_thresholds"]
        if not isinstance(thresh_raw, dict):
            raise ConfigError(f"{where}.spectral_thresholds: expected mapping")
        spectral_thresh = {k: float(v) for k, v in thresh_raw.items()}
        return cls(
            learned_tool_tile_budget=budget,
            spectral_thresholds=spectral_thresh,
            sar_threshold_db=_number(raw, "sar_threshold_db", where),
            rmse_threshold_px=_number(raw, "rmse_threshold_px", where),
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    version: int
    ingest: IngestConfig
    radiometry: RadiometryConfig
    modality: ModalityConfig
    bands: BandsConfig
    sar: SarConfig
    tiling: TilingConfig
    agent: AgentConfig

    @classmethod
    def from_dict(cls, raw: Any) -> "PreprocessingConfig":
        keys = ("version", "ingest", "radiometry", "modality", "bands", "sar", "tiling", "agent")
        raw = _mapping(raw, keys, "preprocessing.yaml")
        version = raw["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ConfigError(
                f"preprocessing.yaml.version: expected a positive int, got {version!r}"
            )
        return cls(
            version=version,
            ingest=IngestConfig.from_dict(raw["ingest"]),
            radiometry=RadiometryConfig.from_dict(raw["radiometry"]),
            modality=ModalityConfig.from_dict(raw["modality"]),
            bands=BandsConfig.from_dict(raw["bands"]),
            sar=SarConfig.from_dict(raw["sar"]),
            tiling=TilingConfig.from_dict(raw["tiling"]),
            agent=AgentConfig.from_dict(raw["agent"]),
        )



def load_preprocessing_config(path: Path | str | None = None) -> PreprocessingConfig:
    path = Path(path) if path is not None else PREPROCESSING_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"preprocessing contract not found at {path}")
    return PreprocessingConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


@functools.cache
def preprocessing_config() -> PreprocessingConfig:
    """The shipped contract, parsed once per process."""
    return load_preprocessing_config()
