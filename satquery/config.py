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


class CapExceededError(ConfigError):
    """A raster reached the model above the frozen `tiling.max_pixels` cap."""


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


def _positive_int(raw: dict[str, Any], key: str, where: str) -> int:
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{where}.{key}: expected a positive int, got {value!r}")
    return value


def _fraction(raw: dict[str, Any], key: str, where: str) -> float:
    value = _number(raw, key, where)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{where}.{key}: {value} outside [0, 1]")
    return value


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
    #: Separate from the optical table, because the two cannot share one: a
    #: two-band raster is VH/VV on radar and means something else entirely on
    #: optical, and one table keyed only on band count could not tell them
    #: apart.
    assumed_sar_orders: dict[int, tuple[str, ...]]

    @classmethod
    def from_dict(cls, raw: Any) -> "BandsConfig":
        where = "bands"
        raw = _mapping(raw, ("assumed_orders", "assumed_sar_orders"), where)

        def _orders(key: str) -> dict[int, tuple[str, ...]]:
            parsed: dict[int, tuple[str, ...]] = {}
            for count, names in (raw.get(key) or {}).items():
                if not isinstance(count, int) or isinstance(count, bool):
                    raise ConfigError(f"{where}.{key}: band count {count!r} is not an int")
                if not isinstance(names, list) or len(names) != count:
                    raise ConfigError(
                        f"{where}.{key}.{count}: expected {count} band names, got {names!r}"
                    )
                parsed[count] = tuple(str(name).lower() for name in names)
            return parsed

        return cls(
            assumed_orders=_orders("assumed_orders"),
            assumed_sar_orders=_orders("assumed_sar_orders"),
        )


def _sar_thresholds(
    raw: Any, where: str
) -> dict[str, dict[str, dict[str, float]] | None]:
    """Parse `sar.fixed_thresholds_db`: band -> polarisation -> {water, builtup}."""
    out: dict[str, dict[str, dict[str, float]] | None] = {}
    for band, table in (raw or {}).items():
        if table is None:
            out[str(band).upper()] = None
            continue
        per_pol: dict[str, dict[str, float]] = {}
        for pol, entry in table.items():
            sub = f"{where}.fixed_thresholds_db.{band}.{pol}"
            entry = _mapping(entry, ("water_max", "builtup_min"), sub)
            water = _number(entry, "water_max", sub)
            builtup = _number(entry, "builtup_min", sub)
            if water >= builtup:
                raise ConfigError(
                    f"{sub}: water is specular and built-up is double-bounce, so "
                    f"water_max ({water}) must sit below builtup_min ({builtup})"
                )
            per_pol[str(pol).upper()] = {"water_max": water, "builtup_min": builtup}
        out[str(band).upper()] = per_pol
    return out


@dataclass(frozen=True)
class SarConfig:
    """Contract for P2 (section 4.2). Frozen now; consumed when P2 lands."""

    chain: tuple[str, ...]
    skip_terrain_correction_if_pregeoreferenced: bool
    pol_dropout_rate: float
    sanity_ranges_db: dict[str, dict[str, tuple[float, float]] | None]
    fixed_thresholds_db: dict[str, dict[str, dict[str, float]] | None]

    @classmethod
    def from_dict(cls, raw: Any) -> "SarConfig":
        where = "sar"
        keys = (
            "chain",
            "skip_terrain_correction_if_pregeoreferenced",
            "pol_dropout_rate",
            "sanity_ranges_db",
            "fixed_thresholds_db",
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
            fixed_thresholds_db=_sar_thresholds(raw["fixed_thresholds_db"], where),
        )

    def sanity_range(self, sar_band: str | None, surface: str) -> tuple[float, float] | None:
        """C-band is the calibrated reference; untuned bands inherit it (4.2)."""
        table = self.sanity_ranges_db.get((sar_band or "C").upper())
        if table is None:
            table = self.sanity_ranges_db.get("C") or {}
        return table.get(surface)

    def inherits_c_band(self, sar_band: str | None) -> bool:
        """True when this band has no tuned table of its own (4.2).

        Callers must warn and lower SAR-side confidence when it is. An X-band
        RISAT-2B scene silently scored against C-band physics is exactly the
        Critical risk row the plan opens with.
        """
        band = (sar_band or "C").upper()
        return band != "C" and self.sanity_ranges_db.get(band) is None

    def fixed_threshold_db(
        self, sar_band: str | None, polarisation: str | None, target: str
    ) -> float | None:
        """Fallback sigma-nought cut for `target` in {water, builtup} (4.6.5).

        Keyed by band then polarisation, per the plan. An untuned band inherits
        C-band and an unknown polarisation falls back to VV; both cases are
        observable through `inherits_c_band` so the caller can warn rather than
        pretend the number is tuned.
        """
        table = self.fixed_thresholds_db.get((sar_band or "C").upper())
        if table is None:
            table = self.fixed_thresholds_db.get("C") or {}
        entry = table.get((polarisation or "VV").upper()) or table.get("VV")
        if entry is None:
            return None
        return entry["water_max"] if target == "water" else entry["builtup_min"]


@dataclass(frozen=True)
class TilingConfig:
    """Contract for P4 (section 4.4). Frozen now; consumed when P4 lands."""

    tile_overlap_fraction: float
    max_pixels: int | None
    tile_side_px: int
    max_tiles: int
    top_k_tiles: int

    @classmethod
    def from_dict(cls, raw: Any) -> "TilingConfig":
        where = "tiling"
        keys = (
            "tile_overlap_fraction",
            "max_pixels",
            "tile_side_px",
            "max_tiles",
            "top_k_tiles",
        )
        raw = _mapping(raw, keys, where)
        overlap = _number(raw, "tile_overlap_fraction", where)
        if not 0.0 <= overlap < 1.0:
            raise ConfigError(f"{where}.tile_overlap_fraction: {overlap} outside [0, 1)")
        max_pixels = raw["max_pixels"]
        if max_pixels is not None and (
            not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels < 1
        ):
            raise ConfigError(f"{where}.max_pixels: expected a positive int or null")
        cfg = cls(
            tile_overlap_fraction=overlap,
            max_pixels=max_pixels,
            tile_side_px=_positive_int(raw, "tile_side_px", where),
            max_tiles=_positive_int(raw, "max_tiles", where),
            top_k_tiles=_positive_int(raw, "top_k_tiles", where),
        )
        if cfg.top_k_tiles > cfg.max_tiles:
            raise ConfigError(f"{where}: top_k_tiles must not exceed max_tiles")
        if max_pixels is not None and cfg.tile_side_px**2 > max_pixels:
            raise ConfigError(
                f"{where}: a {cfg.tile_side_px}x{cfg.tile_side_px} tile is "
                f"{cfg.tile_side_px ** 2:,} px, above the {max_pixels:,} px cap it is "
                f"supposed to keep every source under"
            )
        return cfg

    def check_within_cap(self, width: int, height: int, where: str = "raster") -> None:
        """Fail loudly rather than let the processor silently downsample.

        `max_pixels` is a cap, not a target. The vision tower never upsamples, so
        a source at or below it passes through untouched, while a source above it
        is resized with no log line -- and that happens inside the processor,
        downstream of every curation check, so nothing upstream can observe the
        ground-sample-distance change it causes.

        Tiling (P4) must bring every source under the cap. A raster arriving above
        it is a pipeline defect, not something to quietly resize.
        """
        if self.max_pixels is None:
            return
        pixels = width * height
        if pixels > self.max_pixels:
            factor = (pixels / self.max_pixels) ** 0.5
            raise CapExceededError(
                f"{where}: {width}x{height} = {pixels:,} px exceeds the frozen "
                f"tiling.max_pixels cap of {self.max_pixels:,}. The processor would "
                f"downsample by {factor:.2f}x, coarsening ground sample distance by "
                f"the same factor without any record of it. Tile upstream instead."
            )


@dataclass(frozen=True)
class OtsuGateConfig:
    """The section 4.6.2 bimodality gate."""

    min_between_class_variance_ratio: float
    min_class_fraction: float

    @classmethod
    def from_dict(cls, raw: Any) -> "OtsuGateConfig":
        where = "indices.otsu"
        keys = ("min_between_class_variance_ratio", "min_class_fraction")
        raw = _mapping(raw, keys, where)
        return cls(
            min_between_class_variance_ratio=_fraction(
                raw, "min_between_class_variance_ratio", where
            ),
            min_class_fraction=_fraction(raw, "min_class_fraction", where),
        )


@dataclass(frozen=True)
class FixedThreshold:
    value: float
    above_is_positive: bool
    target: str


@dataclass(frozen=True)
class IndicesConfig:
    """Contract for spectral_index (section 4.6.2)."""

    otsu: OtsuGateConfig
    fixed_thresholds: dict[str, FixedThreshold]

    @classmethod
    def from_dict(cls, raw: Any) -> "IndicesConfig":
        where = "indices"
        raw = _mapping(raw, ("otsu", "fixed_thresholds"), where)
        thresholds: dict[str, FixedThreshold] = {}
        for name, entry in (raw["fixed_thresholds"] or {}).items():
            sub = f"{where}.fixed_thresholds.{name}"
            entry = _mapping(entry, ("value", "above_is_positive", "target"), sub)
            if not isinstance(entry["above_is_positive"], bool):
                raise ConfigError(f"{sub}.above_is_positive: expected a boolean")
            thresholds[str(name).upper()] = FixedThreshold(
                value=_number(entry, "value", sub),
                above_is_positive=entry["above_is_positive"],
                target=str(entry["target"]),
            )
        return cls(otsu=OtsuGateConfig.from_dict(raw["otsu"]), fixed_thresholds=thresholds)

    def fixed(self, index: str) -> FixedThreshold | None:
        return self.fixed_thresholds.get(index.upper())


@dataclass(frozen=True)
class TextureConfig:
    """Contract for texture_seg (4.6.8) and object_box_fallback (4.6.9)."""

    window_px: int
    glcm_levels: int
    opening_radius_px: int
    min_component_fraction: float
    max_proposals: int
    fixed_response_threshold: float

    @classmethod
    def from_dict(cls, raw: Any) -> "TextureConfig":
        where = "texture"
        keys = (
            "window_px",
            "glcm_levels",
            "opening_radius_px",
            "min_component_fraction",
            "max_proposals",
            "fixed_response_threshold",
        )
        raw = _mapping(raw, keys, where)
        window = _positive_int(raw, "window_px", where)
        if window % 2 == 0:
            raise ConfigError(
                f"{where}.window_px: must be odd so the response map stays aligned "
                f"with the source grid, got {window}"
            )
        radius = raw["opening_radius_px"]
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise ConfigError(f"{where}.opening_radius_px: expected a non-negative int")
        return cls(
            window_px=window,
            glcm_levels=_positive_int(raw, "glcm_levels", where),
            opening_radius_px=radius,
            min_component_fraction=_fraction(raw, "min_component_fraction", where),
            max_proposals=_positive_int(raw, "max_proposals", where),
            fixed_response_threshold=_fraction(raw, "fixed_response_threshold", where),
        )


@dataclass(frozen=True)
class CoregConfig:
    """Contract for P3 (section 4.3)."""

    max_rmse_px: float
    refuse_above_rmse_px: float
    mutual_information_bins: int
    upsample_factor: int

    @classmethod
    def from_dict(cls, raw: Any) -> "CoregConfig":
        where = "coreg"
        keys = (
            "max_rmse_px",
            "refuse_above_rmse_px",
            "mutual_information_bins",
            "upsample_factor",
        )
        raw = _mapping(raw, keys, where)
        good = _number(raw, "max_rmse_px", where)
        refuse = _number(raw, "refuse_above_rmse_px", where)
        if good <= 0 or refuse <= good:
            raise ConfigError(
                f"{where}: expected 0 < max_rmse_px ({good}) < refuse_above_rmse_px ({refuse})"
            )
        return cls(
            max_rmse_px=good,
            refuse_above_rmse_px=refuse,
            mutual_information_bins=_positive_int(raw, "mutual_information_bins", where),
            upsample_factor=_positive_int(raw, "upsample_factor", where),
        )


@dataclass(frozen=True)
class FusionConfig:
    """Contract for P7 (sections 4.7.1-4.7.3)."""

    iou_consistent: float
    iou_conflict: float
    disagreement_penalty: float
    threshold_fallback_penalty: float
    deterministic_fallback_ceiling: float

    @classmethod
    def from_dict(cls, raw: Any) -> "FusionConfig":
        where = "fusion"
        keys = (
            "iou_consistent",
            "iou_conflict",
            "disagreement_penalty",
            "threshold_fallback_penalty",
            "deterministic_fallback_ceiling",
        )
        raw = _mapping(raw, keys, where)
        cfg = cls(
            iou_consistent=_fraction(raw, "iou_consistent", where),
            iou_conflict=_fraction(raw, "iou_conflict", where),
            disagreement_penalty=_fraction(raw, "disagreement_penalty", where),
            threshold_fallback_penalty=_fraction(raw, "threshold_fallback_penalty", where),
            deterministic_fallback_ceiling=_fraction(
                raw, "deterministic_fallback_ceiling", where
            ),
        )
        if cfg.iou_conflict >= cfg.iou_consistent:
            raise ConfigError(f"{where}: iou_conflict must sit below iou_consistent")
        return cfg


@dataclass(frozen=True)
class PreprocessingConfig:
    version: int
    ingest: IngestConfig
    radiometry: RadiometryConfig
    modality: ModalityConfig
    bands: BandsConfig
    sar: SarConfig
    tiling: TilingConfig
    indices: IndicesConfig
    texture: TextureConfig
    coreg: CoregConfig
    fusion: FusionConfig

    @classmethod
    def from_dict(cls, raw: Any) -> "PreprocessingConfig":
        keys = (
            "version",
            "ingest",
            "radiometry",
            "modality",
            "bands",
            "sar",
            "tiling",
            "indices",
            "texture",
            "coreg",
            "fusion",
        )
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
            indices=IndicesConfig.from_dict(raw["indices"]),
            texture=TextureConfig.from_dict(raw["texture"]),
            coreg=CoregConfig.from_dict(raw["coreg"]),
            fusion=FusionConfig.from_dict(raw["fusion"]),
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
