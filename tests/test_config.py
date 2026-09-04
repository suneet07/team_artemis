import numpy as np
import pytest
import yaml

from satquery.config import ConfigError, load_preprocessing_config, preprocessing_config
from satquery.ingest import ingest_raster
from satquery.paths import PREPROCESSING_CONFIG_PATH
from tests import fixtures


def test_shipped_contract_loads():
    cfg = preprocessing_config()
    assert cfg.version >= 1
    assert cfg.ingest.overview_max_side == 512
    assert (cfg.radiometry.low_percentile, cfg.radiometry.high_percentile) == (2.0, 98.0)
    assert cfg.bands.assumed_orders[4] == ("blue", "green", "red", "nir")
    assert cfg.agent.learned_tool_tile_budget == 4
    assert cfg.agent.spectral_thresholds["NDWI"] == 0.20
    assert cfg.agent.sar_threshold_db == -18.0
    assert cfg.agent.rmse_threshold_px == 1.5



def test_sar_and_tiling_contracts_are_frozen():
    cfg = preprocessing_config()
    assert cfg.sar.chain[0] == "apply_orbit_file"
    assert cfg.sar.pol_dropout_rate == 0.25
    assert cfg.sar.sanity_range("C", "calm_water") == (-25.0, -20.0)
    assert cfg.tiling.tile_overlap_fraction == 0.10


def test_untuned_sar_band_inherits_the_calibrated_c_band_table():
    cfg = preprocessing_config()
    assert cfg.sar.sanity_ranges_db["X"] is None
    assert cfg.sar.sanity_range("X", "urban") == cfg.sar.sanity_range("C", "urban")


def _write_config(path, mutate) -> str:
    raw = yaml.safe_load(PREPROCESSING_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_unknown_key_is_drift_not_a_warning(tmp_path):
    path = _write_config(tmp_path / "c.yaml", lambda raw: raw["ingest"].update(fudge=1))
    with pytest.raises(ConfigError, match="unknown"):
        load_preprocessing_config(path)


def test_missing_key_is_drift(tmp_path):
    path = _write_config(tmp_path / "c.yaml", lambda raw: raw["radiometry"].pop("low_percentile"))
    with pytest.raises(ConfigError, match="missing"):
        load_preprocessing_config(path)


def test_incoherent_thresholds_rejected(tmp_path):
    def loosen(raw):
        raw["modality"].update(ambiguous_cv_min=9)

    path = _write_config(tmp_path / "c.yaml", loosen)
    with pytest.raises(ConfigError, match="ambiguous_cv_min"):
        load_preprocessing_config(path)


def test_absent_contract_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_preprocessing_config(tmp_path / "nope.yaml")


def test_config_drives_ingest_not_hardcoded_constants(tmp_path):
    """Loosening the speckle thresholds must change what ingest decides."""
    sar = fixtures.sar_single_pol(tmp_path / "sar.tif")
    assert ingest_raster(sar).modality == "sar"

    strict = load_preprocessing_config(
        _write_config(tmp_path / "strict.yaml", lambda raw: raw["modality"].update(
            speckle_cv_min=5.0, speckle_skew_min=5.0, ambiguous_cv_min=5.0, ambiguous_skew_min=5.0
        ))
    )
    assert ingest_raster(sar, strict).modality == "optical"


def test_ambiguous_signature_routes_to_ask_the_user(tmp_path):
    """Between the optical and speckle signatures the answer is 'unknown'."""
    rng = np.random.default_rng(53)
    data = rng.exponential(scale=40.0, size=(1, 96, 96)).astype(np.float32)
    path = fixtures.write_tif(tmp_path / "mid.tif", data, crs="EPSG:32644")

    cfg = load_preprocessing_config(
        _write_config(tmp_path / "mid.yaml", lambda raw: raw["modality"].update(
            speckle_cv_min=2.0, speckle_skew_min=3.0
        ))
    )
    result = ingest_raster(path, cfg)
    assert result.modality == "unknown"
    assert any("ask the user" in w for w in result.compatibility.warnings)
