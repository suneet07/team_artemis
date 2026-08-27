import numpy as np
import pytest

from satquery.ingest import ingest_raster, open_raster, read_overview, read_window
from satquery.ingest.radiometry import percentile_stretch
from tests import fixtures


@pytest.fixture(scope="module")
def optical_tif(tmp_path_factory):
    return fixtures.optical_multispectral(tmp_path_factory.mktemp("fixtures") / "optical_mx.tif")


@pytest.fixture(scope="module")
def sar_tif(tmp_path_factory):
    return fixtures.sar_single_pol(tmp_path_factory.mktemp("fixtures") / "sar_s1.tif")


def test_multispectral_optical_ingest(optical_tif):
    result = ingest_raster(optical_tif)
    assert result.modality == "optical"
    assert result.modality_source == "sensor_tag"
    assert result.meta.bit_depth == 12
    assert result.meta.bit_depth_source == "nbits_metadata"
    assert result.compatibility.bit_depth_source == "nbits_metadata"
    assert result.meta.crs == "EPSG:2056"
    assert result.inventory.bands == {"blue": 1, "green": 2, "red": 3, "nir": 4}
    assert result.inventory.computable_indices == ["NDVI", "NDWI"]
    assert not result.inventory.is_pan_only
    assert abs(result.compatibility.nodata_frac - 400 / 4096) < 0.01
    assert result.compatibility.is_readable
    assert result.compatibility.is_georeferenced


def test_single_band_heavy_tail_detected_as_sar(sar_tif):
    result = ingest_raster(sar_tif)
    assert result.modality == "sar"
    assert result.inventory.polarisations == []
    assert result.inventory.computable_indices == []


def test_sensor_tag_overrides_signature(sar_tif, tmp_path):
    tagged = fixtures.sar_tagged(tmp_path / "risat.tif")
    by_tag = ingest_raster(tagged)
    untagged = ingest_raster(sar_tif)
    assert by_tag.modality == "sar"
    assert any("metadata" in note for note in by_tag.compatibility.warnings)
    assert any("speckle signature" in note for note in untagged.compatibility.warnings)
    assert by_tag.inventory.sar_band == "X"
    assert by_tag.meta.sensor_hint == "RISAT-2B"


def test_quad_pol_sar_is_not_read_as_bgr_nir(tmp_path):
    result = ingest_raster(fixtures.sar_quad_pol(tmp_path / "quad.tif"))
    assert result.modality == "sar"
    assert result.inventory.bands == {}
    assert result.inventory.computable_indices == []


def test_sar_already_in_db_detected_despite_negative_values(tmp_path):
    path = fixtures.sar_single_pol_db(tmp_path / "sar_db.tif")
    with open_raster(path) as src:
        # The fixture deliberately carries a near-zero-backscatter tail below the
        # configured floor. Detection must survive it; keep this property if the
        # fixture is ever regenerated.
        assert src.read(1).min() < -60.0
    result = ingest_raster(path)
    assert result.modality == "sar"
    assert not result.inventory.is_pan_only
    assert any("dB" in w for w in result.compatibility.warnings)


def test_p2_model_input_stack_round_trips_as_sar(tmp_path):
    result = ingest_raster(fixtures.sar_db_stack(tmp_path / "stack.tif"))
    assert result.modality == "sar"
    assert result.inventory.computable_indices == []


def test_ndvi_float_raster_is_not_mistaken_for_db_sar(tmp_path):
    result = ingest_raster(fixtures.ndvi_raster(tmp_path / "ndvi.tif"))
    assert result.modality != "sar"


def test_multispectral_stack_stays_optical(tmp_path):
    result = ingest_raster(fixtures.multispectral_13band(tmp_path / "ms.tif"))
    assert result.modality == "optical"


def test_partial_descriptions_consistent_with_assumed_order_are_completed(tmp_path):
    path = fixtures.partial_descriptions(tmp_path / "ok.tif", ("b1", "b2", "Red", "b4"))
    result = ingest_raster(path)
    assert result.inventory.bands == {"blue": 1, "green": 2, "red": 3, "nir": 4}
    assert result.inventory.computable_indices == ["NDVI", "NDWI"]
    assert any("matched a band description" in w for w in result.compatibility.warnings)


def test_partial_descriptions_conflicting_with_assumed_order_disable_indices(tmp_path):
    path = fixtures.partial_descriptions(tmp_path / "bad.tif", ("Red", "b2", "b3", "b4"))
    result = ingest_raster(path)
    assert result.inventory.bands == {}
    assert result.inventory.computable_indices == []
    assert any("contradict" in w for w in result.compatibility.warnings)


def test_pan_only_optical(optical_tif, tmp_path):
    pan = fixtures.pan_optical(tmp_path / "pan.tif")
    result = ingest_raster(pan)
    assert result.modality == "optical"
    assert result.inventory.is_pan_only
    assert result.inventory.computable_indices == []
    assert result.compatibility.bands_present == []


def test_rgb_only_assumed_order(tmp_path):
    rgb = fixtures.rgb_only(tmp_path / "rgb.tif")
    result = ingest_raster(rgb)
    assert result.modality == "optical"
    assert result.inventory.bands == {"blue": 1, "green": 2, "red": 3}
    assert result.inventory.has_nir is False
    assert any("RGB order" in w for w in result.compatibility.warnings)


def test_geographic_crs_flags_pixel_size(tmp_path):
    geo = fixtures.geographic_crs(tmp_path / "geo.tif")
    result = ingest_raster(geo)
    assert result.meta.pixel_size_m is None
    assert any("degrees" in w for w in result.compatibility.warnings)


def test_modality_override_is_declared_and_recorded(tmp_path):
    pan = fixtures.pan_optical(tmp_path / "pan_override.tif")
    result = ingest_raster(pan, modality_override="sar")
    assert result.modality == "sar"
    assert result.modality_source == "declared"
    assert result.compatibility.modality_source == "declared"
    assert any("declared by caller" in w for w in result.compatibility.warnings)


def test_modality_override_rejects_invalid_value(tmp_path):
    pan = fixtures.pan_optical(tmp_path / "pan_bad.tif")
    with pytest.raises(ValueError):
        ingest_raster(pan, modality_override="unknown")


def test_dark_uint16_reports_container_depth_with_note(tmp_path):
    dark = fixtures.dark_uint16(tmp_path / "dark.tif")
    result = ingest_raster(dark)
    assert result.meta.dtype == "uint16"
    assert result.meta.bit_depth == 16
    assert result.meta.bit_depth_source == "container_dtype"
    assert any("NBITS" in note for note in result.compatibility.warnings)


def test_processing_software_tag_does_not_force_sar(tmp_path):
    from tests.fixtures import write_tif

    rng_data = (
        np.random.default_rng(9).exponential(scale=60.0, size=(1, 96, 96)).astype(np.float32)
    )
    tagged = write_tif(
        tmp_path / "sarscape.tif",
        rng_data,
        tags={"PROCESSING_SOFTWARE": "SARscape"},
    )
    result = ingest_raster(tagged)
    assert result.modality == "sar"
    assert result.modality_source == "intensity_signature"
    assert not any("token" in w for w in result.compatibility.warnings)


def test_windowed_read_matches_full_read(optical_tif):
    with open_raster(optical_tif) as src:
        window = read_window(src, row_off=10, col_off=12, height=8, width=16)
        full = src.read()
    np.testing.assert_array_equal(np.ma.getdata(window), full[:, 10:18, 12:28])


def test_overview_is_decimated(optical_tif):
    with open_raster(optical_tif) as src:
        overview = read_overview(src, max_side=32)
    assert max(overview.shape[-2:]) <= 32
    assert overview.shape[0] == 4


def test_percentile_stretch_records_bounds_and_clamps(optical_tif):
    with open_raster(optical_tif) as src:
        band = src.read(1)
    stretched, (lo, hi) = percentile_stretch(band, nodata=0)
    assert stretched.dtype == np.float32
    assert stretched.min() >= 0.0 and stretched.max() <= 1.0
    valid = band[band != 0].astype(np.float64)
    assert float(valid.min()) <= lo <= float(valid.max())
    assert float(valid.min()) <= hi <= float(valid.max())
    assert lo < hi


def test_percentile_stretch_excludes_nodata_from_bounds(optical_tif):
    with open_raster(optical_tif) as src:
        band = src.read(1)
    _, (lo, _) = percentile_stretch(band, nodata=0)
    assert lo > 0.0


def test_stretch_constant_band_maps_to_zero():
    flat = np.full((8, 8), 5, dtype=np.uint8)
    stretched, _ = percentile_stretch(flat)
    assert float(stretched.max()) == 0.0


