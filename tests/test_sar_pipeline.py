import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio

from satquery.sar.normalisation import SARNormaliser


@pytest.fixture
def dummy_snap_output(tmp_path):
    """Creates a dummy 2-band SNAP output TIFF for testing."""
    snap_out = tmp_path / "snap_out.tif"
    data = np.random.uniform(0.01, 10.0, size=(2, 64, 64)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(0, 0, 10, 10),
    }
    with rasterio.open(snap_out, "w", **profile) as dst:
        dst.write(data)
    return str(snap_out), data


def mock_run_snap_graph(input_path, output_path, is_pregeoreferenced):
    # Instead of running SNAP, we just copy the input to output
    # (assuming test passes the dummy_snap_output path as both input and output)
    # Actually, the normaliser creates a temp dir and passes it as output.
    # So we should create a dummy tif at output_path inside the mock.
    data = np.random.uniform(0.01, 10.0, size=(2, 64, 64)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(0, 0, 10, 10),
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)


@patch("satquery.sar.normalisation.SARNormaliser._run_snap_graph", side_effect=mock_run_snap_graph)
def test_process_dual_pol(mock_snap, tmp_path):
    normaliser = SARNormaliser()
    out_path = str(tmp_path / "final_out.tif")
    normaliser.process("dummy_input.tif", out_path)

    # Verify the output is a 3-channel model stack
    assert os.path.exists(out_path)
    with rasterio.open(out_path) as src:
        data = src.read()
        assert data.shape == (3, 64, 64)
        assert data.dtype == np.float32

        # Verify stretching bounds
        assert np.nanmin(data[0]) >= 0.0
        assert np.nanmax(data[0]) <= 1.0


def test_convert_to_db():
    normaliser = SARNormaliser()
    linear = np.array([1.0, 10.0, 100.0])
    db = normaliser._convert_to_db(linear)
    np.testing.assert_array_almost_equal(db, [0.0, 10.0, 20.0])


def test_percentile_stretch():
    normaliser = SARNormaliser()
    data = np.linspace(-30.0, 10.0, 100)
    stretched = normaliser._percentile_stretch(data)
    assert np.min(stretched) >= 0.0
    assert np.max(stretched) <= 1.0
