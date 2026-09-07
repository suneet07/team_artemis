import numpy as np
import rasterio


def test_mask_conformance_cartosat(geotiff_4band_cartosat, tmp_path):
    # Read the synthetic cartosat
    with rasterio.open(geotiff_4band_cartosat) as src:
        assert src.count == 4
        assert src.dtypes[0] == "uint16"
        assert src.crs == "EPSG:32643"

        # Simulate mask output
        mask = np.ones((src.height, src.width), dtype=np.uint8)

        out_path = tmp_path / "out_mask.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=src.height,
            width=src.width,
            count=1,
            dtype="uint8",
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(mask, 1)

    # Verify the emitted mask conforms
    with rasterio.open(out_path) as verify:
        assert verify.crs == "EPSG:32643"
        assert verify.width == 100
        assert verify.height == 100


def test_mask_conformance_sar(geotiff_single_pol_sar, tmp_path):
    # Read the synthetic SAR
    with rasterio.open(geotiff_single_pol_sar) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.crs == "EPSG:4326"
        assert src.tags().get("SAR_BAND") == "C"

        mask = np.ones((src.height, src.width), dtype=np.uint8)

        out_path = tmp_path / "sar_out_mask.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=src.height,
            width=src.width,
            count=1,
            dtype="uint8",
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(mask, 1)

    with rasterio.open(out_path) as verify:
        assert verify.crs == "EPSG:4326"
        assert verify.width == 100
        assert verify.height == 100
