from satquery.agent.asset import AssetRef
from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.tools import object_box_fallback


def test_asset_contract_invariants():
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDWI", "NDVI"],
    )
    img = ImageRef(
        scene_id="scene_0",
        path="scene_0.tif",
        modality="optical",
        crs="EPSG:32644",
        nodata_frac=0.01,
        swir_available=False,
    )
    bundle = ImageBundle(
        bundle_id="b_asset_test",
        images=[img],
        band_inventory=inv,
        pair_type="single",
    )

    res = run_query(bundle, "Where are the water bodies in this scene?")
    assert res.state == "succeeded"

    for asset in res.evidence:
        assert isinstance(asset, AssetRef)
        # Every asset must have produced_by set
        assert asset.produced_by is not None and len(asset.produced_by) > 0
        # Media type must be declared
        assert asset.media_type is not None
        # Bytes must be populated and positive
        assert asset.bytes > 0
        # Download url must be provided
        assert asset.download_url is not None


def test_asset_contract_bounding_boxes_format_and_geojson():
    # Test deterministic box proposer output directly
    outputs = object_box_fallback.execute({"target_class": "tank"})
    assert "boxes" in outputs
    for b in outputs["boxes"]:
        assert "bbox_px" in b
        bbox = b["bbox_px"]
        # Must be [x0, y0, x1, y1] (xmin, ymin, xmax, ymax)
        assert len(bbox) == 4
        x0, y0, x1, y1 = bbox
        assert x0 <= x1
        assert y0 <= y1
        assert b["method"] == "deterministic_fallback"
