from concurrent.futures import ThreadPoolExecutor, as_completed

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory


def _make_test_bundle(bundle_id: str) -> ImageBundle:
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint="optical_test",
        computable_indices=["NDVI", "NDWI"],
    )
    img = ImageRef(
        scene_id=f"scene_{bundle_id}",
        path=f"img_{bundle_id}.tif",
        modality="optical",
        crs="EPSG:32644",
        pixel_size_m=10.0,
        bands=["blue", "green", "red", "nir"],
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id=bundle_id,
        images=[img],
        band_inventory=inv,
        pair_type="single",
        status="ready",
    )


def test_concurrent_independent_queries():
    # Run 6 queries concurrently
    queries = [
        ("b1", "What is the vegetation extent in scene 1?"),
        ("b2", "Where is the aircraft located?"),
        ("b3", "Locate the bridge across the river"),
        ("b4", "What dominates this scene?"),
        ("b5", "What is the water extent?"),
        ("b6", "Where are the buildings?"),
    ]

    def _execute(item):
        b_id, q = item
        bundle = _make_test_bundle(b_id)
        res = run_query(bundle, q, query_id=f"q_{b_id}")
        return b_id, res

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_execute, q) for q in queries]
        for fut in as_completed(futures):
            b_id, res = fut.result()
            results[b_id] = res

    assert len(results) == 6
    for b_id, res in results.items():
        assert res.state == "succeeded"
        assert res.bundle_id == b_id
        assert res.query_id == f"q_{b_id}"
        assert res.trace["graded"]["parameter_check"]["passed"] is True
        assert len(res.trace["graded"]["permitted_parameters"]) > 0
