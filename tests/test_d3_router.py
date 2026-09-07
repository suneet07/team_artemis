from satquery.agent.router import D3BandRouter
from satquery.ingest.band_inventory import BandInventory


def test_route_vegetation():
    # NIR + Red available
    inv_nir_red = BandInventory(is_pan_only=False, computable_indices=["NDVI"])
    router = D3BandRouter(inv_nir_red)
    res = router.route("vegetation")
    assert res["tool"] == "spectral_index"
    assert res["params"]["index"] == "NDVI"

    # RGB only (no NIR)
    inv_rgb = BandInventory(is_pan_only=False, computable_indices=[])
    router = D3BandRouter(inv_rgb)
    res = router.route("vegetation")
    assert res["tool"] == "texture_seg"

    # Pan only
    inv_pan = BandInventory(is_pan_only=True, computable_indices=[])
    router = D3BandRouter(inv_pan)
    res = router.route("vegetation")
    assert res["tool"] == "refusal"
    assert "Input is Pan-only" in res["reason"]


def test_route_water():
    # SWIR + Green available
    inv_mndwi = BandInventory(is_pan_only=False, computable_indices=["MNDWI", "NDWI"])
    router = D3BandRouter(inv_mndwi)
    res = router.route("water")
    assert res["tool"] == "spectral_index"
    assert res["params"]["index"] == "MNDWI"

    # RGB only
    inv_rgb = BandInventory(is_pan_only=False, computable_indices=[])
    router = D3BandRouter(inv_rgb)
    res = router.route("water")
    assert res["tool"] == "texture_seg"

    # Pan only
    inv_pan = BandInventory(is_pan_only=True, computable_indices=[])
    router = D3BandRouter(inv_pan)
    res = router.route("water")
    assert res["tool"] == "refusal"


def test_route_built_up():
    # SWIR + NIR available
    inv_ndbi = BandInventory(is_pan_only=False, computable_indices=["NDBI"])
    router = D3BandRouter(inv_ndbi)
    res = router.route("built-up")
    assert res["tool"] == "spectral_index"
    assert res["params"]["index"] == "NDBI"

    # No SWIR, but SAR available (e.g. Sentinel-1 or Capella)
    inv_sar = BandInventory(is_pan_only=False, computable_indices=[], sar_band="C")
    router = D3BandRouter(inv_sar)
    res = router.route("built-up")
    assert res["tool"] == "sar_backscatter"

    # No SWIR, No SAR, RGB only
    inv_rgb = BandInventory(is_pan_only=False, computable_indices=[], sar_band=None)
    router = D3BandRouter(inv_rgb)
    res = router.route("built-up")
    assert res["tool"] == "texture_seg"

    # Pan only, no SAR
    inv_pan = BandInventory(is_pan_only=True, computable_indices=[], sar_band=None)
    router = D3BandRouter(inv_pan)
    res = router.route("built-up")
    assert res["tool"] == "refusal"


def test_route_unrecognized():
    inv = BandInventory()
    router = D3BandRouter(inv)
    res = router.route("unknown_intent")
    assert res["tool"] == "refusal"
    assert "Unrecognized routing intent" in res["reason"]
