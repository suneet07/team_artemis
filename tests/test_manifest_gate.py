import pytest

from satquery.ingest.band_inventory import BandInventory
from satquery.tools.manifest import ToolManifest
from satquery.tools.registry import ToolRegistry, check_parameters, effective_params


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.default()


@pytest.fixture(scope="module")
def dummy(registry: ToolRegistry) -> ToolManifest:
    return registry.get("dummy_tool")


@pytest.fixture(scope="module")
def spectral(registry: ToolRegistry) -> ToolManifest:
    return registry.get("spectral_index")


def cartosat_mx() -> BandInventory:
    return BandInventory(bands={"blue": 1, "green": 2, "red": 3, "nir": 4})


def test_registry_loads_builtin_manifests(registry):
    """Every tool section 4.6 names has a manifest, adapter or no adapter.

    The manifest is the contract the section 4.5.4 gate enforces against, so a
    learned tool needs one before its weights exist. Without it the gate rejects
    a perfectly legitimate plan as "not in the registry" and the query refuses
    instead of degrading to the deterministic path.
    """
    assert set(registry.names()) == {
        # deterministic (P6)
        "centroid_prior",
        "change_stats",
        "coreg_check",
        "object_box_fallback",
        "sar_backscatter",
        "spectral_index",
        "texture_seg",
        "tile_scorer",
        # learned (4.6.1) - manifests ship ahead of the adapters
        "change_map",
        "change_vqa",
        "lulc_classifier",
        "optsar_fusion",
        "rs_ground_caption",
        "rs_vqa",
        # phase 0 scaffold
        "dummy_tool",
    }


def test_every_manifest_declaring_a_mask_declares_it_full_scene(registry):
    """C18: a mask is a graded artifact, so its contract must say so."""
    offenders = []
    for name in registry.names():
        manifest = registry.get(name)
        for output, spec in manifest.outputs.items():
            if spec.get("type") != "geotiff":
                continue
            if spec.get("crs") != "source" or spec.get("resolution") != "full_scene":
                offenders.append(f"{name}.{output}: {spec}")
    assert not offenders, (
        "Every mask output must be declared in the source CRS at full scene "
        "resolution (C18): " + "; ".join(offenders)
    )


def test_valid_params_pass(dummy):
    result = check_parameters(dummy, {"index": "ALPHA", "scale": 0.4})
    assert result.passed
    assert result.rejected == []


def test_unknown_param_rejected_not_ignored(dummy):
    result = check_parameters(dummy, {"index": "ALPHA", "scale": 0.4, "gain": 2.0})
    assert not result.passed
    assert any("unknown parameter 'gain'" in r for r in result.rejected)


def test_off_enum_value_rejected_not_clamped(dummy):
    params = {"index": "GAMMA", "scale": 0.4}
    result = check_parameters(dummy, params)
    assert not result.passed
    assert any("GAMMA" in r for r in result.rejected)
    assert params["index"] == "GAMMA"


def test_out_of_range_value_rejected_not_clamped(dummy):
    result = check_parameters(dummy, {"index": "ALPHA", "scale": 47.0})
    assert not result.passed
    assert any("outside permitted range" in r for r in result.rejected)


def test_boundary_values_pass_inclusive_range(dummy):
    assert check_parameters(dummy, {"index": "ALPHA", "scale": 0.0}).passed
    assert check_parameters(dummy, {"index": "ALPHA", "scale": 1.0}).passed


def test_missing_required_param_rejected(dummy):
    result = check_parameters(dummy, {"scale": 0.4})
    assert not result.passed
    assert any("missing required parameter 'index'" in r for r in result.rejected)


def test_wrong_type_rejected(dummy):
    assert not check_parameters(dummy, {"index": "ALPHA", "scale": "high"}).passed
    assert not check_parameters(dummy, {"index": 1, "scale": 0.5}).passed


def test_bool_is_not_a_number(dummy):
    assert not check_parameters(dummy, {"index": "ALPHA", "scale": True}).passed


def test_optional_param_may_be_omitted(dummy):
    assert check_parameters(dummy, {"index": "BETA", "scale": 1.0}).passed


def test_defaults_filled_without_overriding_supplied(dummy):
    merged = effective_params(dummy, {"index": "BETA"})
    assert merged["scale"] == 0.5
    merged = effective_params(dummy, {"index": "BETA", "scale": 0.9})
    assert merged["scale"] == 0.9


def test_requires_bands_blocks_unavailable_index(spectral):
    inv = cartosat_mx()
    ok = check_parameters(spectral, {"index": "NDVI"}, inv)
    ndwi = check_parameters(spectral, {"index": "NDWI"}, inv)
    ndbi = check_parameters(spectral, {"index": "NDBI"}, inv)
    mndwi = check_parameters(spectral, {"index": "MNDWI"}, inv)
    assert ok.passed
    assert ndwi.passed
    assert not ndbi.passed
    assert any("swir" in r for r in ndbi.rejected)
    assert not mndwi.passed


def test_requires_bands_passes_when_swir_present(spectral):
    inv = BandInventory(bands={"green": 2, "nir": 4, "swir": 5}, has_swir=True)
    assert check_parameters(spectral, {"index": "NDBI"}, inv).passed
    assert check_parameters(spectral, {"index": "MNDWI"}, inv).passed


def test_band_check_skipped_without_inventory(spectral):
    assert check_parameters(spectral, {"index": "NDBI"}).passed


def test_modality_mismatch_rejected(spectral):
    result = check_parameters(spectral, {"index": "NDVI"}, cartosat_mx(), modalities=["sar"])
    assert not result.passed
    assert any("requires modality 'optical'" in r and "'sar'" in r for r in result.rejected)


def test_modality_match_passes(spectral):
    result = check_parameters(
        spectral, {"index": "NDVI"}, cartosat_mx(), modalities=["optical", "sar"]
    )
    assert result.passed


def test_modality_accepts_single_string(spectral):
    result = check_parameters(spectral, {"index": "NDWI"}, cartosat_mx(), modalities="optical")
    assert result.passed


def test_modality_check_skipped_without_context(spectral):
    assert check_parameters(spectral, {"index": "NDBI"}).passed


def test_duplicate_registration_raises(registry, dummy):
    with pytest.raises(ValueError):
        registry.register(dummy)


def test_unknown_tool_lookup_raises(registry):
    with pytest.raises(KeyError):
        registry.get("does_not_exist")
