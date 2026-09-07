from satquery.agent.bundle import CoregReport, ImageBundle, ImageRef
from satquery.agent.probe import probe_bundle
from satquery.ingest.band_inventory import BandInventory


def _make_probe_bundle(modalities: list[str], coreg_ok: bool = True) -> ImageBundle:
    images = [
        ImageRef(
            scene_id=f"scene_{i}",
            path=f"file_{i}.tif",
            modality=mod,
            crs="EPSG:32644",
        )
        for i, mod in enumerate(modalities)
    ]
    inv = BandInventory(
        bands={"b": 1, "nir": 2},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=["VV"] if "sar" in modalities else [],
        sar_band="C" if "sar" in modalities else None,
        sensor_hint=None,
        computable_indices=["NDVI"],
    )
    coreg = CoregReport(coregistered=coreg_ok) if len(images) > 1 else None
    return ImageBundle(
        bundle_id="probe_test_bundle",
        images=images,
        band_inventory=inv,
        pair_type=(
            "single"
            if len(images) == 1
            else ("crossmodal" if len(set(modalities)) > 1 else "bitemporal")
        ),
        coreg=coreg,
    )


def test_probe_single_optical_bundle():
    bundle = _make_probe_bundle(["optical"])
    supported, blocked = probe_bundle(bundle)

    assert "single_vqa" in supported
    assert "single_caption" in supported
    assert "single_grounding" in supported

    blocked_tasks = {b["task"]: b["reason"] for b in blocked}
    assert "change_map" in blocked_tasks
    assert "change_description" in blocked_tasks
    assert "change_vqa" in blocked_tasks
    assert "crossmodal_extraction" in blocked_tasks
    assert "crossmodal_vqa" in blocked_tasks


def test_probe_single_sar_bundle():
    bundle = _make_probe_bundle(["sar"])
    supported, blocked = probe_bundle(bundle)

    assert "single_vqa" in supported
    assert "single_grounding" in supported

    blocked_tasks = {b["task"]: b["reason"] for b in blocked}
    assert "single_caption" in blocked_tasks
    assert "crossmodal_extraction" in blocked_tasks


def test_probe_bitemporal_optical_bundle():
    bundle = _make_probe_bundle(["optical", "optical"])
    supported, blocked = probe_bundle(bundle)

    assert "change_map" in supported
    assert "change_description" in supported
    assert "change_vqa" in supported
    assert "single_vqa" in supported


def test_probe_crossmodal_bundle():
    bundle = _make_probe_bundle(["optical", "sar"])
    supported, blocked = probe_bundle(bundle)

    assert "crossmodal_extraction" in supported
    assert "crossmodal_vqa" in supported
