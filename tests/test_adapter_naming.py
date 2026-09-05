"""Rule 9: an adapter name always carries the base model.

    "Adapter names always include the base model -- change_vqa@qwen3vl-4b-v2."

The trace is meant to record which weights produced an answer. A bare `@v2` tag
cannot do that, and §29 Q7 (Qwen3-VL-4B vs Qwen3.5-2B) is still open, so two
traces from either side of that decision would otherwise be indistinguishable.
"""

import os

import pytest

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import run_query
from satquery.ingest.band_inventory import BandInventory
from satquery.serving.client import (
    ADAPTER_VERSIONS,
    base_model,
    qualified_adapter,
)

LEARNED_TOOLS = ["rs_vqa", "rs_ground_caption", "change_vqa", "optsar_fusion"]


def _optical_bundle() -> ImageBundle:
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        sar_band=None,
        sensor_hint=None,
        computable_indices=["NDVI", "NDWI"],
    )
    img = ImageRef(
        scene_id="s0",
        path="scene.tif",
        modality="optical",
        crs="EPSG:32644",
        pixel_size_m=10.0,
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id="b_adapter", images=[img], band_inventory=inv, pair_type="single"
    )


def test_matches_the_spec_worked_example():
    """§5 Rule 9 spells out the exact shape."""
    assert qualified_adapter("change_vqa", "v2") == "change_vqa@qwen3vl-4b-v2"


@pytest.mark.parametrize("tool", LEARNED_TOOLS)
def test_every_learned_tool_reports_a_qualified_adapter(tool: str, monkeypatch):
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    mod = __import__(f"satquery.tools.{tool}", fromlist=["execute"])
    out = mod.execute({"question": "what is here?"}, {"bundle": _optical_bundle()})
    adapter = out["adapter"]
    assert adapter == qualified_adapter(tool, ADAPTER_VERSIONS[tool])
    assert adapter.startswith(f"{tool}@")
    assert base_model() in adapter, f"{adapter} omits the base model (Rule 9)"


def test_graded_tools_invoked_names_the_adapter(monkeypatch):
    """§17: tools_invoked carries the adapter for learned tools."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    res = run_query(_optical_bundle(), "Describe the contents of this satellite image.")
    invoked = res.trace["graded"]["tools_invoked"]
    assert invoked == [qualified_adapter("rs_ground_caption", "v1")]


def test_deterministic_tools_stay_bare(monkeypatch):
    """§17 mixes both: only learned steps carry an adapter."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    res = run_query(_optical_bundle(), "What is the water extent in this scene?")
    invoked = res.trace["graded"]["tools_invoked"]
    assert invoked == ["spectral_index"], "a deterministic tool has no adapter to name"


def test_fusion_block_names_base_and_adapter(monkeypatch):
    """§17: fusion.model is base + adapter, e.g. qwen3vl-4b+optsar_fusion@..."""
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
    res = run_query(_optical_bundle(), "Describe the contents of this satellite image.")
    model = res.trace["fusion"]["model"]
    # §17 shape: "qwen3vl-4b-instruct+optsar_fusion@v3" -- base once, then tool@version.
    assert model == f"{base_model()}+rs_ground_caption@v1"
    assert model.count(base_model()) == 1, "base model repeated in fusion.model"


def test_base_model_is_configurable_because_q7_is_open(monkeypatch):
    """The bake-off has not run; the base must not be a constant."""
    monkeypatch.setenv("SATQUERY_BASE_MODEL", "qwen35-2b")
    assert qualified_adapter("change_vqa", "v2") == "change_vqa@qwen35-2b-v2"


def test_no_unqualified_adapter_literals_remain():
    """Guard against a bare @v tag creeping back into a tool."""
    import pathlib
    import re

    root = pathlib.Path(__file__).parent.parent / "satquery" / "tools"
    bare = re.compile(r'adapter\s*=\s*"[a-z_]+@v\d+"')
    offenders = [
        p.name for p in root.glob("*.py") if bare.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"unqualified adapter literal in {offenders} (Rule 9)"


def test_health_reports_the_base_model():
    """§22: /meta/health exposes what is serving."""
    from satquery.serving.client import ServingClient

    health = ServingClient().get_health()
    assert health["base_model"] == base_model()
    # With nothing reachable, claim no adapters rather than a fictional roster.
    assert health["serving"] != "vllm"
    assert health["adapters_loaded"] == []


def test_env_override_is_not_leaked_between_tests():
    assert os.environ.get("SATQUERY_BASE_MODEL") in (None, base_model())
