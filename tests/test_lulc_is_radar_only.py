"""`lulc_classifier` must never answer from the optical model.

The optical arm scores **0.4968 on 4,000 rows** -- chance -- against radar's
0.7525, and radar wins 87% of their disagreements (`logs/bifold_duet.json`).
That decision is recorded in the manifest as `required_modalities: [sar]`.

The tool nonetheless carried an `elif optical is not None` fallback that loaded
`resnet50-s2` behind a warning. Two guards made it unreachable -- the router
requires `has_sar` before planning the tool, and the parameter gate enforces the
manifest -- but unreachable is not safe: relaxing either guard would have served
a chance-level classifier as a measurement, and the warning explaining that is a
panel most readers never open.

Three tests, because the protection is three independent layers and any one of
them silently failing is what would let this back in.
"""

import pytest

from satquery.agent.executor import check_plan
from satquery.agent.router import QueryContext, route
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.registry import ToolRegistry

OPTICAL = BandInventory(bands={"blue": 1, "green": 2, "red": 3, "nir": 4})


def test_the_manifest_requires_radar():
    manifest = ToolRegistry.default().get("lulc_classifier")
    assert list(manifest.required_modalities) == ["sar"]


def test_the_router_does_not_plan_it_on_optical_only():
    decision = route(
        QueryContext(
            query_text="Is there forest in this image?",
            modalities=["optical"],
            inventories=[OPTICAL],
            image_count=1,
        )
    )
    assert "lulc_classifier" not in [step["tool"] for step in decision.plan]


def test_the_gate_refuses_it_on_optical_only():
    outcome = check_plan(
        [{"tool": "lulc_classifier", "params": {}}],
        ToolRegistry.default(),
        band_inventory=OPTICAL,
        modalities=["optical"],
    )
    assert not outcome.runnable
    assert any("sar" in reason for reason in outcome.rejected)


def test_the_tool_itself_refuses_without_radar():
    """The last line of defence, and the one that was missing. Even called
    directly with an optical scene, it must refuse rather than fall back."""
    import numpy as np

    from satquery.config import preprocessing_config
    from satquery.tools.base import Scene, ToolContext
    from satquery.tools.catalog import implementation

    scene = Scene(
        name="optical.tif",
        modality="optical",
        bands={n: np.zeros((8, 8)) for n in ("blue", "green", "red", "nir")},
        inventory=OPTICAL,
    )
    tool = implementation("lulc_classifier")
    context = ToolContext(
        query_text="Is there forest in this image?",
        scenes=[scene],
        params={},
        config=preprocessing_config(),
    )
    with pytest.raises(ValueError, match="radar only"):
        tool.run(context)
