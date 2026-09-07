"""BIFOLD stays float32 even when a VLM has changed torch's default dtype.

`lulc_classifier` scored 74.95% as a standalone script and then failed on every
request inside the deployed API with

    RuntimeError: Expected weight to have type Float but got BFloat16

Nothing about the tool had changed. Loading Qwen3-VL with `dtype=bfloat16`
leaves torch's *global* default dtype at bfloat16, and `timm.create_model` then
builds a bf16 ResNet-50 -- so the bug only appears when the classifier shares a
process with the VLM, which is exactly what the deployment does and what a
standalone benchmark never does.

The executor records a tool failure as a trace warning and answers from
whatever else ran, so this cost nothing visible: health reported the tool
available, the answer came back confidently, and the measured 74.95% model was
silently absent from every response.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from satquery.ingest.band_inventory import BandInventory  # noqa: E402
from satquery.tools.base import Scene, ToolContext  # noqa: E402


@pytest.fixture
def bfloat16_default():
    """A process where something else has already set the global dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _sar_scene() -> Scene:
    bands = {
        "vv": np.random.normal(-12.0, 3.0, (64, 64)).astype("float32"),
        "vh": np.random.normal(-18.0, 3.0, (64, 64)).astype("float32"),
    }
    return Scene(
        name="s1",
        modality="sar",
        bands=bands,
        inventory=BandInventory(bands=("vv", "vh")),
        pixel_size_m=10.0,
    )


def test_the_model_is_float32_even_under_a_bfloat16_default(bfloat16_default):
    """The regression, reproduced without downloading BIFOLD's weights."""
    timm = pytest.importorskip("timm")
    from satquery.tools.lulc import CLASSES

    # What `_load` does, with the guard the fix added. Without setting the
    # dtype explicitly this model comes out bfloat16.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = timm.create_model("resnet50", in_chans=2, num_classes=len(CLASSES))
    finally:
        torch.set_default_dtype(previous)

    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, (
        f"the classifier built as {dtypes} under a bfloat16 default; BIFOLD "
        "publishes float32 weights and its forward pass rejects the mismatch"
    )


def test_an_unguarded_build_really_would_have_been_bfloat16(bfloat16_default):
    """Proves the fixture reproduces the original condition, not a no-op test."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet50", in_chans=2, num_classes=19)
    assert any(p.dtype is torch.bfloat16 for p in model.parameters()), (
        "the global default dtype no longer leaks into timm, so the guard in "
        "lulc.py may be removable -- check before deleting it"
    )


def test_the_tool_reports_a_missing_band_rather_than_padding_it():
    """A zero channel is indistinguishable downstream from a band reading zero."""
    from satquery.tools.lulc import LulcClassifierTool

    scene = _sar_scene()
    del scene.bands["vh"]
    scene.inventory = BandInventory(bands=("vv",))
    # Band names are addressed as BIFOLD orders them, `("VH", "VV")`.
    with pytest.raises(ValueError, match="VH"):
        LulcClassifierTool().run(
            ToolContext(query_text="what land cover?", scenes=[scene], params={})
        )


def test_the_serving_path_matches_the_benchmarked_preprocessing():
    """The tool must preprocess exactly as the script that measured 74.95% did.

    `scripts/eval_bifold.py` produced the number; `satquery/tools/lulc.py`
    serves it. They are separate implementations, so any drift between them
    silently invalidates the reported score -- the tool would still answer, just
    not the way it was graded.
    """
    import re
    from pathlib import Path

    from satquery.tools.lulc import _S1_ORDER, _S1_STATS, _S2_ORDER, _S2_STATS

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_bifold.py"
    source = script.read_text(encoding="utf-8")

    def floats(name: str) -> list[float]:
        block = re.search(name + r"\s*=\s*\{(.*?)\}", source, re.S)
        assert block, f"{name} not found in eval_bifold.py"
        return [float(x) for x in re.findall(r"-?\d+\.\d+", block.group(1))]

    served_s1 = [v for band in ("VV", "VH") for v in _S1_STATS[band]]
    assert sorted(floats("S1_STATS")) == sorted(served_s1), (
        "S1 normalisation drifted from the benchmarked values"
    )
    served_s2 = [v for band in _S2_ORDER for v in _S2_STATS[band]]
    assert sorted(floats("S2_STATS")) == sorted(served_s2)

    s1_order = re.search(r"S1_ORDER\s*=\s*\[(.*?)\]", source).group(1)
    assert [b.strip().strip('"') for b in s1_order.split(",")] == list(_S1_ORDER), (
        "band order drifted; a [VV, VH] control scored exactly 49.9% -- chance"
    )
    s2_order = re.search(r"S2_ORDER\s*=\s*\[(.*?)\]", source).group(1)
    assert [b.strip().strip('"') for b in s2_order.split(",")] == list(_S2_ORDER)


def test_the_answer_threshold_is_the_one_that_was_measured():
    """0.5 is where 74.95% was scored; showing 0.25 would over-report."""
    import re
    from pathlib import Path

    from satquery.agent.pipeline import _LULC_THRESHOLD

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_bifold.py"
    defaults = re.findall(
        r'--threshold", type=float, default=([\d.]+)',
        script.read_text(encoding="utf-8"),
    )
    assert defaults, "eval_bifold.py no longer declares a threshold default"
    assert {float(d) for d in defaults} == {_LULC_THRESHOLD}
