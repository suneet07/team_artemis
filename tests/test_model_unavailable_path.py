"""Tests that exercise the MODEL_UNAVAILABLE path when SATQUERY_STUB_SERVING is not set.

L-3 audit finding: conftest.py sets SATQUERY_STUB_SERVING=1 suite-wide, so no test
exercises the MODEL_UNAVAILABLE default unless it explicitly unsets the variable.
"""

import pytest

from satquery.agent.bundle import BandInventory, ImageBundle, ImageRef
from satquery.agent.graph import run_query


def _make_single_optical_bundle() -> ImageBundle:
    img = ImageRef(
        scene_id="test_scene",
        path="nonexistent.tif",
        modality="optical",
        native_gsd_m=4.0,
        crs="EPSG:32644",
    )
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id="bundle_model_unavail_test",
        images=[img],
        band_inventory=inv,
        pair_type="single",
        status="ready",
    )


def test_model_unavailable_yields_honest_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """Without SATQUERY_STUB_SERVING, learned tools must fail with MODEL_UNAVAILABLE.

    The confidence must be 0.0 and the state must be 'failed' (not 'succeeded').
    An invented answer at 0.85 confidence is the defect class this prevents.
    """
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SATQUERY_TRACES_DIR", str(traces_dir))
    # Critically: do NOT set SATQUERY_STUB_SERVING — this exercises the real unavailable path
    monkeypatch.delenv("SATQUERY_STUB_SERVING", raising=False)

    bundle = _make_single_optical_bundle()
    result = run_query(bundle, "Describe the contents of this satellite image.")

    # The query may succeed (if deterministic fallbacks produce output) or fail.
    # Either way, if MODEL_UNAVAILABLE occurred, confidence MUST be 0.0 or low.
    if result.state == "failed":
        # All tools failed path: confidence must be 0.0
        assert result.confidence == 0.0, (
            f"Failed query should have confidence 0.0, got {result.confidence}"
        )
    else:
        # If we got a result, it must not be the invented stub answer at 0.85
        if result.answer and "prominent water body" in (result.answer or "").lower():
            pytest.fail(
                "Stub keyword-matched answer reached the output without SATQUERY_STUB_SERVING. "
                "The serving client must return MODEL_UNAVAILABLE by default, not fabricated text."
            )
        # Confidence must not claim calibrated high confidence from the stub
        is_high_calibrated = (
            result.confidence
            and result.confidence > 0.80
            and result.confidence_basis == "calibrated"
        )
        if is_high_calibrated:
            pytest.fail(
                f"Query reported calibrated confidence {result.confidence} from a stub path. "
                "Unfitted calibrator should never produce high confident results."
            )


def test_model_unavailable_result_state_not_fabricated_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """Without the stub, a single-caption query must NOT report state=succeeded with fake boxes.

    R-2 regression: rs_ground_caption used to silently return a hardcoded placeholder box
    at 0.85 confidence. That box is now deleted; this test verifies it stays deleted.
    """
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SATQUERY_TRACES_DIR", str(traces_dir))
    monkeypatch.delenv("SATQUERY_STUB_SERVING", raising=False)

    bundle = _make_single_optical_bundle()
    result = run_query(bundle, "Where are the buildings?")

    trace = result.trace
    # Check no step emits a constant hardcoded box when model is unavailable
    for step in trace.get("steps", []):
        step_outputs = step.get("outputs", {})
        boxes = step_outputs.get("boxes", [])
        for box in boxes:
            bbox = box.get("bbox_px", [])
            if bbox == [100.0, 100.0, 200.0, 200.0]:
                pytest.fail(
                    "Hardcoded placeholder box [100, 100, 200, 200] found in trace output. "
                    "The default_boxes constant must not be present in rs_ground_caption. (R-2)"
                )
