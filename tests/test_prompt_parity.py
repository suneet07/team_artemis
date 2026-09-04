"""Prompt parity tests: inference assembler vs. training collator vs. golden fixture.

§13 requires that train/serve prompt assembly byte-matches. This is enforced by:
1. Comparing assemble_prompt() output against the committed golden fixture.
2. Comparing QGPCollator.collate_example() against the SAME golden fixture.

The golden fixture in training/eval/prompt_fixture.json was generated once by running
assemble_prompt() and checked in.  To update the format, you must:
  1. Regenerate the fixture intentionally (see training/eval/generate_prompt_fixture.py).
  2. Update training/ imports and any grounding-prior format with the QGP owner (§13 Q5).
  3. Re-check both sides agree before committing.

Failure of these tests means the train/serve contract has drifted — the highest-risk
silent failure in the whole component (§13).
"""

import json
from pathlib import Path

import pytest

from satquery.agent.bundle import ImageRef
from satquery.agent.prompt import assemble_prompt
from training.collator import QGPCollator

# Path to the committed golden fixture, relative to the repo root
FIXTURE_PATH = Path(__file__).parent.parent / "training" / "eval" / "prompt_fixture.json"


@pytest.fixture(scope="module")
def golden() -> dict[str, str]:
    """Load the golden prompt fixture checked into training/eval/."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Golden fixture not found at {FIXTURE_PATH}. "
            "Generate it by running: python training/eval/generate_prompt_fixture.py"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_prompt_parity_bitemporal_assembler_vs_golden(golden: dict[str, str]) -> None:
    """assemble_prompt() must byte-match the golden fixture for a bitemporal pair."""
    img0 = ImageRef(scene_id="s0", path="p0.tif", modality="optical", native_gsd_m=4.0)
    img1 = ImageRef(scene_id="s1", path="p1.tif", modality="optical", native_gsd_m=4.0)

    prompt = assemble_prompt(
        images=[img0, img1],
        roles=["t0", "t1"],
        modalities=["optical", "optical"],
        effective_gsd_m=[4.0, 4.0],
        question="What changed between these two acquisitions?",
    )

    expected = golden["bitemporal"]
    assert prompt == expected, (
        "assemble_prompt() output does not match the golden fixture. "
        "If the format changed intentionally, regenerate training/eval/prompt_fixture.json "
        "and confirm the QGP owner is aligned (§13).\n"
        f"Expected: {expected!r}\n"
        f"Got:      {prompt!r}"
    )


def test_prompt_parity_bitemporal_collator_vs_golden(golden: dict[str, str]) -> None:
    """QGPCollator must produce the same prompt as the golden fixture for a bitemporal pair."""
    collator = QGPCollator()
    example = {
        "images": [
            {"scene_id": "s0", "path": "p0.tif", "modality": "optical", "native_gsd_m": 4.0},
            {"scene_id": "s1", "path": "p1.tif", "modality": "optical", "native_gsd_m": 4.0},
        ],
        "roles": ["t0", "t1"],
        "modalities": ["optical", "optical"],
        "effective_gsd_m": [4.0, 4.0],
        "question": "What changed between these two acquisitions?",
    }
    collated = collator.collate_example(example)

    expected = golden["bitemporal"]
    assert collated["prompt"] == expected, (
        "QGPCollator output does not match the golden fixture — train/serve skew detected! "
        "Regenerate training/eval/prompt_fixture.json and align both pipeline sides (§13).\n"
        f"Expected: {expected!r}\n"
        f"Got:      {collated['prompt']!r}"
    )


def test_prompt_parity_grounding_assembler_vs_golden(golden: dict[str, str]) -> None:
    """assemble_prompt() must byte-match the golden fixture for a grounding query with prior."""
    img0 = ImageRef(scene_id="s0", path="p0.tif", modality="optical", native_gsd_m=0.5)

    prompt = assemble_prompt(
        images=[img0],
        roles=["single"],
        modalities=["optical"],
        effective_gsd_m=[0.5],
        question="Where is the aircraft?",
        point_prior=(0.450, 0.625),
    )

    expected = golden["grounding_with_prior"]
    assert prompt == expected, (
        "assemble_prompt() grounding output does not match the golden fixture. "
        "Note: the [prior: (x, y)] format must be confirmed with the QGP owner (§13 Q5).\n"
        f"Expected: {expected!r}\n"
        f"Got:      {prompt!r}"
    )


def test_prompt_parity_grounding_collator_vs_golden(golden: dict[str, str]) -> None:
    """QGPCollator must produce the same grounding prompt as the golden fixture."""
    collator = QGPCollator()
    example = {
        "images": [
            {"scene_id": "s0", "path": "p0.tif", "modality": "optical", "native_gsd_m": 0.5}
        ],
        "roles": ["single"],
        "modalities": ["optical"],
        "effective_gsd_m": [0.5],
        "question": "Where is the aircraft?",
        "point_prior": (0.450, 0.625),
    }
    collated = collator.collate_example(example)

    expected = golden["grounding_with_prior"]
    assert collated["prompt"] == expected, (
        "QGPCollator grounding output does not match the golden fixture — train/serve skew!\n"
        f"Expected: {expected!r}\n"
        f"Got:      {collated['prompt']!r}"
    )
