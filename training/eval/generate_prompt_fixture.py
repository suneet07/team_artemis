"""Script to regenerate the golden prompt fixture.

Run this script when the prompt format changes INTENTIONALLY and both sides
(training pipeline and the QGP owner) have been aligned per §13.

Usage:
    python training/eval/generate_prompt_fixture.py
"""

import json
from pathlib import Path

from satquery.agent.bundle import ImageRef
from satquery.agent.prompt import assemble_prompt

FIXTURE_PATH = Path(__file__).parent / "prompt_fixture.json"


def main() -> None:
    img0 = ImageRef(scene_id="s0", path="p0.tif", modality="optical", native_gsd_m=4.0)
    img1 = ImageRef(scene_id="s1", path="p1.tif", modality="optical", native_gsd_m=4.0)
    prompt_bitemporal = assemble_prompt(
        images=[img0, img1],
        roles=["t0", "t1"],
        modalities=["optical", "optical"],
        effective_gsd_m=[4.0, 4.0],
        question="What changed between these two acquisitions?",
    )

    img_single = ImageRef(scene_id="s0", path="p0.tif", modality="optical", native_gsd_m=0.5)
    prompt_grounding = assemble_prompt(
        images=[img_single],
        roles=["single"],
        modalities=["optical"],
        effective_gsd_m=[0.5],
        question="Where is the aircraft?",
        point_prior=(0.450, 0.625),
    )

    fixture = {
        "bitemporal": prompt_bitemporal,
        "grounding_with_prior": prompt_grounding,
    }

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"Golden fixture written to {FIXTURE_PATH}")
    print(json.dumps(fixture, indent=2))


if __name__ == "__main__":
    main()
