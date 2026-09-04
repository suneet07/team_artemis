"""Training collator implementing the QGP §2 prompt assembly contract.

Shared with the agentic controller (satquery/agent/prompt.py) to guarantee
train/inference prompt parity per §13 of the P5 Build Specification.
"""

from typing import Any

from satquery.agent.bundle import ImageRef
from satquery.agent.prompt import assemble_prompt


class QGPCollator:
    """Collator that prepares training batches by assembling prompts from stored parts."""

    def __init__(self) -> None:
        self.assembler = assemble_prompt

    def collate_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Assembles prompt string for a single training sample from its components."""
        images = [
            ImageRef(
                scene_id=img.get("scene_id", f"img_{i}"),
                path=img.get("path", f"img_{i}.tif"),
                modality=img.get("modality", "optical"),
                native_gsd_m=img.get("native_gsd_m"),
            )
            for i, img in enumerate(example.get("images", []))
        ]
        prompt = self.assembler(
            images=images,
            roles=example.get("roles", ["single"]),
            modalities=example.get("modalities", [img.modality for img in images]),
            effective_gsd_m=example.get(
                "effective_gsd_m", [img.native_gsd_m or 4.0 for img in images]
            ),
            question=example["question"],
            point_prior=example.get("point_prior"),
        )
        res = dict(example)
        res["prompt"] = prompt
        return res

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.collate_example(ex) for ex in batch]
