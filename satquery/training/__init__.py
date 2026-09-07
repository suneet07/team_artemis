"""Training machinery, inside the package rather than beside it.

Until now the timing harness lived entirely in ``scripts/`` and imported nothing
from ``satquery``. The knowledge graph shows the consequence plainly: no path
from ``TimingModule`` to ``PreprocessingConfig``, none from the harness dataset
to ``load_scene``. It restated ``max_pixels``, the composite count and the LoRA
hyperparameters instead of reading them, and drifted -- its docstring still
described ``max_pixels: null`` after the config had been frozen at 262,144.

That is the timing review's defect 3 ("config measured != config that will run")
with nothing structural to stop it recurring. Guard rail 4 asks for the section
5.5 budget table to be re-derived from a real measurement; a measurement of a
config the deployed system will not use does not satisfy it.

So the pieces every training entry point shares -- the config, the dataset, the
label-masking collator, the model/LoRA build -- live here, are imported by the
sweep, the trainer and the evaluation harnesses alike, and are covered by tests
that run on CPU.

Torch is imported lazily inside functions, never at module scope. CI installs
``[dev]`` on a CPU runner and imports the whole package; a top-level torch
import would add minutes to every PR for code CI cannot exercise.
"""

from satquery.training.config import TrainingConfig, adapter_composites
from satquery.training.dataset import (
    CanonicalSample,
    load_canonical_manifest,
    resolve_sample_images,
)

__all__ = [
    "CanonicalSample",
    "TrainingConfig",
    "adapter_composites",
    "load_canonical_manifest",
    "resolve_sample_images",
]
