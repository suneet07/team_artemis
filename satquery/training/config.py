"""Training configuration, derived from the frozen preprocessing contract.

Section 5.2 freezes the input representation and the LoRA hyperparameters;
``configs/preprocessing.yaml`` freezes ``max_pixels`` and the SAR chain. Nothing
in a training run may restate either. Where a value exists in the contract it is
read from the contract, and the field carries no default that could win.

The one thing this adds on top is the **composite count per adapter**, which
section 5.2 states in prose rather than in YAML: three composites on
multispectral sources under C22, two on ``rs_ground_caption`` because Cartosat
has no SWIR so the third view never exists at inference. Pricing every adapter
at three is what made the first sweep overstate the budget.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from satquery.config import PreprocessingConfig, preprocessing_config

__all__ = [
    "ADAPTERS",
    "LORA_TARGETS",
    "TrainingConfig",
    "adapter_composites",
]

#: The four adapters of section 5.3, in the order section 5.5 guard rail 3 runs
#: them: ``rs_vqa`` first, because it is cheapest and exercises the whole
#: data -> train -> serve path. A broken data format surfaces at hour 4, not 16.
ADAPTERS = ("rs_vqa", "change_vqa", "optsar_fusion", "rs_ground_caption")

#: Section 5.2: plain LoRA on the unmodified architecture (C1). ``all-linear``
#: rather than a hand-picked module list -- the dual-ViT recipe that motivated
#: targeted injection is explicitly not what we run, and vLLM multi-LoRA needs
#: the architecture unmodified.
LORA_TARGETS = "all-linear"

#: Views per *sample* — not per date. Section 5.2 + C22. ``rs_ground_caption``
#: is two because its sub-metre sources are RGB and the deployment sensor has no
#: SWIR. ``change_vqa`` is **six**: three composites at each of two dates, which
#: is what section 5.5 prices at 1,536 vision tokens per sample and what makes
#: the task answerable at all -- a change question with one date is unanswerable
#: except from the answer prior.
_COMPOSITES: dict[str, int] = {
    "rs_vqa": 3,
    "change_vqa": 6,
    "optsar_fusion": 3,
    "rs_ground_caption": 2,
}


def adapter_composites(adapter: str) -> int:
    """Composite views per sample for ``adapter``.

    Raises for an unknown adapter rather than defaulting to three: a typo that
    silently prices an adapter 50% high is exactly the error the first budget
    table shipped with.
    """
    try:
        return _COMPOSITES[adapter]
    except KeyError:
        raise ValueError(
            f"unknown adapter {adapter!r}; section 5.3 defines {ADAPTERS}"
        ) from None


@dataclass
class TrainingConfig:
    """Everything a training or timing run needs, with its provenance.

    Construct with :meth:`from_frozen` so ``max_pixels`` comes from the YAML.
    The constructor still accepts an override because the timing sweep's whole
    purpose is to price ``max_pixels`` as a variable -- but an override is
    recorded in :attr:`overrides` and reported, so a swept value can never be
    mistaken for the frozen one.
    """

    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"

    # --- from configs/preprocessing.yaml -----------------------------------
    max_pixels: int = 0
    config_version: int = 0

    # --- from master plan section 5.2 --------------------------------------
    composites: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: str = LORA_TARGETS
    peak_lr: float = 1e-4
    init_lr: float = 1e-6
    warmup_frac: float = 0.01
    epochs: int = 1

    # --- run shape ---------------------------------------------------------
    micro_batch: int = 4
    grad_accum: int = 1
    devices: int = 1
    num_workers: int = 4
    grad_checkpointing: bool = True
    max_text_tokens: int = 256
    seed: int = 0

    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frozen(
        cls, config: PreprocessingConfig | None = None, **overrides: Any
    ) -> "TrainingConfig":
        """Build from the frozen contract, recording anything overridden."""
        cfg = config if config is not None else preprocessing_config()
        base = cls(
            max_pixels=cfg.tiling.max_pixels,
            config_version=getattr(cfg, "version", 0),
        )
        if not overrides:
            return base
        recorded = {
            name: {"frozen": getattr(base, name, None), "used": value}
            for name, value in overrides.items()
            if getattr(base, name, None) != value
        }
        return replace(base, **overrides, overrides=recorded)

    @classmethod
    def for_adapter(
        cls, adapter: str, config: PreprocessingConfig | None = None, **overrides: Any
    ) -> "TrainingConfig":
        """Frozen config at the composite count this adapter actually uses."""
        return cls.from_frozen(
            config, composites=adapter_composites(adapter), **overrides
        )

    # --- derived -----------------------------------------------------------

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum * self.devices

    @property
    def vision_tokens_per_image(self) -> int:
        """Qwen3-VL: 16 px patches, 2x2 spatial merge, so ``max_pixels / 1024``.

        This is a **cap**, not a target. Qwen3-VL never upsamples, so a 120 px
        BigEarthNet patch costs 16 tokens no matter how high ``max_pixels`` is.
        Use :meth:`vision_tokens_for` when the source size is known.
        """
        return self.max_pixels // 1024

    def vision_tokens_for(self, source_px: int) -> int:
        """Vision tokens for a ``source_px`` square source, cap applied."""
        return min(source_px * source_px, self.max_pixels) // 1024

    @property
    def frozen_max_pixels(self) -> bool:
        """Whether ``max_pixels`` is the contract value or a swept override."""
        return "max_pixels" not in self.overrides

    def provenance(self) -> str:
        """One line naming where the numbers came from, for every report."""
        source = (
            f"configs/preprocessing.yaml v{self.config_version}"
            if self.frozen_max_pixels
            else (
                f"OVERRIDDEN (frozen value {self.overrides['max_pixels']['frozen']}, "
                f"swept at {self.max_pixels})"
            )
        )
        return (
            f"max_pixels={self.max_pixels} from {source}; "
            f"LoRA r={self.lora_r} alpha={self.lora_alpha} dropout={self.lora_dropout} "
            f"target={self.lora_targets}; LR {self.init_lr:g} -> {self.peak_lr:g} over "
            f"{self.warmup_frac:.0%} then cosine (master plan section 5.2)"
        )
