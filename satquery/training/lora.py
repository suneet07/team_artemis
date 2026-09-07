"""Model and LoRA construction, with the two silent failures made loud.

**Failure one: eval mode.** ``transformers.from_pretrained`` returns a model in
eval mode and PEFT preserves it. Lightning never calls ``.train()`` at the start
of training -- it only warns. With the layers in eval:

* gradient checkpointing silently no-ops, because transformers gates it on
  ``gradient_checkpointing AND self.training``. Measured **0 of 60** layers
  active, with ``--grad-checkpointing`` and ``--no-grad-checkpointing``
  producing byte-identical 15.9 GB of activations, so every OOM in the first
  cost surface was an artifact rather than a hardware limit;
* every ``nn.Dropout`` is inert, so the LoRA dropout 0.1 that section 5.2
  specifies does nothing at all.

Both fail in the wrong direction: the run completes, the numbers look
plausible, and the config measured is not the config described.
:func:`build_lora_model` calls ``.train()`` itself and :func:`audit_checkpointing`
reports how many layers will actually checkpoint, so a repeat is visible in the
report rather than inferred from a memory-scaling curve months later.

**Failure two: precision.** Loading weights in bf16 by hand and leaving the
trainer at its default ``32-true`` gives pure bf16 with bf16 optimizer states
and no master weights. Base weights load bf16, trainable LoRA parameters are
cast back to fp32, and the caller runs ``bf16-mixed`` -- bf16 compute with fp32
master weights on exactly the parameters the optimizer owns.
"""

from dataclasses import dataclass
from typing import Any

from satquery.training.config import TrainingConfig

__all__ = [
    "CheckpointAudit",
    "ModelBundle",
    "audit_checkpointing",
    "build_lora_model",
    "lr_lambda_for",
    "select_attention",
]


@dataclass
class CheckpointAudit:
    """How many checkpointable layers will actually checkpoint."""

    layers: int
    flag_set: int
    in_train_mode: int
    active: int

    @property
    def fully_active(self) -> bool:
        return self.layers > 0 and self.active == self.layers

    def warning(self) -> str:
        if self.layers == 0:
            return "no checkpointable layers found; the audit could not run"
        if self.fully_active:
            return ""
        return (
            f"{self.layers - self.active} of {self.layers} layer(s) will NOT "
            "checkpoint. Activation memory reflects full storage, so every OOM "
            "below is an artifact rather than a hardware limit."
        )


@dataclass
class ModelBundle:
    """A ready-to-train model plus what was decided while building it."""

    model: Any
    processor: Any
    attention: str
    trainable_params: int
    total_params: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0


def select_attention() -> str:
    """FlashAttention-2 on Ampere and later, SDPA below.

    Section 5.5 turns FA2 on for the A100 and notes Turing cannot have it. The
    capability check keeps one code path working on both rather than failing the
    T4 fallback at import.
    """
    import torch

    if not torch.cuda.is_available():
        return "sdpa"
    major, _ = torch.cuda.get_device_capability()
    return "flash_attention_2" if major >= 8 else "sdpa"


def build_lora_model(cfg: TrainingConfig, *, verbose: bool = True) -> ModelBundle:
    """Load the base model, attach LoRA, and put it in the state it must run in."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    wanted = select_attention()
    try:
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_id, dtype=torch.bfloat16, attn_implementation=wanted
        )
        attention = wanted
    except (ImportError, ValueError) as error:
        if verbose:
            print(f"  [warn] {wanted} unavailable ({error}); falling back to sdpa")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        attention = "sdpa"

    if cfg.grad_checkpointing:
        # use_reentrant=False is REQUIRED. The reentrant path silently declines
        # to checkpoint when the base is frozen and only LoRA params require
        # grad, keeping activations in full.
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
    base.config.use_cache = False

    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_targets,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # fp32 master weights, trainable parameters only.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    # The line Lightning will not call for us. Without it, checkpointing and
    # LoRA dropout are both silently inert.
    model.train()

    processor = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)

    return ModelBundle(
        model=model,
        processor=processor,
        attention=attention,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )


def audit_checkpointing(model) -> CheckpointAudit:
    """Count layers that will genuinely checkpoint, not merely carry the flag."""
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except ImportError:
        return CheckpointAudit(layers=0, flag_set=0, in_train_mode=0, active=0)

    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
    flag = sum(bool(getattr(m, "gradient_checkpointing", False)) for m in layers)
    training = sum(m.training for m in layers)
    active = sum(
        bool(getattr(m, "gradient_checkpointing", False)) and m.training for m in layers
    )
    return CheckpointAudit(
        layers=len(layers), flag_set=flag, in_train_mode=training, active=active
    )


def lr_lambda_for(cfg: TrainingConfig, total_steps: int):
    """Section 5.2 schedule as a ``LambdaLR`` multiplier on ``cfg.init_lr``.

    Warmup ``init_lr -> peak_lr`` over the first ``warmup_frac`` of steps, then
    cosine decay. The first harness used a flat 2e-5, which is a full-finetune
    learning rate roughly 5-10x too low for LoRA.
    """
    import math

    warm = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warm:
            lr = cfg.init_lr + (step / warm) * (cfg.peak_lr - cfg.init_lr)
        else:
            progress = (step - warm) / max(1, total_steps - warm)
            lr = cfg.peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return lr / cfg.init_lr

    return lr_lambda
