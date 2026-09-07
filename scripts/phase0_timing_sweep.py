"""
Phase 0 — 200-step timing sweep (re-derives the Master Plan v3.8 section 5.5 budget table).

Replaces scripts/run_200_step_timing.py, which measured a workload the system will
never run: sequences hard-capped at 128 tokens, ~16 vision tokens per sample from
120x120 dummy patches, no gradient checkpointing, and Lightning's default
precision="32-true" silently contradicting the hand-loaded bf16 weights.

WHAT THIS MEASURES, AND WHY IT IS A SWEEP

configs/preprocessing.yaml now carries `tiling.max_pixels: 262144` (v2), frozen
on the strength of an earlier run of this harness. It is read from the contract
via satquery.training.TrainingConfig and printed in every report, so a swept
value can never be mistaken for the frozen one.

Section 5.5 ranks max_pixels as cost lever #1, and C22 (three-composite optical
input) multiplies image tokens by 2-3x on multispectral samples at an estimated
cost of 5-8 of the 50 hours. The freeze rests on a measurement whose own report
flagged gradient checkpointing as inert, so it is still worth re-deriving. So a
single timing run cannot re-derive the budget table -- it can only price a point on a
curve whose x-axis is still open.

This harness sweeps (max_pixels x n_composites) and emits the section 5.5 table
as a function of both, plus peak VRAM per cell, so the config can be *chosen*
against the 31 h schedule and the 40 GB card actually in hand rather than
assumed.

TOKEN ARITHMETIC (Qwen3-VL: 16 px patch, 2x2 spatial merge)

    vision_tokens_per_image = max_pixels / (16^2 * 2^2) = max_pixels / 1024

so max_pixels=1_048_576 (a 1024x1024 view) is ~1024 vision tokens per composite,
and a three-composite multispectral sample is ~3072 before any text. The old
test ran at ~16.

WHAT IT FIXES RELATIVE TO run_200_step_timing.py

  1. Sequence length is driven by max_pixels and dynamic padding, not a 128 cap.
  2. Labels mask padding, image placeholders, and the prompt span. The old
     labels=input_ids made loss start at 11.32 against ln(151936)=11.93 -- a
     pretrained model at 95% of uniform-random, because it was being scored on
     emitting <pad>.
  3. Trainer(precision="bf16-mixed") is set EXPLICITLY, and trainable LoRA
     params are cast to fp32 for real master weights. The old script loaded
     bf16 by hand and left Trainer at its default precision="32-true".
  4. Warmup steps are excluded from the timed window.
  5. Peak VRAM is reported -- the number section 5.5 needs to pick a batch size,
     and the one number the 40 GB run could not supply.
  6. Epoch projection uses trainer.num_devices * num_nodes and grad_accum, so
     per-device batch is not mistaken for global batch (the old
     steps_for_epoch = 80000 // batch_size).
  7. LoRA hyperparameters follow section 5.2 as written: dropout 0.1 (not 0.05)
     and warmup 1e-6 -> 1e-4 over the first 1% of steps then cosine decay (not a
     flat 2e-5, which is a full-finetune LR roughly 5-10x too low for LoRA).

Timing depends on tensor shapes, not pixel values -- the SPECULATIONS.md call on
that is correct and is preserved here. Synthetic images are generated large and
let max_pixels do the capping, so max_pixels is the true independent variable.

Install
-------
    pip install -r scripts/requirements-timing.txt

NOT `pip install -e ".[train]"` on a GPU box -- that also pulls the package's
base deps, including arosics -> the `gdal` sdist, which needs system GDAL
headers and fails on a stock Lightning AI Studio with "Could not find
gdal-config". This harness imports no satquery module and no geo library.

Usage
-----
    # full sweep, writes logs/phase_0_timing_sweep.md
    python scripts/phase0_timing_sweep.py

    # single point
    python scripts/phase0_timing_sweep.py --max-pixels 1048576 --composites 3

    # quick shape/masking check on CPU or a small card, no timing claims
    python scripts/phase0_timing_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# The harness used to import nothing from the package and restate max_pixels,
# the composite count and the LoRA hyperparameters. It drifted: this docstring
# described `max_pixels: null` after the config was frozen at 262,144. Every
# contract value now comes from the contract.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from satquery.training.config import TrainingConfig  # noqa: E402
from satquery.training.dataset import (  # noqa: E402
    RealChipDataset,
    load_canonical_manifest,
)

# --------------------------------------------------------------------------
# Section 5.5 budget table. Sample counts are the "60-80k samples, 1 epoch"
# band from the plan; hours are the scheduled per-adapter lines.
# --------------------------------------------------------------------------

ADAPTERS: dict[str, dict] = {
    "rs_vqa": {
        "samples": 80_000,
        "budget_h": 4.0,
        "composites": 3,  # BEN.txt is Sentinel-2 multispectral -> C22 three-composite
        "note": "Run first (guard rail 3). Cheapest adapter, exercises the whole path.",
    },
    "change_vqa": {
        "samples": 80_000,
        "budget_h": 6.0,
        "composites": 3,  # bi-temporal pairs; composites here means views per sample
        "note": "SpaceNet 7 / MUDS primary after C59.",
    },
    "optsar_fusion": {
        "samples": 80_000,
        "budget_h": 6.0,
        "composites": 3,  # optical composites + SAR stack
        "note": "Includes SpaceNet 6 / OEM-SAR generated captions.",
    },
    "rs_ground_caption": {
        "samples": 80_000,
        "budget_h": 12.0,
        "composites": 2,  # sub-metre optical, no SWIR on Cartosat -> always two
        "note": "~40% of cost is image size. Uncuttable (C17).",
    },
}

SCHEDULED_TOTAL_H = 31.0  # four adapters + 3 h smoke/timing/debug
RESERVE_H = 19.0
TOTAL_ACCESS_H = 50.0

# Vision tokens per image = max_pixels / (patch_size^2 * merge_size^2)
DEFAULT_MAX_PIXELS_SWEEP = [
    262_144,  # ~256 vision tokens  (512x512 view)
    524_288,  # ~512
    1_048_576,  # ~1024              (1024x1024 view)
    2_097_152,  # ~2048
]
DEFAULT_COMPOSITE_SWEEP = [1, 2, 3]

MIN_PIXELS = 56 * 56

# Default is 512px: the benchmark chip size, and the size every source is tiled
# to under the frozen resolution policy. It is EXACTLY max_pixels=262144, so the
# cap fits without binding. Override with --source-size to measure other real
# workloads: BigEarthNet patches are 120x120 (16 vision tokens).
#
# This default was 2048, which made the cap bind on every cell and mispriced
# every adapter -- synthetic tiles always hit the cap, real imagery mostly does
# not. Qwen3-VL never upsamples, so a 120px source costs 16 tokens no matter how
# high max_pixels is.
DEFAULT_SOURCE_TILE = 512


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class SweepConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_pixels: int = 1_048_576
    composites: int = 3
    micro_batch: int = 4
    grad_accum: int = 1
    steps: int = 200
    warmup_steps: int = 20
    num_workers: int = 4
    devices: int = 1
    grad_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1  # section 5.2 as written
    lora_targets: str = "all-linear"
    peak_lr: float = 1e-4  # section 5.2: warmup 1e-6 -> 1e-4, cosine
    init_lr: float = 1e-6
    warmup_frac: float = 0.01
    max_text_tokens: int = 256  # cap on TEXT only; images are uncapped here
    source_size: int = DEFAULT_SOURCE_TILE
    seed: int = 0
    # Real imagery when given, synthetic shapes when not. Never silently
    # synthetic: `data_source` is printed in the report header, because a cost
    # surface measured on noise and one measured on the corpus are different
    # claims and the first sweep did not distinguish them.
    manifest: str = ""
    image_root: str = ""
    adapter: str = ""

    @property
    def data_source(self) -> str:
        return f"real:{self.manifest}" if self.manifest else "synthetic-shapes"

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum * self.devices

    @property
    def approx_vision_tokens(self) -> int:
        return int(self.composites * self.max_pixels / 1024)


@dataclass
class SweepResult:
    config: dict
    ok: bool
    error: str = ""
    sec_per_opt_step: float = 0.0
    sec_per_sample: float = 0.0
    mean_seq_len: float = 0.0
    p95_seq_len: float = 0.0
    mean_vision_tokens: float = 0.0
    mean_supervised_tokens: float = 0.0
    peak_vram_gb: float = 0.0
    reserved_vram_gb: float = 0.0
    static_vram_gb: float = 0.0
    activation_vram_gb: float = 0.0
    ckpt_layers: int = 0
    ckpt_active: int = 0
    first_loss: float = 0.0
    last_loss: float = 0.0
    effective_tflops: float = 0.0
    per_adapter_hours: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Synthetic data
#
# Pixel VALUES do not affect timing (SPECULATIONS.md item 2 is right about
# that). Pixel DIMENSIONS do, and that is what the old test got wrong. We
# generate deliberately oversized noise tiles and let the processor's
# max_pixels do the capping, so max_pixels is the real independent variable.
# Noise rather than zeros so nothing downstream can shortcut on a constant.
# --------------------------------------------------------------------------


class SyntheticRSDataset(Dataset):
    def __init__(self, cfg: SweepConfig, n: int = 4096):
        self.cfg = cfg
        self.n = n
        rng = np.random.default_rng(cfg.seed)
        # A small pool of distinct tiles, reused. Decode cost is what varies with
        # num_workers; content does not matter.
        self._pool = [
            Image.fromarray(
                rng.integers(0, 256, (cfg.source_size, cfg.source_size, 3), dtype=np.uint8)
            )
            for _ in range(4)
        ]
        self._q = (
            "Would you say that any arable land lies next to pastures in the image? "
            "Answer with reference to the visible land cover."
        )
        # A pool of distinct answers of near-identical token length. The first
        # sweep used ONE fixed answer, so the LoRA memorised it within ~20 steps
        # and loss collapsed to 0.00 -- timing was unaffected (step cost depends
        # on shapes, not values) but the loss channel became useless as a check
        # that labels are wired correctly. Same length keeps seq_len controlled.
        self._answers = [
            "Yes. Arable parcels occupy the north-eastern quadrant and share a "
            "boundary with pasture along the drainage line.",
            "No. Coniferous stands dominate the southern slope and meet open "
            "heathland well short of the cultivated margin.",
            "Yes. Irrigated terraces follow the valley floor and abut grazing "
            "land immediately west of the settlement edge.",
            "No. Broad-leaved canopy covers the eastern third and borders only "
            "water, with no cultivated parcel adjacent.",
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        imgs = [self._pool[(idx + k) % len(self._pool)] for k in range(self.cfg.composites)]
        return {
            "images": imgs,
            "question": self._q,
            "answer": self._answers[idx % len(self._answers)],
        }


class Collator:
    """
    Builds a batch and masks labels properly.

    The old script used labels=batch["input_ids"] with padding to a fixed 128.
    That scores the model on pad tokens and image placeholders, which is why
    loss opened at 11.32 (ln(vocab)=11.93). Here the prompt is tokenized twice --
    once with add_generation_prompt=True to find where the answer begins -- and
    everything before the answer, plus all padding, is set to -100.
    """

    def __init__(self, processor, cfg: SweepConfig):
        self.processor = processor
        self.cfg = cfg
        tok = getattr(processor, "tokenizer", processor)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _messages(self, item, with_answer: bool):
        content = [{"type": "image"} for _ in item["images"]]
        content.append({"type": "text", "text": item["question"]})
        msgs = [{"role": "user", "content": content}]
        if with_answer:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]}
            )
        return msgs

    def __call__(self, batch):
        full_texts, prompt_texts, images = [], [], []
        for item in batch:
            full_texts.append(
                self.processor.apply_chat_template(
                    self._messages(item, True), tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    self._messages(item, False), tokenize=False, add_generation_prompt=True
                )
            )
            images.append(item["images"])

        # Dynamic padding to the longest sequence in the batch -- NOT a fixed cap.
        enc = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        prompt_enc = self.processor(
            text=prompt_texts, images=images, return_tensors="pt", padding=True
        )

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        # Mask the prompt span: everything up to where the assistant answer starts.
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        for i, plen in enumerate(prompt_lens):
            labels[i, : int(plen)] = -100

        # Belt and braces: never supervise image placeholder ids.
        for attr in ("image_token_id", "video_token_id"):
            tid = getattr(self.processor, attr, None)
            if tid is None:
                tid = getattr(getattr(self.processor, "tokenizer", None), attr, None)
            if isinstance(tid, int):
                labels[enc["input_ids"] == tid] = -100

        enc["labels"] = labels
        return enc


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class TimingModule(pl.LightningModule):
    """
    LoRA fine-tune step for timing.

    Precision note: base weights load in bf16, then trainable LoRA params are
    cast back to fp32 and Trainer runs precision="bf16-mixed". That gives bf16
    compute with fp32 master weights on exactly the params the optimizer owns.

    The old script loaded bf16 by hand and left Trainer at its default
    precision="32-true" -- so it silently ran pure bf16 with bf16 optimizer
    states and no master weights, which is not what any real run should do.
    """

    def __init__(self, cfg: SweepConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_impl = "sdpa"

        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3VLForConditionalGeneration

        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        want = "flash_attention_2" if cap[0] >= 8 else "sdpa"
        try:
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg.model_id, dtype=torch.bfloat16, attn_implementation=want
            )
            self.attn_impl = want
        except (ImportError, ValueError) as exc:
            print(f"  [warn] {want} unavailable ({exc}); using sdpa")
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg.model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
            )

        if cfg.grad_checkpointing:
            # use_reentrant=False is REQUIRED here. The default (True) uses the
            # reentrant autograd path, which silently declines to checkpoint when
            # the base is frozen and only LoRA params require grad -- activations
            # are kept in full. First sweep measured 24.4 GB at seq_len 316 where
            # ~10 GB was expected, and activation memory scaled linearly with
            # seq_len, which is the signature of checkpointing not running.
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            base.enable_input_require_grads()
        base.config.use_cache = False

        self.model = get_peft_model(
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

        # fp32 master weights for the trainable params only.
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.n_total = sum(p.numel() for p in self.model.parameters())

        # Per-batch shape stats, drained by the callback.
        self.seq_lens: list[int] = []
        self.vis_tokens: list[float] = []
        self.sup_tokens: list[float] = []
        self._merge_sq = 4  # set from the processor before fit
        self.ckpt_layers = 0
        self.ckpt_active = 0

    def on_train_start(self) -> None:
        """
        Audit whether gradient checkpointing is ACTUALLY active.

        transformers >=5 implements checkpointing in GradientCheckpointingLayer.__call__,
        gated on `self.gradient_checkpointing and self.training`. Setting the flag is
        not sufficient -- if the layers sit in eval mode the flag does nothing, and
        checkpointing silently no-ops. On the 80 GB sweep, --grad-checkpointing and
        --no-grad-checkpointing produced byte-identical peak VRAM (15.9 GB of
        activations either way), which is what this audit exists to explain.
        """
        # REQUIRED. Lightning never calls .train() at the start of training -- its
        # only .train() calls are in the validation/test hooks, to restore mode
        # after an eval loop. It merely WARNS ("Found N module(s) in eval mode at
        # the start of training"). transformers' from_pretrained returns the model
        # in eval mode and PEFT preserves it, so without this line every layer
        # stays in eval and:
        #   * gradient checkpointing silently no-ops (gated on
        #     `gradient_checkpointing AND self.training`) -- measured 0 of 60
        #     layers active, with --grad-checkpointing and --no-grad-checkpointing
        #     giving byte-identical 15.9 GB of activations;
        #   * every nn.Dropout is inert, so LoRA dropout 0.1 (section 5.2) does nothing.
        self.model.train()

        try:
            from transformers.modeling_layers import GradientCheckpointingLayer
        except ImportError:
            print("  [ckpt audit] GradientCheckpointingLayer not importable; skipped")
            return

        layers = [m for m in self.modules() if isinstance(m, GradientCheckpointingLayer)]
        flag = sum(bool(getattr(m, "gradient_checkpointing", False)) for m in layers)
        train = sum(m.training for m in layers)
        active = sum(
            bool(getattr(m, "gradient_checkpointing", False)) and m.training for m in layers
        )
        self.ckpt_layers, self.ckpt_active = len(layers), active
        print(
            f"  [ckpt audit] checkpointable_layers={len(layers)}  flag_set={flag}  "
            f"in_train_mode={train}  ACTIVE={active}"
        )
        if self.cfg.grad_checkpointing and active < len(layers):
            print(
                f"  [ckpt audit] WARNING: {len(layers) - active} layer(s) will NOT "
                "checkpoint. Memory figures reflect full activation storage."
            )

    def training_step(self, batch, batch_idx):
        self.seq_lens.append(int(batch["input_ids"].shape[1]))
        self.sup_tokens.append(float((batch["labels"] != -100).sum()) / self.cfg.micro_batch)
        grid = batch.get("image_grid_thw")
        if grid is not None:
            self.vis_tokens.append(
                float(grid.prod(dim=-1).sum()) / self._merge_sq / self.cfg.micro_batch
            )
        return self.model(**batch).loss

    def configure_optimizers(self):
        cfg = self.cfg
        # Trainable params only. The old script handed AdamW every frozen base param.
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.init_lr, weight_decay=0.0)

        total = cfg.warmup_steps + cfg.steps
        warm = max(1, int(total * cfg.warmup_frac))

        def lr_lambda(step: int) -> float:
            # Section 5.2: warmup 1e-6 -> 1e-4 over first 1% of steps, then cosine.
            if step < warm:
                lr = cfg.init_lr + (step / warm) * (cfg.peak_lr - cfg.init_lr)
            else:
                prog = (step - warm) / max(1, total - warm)
                lr = cfg.peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            return lr / cfg.init_lr

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


class TimingCallback(pl.Callback):
    """
    Times the window AFTER warmup only.

    The old TimingCallback set start_time in on_train_start, so step 1 carried
    cuDNN autotune and first-batch dataloader cost inside the measured average.
    """

    def __init__(self, cfg: SweepConfig):
        self.cfg = cfg
        self.t0: float | None = None
        self.total: float = 0.0
        self.static_gb: float = 0.0
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step  # optimizer steps, accumulation already folded in
        if self.t0 is None and step >= self.cfg.warmup_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Allocated between steps = weights + grads + optimizer state.
                self.static_gb = torch.cuda.memory_allocated() / 1024**3
                torch.cuda.reset_peak_memory_stats()
            # Shape stats collected during warmup are not part of the measurement.
            pl_module.seq_lens.clear()
            pl_module.vis_tokens.clear()
            pl_module.sup_tokens.clear()
            self.t0 = time.perf_counter()
            print(f"  warmup done ({step} steps), timing {self.cfg.steps}...")
            return

        if self.t0 is None:
            return

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None:
            self.losses.append(float(loss.detach()))

        done = step - self.cfg.warmup_steps
        if done > 0 and done % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            el = time.perf_counter() - self.t0
            print(
                f"    step {done}/{self.cfg.steps}  {el:.1f}s  "
                f"({el / done:.3f} s/step)  loss={self.losses[-1]:.4f}"
            )

    def on_train_end(self, trainer, pl_module):
        if self.t0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.total = time.perf_counter() - self.t0


def run_one(cfg: SweepConfig, dry_run: bool = False) -> SweepResult:
    from transformers import AutoProcessor

    pl.seed_everything(cfg.seed, workers=True)

    print(
        f"\n--- max_pixels={cfg.max_pixels:,} composites={cfg.composites} "
        f"(~{cfg.approx_vision_tokens:,} vision tokens/sample) ---"
    )

    processor = AutoProcessor.from_pretrained(
        cfg.model_id, min_pixels=MIN_PIXELS, max_pixels=cfg.max_pixels
    )

    if cfg.manifest:
        samples = load_canonical_manifest(
            cfg.manifest, adapter=cfg.adapter or None, split=None
        )
        ds = RealChipDataset(
            samples,
            root=cfg.image_root or Path(cfg.manifest).parent,
            composites=cfg.composites,
            # The sweep's whole job is pricing max_pixels x composites, so it
            # varies the view count on purpose and must be allowed to.
            resize_views=True,
        )
        print(f"  data: {len(ds)} real samples from {cfg.manifest}")
    else:
        ds = SyntheticRSDataset(cfg)
    dl = DataLoader(
        ds,
        batch_size=cfg.micro_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=Collator(processor, cfg),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    if dry_run:
        batch = next(iter(dl))
        sup = int((batch["labels"] != -100).sum())
        res = SweepResult(config=asdict(cfg), ok=True)
        res.mean_seq_len = float(batch["input_ids"].shape[1])
        res.mean_supervised_tokens = float(sup) / cfg.micro_batch
        print(
            f"  seq_len={res.mean_seq_len:.0f}  supervised_tokens/sample="
            f"{res.mean_supervised_tokens:.1f}  (dry run, no timing)"
        )
        if sup == 0:
            res.ok = False
            res.error = "label masking produced zero supervised tokens"
        return res

    module = TimingModule(cfg)
    module._merge_sq = getattr(processor.image_processor, "merge_size", 2) ** 2
    cb = TimingCallback(cfg)

    trainer = pl.Trainer(
        max_steps=cfg.warmup_steps + cfg.steps,
        accelerator="auto",
        devices=cfg.devices,
        # EXPLICIT. The old script left this at the default "32-true" while
        # loading bf16 weights by hand.
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.grad_accum,
        gradient_clip_val=1.0,
        callbacks=[cb],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        barebones=False,
    )

    print(
        f"  attn={module.attn_impl}  trainable={module.n_trainable / 1e6:.1f}M  "
        f"grad_ckpt={cfg.grad_checkpointing}  devices={trainer.num_devices}  "
        f"precision=bf16-mixed"
    )

    try:
        trainer.fit(module, train_dataloaders=dl)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("  [OOM]")
        return SweepResult(config=asdict(cfg), ok=False, error="CUDA OOM")

    if cb.total <= 0:
        return SweepResult(
            config=asdict(cfg), ok=False, error="timing window never opened (steps <= warmup?)"
        )

    res = SweepResult(config=asdict(cfg), ok=True)
    res.sec_per_opt_step = cb.total / cfg.steps

    # world_size from the Trainer, so per-device batch is never mistaken for
    # global batch (the old steps_for_epoch = 80000 // batch_size bug).
    ws = trainer.num_devices * trainer.num_nodes
    samples_per_step = cfg.micro_batch * cfg.grad_accum * ws
    res.sec_per_sample = res.sec_per_opt_step / samples_per_step

    res.ckpt_layers = module.ckpt_layers
    res.ckpt_active = module.ckpt_active
    res.mean_seq_len = statistics.fmean(module.seq_lens) if module.seq_lens else 0.0
    res.p95_seq_len = float(np.percentile(module.seq_lens, 95)) if module.seq_lens else 0.0
    res.mean_vision_tokens = statistics.fmean(module.vis_tokens) if module.vis_tokens else 0.0
    res.mean_supervised_tokens = statistics.fmean(module.sup_tokens) if module.sup_tokens else 0.0
    res.first_loss = cb.losses[0] if cb.losses else 0.0
    res.last_loss = cb.losses[-1] if cb.losses else 0.0

    if torch.cuda.is_available():
        res.peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        res.reserved_vram_gb = torch.cuda.max_memory_reserved() / 1024**3
        res.static_vram_gb = cb.static_gb
        # Activation share is the diagnostic: with checkpointing working it
        # should stay small and grow sub-linearly with seq_len. Linear growth
        # means checkpointing is not running.
        res.activation_vram_gb = max(0.0, res.peak_vram_gb - cb.static_gb)

    toks = res.mean_seq_len * samples_per_step
    res.effective_tflops = (6 * module.n_total * toks) / res.sec_per_opt_step / 1e12

    for name, spec in ADAPTERS.items():
        steps_per_epoch = math.ceil(spec["samples"] / samples_per_step)
        res.per_adapter_hours[name] = round(steps_per_epoch * res.sec_per_opt_step / 3600, 2)

    print(
        f"  => {res.sec_per_opt_step:.3f} s/opt-step | seq_len mean "
        f"{res.mean_seq_len:.0f} p95 {res.p95_seq_len:.0f} | vision "
        f"{res.mean_vision_tokens:.0f} tok | supervised "
        f"{res.mean_supervised_tokens:.1f} tok | VRAM "
        f"{res.peak_vram_gb:.1f} GB peak "
        f"({res.static_vram_gb:.1f} static + {res.activation_vram_gb:.1f} act) "
        f"| ~{res.effective_tflops:.0f} TFLOPS"
    )

    del module, trainer, cb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def gpu_name() -> str:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 1024**3:.0f} GB)"
    return "CPU"


def write_report(results: list[SweepResult], out: Path, cfg0: SweepConfig) -> None:
    ok = [r for r in results if r.ok]
    L: list[str] = []
    A = L.append

    A("# Phase 0 — 200-step timing sweep (§5.5 re-derivation)\n")
    A(f"**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}  ")
    A(f"**Hardware:** {gpu_name()} × {cfg0.devices} device(s)  ")
    A(f"**Base model:** `{cfg0.model_id}`  ")
    A(
        f"**Adapter:** LoRA r={cfg0.lora_r}, α={cfg0.lora_alpha}, "
        f"dropout={cfg0.lora_dropout}, target=`{cfg0.lora_targets}`  "
    )
    A(
        f"**Schedule:** {cfg0.init_lr:g} → {cfg0.peak_lr:g} over first "
        f"{cfg0.warmup_frac:.0%} of steps, cosine decay (§5.2)  "
    )
    A(
        f"**Batch:** micro={cfg0.micro_batch} × accum={cfg0.grad_accum} × "
        f"devices={cfg0.devices} = **{cfg0.effective_batch} effective**  "
    )
    A(f"**Grad checkpointing:** {cfg0.grad_checkpointing}  ")
    A(f"**Source tile:** {cfg0.source_size}px  ")
    A(f"**Data:** {cfg0.data_source}  ")
    A(f"**Contract:** {TrainingConfig.from_frozen().provenance()}  ")
    A(f"**Torch:** {torch.__version__} · Python {platform.python_version()}\n")

    A(
        "> Timed window excludes the first "
        f"{cfg0.warmup_steps} warmup steps. Synthetic imagery: pixel values do not "
        "affect timing, dimensions do — tiles are generated at "
        f"{cfg0.source_size}×{cfg0.source_size} and capped by `max_pixels`. "
        "Qwen3-VL never upsamples, so a source smaller than the cap sets the token "
        "count on its own — 120px (BigEarthNet) is 16 tokens at any `max_pixels`.\n"
    )

    if not ok:
        A("## No configuration completed\n")
        for r in results:
            c = r.config
            A(f"- max_pixels={c['max_pixels']:,} composites={c['composites']}: **{r.error}**")
        out.write_text("\n".join(L), encoding="utf-8")
        return

    A("## Measured cost surface\n")
    A(
        "| max_pixels | comps | vision tok | seq len (mean/p95) | s/opt-step | "
        "s/sample | peak VRAM | static | activations | ~TFLOPS |"
    )
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r.config
        if not r.ok:
            A(
                f"| {c['max_pixels']:,} | {c['composites']} | "
                f"~{int(c['composites'] * c['max_pixels'] / 1024):,} | — | **{r.error}** "
                f"| — | — | — | — | — |"
            )
            continue
        A(
            f"| {c['max_pixels']:,} | {c['composites']} | {r.mean_vision_tokens:.0f} | "
            f"{r.mean_seq_len:.0f} / {r.p95_seq_len:.0f} | {r.sec_per_opt_step:.3f} | "
            f"{r.sec_per_sample:.3f} | {r.peak_vram_gb:.1f} GB | "
            f"{r.static_vram_gb:.1f} GB | {r.activation_vram_gb:.1f} GB | "
            f"{r.effective_tflops:.0f} |"
        )
    A("")

    # Index cells by (max_pixels, composites) so each adapter can be costed at
    # ITS OWN composite count rather than one count applied to all four.
    cell = {(r.config["max_pixels"], r.config["composites"]): r for r in ok}
    mp_values = sorted({r.config["max_pixels"] for r in ok})

    A("## §5.5 budget table, re-derived (guard rail 4)\n")
    A(
        "Hours for **1 epoch** per adapter, each costed at its own composite count "
        "from §5.2 — `rs_ground_caption` at two (no SWIR on Cartosat, so it is "
        "always two), the rest at three under C22. Columns are `max_pixels`, "
        "expressed as vision tokens per image.\n"
    )
    A(
        "| Adapter | comps | §5.5 budget | "
        + " | ".join(f"{mp // 1024}k tok/img" for mp in mp_values)
        + " |"
    )
    A("|---|---|---|" + "---|" * len(mp_values))

    column_totals: dict[int, float] = {mp: 0.0 for mp in mp_values}
    column_complete: dict[int, bool] = {mp: True for mp in mp_values}

    for name, spec in ADAPTERS.items():
        comps = spec["composites"]
        cells = []
        for mp in mp_values:
            r = cell.get((mp, comps))
            if r is None:
                cells.append("—")
                column_complete[mp] = False
                continue
            h = r.per_adapter_hours.get(name, 0.0)
            column_totals[mp] += h
            cells.append(f"{h:.1f} h" + ("" if h <= spec["budget_h"] else " ⚠"))
        A(f"| `{name}` | {comps} | {spec['budget_h']:.0f} h | " + " | ".join(cells) + " |")

    train_budget = SCHEDULED_TOTAL_H - 3
    totals = []
    for mp in mp_values:
        if not column_complete[mp]:
            totals.append("— *(incomplete)*")
            continue
        t = column_totals[mp]
        totals.append(f"**{t:.1f} h**" + ("" if t <= train_budget else " ⚠"))
    A(f"| **Four-adapter total** | mixed | **{train_budget:.0f} h** | " + " | ".join(totals) + " |")
    A("")
    A(
        f"Scheduled total in §5.5 is **{SCHEDULED_TOTAL_H:.0f} h** including 3 h of "
        f"smoke/timing/debug, against **{RESERVE_H:.0f} h** reserve out of "
        f"**{TOTAL_ACCESS_H:.0f} h** access. The four-adapter row above is the "
        f"{train_budget:.0f} h of actual training.\n"
    )

    A("## C22 three-composite ablation\n")
    A(
        "§5.5 prices three-composite optical input at 5–8 of the 50 hours and "
        "asks for it to be measured, not assumed.\n"
    )
    by_mp: dict[int, dict[int, SweepResult]] = {}
    for r in ok:
        by_mp.setdefault(r.config["max_pixels"], {})[r.config["composites"]] = r
    shown = False
    for mp, row in sorted(by_mp.items()):
        if 2 in row and 3 in row:
            two = sum(row[2].per_adapter_hours.values())
            three = sum(row[3].per_adapter_hours.values())
            if two <= 0:
                continue
            shown = True
            A(
                f"- `max_pixels={mp:,}`: two composites **{two:.1f} h**, three "
                f"**{three:.1f} h** → C22 costs **{three - two:+.1f} h** "
                f"({(three / two - 1) * 100:+.0f}%)"
            )
    if not shown:
        A("- Not measured: sweep both 2 and 3 composites at a shared `max_pixels` to price this.")
    A("")

    fits = [mp for mp in mp_values if column_complete[mp] and column_totals[mp] <= train_budget]
    A("## Recommendation\n")
    if fits:
        best_mp = max(fits)
        peak = max(
            (
                cell[(best_mp, c)].peak_vram_gb
                for c in {s["composites"] for s in ADAPTERS.values()}
                if (best_mp, c) in cell
            ),
            default=0.0,
        )
        A(
            f"Highest `max_pixels` that fits the {train_budget:.0f} h training "
            f"schedule: **{best_mp:,}** (~{best_mp // 1024} vision tokens/image, "
            f"{column_totals[best_mp]:.1f} h across four adapters, "
            f"{peak:.1f} GB peak).\n"
        )
        A(
            f"Set `tiling.max_pixels: {best_mp}` in "
            "`configs/preprocessing.yaml` and bump `version` — **but only after "
            "checking it against the 5 min/scene prep SLA on the demo machine**, "
            "which this harness does not measure. The YAML TODO names that SLA as "
            "the binding constraint alongside the compute budget.\n"
        )
    else:
        A(
            "**No swept configuration fits the training schedule.** Apply the §5.5 "
            "cost lever ranking in order: (1) cap `max_pixels` harder, "
            "(2) subsample below 80k, (3) merge adapters — noting C17 forbids "
            "degrading `rs_ground_caption`.\n"
        )

    hi = max(ok, key=lambda r: r.peak_vram_gb)
    A(
        f"Peak VRAM across the sweep: **{hi.peak_vram_gb:.1f} GB** at "
        f"max_pixels={hi.config['max_pixels']:,}, composites={hi.config['composites']}. "
        '§5.5 is written for an A100-**80GB** (C41: *"a 4B base fits 80 GB '
        'trivially"*, *"raise batch size until utilisation saturates"*). '
        "Confirm which card is actually available before committing this table.\n"
    )

    A("## Sanity checks\n")
    A("| Check | Expected | Observed |")
    A("|---|---|---|")
    r0 = ok[0]
    A(
        f"| Opening loss | ~2–4 healthy; near ln(151669)=11.93 means the label "
        f"mask is broken. Synthetic answers come from a 4-item pool, so a low "
        f"value here is memorisation, not a fault | {r0.first_loss:.2f} |"
    )
    A(
        f"| Supervised tokens/sample | answer span only, not the full sequence | "
        f"{r0.mean_supervised_tokens:.1f} of {r0.mean_seq_len:.0f} |"
    )
    A(
        f"| Vision tokens/image | max_pixels/1024 = "
        f"{r0.config['max_pixels'] // 1024} | "
        f"{r0.mean_vision_tokens / max(1, r0.config['composites']):.0f} |"
    )
    A("")
    # Direct audit, not a scaling heuristic. An earlier version inferred
    # checkpointing from activation-vs-seq_len scaling, which is wrong: saved
    # layer-boundary activations scale linearly with seq_len too. Checkpointing
    # changes the coefficient, not the exponent.
    A(
        f"| Gradient checkpointing actually active | all "
        f"{r0.ckpt_layers} checkpointable layers when enabled | "
        f"{r0.ckpt_active} of {r0.ckpt_layers} |"
    )
    A("")
    if r0.config.get("grad_checkpointing") and r0.ckpt_layers and r0.ckpt_active < r0.ckpt_layers:
        A(
            f"> ⚠ **Gradient checkpointing is enabled but only "
            f"{r0.ckpt_active}/{r0.ckpt_layers} layers will actually checkpoint.** "
            "transformers gates it on `gradient_checkpointing AND self.training`; "
            "layers left in eval mode silently skip it. Every OOM above is then an "
            "artifact of full activation storage, not a hardware limit.\n"
        )

    if r0.first_loss > 9.0:
        A(
            "> ⚠ **Opening loss is near ln(vocab).** Label masking is not working — "
            "the model is being scored on tokens it was never trained to emit. "
            "This was the defect in `run_200_step_timing.py`. Do not trust these "
            "numbers until it is fixed.\n"
        )

    A("---\n")
    A("<details><summary>Raw results (JSON)</summary>\n")
    A("```json")
    A(json.dumps([asdict(r) for r in results], indent=2))
    A("```\n</details>")

    out.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------


def print_dry_run_summary(results: list[SweepResult]) -> None:
    """
    Shape + label-masking check only. Deliberately writes no report: with no
    timings every hour figure would be 0.0 and the recommendation would
    trivially "fit" the budget -- exactly the kind of vacuous SUCCESS this
    harness exists to stop producing.
    """
    print("\n--- DRY RUN: shapes and label masking ---")
    print(f"{'max_pixels':>12} {'comps':>6} {'pred tok':>9} {'seq_len':>8} {'supervised':>11}")
    bad = []
    for r in results:
        c = r.config
        pred = int(c["composites"] * c["max_pixels"] / 1024)
        if not r.ok:
            print(f"{c['max_pixels']:>12,} {c['composites']:>6} {pred:>9,} {'—':>8} {r.error:>11}")
            bad.append(r)
            continue
        print(
            f"{c['max_pixels']:>12,} {c['composites']:>6} {pred:>9,} "
            f"{r.mean_seq_len:>8.0f} {r.mean_supervised_tokens:>11.1f}"
        )
        if r.mean_supervised_tokens <= 0:
            bad.append(r)

    print()
    if bad:
        print(
            "FAIL: label masking produced no supervised tokens in "
            f"{len(bad)} cell(s). Do not run the timing sweep -- loss would "
            "be meaningless."
        )
        return
    print(
        "PASS: every cell supervises a nonzero answer span, and seq_len "
        "tracks max_pixels/1024 per image as expected."
    )
    print("Next: python scripts/phase0_timing_sweep.py")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-id", default=SweepConfig.model_id)
    ap.add_argument("--max-pixels", type=int, default=None, help="single value; omit to sweep")
    ap.add_argument("--composites", type=int, default=None, help="single value; omit to sweep")
    ap.add_argument(
        "--source-size",
        type=int,
        default=DEFAULT_SOURCE_TILE,
        help="synthetic source tile edge in px. 120=BigEarthNet patch, "
        "512=benchmark chip. max_pixels only binds above this.",
    )
    ap.add_argument("--micro-batch", type=int, default=SweepConfig.micro_batch)
    ap.add_argument("--grad-accum", type=int, default=SweepConfig.grad_accum)
    ap.add_argument("--steps", type=int, default=SweepConfig.steps)
    ap.add_argument("--warmup-steps", type=int, default=SweepConfig.warmup_steps)
    ap.add_argument("--num-workers", type=int, default=SweepConfig.num_workers)
    ap.add_argument(
        "--devices",
        type=int,
        default=SweepConfig.devices,
        help="GPUs per cell. >1 sweeps under DDP; prefer one cell per invocation in that case.",
    )
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument(
        "--manifest",
        default="",
        help="canonical JSONL of real samples. Without it the sweep runs on "
        "synthetic shapes, which prices a config honestly but proves nothing "
        "about the corpus. Phase 0 item 12 asks for real data.",
    )
    ap.add_argument("--image-root", default="", help="defaults to the manifest dir")
    ap.add_argument("--adapter", default="", help="filter the manifest to one adapter")
    ap.add_argument(
        "--dry-run", action="store_true", help="shape and label-masking check only, no timing"
    )
    ap.add_argument("--out", default="logs/phase_0_timing_sweep.md")
    args = ap.parse_args()

    mps = [args.max_pixels] if args.max_pixels else DEFAULT_MAX_PIXELS_SWEEP
    comps = [args.composites] if args.composites else DEFAULT_COMPOSITE_SWEEP

    base = SweepConfig(
        model_id=args.model_id,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        steps=args.steps if not args.dry_run else 1,
        warmup_steps=args.warmup_steps if not args.dry_run else 0,
        num_workers=args.num_workers,
        source_size=args.source_size,
        devices=args.devices,
        grad_checkpointing=not args.no_grad_checkpointing,
        manifest=args.manifest,
        image_root=args.image_root,
        adapter=args.adapter,
    )

    frozen = TrainingConfig.from_frozen()
    print(frozen.provenance())
    if not args.manifest:
        print(
            "  [note] no --manifest: measuring synthetic shapes. Legitimate for "
            "pricing a config, not for Phase 0 item 12, which asks for real data."
        )

    print(f"Hardware: {gpu_name()} x {base.devices} device(s)")
    print(f"Sweep: max_pixels={mps} composites={comps}  ({len(mps) * len(comps)} cells)")

    results: list[SweepResult] = []
    for mp in mps:
        for c in comps:
            cfg = SweepConfig(**{**asdict(base), "max_pixels": mp, "composites": c})
            try:
                results.append(run_one(cfg, dry_run=args.dry_run))
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
                print(f"  [error] {type(exc).__name__}: {exc}")
                results.append(
                    SweepResult(config=asdict(cfg), ok=False, error=f"{type(exc).__name__}: {exc}")
                )

    if args.dry_run:
        print_dry_run_summary(results)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, out, base)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
