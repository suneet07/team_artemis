# Phase 0 timing test — review verdict (2026-08-26)

Review of `logs/phase_0_timing_report.log` (200-step LoRA timing run, A100-SXM4-40GB)
against Master Plan v3.8 §5.5.

## Verdict: does NOT clear the Phase 0 gate

The run executed cleanly and 0.48 s/step is an honest measurement of the config as
written. But the `SUCCESS` verdict does not hold, for three independent reasons.

**1. Apples-to-oranges comparison.** The log compares a single-adapter epoch estimate
(2.69 h) against §5.5's **31 h**, which is the scheduled total for all four adapters
plus 3 h of smoke/debug. The right comparison for a BEN VQA proxy is the `rs_vqa`
line: **4 h**. Real headroom is ~1.5×, not the implied 11.5×.

**2. Wrong GPU.** §5.5 is titled "A100-**80GB** budget (C41)" and every config decision
in it (drop QLoRA, raise batch size until utilisation saturates) is sized for 80 GB.
The run used a **40 GB** card — half the VRAM, ~20% less memory bandwidth. Neither
bound bites at 128-token sequences; both bite at real `max_pixels`.

**3. Config measured ≠ config that will run.** Sequences hard-capped at 128 tokens with
~16 vision tokens per sample (120×120 dummy patches), no gradient checkpointing, no
gradient accumulation, `precision` unset. Guard rail 4 ("re-derive this entire table
from the Phase 0 measurement") is therefore not satisfied. **Checklist line 1583 should
stay unchecked.**

## Confirmed defect: label masking

`labels=batch["input_ids"]` with `padding="max_length"` scored the model on pad tokens
and image placeholders. Evidence: opening loss **11.32** against `ln(151669) = 11.93`
for Qwen3-VL's vocab — a pretrained model at 95% of uniform-random on its own chat
template. Loss then collapsed to ~4.4 by step 100 and flatlined into noise. It could
not have been learning anything visual: all 21 micro-split images are constant black
(`getextrema()` = `((0,0),(0,0),(0,0))`). **The loss curve is not evidence that
training works.**

## Other findings

- LoRA dropout 0.05 in code vs **0.1** in §5.2; flat LR 2e-5 vs §5.2's **warmup
  1e-6 → 1e-4 over first 1% then cosine** (2e-5 is a full-finetune LR, ~5–10× low for LoRA).
- `steps_for_epoch = 80000 // batch_size` treats per-device batch as global — wrong by
  N× under DDP.
- `AdamW(self.model.parameters())` includes frozen base params.
- No peak VRAM reported — the one number needed to pick a batch size on a 40 GB card.
- Licence: `ben-micro-split/dataset-metadata.json` declares **CC0-1.0** for Kaggle upload,
  but BigEarthNet.txt's annotation layer is **CDLA-Permissive 1.0** per ../../CREDITS.md.
  Derived annotations cannot be relicensed to CC0. Caught by their own C45
  Provenance Traceability Rule.
- Repo is **not under version control** (no `.git`).

## Replacement harness

`scripts/phase0_timing_sweep.py` (replaces `scripts/run_200_step_timing.py`).

Built as a **sweep**, not a single run, because `configs/preprocessing.yaml` has
`tiling.max_pixels: null` with an explicit "must be measured, not guessed" TODO, and
§5.5 ranks `max_pixels` as cost lever #1. A single point cannot re-derive a table whose
x-axis is undecided.

Sweeps `max_pixels × n_composites`, emits the §5.5 table per adapter (each at its own
composite count — `rs_ground_caption` at 2, no SWIR on Cartosat; rest at 3 under C22),
prices the C22 ablation, and reports peak VRAM per cell.

Token arithmetic (verified against the real processor: `patch_size=16, merge_size=2`):
`vision_tokens_per_image = max_pixels / 1024`.

Validated: ruff clean; label masking verified against the real Qwen3-VL processor
(pads and image placeholders excluded, supervised span decodes to the answer only);
report renders for mixed success/OOM/all-failed cells; broken-mask warning fires at
opening loss > 9.

## Open decisions

1. **Which card** — confirm 80 GB access or re-derive §5.5 for 40 GB.
2. **`max_pixels`** — must clear *both* the compute budget and the 5 min/scene prep SLA
   on the demo machine. The harness measures the first, not the second.
3. **C22 two vs three composites** — §5.5 prices it at 5–8 of the 50 h; now measurable.
4. **Base model bake-off** — Qwen3-VL-4B vs Qwen3.5-2B still open per §5.1.
