# Phase 0 — 200-step timing sweep (§5.5 re-derivation)

**Date:** 2026-08-27 08:44 UTC  
**Hardware:** NVIDIA A100-SXM4-80GB (79 GB) × 1 device(s)  
**Base model:** `Qwen/Qwen3-VL-4B-Instruct`  
**Adapter:** LoRA r=16, α=32, dropout=0.1, target=`all-linear`  
**Schedule:** 1e-06 → 0.0001 over first 1% of steps, cosine decay (§5.2)  
**Batch:** micro=4 × accum=1 × devices=1 = **4 effective**  
**Grad checkpointing:** False  
**Torch:** 2.8.0+cu128 · Python 3.12.10

> Timed window excludes the first 5 warmup steps. Synthetic imagery: pixel values do not affect timing, dimensions do — tiles are generated at 2048×2048 and capped by `max_pixels`, which is therefore the true independent variable.

## Measured cost surface

| max_pixels | comps | vision tok | seq len (mean/p95) | s/opt-step | s/sample | peak VRAM | static | activations | ~TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| 262,144 | 1 | 256 | 317 / 317 | 0.538 | 0.134 | 24.8 GB | 8.9 GB | 15.9 GB | 63 |

## §5.5 budget table, re-derived (guard rail 4)

Hours for **1 epoch** per adapter, each costed at its own composite count from §5.2 — `rs_ground_caption` at two (no SWIR on Cartosat, so it is always two), the rest at three under C22. Columns are `max_pixels`, expressed as vision tokens per image.

| Adapter | comps | §5.5 budget | 256k tok/img |
|---|---|---|---|
| `rs_vqa` | 3 | 4 h | — |
| `change_vqa` | 3 | 6 h | — |
| `optsar_fusion` | 3 | 6 h | — |
| `rs_ground_caption` | 2 | 12 h | — |
| **Four-adapter total** | mixed | **28 h** | — *(incomplete)* |

Scheduled total in §5.5 is **31 h** including 3 h of smoke/timing/debug, against **19 h** reserve out of **50 h** access. The four-adapter row above is the 28 h of actual training.

## C22 three-composite ablation

§5.5 prices three-composite optical input at 5–8 of the 50 hours and asks for it to be measured, not assumed.

- Not measured: sweep both 2 and 3 composites at a shared `max_pixels` to price this.

## Recommendation

**No swept configuration fits the training schedule.** Apply the §5.5 cost lever ranking in order: (1) cap `max_pixels` harder, (2) subsample below 80k, (3) merge adapters — noting C17 forbids degrading `rs_ground_caption`.

Peak VRAM across the sweep: **24.8 GB** at max_pixels=262,144, composites=1. §5.5 is written for an A100-**80GB** (C41: *"a 4B base fits 80 GB trivially"*, *"raise batch size until utilisation saturates"*). Confirm which card is actually available before committing this table.

## Sanity checks

| Check | Expected | Observed |
|---|---|---|
| Opening loss | ~2–4 healthy; near ln(151669)=11.93 means the label mask is broken. Synthetic answers come from a 4-item pool, so a low value here is memorisation, not a fault | 2.47 |
| Supervised tokens/sample | answer span only, not the full sequence | 24.0 of 317 |
| Vision tokens/image | max_pixels/1024 = 256 | 256 |

---

<details><summary>Raw results (JSON)</summary>

```json
[
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 262144,
      "composites": 1,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 20,
      "warmup_steps": 5,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": false,
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.1,
      "lora_targets": "all-linear",
      "peak_lr": 0.0001,
      "init_lr": 1e-06,
      "warmup_frac": 0.01,
      "max_text_tokens": 256,
      "seed": 0
    },
    "ok": true,
    "error": "",
    "sec_per_opt_step": 0.5377503916999998,
    "sec_per_sample": 0.13443759792499996,
    "mean_seq_len": 316.8,
    "p95_seq_len": 317.0,
    "mean_vision_tokens": 256.0,
    "mean_supervised_tokens": 24.0,
    "peak_vram_gb": 24.840688228607178,
    "reserved_vram_gb": 25.859375,
    "static_vram_gb": 8.905538558959961,
    "activation_vram_gb": 15.935149669647217,
    "first_loss": 2.47464919090271,
    "last_loss": 0.30103442072868347,
    "effective_tflops": 63.31524211621697,
    "per_adapter_hours": {
      "rs_vqa": 2.99,
      "change_vqa": 2.99,
      "optsar_fusion": 2.99,
      "rs_ground_caption": 2.99
    }
  }
]
```
</details>