# Phase 0 — 200-step timing sweep (§5.5 re-derivation)

**Date:** 2026-08-27 08:40 UTC  
**Hardware:** NVIDIA A100-SXM4-80GB (79 GB) × 1 device(s)  
**Base model:** `Qwen/Qwen3-VL-4B-Instruct`  
**Adapter:** LoRA r=16, α=32, dropout=0.1, target=`all-linear`  
**Schedule:** 1e-06 → 0.0001 over first 1% of steps, cosine decay (§5.2)  
**Batch:** micro=4 × accum=1 × devices=1 = **4 effective**  
**Grad checkpointing:** True  
**Torch:** 2.8.0+cu128 · Python 3.12.10

> Timed window excludes the first 10 warmup steps. Synthetic imagery: pixel values do not affect timing, dimensions do — tiles are generated at 2048×2048 and capped by `max_pixels`, which is therefore the true independent variable.

## Measured cost surface

| max_pixels | comps | vision tok | seq len (mean/p95) | s/opt-step | s/sample | peak VRAM | static | activations | ~TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| 262,144 | 1 | 256 | 317 / 317 | 0.529 | 0.132 | 24.8 GB | 8.9 GB | 15.9 GB | 64 |
| 262,144 | 2 | 512 | 575 / 575 | 0.770 | 0.193 | 38.8 GB | 8.9 GB | 29.9 GB | 80 |
| 262,144 | 3 | 768 | 833 / 833 | 1.135 | 0.284 | 52.9 GB | 9.0 GB | 43.9 GB | 79 |
| 524,288 | 1 | 484 | 545 / 545 | 0.778 | 0.194 | 37.2 GB | 8.9 GB | 28.3 GB | 75 |
| 524,288 | 2 | 968 | 1031 / 1031 | 1.438 | 0.360 | 63.7 GB | 9.0 GB | 54.7 GB | 77 |
| 524,288 | 3 | ~1,536 | — | **CUDA OOM** | — | — | — | — | — |
| 1,048,576 | 1 | 1024 | 1085 / 1085 | 1.536 | 0.384 | 66.7 GB | 9.0 GB | 57.7 GB | 76 |
| 1,048,576 | 2 | ~2,048 | — | **CUDA OOM** | — | — | — | — | — |
| 1,048,576 | 3 | ~3,072 | — | **CUDA OOM** | — | — | — | — | — |
| 2,097,152 | 1 | ~2,048 | — | **CUDA OOM** | — | — | — | — | — |
| 2,097,152 | 2 | ~4,096 | — | **CUDA OOM** | — | — | — | — | — |
| 2,097,152 | 3 | ~6,144 | — | **CUDA OOM** | — | — | — | — | — |

## §5.5 budget table, re-derived (guard rail 4)

Hours for **1 epoch** per adapter, each costed at its own composite count from §5.2 — `rs_ground_caption` at two (no SWIR on Cartosat, so it is always two), the rest at three under C22. Columns are `max_pixels`, expressed as vision tokens per image.

| Adapter | comps | §5.5 budget | 256k tok/img | 512k tok/img | 1024k tok/img |
|---|---|---|---|---|---|
| `rs_vqa` | 3 | 4 h | 6.3 h ⚠ | — | — |
| `change_vqa` | 3 | 6 h | 6.3 h ⚠ | — | — |
| `optsar_fusion` | 3 | 6 h | 6.3 h ⚠ | — | — |
| `rs_ground_caption` | 2 | 12 h | 4.3 h | 8.0 h | — |
| **Four-adapter total** | mixed | **28 h** | **23.2 h** | — *(incomplete)* | — *(incomplete)* |

Scheduled total in §5.5 is **31 h** including 3 h of smoke/timing/debug, against **19 h** reserve out of **50 h** access. The four-adapter row above is the 28 h of actual training.

## C22 three-composite ablation

§5.5 prices three-composite optical input at 5–8 of the 50 hours and asks for it to be measured, not assumed.

- `max_pixels=262,144`: two composites **17.1 h**, three **25.2 h** → C22 costs **+8.1 h** (+47%)

## Recommendation

Highest `max_pixels` that fits the 28 h training schedule: **262,144** (~256 vision tokens/image, 23.2 h across four adapters, 52.9 GB peak).

Set `tiling.max_pixels: 262144` in `configs/preprocessing.yaml` and bump `version` — **but only after checking it against the 5 min/scene prep SLA on the demo machine**, which this harness does not measure. The YAML TODO names that SLA as the binding constraint alongside the compute budget.

Peak VRAM across the sweep: **66.7 GB** at max_pixels=1,048,576, composites=1. §5.5 is written for an A100-**80GB** (C41: *"a 4B base fits 80 GB trivially"*, *"raise batch size until utilisation saturates"*). Confirm which card is actually available before committing this table.

## Sanity checks

| Check | Expected | Observed |
|---|---|---|
| Opening loss | ~2–4 healthy; near ln(151669)=11.93 means the label mask is broken. Synthetic answers come from a 4-item pool, so a low value here is memorisation, not a fault | 1.29 |
| Supervised tokens/sample | answer span only, not the full sequence | 24.0 of 317 |
| Vision tokens/image | max_pixels/1024 = 256 | 256 |

| Activation memory scaling | sub-linear in seq_len if gradient checkpointing is active | seq ×3.43 → activations ×3.62 |

> ⚠ **Activation memory is scaling linearly with sequence length.** Gradient checkpointing is probably not in effect — check that `use_reentrant=False` reached the model. Every OOM in the table above is then an artifact, not a hardware limit.

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
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "sec_per_opt_step": 0.52942553774,
    "sec_per_sample": 0.132356384435,
    "mean_seq_len": 316.64,
    "p95_seq_len": 317.0,
    "mean_vision_tokens": 256.0,
    "mean_supervised_tokens": 23.97,
    "peak_vram_gb": 24.846734523773193,
    "reserved_vram_gb": 25.880859375,
    "static_vram_gb": 8.905538558959961,
    "activation_vram_gb": 15.941195964813232,
    "first_loss": 1.2880665063858032,
    "last_loss": 0.00032103192643262446,
    "effective_tflops": 64.27835071416062,
    "per_adapter_hours": {
      "rs_vqa": 2.94,
      "change_vqa": 2.94,
      "optsar_fusion": 2.94,
      "rs_ground_caption": 2.94
    }
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 262144,
      "composites": 2,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "sec_per_opt_step": 0.7702232490399998,
    "sec_per_sample": 0.19255581225999996,
    "mean_seq_len": 574.64,
    "p95_seq_len": 575.0,
    "mean_vision_tokens": 512.0,
    "mean_supervised_tokens": 23.97,
    "peak_vram_gb": 38.83460092544556,
    "reserved_vram_gb": 40.52734375,
    "static_vram_gb": 8.929006576538086,
    "activation_vram_gb": 29.90559434890747,
    "first_loss": 1.3112291097640991,
    "last_loss": 0.005073699168860912,
    "effective_tflops": 80.18314090547204,
    "per_adapter_hours": {
      "rs_vqa": 4.28,
      "change_vqa": 4.28,
      "optsar_fusion": 4.28,
      "rs_ground_caption": 4.28
    }
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 262144,
      "composites": 3,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "sec_per_opt_step": 1.1345512674399998,
    "sec_per_sample": 0.28363781685999995,
    "mean_seq_len": 832.64,
    "p95_seq_len": 833.0,
    "mean_vision_tokens": 768.0,
    "mean_supervised_tokens": 23.97,
    "peak_vram_gb": 52.86042404174805,
    "reserved_vram_gb": 55.349609375,
    "static_vram_gb": 8.952476501464844,
    "activation_vram_gb": 43.9079475402832,
    "first_loss": 1.2985241413116455,
    "last_loss": 0.014697214588522911,
    "effective_tflops": 78.87456546844612,
    "per_adapter_hours": {
      "rs_vqa": 6.3,
      "change_vqa": 6.3,
      "optsar_fusion": 6.3,
      "rs_ground_caption": 6.3
    }
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 524288,
      "composites": 1,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "sec_per_opt_step": 0.7778520710199995,
    "sec_per_sample": 0.19446301775499988,
    "mean_seq_len": 544.64,
    "p95_seq_len": 545.0,
    "mean_vision_tokens": 484.0,
    "mean_supervised_tokens": 23.97,
    "peak_vram_gb": 37.24086856842041,
    "reserved_vram_gb": 38.849609375,
    "static_vram_gb": 8.926441669464111,
    "activation_vram_gb": 28.3144268989563,
    "first_loss": 1.235221266746521,
    "last_loss": 0.0005591787630692124,
    "effective_tflops": 75.25170705027254,
    "per_adapter_hours": {
      "rs_vqa": 4.32,
      "change_vqa": 4.32,
      "optsar_fusion": 4.32,
      "rs_ground_caption": 4.32
    }
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 524288,
      "composites": 2,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "sec_per_opt_step": 1.4381085975799999,
    "sec_per_sample": 0.35952714939499997,
    "mean_seq_len": 1030.64,
    "p95_seq_len": 1031.0,
    "mean_vision_tokens": 968.0,
    "mean_supervised_tokens": 23.97,
    "peak_vram_gb": 63.711806297302246,
    "reserved_vram_gb": 66.6875,
    "static_vram_gb": 8.970809936523438,
    "activation_vram_gb": 54.74099636077881,
    "first_loss": 1.2690517902374268,
    "last_loss": 0.0006707272841595113,
    "effective_tflops": 77.02277217642668,
    "per_adapter_hours": {
      "rs_vqa": 7.99,
      "change_vqa": 7.99,
      "optsar_fusion": 7.99,
      "rs_ground_caption": 7.99
    }
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 524288,
      "composites": 3,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "ok": false,
    "error": "CUDA OOM",
    "sec_per_opt_step": 0.0,
    "sec_per_sample": 0.0,
    "mean_seq_len": 0.0,
    "p95_seq_len": 0.0,
    "mean_vision_tokens": 0.0,
    "mean_supervised_tokens": 0.0,
    "peak_vram_gb": 0.0,
    "reserved_vram_gb": 0.0,
    "static_vram_gb": 0.0,
    "activation_vram_gb": 0.0,
    "first_loss": 0.0,
    "last_loss": 0.0,
    "effective_tflops": 0.0,
    "per_adapter_hours": {}
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 1048576,
      "composites": 1,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "sec_per_opt_step": 1.5356692918,
    "sec_per_sample": 0.38391732295,
    "mean_seq_len": 1084.64,
    "p95_seq_len": 1085.0,
    "mean_vision_tokens": 1024.0,
    "mean_supervised_tokens": 23.97,
    "peak_vram_gb": 66.70282793045044,
    "reserved_vram_gb": 69.859375,
    "static_vram_gb": 8.975942611694336,
    "activation_vram_gb": 57.7268853187561,
    "first_loss": 1.19566810131073,
    "last_loss": 0.0005616574198938906,
    "effective_tflops": 75.90873447303821,
    "per_adapter_hours": {
      "rs_vqa": 8.53,
      "change_vqa": 8.53,
      "optsar_fusion": 8.53,
      "rs_ground_caption": 8.53
    }
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 1048576,
      "composites": 2,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "ok": false,
    "error": "CUDA OOM",
    "sec_per_opt_step": 0.0,
    "sec_per_sample": 0.0,
    "mean_seq_len": 0.0,
    "p95_seq_len": 0.0,
    "mean_vision_tokens": 0.0,
    "mean_supervised_tokens": 0.0,
    "peak_vram_gb": 0.0,
    "reserved_vram_gb": 0.0,
    "static_vram_gb": 0.0,
    "activation_vram_gb": 0.0,
    "first_loss": 0.0,
    "last_loss": 0.0,
    "effective_tflops": 0.0,
    "per_adapter_hours": {}
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 1048576,
      "composites": 3,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "ok": false,
    "error": "CUDA OOM",
    "sec_per_opt_step": 0.0,
    "sec_per_sample": 0.0,
    "mean_seq_len": 0.0,
    "p95_seq_len": 0.0,
    "mean_vision_tokens": 0.0,
    "mean_supervised_tokens": 0.0,
    "peak_vram_gb": 0.0,
    "reserved_vram_gb": 0.0,
    "static_vram_gb": 0.0,
    "activation_vram_gb": 0.0,
    "first_loss": 0.0,
    "last_loss": 0.0,
    "effective_tflops": 0.0,
    "per_adapter_hours": {}
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 2097152,
      "composites": 1,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "ok": false,
    "error": "CUDA OOM",
    "sec_per_opt_step": 0.0,
    "sec_per_sample": 0.0,
    "mean_seq_len": 0.0,
    "p95_seq_len": 0.0,
    "mean_vision_tokens": 0.0,
    "mean_supervised_tokens": 0.0,
    "peak_vram_gb": 0.0,
    "reserved_vram_gb": 0.0,
    "static_vram_gb": 0.0,
    "activation_vram_gb": 0.0,
    "first_loss": 0.0,
    "last_loss": 0.0,
    "effective_tflops": 0.0,
    "per_adapter_hours": {}
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 2097152,
      "composites": 2,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "ok": false,
    "error": "CUDA OOM",
    "sec_per_opt_step": 0.0,
    "sec_per_sample": 0.0,
    "mean_seq_len": 0.0,
    "p95_seq_len": 0.0,
    "mean_vision_tokens": 0.0,
    "mean_supervised_tokens": 0.0,
    "peak_vram_gb": 0.0,
    "reserved_vram_gb": 0.0,
    "static_vram_gb": 0.0,
    "activation_vram_gb": 0.0,
    "first_loss": 0.0,
    "last_loss": 0.0,
    "effective_tflops": 0.0,
    "per_adapter_hours": {}
  },
  {
    "config": {
      "model_id": "Qwen/Qwen3-VL-4B-Instruct",
      "max_pixels": 2097152,
      "composites": 3,
      "micro_batch": 4,
      "grad_accum": 1,
      "steps": 50,
      "warmup_steps": 10,
      "num_workers": 4,
      "devices": 1,
      "grad_checkpointing": true,
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
    "ok": false,
    "error": "CUDA OOM",
    "sec_per_opt_step": 0.0,
    "sec_per_sample": 0.0,
    "mean_seq_len": 0.0,
    "p95_seq_len": 0.0,
    "mean_vision_tokens": 0.0,
    "mean_supervised_tokens": 0.0,
    "peak_vram_gb": 0.0,
    "reserved_vram_gb": 0.0,
    "static_vram_gb": 0.0,
    "activation_vram_gb": 0.0,
    "first_loss": 0.0,
    "last_loss": 0.0,
    "effective_tflops": 0.0,
    "per_adapter_hours": {}
  }
]
```
</details>