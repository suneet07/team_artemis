# Phase 0 cost model — how to price a SatQuery training run

Supersedes the hours column of `../../logs/phase_0_timing_sweep_a100_80gb.md`. That sweep
measured synthetic 2048×2048 tiles, which always hit the `max_pixels` cap. Real
imagery mostly does not, so the sweep's per-adapter hours are wrong for any adapter
whose source images are smaller than the cap. Its **cost per token** is correct and
transferable, and that is what this document is built on.

## 1. `max_pixels` is a cap, not a target

Qwen3-VL uses native dynamic resolution (§5.1) and **never upsamples**. Measured
against the real processor at `max_pixels = 262,144`:

| source image | vision tokens |
|---|---|
| 120×120 (BigEarthNet patch) | **16** |
| 256×256 | 64 |
| 512×512 | 256 |
| 1024×1024 | 256 (cap binds) |
| 2048×2048 | 256 (cap binds) |

Consequences:

1. **Raising `max_pixels` above a source's native size costs nothing and buys
   nothing.** It is inert on BigEarthNet.
2. **`max_pixels = 262,144` is exactly a 512×512 view** — the size §5.5's cut-list
   note calls the benchmark chip size ("Benchmark chips (512px) never need
   tiling"). It preserves chips exactly, with no waste and no loss.
3. The budget is therefore driven almost entirely by the **sub-metre adapters**,
   not by BigEarthNet. §5.5 said this already: *"`rs_ground_caption` is ~40% of
   the budget purely from image size."* With BEN's real token count it is more
   than 40%.

Token arithmetic (patch 16, merge 2, both verified):

    vision_tokens_per_image = min(source_pixels, max_pixels) / 1024

## 2. Cost per token — the transferable measurement

From the A100-80GB sweep, `micro_batch=4`, bf16-mixed, **no** gradient
checkpointing, ~25% MFU:

| seq_len | s/opt-step | ms per token |
|---|---|---|
| 317 | 0.529 | 0.42 |
| 833 | 1.135 | 0.34 |
| 1085 | 1.536 | 0.35 |

Roughly constant at **~0.35 ms/token** above short sequences. So:

    epoch_hours ≈ 80,000 × tokens_per_sample × 0.35e-3 / 3600
                ≈ tokens_per_sample × 0.0078 hours

Sanity check: 575 tokens → 4.5 h, and the sweep measured 4.3 h. Model holds.

**This is pessimistic.** 25% MFU at `micro_batch=4` leaves ~70 GB of an 80 GB card
unused. §5.5 C41 says to "raise batch size until utilisation saturates." At 40% MFU
the constant drops to ~0.22 ms/token, i.e. **×0.63 on every figure below**.

## 3. Tokens per sample, by source (at `max_pixels = 262,144`)

| Source | native size | views | vision tokens |
|---|---|---|---|
| BEN.txt S2 (10 m) | 120×120 | 3 composites (C22) | **48** |
| BEN.txt S1+S2 pair | 120×120 | 3 optical + SAR stack | ~64 |
| RSVQA-LR (Sentinel-2) | 256×256 | 1 | 64 |
| RSVQA-HR (USGS 15 cm, C54) | 512×512 | 1 | 256 |
| RarePlanes (30 cm, C36) | 512×512 tiles | 1 | 256 |
| SpaceNet 6 MSAW (0.5 m, C31) | ~900×900 | optical + SAR | 512 |
| SpaceNet 7 / MUDS (4 m, C59) | 1024×1024 | bi-temporal | 512 |
| HRSCD (0.5 m, C60) | tiled | bi-temporal | 512 |
| OpenEarthMap-SAR (0.15–0.5 m, C37) | ~1024×1024 | optical + SAR | 512 |
| Cartosat-2S (inference) | §4.4 tiling output | 2 (no SWIR) | 512 |

Add ~60 text tokens for VQA, ~150 for captioning.

## 4. Per-adapter estimate

Mix weights below are **assumptions** — they come from §5.3's "BEN.txt majority"
rule, not from a built manifest. Replace them once the manifests exist.

| Adapter | assumed mix | tokens/sample | @25% MFU | @40% MFU | §5.5 |
|---|---|---|---|---|---|
| `rs_vqa` | 70% BEN, 20% RSVQA-HR, 10% LR | ~150 | **1.2 h** | 0.7 h | 4 h |
| `optsar_fusion` | 60% BEN pairs, 40% SpaceNet 6 / OEM-SAR | ~330 | **2.6 h** | 1.6 h | 6 h |
| `change_vqa` | SpaceNet 7 / HRSCD / self-generated Sentinel | ~450 | **3.5 h** | 2.2 h | 6 h |
| `rs_ground_caption` | 40% BEN referring, 60% sub-metre | ~320 | **2.5 h** | 1.6 h | 12 h |
| **Four-adapter total** | | | **~9.8 h** | **~6.1 h** | **28 h** |

Against §5.5's 28 h of scheduled training, this is **~3× headroom** — a very
different picture from the sweep's naive 23.2 h, which priced every adapter as if
its imagery were 512×512.

## 5. What this frees up, and where it should go

C42 already names the rule: if the pass runs clean with reserve left, spend it on
`rs_ground_caption` sample count, because referring detection is the weakest §6.1
target (≥45 mIoU against RS-InternVL's 65.84) and more data is the only lever.

This cost model says the reserve is much larger than 19 h. Options, in the order
C42 implies:

1. **Raise `rs_ground_caption` sample count** well beyond 80k — RarePlanes alone
   carries ~630k synthetic annotations across 50k synthetic images (C36).
2. **Raise `max_pixels` for the grounding adapter only**, if its sources exceed
   512 px natively. Costs nothing on BEN, everything is a cap.
3. Only then consider more epochs or the §5.6 stretch experiment.

## 6. Confidence and open inputs

**Measured:** token counts per source size; ms/token at 25% MFU; peak VRAM per
config; that checkpointing was inert (see below).

**Assumed:** 40% MFU is reachable (needs a batch-size sweep — 2–3 single-cell runs,
under a dollar); the dataset mix weights in §4; chip sizes for RarePlanes,
SpaceNet 6/7, HRSCD and OEM-SAR taken from their published tiling, not verified
against downloaded files.

**Not modelled at all — the dominant remaining risk:** data loading. All timings
used synthetic tiles from a memory cache. If the §4.2 SAR chain (orbit file,
calibrate σ⁰, multilook, Refined Lee, terrain correction, dB, stretch) runs online
per sample, it can exceed GPU time entirely. **Precompute preprocessed tensors
offline and this cost model holds; run the chain online and it does not.** That is
a design decision, not a measurement.

Also unmodelled: checkpoint I/O (guard rail 2, ~every 500 steps), and captioning
targets being longer than the 24-token synthetic answers used in the harness.

## 7. The bug worth remembering

Lightning **never calls `.train()`** at the start of training — its only `.train()`
calls are in the validation/test hooks, to restore mode after an eval loop. It only
*warns*: "Found N module(s) in eval mode at the start of training."

`transformers.from_pretrained` returns a model in **eval** mode and PEFT preserves
it. So without an explicit `model.train()`:

* gradient checkpointing silently no-ops — transformers gates it on
  `gradient_checkpointing AND self.training`. Measured **0 of 60** layers active,
  with `--grad-checkpointing` and `--no-grad-checkpointing` producing byte-identical
  15.9 GB of activations;
* every `nn.Dropout` is inert, so **LoRA dropout 0.1 (§5.2) does nothing**.

Both fail silently and in the wrong direction. Fixing it took activations from
15.9 GB to 2.6 GB (6.1×) at a 58% step-time cost.

**Put `model.train()` in the real training script explicitly.** Do not rely on
Lightning, and do not scroll past that warning.

## 8. Next actions

1. Batch-size sweep at `max_pixels=262,144` — raise `--micro-batch` until VRAM
   approaches ~70 GB. Settles the 25% vs 40% MFU question. Under $1.
2. Decide **precompute vs online preprocessing**. Bigger effect than anything else
   in this document.
3. Confirm real chip sizes for the sub-metre datasets; replace §3's assumptions.
4. Freeze `tiling.max_pixels: 262144` in `configs/preprocessing.yaml`, bump
   `version` — but only after checking the 5 min/scene prep SLA, which the YAML
   TODO names as the other binding constraint and which no run so far has measured.
5. Solve the BigEarthNet imagery extraction problem (../04-open/SPECULATIONS.md item 2). It
   blocks everything above.
