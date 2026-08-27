# Phase 0 — resolution policy: verified chip sizes, per-dataset GSD decisions

Replaces the assumed chip sizes in `../03-compute/phase-0-cost-model.md` §3 (marked "not verified
against downloaded files"). All figures from primary sources, cited at the end.
**Revision 2:** the first version of this doc recommended blanket 512² native-GSD tiling.
That fixed silent degradation but overspent where the task doesn't use native resolution.
This version decides GSD per dataset against an explicit criterion.

## 1. The two failure modes

**`max_pixels` is not a resolution setting. It is a silent resampler for anything above it.**
Qwen3-VL never upsamples, so below the cap it is inert; above it, the processor downsamples
invisibly, downstream of every curation check. §5.4's GSD stratification cannot see it.

The symmetric failure: **resolution finer than the task uses is pure token spend.** A 512²
view costs 256 tokens whether or not the extra pixels carry task-relevant signal. RSVQA-HR
at its native 15 cm is the worst case — 4× the tokens of a 30 cm version that answers the
same questions *and* sits closer to the deployment sensor.

## 2. The decision criterion

For each dataset, the coarsest GSD that satisfies both:

1. **Pixels-on-target.** The smallest scoring-relevant object spans enough pixels for the
   task type: **≥12–16 px** for bbox grounding / segmentation, **≥3–4 px** for detection
   presence and change, VQA/captioning tolerant above ~4 px on queried objects.
2. **Deployment match.** Effective GSD at or near the hidden-set sensors — Cartosat-2S
   (0.65 m pan / 2 m MX) and RISAT (~1–2 m). Training *finer* than both the task needs and
   the eval sensor provides buys nothing (S3, transposed).

Chips sized 512² max (=262,144 px, the cap, exactly). Coarser sources keep their smaller
native chips.

## 3. Verified chip dimensions (primary sources)

| Source | Native GSD | Shipped chip | Cap binds at 262k? |
|---|---|---|---|
| BEN.txt S2 / S1 | 10 m | 120×120 (1200 m footprint) | no — inert |
| RSVQA-LR | 10 m | 256×256 | no |
| RSVQA-HR | 0.15 m (USGS HRO aerial) | 512×512 | exact fit |
| RarePlanes real | 0.31–0.39 m | 512×512, 20% overlap | exact fit |
| RarePlanes synthetic | 0.30 m | **1920×1080** | **yes — 7.9×** |
| SpaceNet 6 MSAW | 0.5 m | **900×900** (450 m tiles) | **yes — 3.1×** |
| SpaceNet 7 / MUDS | 4 m | 1024×1024 | **yes — 4.0×** |
| OpenEarthMap-SAR | 0.15–0.5 m | 1024×1024 | **yes — 4.0×** |
| LS-SSDD-v1.0 | 10 m | 800×800 (from 24,000×16,000 scenes) | **yes — 2.4×** |
| OSCD | 10 m | ~600×600 | **yes — 1.4×** |
| HRSCD | 0.5 m | **10000×10000** (100 MP, untiled) | **yes — 381×** |

Worst casualties of the 262k cap as-is: **LS-SSDD** ships drop to 1.9 px (below detection
floor — the capability C53 bought does not survive), **SpaceNet 6** (0.88 m) and **OEM-SAR**
(0.60 m) stop being sub-metre — nullifying C31/C37's selection rationale.

## 4. Per-dataset decisions

| Source | Decision | Eff. GSD | Chip | Vis tok | Why this and not finer/coarser |
|---|---|---|---|---|---|
| BEN.txt S2 | as shipped | 10 m | 120² ×3 comps | 48 | nothing to decide; cap inert |
| BEN.txt S1 | as shipped | 10 m | 120² | 16 | — |
| RSVQA-LR | as shipped | 10 m | 256² | 64 | — |
| **RSVQA-HR** | **downsample 2×** | **0.30 m** | **256²** | **64** (was 256) | VQA on buildings/roads: 10 m object = 33 px at 0.3 m, far above need. 0.15 m is finer than Cartosat-2S pan (0.65 m) — training there *widens* the domain gap C27 exists to close. 4× token cut on 20% of the `rs_vqa` mix |
| RarePlanes real | as shipped | 0.35 m | 512² | 256 | bbox grounding: small aircraft ~8 m = 23 px — near the 12–16 px floor, no headroom to coarsen |
| RarePlanes synthetic | **crop 512² around annotations** | 0.30 m | 512² | 256 | never feed 1920×1080 whole (silently becomes 0.84 m — 2.4× off the real half, S3 via the processor). Same class must train at one resolution |
| SpaceNet 6 | **crop 512² from 900²** | 0.50 m | 512² | 256/view | sub-metre is the point (C31); buildings 16 px at 0.5 m, floor at 1 m would be 8 px — fails grounding need |
| OpenEarthMap-SAR | **resample to 0.5 m, crop 512²** | 0.50 m | 512² | 256/view | harmonises the 0.15–0.5 m spread (one source, one distribution); at 0.15 m a 512² crop covers only 77 m — too little context for land-cover classes; 0.5 m keeps sub-metre + 16 px buildings + 256 m context |
| SpaceNet 7 | **crop 512² from 1024²** | 4 m | 512² | 256/view | buildings 5 px at native — *already at the floor*, zero coarsening headroom; crop keeps native + 2 km context per view |
| LS-SSDD | **crop 512² around ships + background negatives** | 10 m | 512² | 256 | ships 3 px at native — the floor exactly; ships sparse, so crop around annotations, keep pure-background crops (the dataset's own design point) as negatives |
| OSCD | **crop 512² from ~600²** | 10 m | 512² | 256/view | urban change at 10 m already marginal; no headroom |
| HRSCD | **native 0.5 m, 512² tiles, change-stratified selection** | 0.50 m | 512² | 256/view | it is our *only* sub-metre semantic change source (C60) — coarsening forfeits that. Cost is controlled by which tiles enter the manifest, not by resolution: 291 pairs × ~380 tiles = ~220k candidates, we need ~10–20k. Select change-containing tiles + stratified no-change |
| VRSBench / CDVQA (eval) | as published | — | — | — | never alter eval imagery; preprocessing at eval must match training per-source (frozen contract) |

**Consistency rules that make this safe:**
- The per-source GSD is part of the frozen preprocessing contract — applied identically at
  train and inference (byte-identical chain, §4.2's own argument).
- **GSD-conditioned prompts (§5.4) must inject the *effective* GSD, not the native one.**
- `max_pixels: 262144` stays in `preprocessing.yaml` as a safety net **with a hard assert
  that it never binds** — a source arriving above the cap is a pipeline bug, not a resize.

## 5. Budget effect

Per-sample tokens change only where a decision changed them:

- **`rs_vqa`: 151 → 113 tok/sample** (RSVQA-HR at 64 vis tokens) → **1.18 h → 0.88 h**/epoch.
- Other adapters: unchanged — the cost model already priced the sub-metre sources at 512²
  equivalents, and their floors leave no room to cut.
- Four-adapter total: **~9.8 h → ~9.5 h** at 25% MFU. Modest, because blanket-512 was
  already near cost-optimal; the value of this revision is *correctness*: RSVQA-HR now
  matches the deployment sensor instead of overshooting it, OEM-SAR trains as one
  distribution instead of a 3.3× internal GSD spread, and HRSCD/LS-SSDD keep the
  properties they were selected for.
- Non-GPU resources move more: HRSCD staged as selected tiles (~10–20k × 0.5 MB) instead
  of 291 × 100 MP pairs; RSVQA-HR at 256² is 4× less storage/prep on a 10,659-image set.

## 6. Second-order finding: the C22 ablation is nearly free

BEN is 16 tokens *per composite*; three vs two composites is a 16-token delta on the
majority source. The Modal sweep's "+8.1 h (+47%)" came from 2048² synthetic tiles that
always hit the cap. On real BEN imagery, C43's ablation is a **sub-hour** experiment, and
cut-list item 3 ("third composite → two") frees almost nothing — re-order or drop it.

## 7. Closed checklist item

**Umbra is X-band (9.8 GHz nominal centre frequency)** — closes the v3.4 line "Umbra SAR
band confirmed (C37)" and its risk row. OEM-SAR is therefore a second independent X-band
source alongside SpaceNet 6 Capella, so RISAT-2B σ⁰ tuning is not single-sourced on
Rotterdam — provided OEM-SAR is cropped/resampled per §4, not cap-downsampled.

## 8. Actions

1. Freeze `tiling.max_pixels: 262144` + never-binds assert; bump `version`. Still gated on
   the 5 min/scene prep SLA, which nothing has measured.
2. Add the per-source effective-GSD table (§4) to the frozen preprocessing contract.
3. RSVQA-HR: 2× downsample to 0.30 m at staging; same resample applied at eval time.
4. HRSCD: build the change-stratified 512² tile selector (this also defines the previously
   undefined tiling for a 100 MP source).
5. C42 reserve rule: RarePlanes synthetic enters as 512² annotation-centred crops only.
6. Re-run the batch-size sweep at `--source-size 512` — what the sub-metre adapters
   actually see.
7. Re-price C22/C43 at BEN's real 16 tok/composite; fix cut-list item 3.

## Sources

- RarePlanes: arXiv 2006.02963 — real 512×512 @ 0.31–0.39 m, 20% overlap; synthetic 1920×1080 @ 0.3 m
- SpaceNet 6: spacenet.ai/sn6-challenge — ~450 m tiles @ 0.5 m; Capella X-band quad-pol
- SpaceNet 7 / MUDS: Van Etten et al., CVPR 2021 — "1024 × 1024 pixels", "GSD ≈ 4 meters", 11,079,262 labels
- OpenEarthMap-SAR: github.com/cliffbb/OpenEarthMap-SAR — "1024x1024 pixels at a ground sampling distance of 0.15m--0.5m"
- HRSCD: arXiv 1810.08452 — "291 RGB image pairs of 10000x10000 pixels", "50 cm per pixel"
- LS-SSDD-v1.0: Remote Sensing 12(18):2997 — 24,000×16,000 scenes cut to 800×800; 250 km swath ⇒ 10 m spacing
- RSVQA: arXiv 2003.07333 — LR 256² Sentinel-2 @ 10 m; HR 512² @ "15cm resolution aerial RGB… USGS" HRO, NE USA
- BigEarthNet v2.0 / reBEN: arXiv 2407.03653 — 549,488 pairs, "1200 m x 1200 m" patches ⇒ 120² @ 10 m
- OSCD: rcdaudt.github.io/oscd — 24 Sentinel-2 pairs, 13 bands
- Umbra: eoPortal — "Nominal centre frequency (GHz): 9.8 (X-band)"
