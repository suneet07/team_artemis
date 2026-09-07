# Segment 4 — SAR and optical–SAR fusion (gate G5)

Cross-modal pair analysis: a co-registered optical + SAR pair, answering
*"use the optical and SAR images together to identify built-up and water-covered
regions"* — the problem statement's own representative query.

**Status: shipped as a tool + a deterministic floor. No adapter, deliberately.**

**`optsar_fusion` is not an adapter.** It ships **BIFOLD** (`lulc_classifier`,
74.95% measured) sitting on top of **D1** (deterministic decision-level fusion).

---

## 1. Result

```
resnet50-s1 on radar    74.95%  on 6,000 held-out val rows
majority-answer floor   50.1%
                        +24.85 points, from radar alone, zero training
```

Threshold **0.5** (the honest operating point). At 0.3, swept and optimistic,
76.47%.

| slice | accuracy |
|---|---|
| forest questions | 78.2% |
| water / urban / industrial presence | 69.0% |

The presence slice is the one that matters — `Urban fabric`, `Inland waters`,
`Marine waters`, `Industrial units` is precisely the PS's example query.

### The train-leakage catch

An earlier probe reported **81.7%**. That was on reBEN **train**, which BIFOLD
was fitted on. Held-out val is **74.95%**. The 6.7-point gap is exactly what
training-set contamination looks like — caught before it reached a claim.

---

## 2. The model

```
RADAR_MODEL  = "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0"
OPTICAL_MODEL = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
```

**BIFOLD** = Berlin Institute for the Foundations of Learning and Data, TU
Berlin. `BIFOLD-BigEarthNetv2-0` is their official HuggingFace org — the
**authors' own**, not a third-party re-upload, which matters under C45
(*"a re-uploader cannot grant rights Planet did not"*).

- **MIT licence.** Zero training required.
- 40+ models already fine-tuned for the **19 CORINE classes** on reBEN v2.
- Loads through plain **`timm`** — no `configilm` dependency, no GitLab clone at
  container start. Every tensor is a plain timm ResNet-50 under a wrapper prefix
  (`model.vision_encoder.`), stripped at load.

### Published AP on reBEN v2 (official test split)

```
resnet50-s2   (optical)  0.714     <- strongest
resnet50-all  (fused)    0.711
resnet50-s1   (radar)    0.628
```

**Fusion adds nothing.** 0.711 fused against 0.714 optical alone. That
independently confirms the project's existing decision-level choice and means
there is no GPU spend left on the fused path.

---

## 3. The preprocessing contract — pinned from source, not guessed

This is the part that must never drift. All of it lives in
`satquery/tools/lulc.py` and is mirrored in `scripts/eval_bifold.py`.

### Band order

```python
_S1_ORDER = ("VH", "VV")
_S2_ORDER = ("B02","B03","B04","B08","B05","B06","B07","B11","B12","B8A")
# fused 12ch = S2 order above + ("VH", "VV")
```

**S1 puts VH first** — which contradicts both the filenames *and* the Sentinel-1
convention. **S2 is not sorted** — B08 sits fourth.

From `configilm.extra.BEN_lmdb_utils.BAND_COMBINATION_PREDEFINTIONS`.
(`BEN2DataSet` inherits from the v1 `BENDataSet` and reuses `BEN_lmdb_utils`, so
the v1 order is correct for v2.)

### Band order verified empirically, not assumed

| S1 band order | accuracy | floor |
|---|---|---|
| **`vh_vv`** (documented) | **81.7%** (train) → **74.95%** (val) | 50% |
| `vv_vh` (control) | **49.9%** | 50% |

**The wrong order scores exactly chance.** That control is what makes the band
order a measured fact rather than a reading of documentation.

### Normalisation

```python
_S1_STATS = {
    "VV": (-12.619993741972035, 5.115911777546365),
    "VH": (-19.29044597721542,  5.464428464912864),
}
```

S1 is **calibrated sigma-nought in decibels**; S2 is L2A reflectance scaled by
10,000. Verified against our staged reBEN: S1 float32 spanning **−44.5 dB**
(specular water) to **+18.7 dB** (double-bounce) — real radar physics, not a
stretched image.

### Input size and threshold

```
input      120 px, bicubic  (reBEN patches are natively 120 px, so a no-op there)
threshold  0.5              <- the operating point 74.95% was measured at
```

**The threshold was wrong in serving** — the answer composer displayed classes at
0.25, which shows classes the reported number never covered. Fixed to 0.5, with
`_LULC_THRESHOLD` in both composers and a test that reads `eval_bifold.py`'s
argparse default and fails on drift.

Two tests now compare the served constants against the scorer's, digit for digit,
including a check that a `[VV, VH]` swap cannot creep back.

---

## 4. The bf16 bug — measured good, silently broken in production

`lulc_classifier` scored 74.95% standalone and **failed on every deployed
request**:

```
RuntimeError: Expected weight to have type Float but got BFloat16
```

**Cause:** loading Qwen3-VL with `dtype=bfloat16` leaves torch's **global**
default dtype at bfloat16. `timm.create_model` then built a bf16 ResNet-50, and
BIFOLD's float32 weights were cast down into it. The bug exists **only** when the
classifier shares a process with the VLM — which is exactly what the deployment
does and what a standalone benchmark never does.

**Why it stayed invisible:** the executor records a tool failure as a trace
warning and answers from whatever else ran. `/meta/health` reported the tool
available, the answer came back confidently from the VLM, and nothing said the
74.95% model had not run. A GPU was billed for a model that was never called.

**Fix:** build the model under an explicit `torch.set_default_dtype(float32)`
guard, `model.float()` after loading, and cast the input batch explicitly.
`tests/test_lulc_dtype.py` reproduces the condition and asserts float32 — plus a
test proving the fixture really does reproduce the original failure, so it cannot
become a no-op.

---

## 5. D1 — the deterministic fusion floor

`satquery/fusion/fusion.py`. **The plan makes D1 carry gate G5; the adapter is
explicitly upside.** The tool config says it plainly: *"an upgrade layered over
the deterministic D1 fusion floor, never a dependency of it."*

```
optical  →  spectral_index (MNDWI / NDWI / NDBI / NDVI)  →  mask
radar    →  sar_backscatter (sigma-nought dB thresholds) →  mask
            ↓
       D1 reconciliation + DISAGREEMENT_RULES
            e.g. cloud_over_water -> trusts sar
```

**Decision-level, not feature-level, and the reason is published:** EarthMind's
own ablation showed naive token concatenation fails on optical–SAR heterogeneity.
Decision-level fusion is more robust to residual misregistration and gives
interpretable confidence. Our own measurement agrees — published fused AP (0.711)
is *below* optical alone (0.714).

D1 is **resolution-agnostic and transfers to any sensor pair**, which is the
right hedge against a hidden set we know nothing about.

### The units bug — the floor was broken and nobody had noticed

Our own `DATASET_SPECS` said dB thresholds *"will not work directly on this
dataset without reverse-calibration"* for 8-bit SAR (OpenEarthMap-SAR ships
exactly that).

Three call sites applied dB thresholds to whatever array they were handed, with
no check:

- The water cut `values < -18.0` is **never true** on 0–255 pixels → **empty
  mask, reported as a finding**.
- `texture_seg`'s SAR branch has no Otsu gate and compares `>= 0.0` → marks the
  **entire frame** as built-up.

**Fixed:** `assert_decibels()` at all three sites. It **raises rather than
warns**, because a dB cut on 0–255 pixels is not provisional, it is meaningless.
Two regression tests.

**This was the highest-value SAR work available and it was cheap.** The floor
matters more than the ceiling — and it had never seen a radar pixel, because we
had never staged any.

---

## 6. The routing decision

```
optical  →  spectral_index (masks)  +  rs_vqa VLM (language)        both measured, both work
radar    →  sar_backscatter (dB)    +  lulc_classifier (19 classes)  74.95% measured
reconcile →  D1 disagreement table
```

**`lulc_classifier` was rescoped to `required_modalities: [sar]`.**

Three reasons:

1. **Measured.** On 4,000 held-out cross-modal rows with both models answering
   the same questions, radar scores **0.7525** and optical **0.4968** — chance.
   Where they disagree, **radar is right 87% of the time**.
2. **Optical is already covered twice** — `spectral_index` gives deterministic
   MNDWI/NDBI masks and the VLM answers in language. Adding `resnet50-s2` was
   arguably redundant even if it had worked.
3. **The fused path never worked in our repack** (see §7).

The manifest originally declared `required_modalities: [optical]`, which the
validator enforces — so a measured 74.95% radar model sat behind a contract that
*refused radar*, while the PS explicitly permits SAR-only input (C26).

**Radar-only is a first-class path with an honest warning.** When no optical view
exists the tool records that its published AP is 0.628 against optical's 0.714. A
trace that hid which model answered could not be audited.

### The routing gap that hid it for a while

`add_lulc()` ran for `CROSSMODAL_*` and `SINGLE_CAPTION` but **not
`SINGLE_VQA`** — so *"What land cover is in this radar scene?"*, the question the
classifier is measurably best at, planned a lone VLM step and answered from an
impression of speckle. Fixed.

Before that, `lulc_classifier` was registered and working but appeared in **no
plan at all** — reachable, never reached.

---

## 7. The fused path — dead, and honestly so

```
s1=vh_vv,  s2=configilm   0.5007     <- exactly chance
s1=vv_vh,  s2=configilm   0.4980
s1=vv_vh,  s2=sorted      0.4980
s1=vh_vv,  s2=sorted      0.4960
```

Since `vh_vv` was demonstrably correct for the S1-only model, the radar half is
right — which means our **repacked S2 is not in configilm's order**, and neither
candidate ordering fixed it.

**Abandoned after three probe runs and a fingerprint.** The returns went
negative, published numbers say fusion adds nothing anyway (0.711 vs 0.714), and
D1 covers the mandatory bullet deterministically.

Worth recording that the repack itself is the suspect: converting to a stacked
raster is a normal convenience, but two things can be lost — **bands dropped**, or
**dtype narrowed** (uint16 reflectance → uint8, float32 dB → uint8, which is
exactly the failure D1 was fixed for). The verification checklist that came out
of this:

- **Band count** — S2 must carry ≥10 bands; fewer means the repack dropped some.
- **dtype** — S2 uint16, S1 float32. uint8 means re-quantised.
- **Sign and range** — S1 sigma-nought is negative over nearly every surface. No
  negative values anywhere means it has been stretched.

---

## 8. Why not train an `optsar_fusion` adapter

The user asked directly: *"do you think we should fine tune our own maybe fine
tune this bifold itself maybe the ssl4eo-s12..your call to make"*

**Answer: no, and the reasoning is measurement, not effort.**

1. **There is no public benchmark for cross-modal in the PS.** VRSBench and RSVQA
   are single-image; CDVQA is change. Cross-modal is graded **only on the hidden
   ISRO set** — Cartosat-2S + RISAT, annotations undisclosed. So any optsar
   training is **unmeasurable**: we would spend GPU hours and never know if it
   helped. Every other decision this session was settled by measuring; this one
   cannot be, and that is an argument against spending, not for guessing harder.
2. **An optsar LoRA would train on reBEN** — the same data BIFOLD's authors
   already trained a supervised specialist on. We would reproduce their result,
   worse and slower.
3. **Fine-tuning BIFOLD on reBEN is circular.**
4. **Fine-tuning SSL4EO is a research bet with nothing to prove it on.**
5. **The PS asks for exactly this architecture** — *"The system may use multiple
   specialised components, such as… an optical–SAR fusion or information-
   extraction model."* A registered classifier tool is the intended design, not a
   shortcut around it.

**Zero SAR in the public benchmarks.** VRSBench optical, RSVQA optical, CDVQA
optical bi-temporal, BEN.txt named for *adaptation* not scoring. **Nothing in the
graded public set contains a SAR image.**

---

## 9. SSL4EO-S12 — the considered alternative

| | code | weights | data |
|---|---|---|---|
| **SSL4EO-S12** | Apache-2.0 | **CC-BY-4.0** | 251,079 locations, S1 dual-pol + S2, 4 seasons |
| **CROMA** | MIT | **MIT** | joint radar–optical encoder, S1 2ch + S2 12ch |
| **DOFA** | MIT | released | S1, S2, NAIP, Gaofen, EnMAP |
| **TerraMind** (IBM/ESA) | Apache-2.0 | released | optical + SAR + DEM |
| **UnigeoCLIP** | MIT | — | 2.5M aligned optical + SAR + captions |

**Why BIFOLD over SSL4EO for the deadline:** SSL4EO gives you *features* and you
must train a probe first. BIFOLD gives **19-class predictions immediately**. Same
underlying data, same 10 m C-band domain, MIT licence, zero training.
`B2_vits16_mae_ep99_enc.pth` is only 86 MB — but it has no classification head.

**Why the published numbers cannot be compared.** SSL4EO's BigEarthNet figures
are fine-tuning/linear-probe results on **v1** with the **SeCo split**. BIFOLD's
0.711/0.859 is **reBEN v2** on the **official test split** — cleaner labels,
removed patches. Putting 91.8% beside 0.859 is the "comparing on different
samples" error already logged in `TEAM_CONTEXT.md`.

**Band mismatch too:** SSL4EO's S2 models expect **13-band L1C**; our verified
reBEN patches are **10-band L2A**. Not a drop-in.

**After the deadline** SSL4EO becomes a real candidate: fine-tune `B2_vits16_mae`
on our reBEN S1 and see if it beats `resnet50-s1`'s 0.628. An open question worth
answering with a run rather than a table.

---

## 10. The domain gap — X-band vs C-band, and why it matters less than feared

The hidden set is **RISAT** — X-band, sub-metre, with modes including single-pol
and hybrid circular polarimetry. BIFOLD is **C-band, dual-pol VV/VH, 10 m**.

Three specific mismatches:

1. **Label space.** BIFOLD emits 19 CORINE classes.
2. **Channel count.** BIFOLD-s1 wants 2 channels dual-pol. SpaceNet 6 is
   **quad-pol** (4 intensity + 2 Pauli); OEM-SAR is **single-pol**. Neither is
   2-channel dual-pol, so `conv1` needs surgery in both directions.
3. **Units.** OEM-SAR ships **8-bit stretched**, not dB. BIFOLD subtracts a VV
   mean of −12.62 dB. Feed 8-bit into that and you get the band-order failure
   again — **exactly chance, confidently**.

**But the gap matters less than first argued.** The hidden set is *paired*, so
optical carries the semantic load and SAR contributes complementary structure. A
C-band-trained tool reading *"bright = built-up, dark = water"* on X-band still
gets that right — those scattering mechanisms are **wavelength-independent**.
Band matters for *penetration*, not for basic scattering class.

**Ships are the exception and work even at 10 m.** A vessel is a strong point
scatterer on specular near-black water; the contrast is enormous, which is why
operational ship detection runs on Sentinel-1 at exactly this resolution.
Threshold-bright-returns plus connected components — and D1 already has those
pieces in `texture_seg`'s SAR branch and `object_box_fallback`.

**What is not covered, and should be said plainly rather than discovered by a
judge:** SAR object detection beyond ships.

---

## 11. The optsar corpus — built, measured, unused for training

**85,096 rows** from 13,683 verified reBEN pairs, via `gen_crossmodal_qa` —
which emits the same question three ways: **optical-only, SAR-only, fused**.

**Every arm at exactly 50.0% majority-answer floor**, reached over three
iterations: 60.6% → 53.9% → **50.0%**. A corpus that cannot be gamed by the
answer prior.

Val split: **41,850 rows with ground-truth answers** — which is what made the
BIFOLD bake-off possible without any training.

**The C26 point the generator's own docstring flags:** without deliberately
withheld-modality rows, a **SAR-only input at inference is out of distribution for
every adapter**, not just this one. The PS permits *"one optical/multispectral
**or** SAR image"* as input. D1 handles mask-level fusion fine — but any
*question* about a SAR image hits an untrained path, because **D1 fuses masks, it
does not answer questions**.

---

## 12. SAR data sources

| source | resolution | licence | state |
|---|---|---|---|
| **reBEN (BigEarthNet v2)** | 10 m C-band S1 + S2 | CDLA-Permissive 1.0 | ✅ staged, 13,683 verified pairs |
| **SpaceNet 6 MSAW** | **0.5 m** Capella X-band quad-pol + Maxar optical, co-registered | CC BY-SA 4.0 | ⚠ optical half staged (6,802 files); **`SAR-Intensity` never fetched (41 GB)** |
| **OpenEarthMap-SAR** | 0.15–0.5 m | cleared | not staged; ships 8-bit, needs reverse-calibration |
| **GEOID-Flood** | 10 m, bi-temporal S1 + S2 + DEM, 14,282 tiles, 219 events | CC BY 4.0 | the thing C28 said did not exist |
| **Umbra** | 0.16–1 m | CC BY 4.0 | X-band, no labels |
| **M3LEO** | 10 m + InSAR/coherence/polarimetry | CC BY-SA 4.0 | clean, but does not close the metre-class gap |
| **LS-SSDD-v1.0** | Sentinel-1 ships | Apache-2.0 **badge only** | ❌ repo is a pointer; CAS portal is research/teaching only |
| **SOMA-1M** | 0.5–10 m | **none** | ❌ Google Earth optical + PIESAT/Capella SAR, not fully released |
| **SARLANG-1M** | — | split licence | ❌ clean imagery, unlicensed annotations |

**SpaceNet 6 is the structural match to the ISRO set** — 0.5 m co-registered
optical + SAR, Cartosat-2S + RISAT in miniature. It also carries **PAN**, and
Cartosat-2S is commonly panchromatic (C15), so it is pan-only training data too.

C28's original claim — *"no high-resolution optical–SAR paired training data
available at any price"* — was **wrong**, corrected by C31 and confirmed twice
over.

---

## 13. Files

| | |
|---|---|
| tool | `satquery/tools/lulc.py` — `LulcClassifierTool`, `CLASSES`, `_S1_ORDER`, `_S1_STATS` |
| scorer | `scripts/eval_bifold.py` — probe / score / bands / duet modes |
| manifest | `configs/tools/lulc_classifier.yaml` — `required_modalities: [sar]` |
| deterministic SAR | `satquery/tools/deterministic.py` — `SarBackscatterTool`, `assert_decibels()` |
| D1 | `satquery/fusion/fusion.py` — `DISAGREEMENT_RULES` |
| corpus generator | `gen_crossmodal_qa` |
| staging probe | `scripts/stage_reben_probe.py` — inspect / stage / verify |
| dtype tests | `tests/test_lulc_dtype.py` |
| routing tests | `tests/test_lulc_routing.py` |

Modal entrypoints: `bifold`, `reben_*`, `gen_optsar`.

---

## 14. Open items

- **SpaceNet 6 `SAR-Intensity` was never fetched** — 41 GB, our only sub-metre
  co-registered optical+SAR. `eval/sn6` was staged as
  `modalities: ["PS-RGB", "geojson_buildings"]` only. **The SAR path has
  therefore never run on real SAR imagery** — only on a synthetic VV/VH scene.
- **Cross-region transfer never tested.** The cheap decisive experiment: train on
  SN6 (Rotterdam), hold out OEM-SAR's regions (Japan, France, USA). If it does
  not survive Rotterdam→Japan it will not survive Rotterdam→India.
- **Every training source is Western; the hidden set is Indian.** Documented in
  our own code, unclosable with public data. The mitigation is measuring
  transfer, not pretending it is closed.
- **The fused 12-channel path is at chance and unexplained** — worthless anyway
  per published AP, but the cause was never found.
- **After the deadline:** an OEM-SAR fine-tune warm-started from BIFOLD — ~700
  hand-annotated images, 8 classes, aimed squarely at X-band sub-metre, after
  solving the dB reverse-calibration. The one thing nothing else in our stack
  covers.
