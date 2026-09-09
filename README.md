<div align="center">

# SatQuery AI

**Ask satellite imagery a question. Get an answer, the evidence, and the reasoning.**

`SIH26167` · ISRO / Space Applications Centre · Space Technology

[![tests](https://img.shields.io/badge/tests-451%20passing-brightgreen)]()
[![routes](https://img.shields.io/badge/route%20checks-50%2F50-brightgreen)]()
[![base](https://img.shields.io/badge/base-Qwen3--VL--4B-blue)]()
[![licence](https://img.shields.io/badge/sources-all%20cleared-blue)](CREDITS.md)

</div>

---

## 1. Project Information

| | |
|---|---|
| **Project Title** | SatQuery AI — agentic vision–language interpretation of remote-sensing imagery |
| **PS ID** | SIH26167 |
| **PS Title** | Vision-Language Model for Remote Sensing Imagery Interpretation and Analysis |
| **Category** | Software |
| **Theme** | Space Technology |
| **Organisation** | ISRO / Space Applications Centre (SAC) |
| **Team Name** | *(fill from portal — must match registration character for character)* |
| **Team ID** | *(fill from portal)* |

## 2. Problem Statement

ISRO's archives grow faster than anyone can read them. Answering a single
question about one scene is a manual GIS session: open the file, choose the
bands, compute an index, pick a threshold by eye, write up what was seen.
**The scene is not the bottleneck — the session is**, and it requires a
remote-sensing specialist to run.

A general vision–language model does not solve this, for three measurable
reasons:

- **It has no physical scale.** A 10 m Sentinel-2 pixel and a 0.3 m aerial pixel
  are the same pixel to a general model.
- **It answers from dataset priors, not pixels.** RSVQA presence questions are
  **76.3% "yes"** — a model that never opens the image scores that.
- **Its reasoning is unverifiable.** An analyst cannot act on "15 buildings"
  without knowing the threshold that produced it.

The problem statement itself is explicit that *"a generic LLM or VLM without
remote-sensing adaptation will not satisfy the requirements."*

## 3. Proposed Solution

**One question. One 4B backbone. Four capabilities, one deterministic
orchestrator. Every answer traceable.**

The user uploads optical, SAR, bi-temporal or cross-modal GeoTIFFs and asks a
question in plain language. A **rules-first router** — not an LLM — selects the
task, a **parameter gate** refuses any plan its tool manifests cannot support,
and a dependency-wave executor runs the tools. Learned adapters answer what only
a model can; deterministic geospatial tools answer what arithmetic can measure.
Where optical and SAR disagree, **D1 decision-level fusion** applies five
physical rules to decide which sensor to trust.

Every answer ships with the mask, the statistic, and a trace naming each tool,
its parameters, the threshold chosen and why.

Where the imagery cannot support a question, the system says so and names the
input that would answer it. A refusal is a graded output, not an error.

## 4. Key Features

| # | capability | what it does |
|---|---|---|
| 1 | **Single-image VQA** (`rs_vqa`) | Presence, counting, comparison and area questions on optical scenes, with Ground Sample Distance injected into every prompt to enforce physical scale |
| 2 | **Referring grounding** (`rs_ground_caption`) | Locates the object a sentence describes and returns a bounding box — **no training, zero additional parameters** |
| 3 | **Bi-temporal change VQA** (`change_vqa`) | What changed between two dates, on a licence-clean corpus |
| 4 | **SAR land cover** (`lulc_classifier`) | 19-class multi-label land cover from radar, which sees through cloud |
| 5 | **Cross-modal fusion** | Five physical rules reconcile optical against SAR; when no rule applies, **neither sensor wins** and confidence drops |
| 6 | **Agentic orchestration** | Rules-first router, manifest-enforced parameter gate, dependency-wave execution |
| 7 | **Full audit trace** | Every tool, parameter, threshold and decision, exportable as a report |
| 8 | **Evidence export** | Masks written as georeferenced GeoTIFF, not screenshots |
| 9 | **Honest refusal** | When the imagery cannot support the question, it says so and names the fix |
| 10 | **Held-out testing corpus** | 200 rows from public test splits, replayable live through the same router |

## 5. Technology Stack

| layer | what |
|---|---|
| **Backbone** | Qwen3-VL-4B-Instruct (Apache 2.0) — **one** model, loaded once |
| **Adapters** | 2 × LoRA, r=16 α=32, all-linear · **40,271,872 params (0.899%)** each |
| **Radar** | BIFOLD `resnet50-s1` — 19-class land cover (MIT) |
| **Deterministic** | NDVI / NDWI / MNDWI / NDBI · SAR backscatter dB · texture segmentation · co-registration · set arithmetic · centroid prior |
| **Geospatial** | rasterio · GDAL · scikit-image · AROSICS |
| **Agent** | rules-first router · parameter gate · dependency-wave executor · trace builder |
| **Serving** | Modal (L4 GPU) · PEFT multi-adapter hot-swap · FastAPI |
| **Console** | React · TypeScript · Vite · deck.gl / MapLibre · Vercel |
| **Verification** | 451 unit tests · 50 route checks · 200-row held-out replay |

**The constraint that shaped everything:** one shared base with adapters swapped
over it. Adapters are tens of MB; the base is gigabytes. **Four capabilities on
a single L4.**

## 6. Architecture

The full routing tree. **Read it in one direction: input shape decides first,
question wording second, available bands third.** Cut components are drawn where
they would have hung, with the measurement that cut them.

```
                              QUESTION  +  SCENE FILES
                    ingest: modality · bands · GSD · CRS · nodata
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │  ①  optical AND SAR?           │  ②  ≥2 same-modality images    │  ③  ONE IMAGE
        │     input shape wins —         │     (or change words + ≥2)     │     words decide
        │     no wording overrides       │                                │
        ▼                                ▼                                ▼
  ┌───────────┐                   ┌───────────┐              ┌─────────────────────────┐
  │ wants a   │                   │ wants a   │              │ locate / where is /     │
  │ mask or   │                   │ mask?     │              │ bbox / segment words,   │
  │ a number? │                   │           │              │ OR a referring          │
  └─┬───────┬─┘                   └─┬───────┬─┘              │ expression?             │
    │yes    │no                     │yes    │no              │  (not a question ·      │
    ▼       ▼                       ▼       ▼                │   ≥6 words · names a    │
 CROSS-  CROSS-                  CHANGE_  how much/ratio?    │   known object · places │
 MODAL_  MODAL_                  MAP      → CHANGE_VQA       │   it)  ← 12% → 80%      │
 EXTRAC-  VQA                    ✗ cut    describe?          └─┬────────────────────┬──┘
 TION                            F1 .29   → CHANGE_DESCR       │yes              no │
    └───┬───┘                    wrong    else                 ▼                    ▼
        │                        task     → CHANGE_VQA   SINGLE_GROUNDING    ┌──────────────┐
        ▼                            └───────┬───────┘          │            │ describe /   │
  coreg_check                                ▼                  │            │ caption      │
  [alignment error, px]              coreg_check                │            │ words?       │
        ▼                            [alignment error, px]      │            └─┬──────────┬─┘
  lulc_classifier                            ▼                  │              │yes    no │
  [19-class radar land cover]         change_vqa                │              ▼          ▼
        ▼                             [trained LoRA,            │        ends in "?"   SINGLE_
  ┌─────┴──────┐                       pair answers]            │        ┌────┴────┐    VQA
  │            │                       AA 68.0                  │      no│         │yes   │
  optical arm  radar arm                                        │        ▼         ▼      │
  │            │                    change_stats not planned    │  SINGLE_    SINGLE_VQA  │
  ▼            ▼                    — needs a class map         │  CAPTION    ⚡ THE ONLY │
 ┌──────────────────────┐             nothing produces          │      │      LLM TIE-    │
 │ D3 BAND LADDER       │             (100% on 2,012 rows       │      │      BREAK       │
 │ water  MNDWI→NDWI→   │              given ground truth)      │      │      15 of 16    │
 │        texture_seg   │                                       │      │      never reach │
 │ veg    NDVI→         │                                       │      │      it          │
 │        texture_seg   │                                       │      │         │        │
 │ built  NDBI→SAR→     │                                       │      ▼         └────┬───┘
 │        texture_seg   │                                       │  has SAR?           │
 │ none of them → REFUSE│                                       │  ├yes→ lulc_        ▼
 └──────────┬───────────┘                                       │  │     classifier  per target:
            │                                                   │  ▼        ▼        D3 ladder
            ▼                                                   │ rs_ground_caption      ▼
  sar_backscatter                                               │ (mode=caption)    lulc_classifier
  [radar brightness, dB]                                        │ [base model,       (if SAR)
            │                                                   │  no adapter]           ▼
            ▼                                                   ▼                  radar-only AND
 ┌──────────────────────────┐                     noun in vocabulary?              names a known
 │ D1 FUSION — both masks   │                     ├ yes → spectral_index /         class?
 │ exist?                   │                     │       sar_backscatter          ├yes→ STOP.
 │ IoU ≥0.60 union, conf ↑  │                     │            ▼                   │  classifier
 │ IoU <0.60 → 5 rules,     │                     │       centroid_prior           │  answers
 │   first match wins:      │                     │       [mask centre point] D2   │  alone
 │   cloud/water   → SAR    │                     │            ▼                   └no→ rs_vqa
 │   wet soil      → optical│                     └ no ──→ rs_ground_caption           [trained
 │   radar shadow  → optical│                             (mode=grounding)             LoRA,
 │   wind on water → optical│                             [base model +                single-
 │   dry sand      → optical│                              PRECISE_PROMPT,             image]
 │ no rule → INTERSECTION,  │                              no adapter]  62.7%          85.06
 │   confidence × 0.6       │                     object_box_fallback ✗ removed
 └──────────┬───────────────┘                                  │
            │                                                  │
            └──────────────────┬───────────────────────────────┘
                               ▼
              ┌────────────────────────────────────┐
              │ COMPOSE — a learned adapter spoke? │
              │   yes → its answer IS the answer   │
              │   no  → deterministic sentences,   │
              │         each naming its sensor     │
              └────────────────┬───────────────────┘
                               ▼
              ANSWER  ·  EVIDENCE (GeoTIFF masks)  ·  TRACE
              every step · threshold + why · RMSE px · IoU · warnings

  ✗ REFUSAL TERMINALS, drawn in red wherever they hang off the tree:
    change words + 1 image → "upload the second acquisition"   ·  SAR + colour
    question → "radar measures backscatter, not light"  ·  cross-modal question
    + 1 modality  ·  RGB/pan-only + "vegetation health"  ·  no band, no tool
```

**One 4B backbone. Two LoRA adapters at 40.3M parameters each (0.899%). Four
capabilities. One L4 GPU.**

## 7. Repository Structure

```text
satquery/               the package
├── agent/              router · planner · parameter gate · executor · trace
├── tools/              spectral_index · texture_seg · sar_backscatter
│                       lulc_classifier · coreg_check · change_stats · learned
├── ingest/             reader · modality · band inventory · radiometry
├── fusion/             D1 decision-level fusion + the physical rule table
├── evalcli/            headless eval + the answer-contract formatters
├── api/                FastAPI server, trace and evidence endpoints
├── training/           dataset, config, generation, collator
├── qgen/               corpus generators — questions computed from annotations
├── sar/ coreg/ tiling/ preprocessing: sigma-nought, co-registration, P4 tiling
└── report/ serving/    evidence writers, overlays, multi-adapter serving

frontend/               React + TypeScript console (Vite)
├── src/features/       workspace · chat · evidence · trace · gallery · system
├── src/map/            deck.gl / MapLibre scene + overlay rendering
└── contracts/          the frozen API contract the backend is checked against

scripts/                staging · training · evaluation · deployment
├── build_gallery.py    the 200-row held-out testing corpus
├── verify_gallery.py   replays it through the live API
├── verify_routes.py    every endpoint, every task, one executed query
└── deploy_verify.sh    tests → snapshot → deploy → verify

configs/                preprocessing.yaml · per-tool manifests · trace schema
tests/                  451 tests, incl. the licence blocklist gate
logs/                   curated measurement reports — every number traces here
training/               LoRA training entry points and evaluation harnesses
CREDITS.md              every model, dataset and method with its licence
```

### What goes where?

| Item | Location |
|---|---|
| Source code | `satquery/`, `frontend/`, `scripts/`, `training/` |
| Architecture / technical documentation | `docs/` |
| Project screenshots | `assets/screenshots/` |
| Final PPT / presentation | `submission/` |
| Demo video link | `submission/DEMO.md` |
| Project overview | `README.md` |

## 8. Final Presentation

See [`submission/PRESENTATION.md`](submission/PRESENTATION.md).

The deck is six slides, one per required heading, and is authored slide-by-slide
in Markdown before export.

## 9. Demo Video

See [`submission/DEMO.md`](submission/DEMO.md).

## 10. Screenshots / Prototype Photos

See [`assets/screenshots/`](assets/screenshots/) — console screenshots showing a
grounded box, a disagreement panel and a full execution trace.

## 11. Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd "Sat Query"
pip install -e ".[dev]"
```

CPU-only is enough for the whole test suite, the router and every deterministic
tool. A GPU is needed only to serve the learned adapters.

Frontend:

```bash
cd frontend && npm install
```

## 12. Run

```bash
# the whole suite — 451 tests, ~20 s, no GPU
python -m pytest -q

# every task branch + one executed crossmodal query — ~6 s, no GPU, no server
python scripts/verify_routes.py --offline-only --stub-adapters

# ask a question headlessly
python -m satquery.evalcli \
  --images path/to/scene.tif \
  --question "Is there water in this image?" \
  --evidence

# the console
cd frontend && npm run dev
```

Deploying the GPU backend is an operator task; the serving entry points are
`scripts/modal_phase0.py` and `scripts/deploy_verify.sh`, both of which take the
target workspace from `MODAL_PROFILE`.

## 13. Future Scope

### Who asks, and what changes for them

ISRO's archives grow faster than anyone can read them. Every question asked of a
scene today is a manual session in a GIS: open the file, choose the bands,
compute an index, pick a threshold by eye, write up what you saw. The scene is
not the bottleneck — the session is.

| who asks | the question | today | with SatQuery |
|---|---|---|---|
| **ISRO / SAC analyst** | *"What land cover is in this radar scene?"* | a SAR specialist reads it, or it stays unread | answered in seconds — archive triage at scale |
| **Disaster response** | *"What changed between these two dates?"* | hours to trace a flood boundary by hand | extent in seconds, mask exported as GeoTIFF |
| **Agriculture** | *"How much of this scene is vegetation?"* | an index computed, then judged by eye | measured coverage, with the threshold recorded |
| **Urban planning** | *"Where is the built-up area?"* | one analyst's reading, hard to re-check | reproducible, and auditable months later |
| **District officer** | plain language, no band mathematics | ask a remote-sensing specialist, and wait | ask the system directly |

```text
TODAY      open QGIS → choose bands → compute an index → set a threshold by eye → write it up
           minutes to hours per scene · you must be an expert to ask

SATQUERY   type the question
           seconds per scene · anyone can ask · threshold recorded, evidence exported, every step traced
```

**Cloud is the case nobody handles properly.** Optical sensors cannot see through
monsoon cloud. Radar can. SatQuery does not average the two into a compromise —
it decides which one to believe and states why: cloud over water goes to radar,
because cloud is opaque to visible light and transparent to C-band. That is
monsoon-season India, settled by physics rather than preference.

### Benefits

**Operational**

- **Minutes → seconds** per question
- Runs on **one L4 GPU** — deployable on-premises, no cluster, no data leaves the building
- Four capabilities from **one 4B model**; adapters are tens of MB
- **Every answer auditable**: tool, parameters, threshold and its reason, alignment error in pixels
- Masks export as **GeoTIFF straight into QGIS** — the output is usable, not a screenshot

**Economic**

- The entire system trains in **under 6 GPU-hours**
- Fine-tuning on ISRO's own archives: **~3 GPU-hours per adapter**
- **Zero licence exposure** — every source cleared and recorded, enforced by a build-failing test
- No per-query API cost, no vendor lock-in — Apache-2.0 backbone, self-hosted

**Social and environmental**

- **Faster disaster assessment** — flood and change extent in seconds
- **Crop and vegetation monitoring** at archive scale
- **Access widened** — a district officer can ask what previously needed a specialist
- **Small model, small footprint**: 4B on one L4, not a 70B on a cluster

**Strategic**

- Handed over as **code and weights**, not a hosted service
- Built to be **fine-tuned on Cartosat-2S and RISAT** the day ISRO supplies them
- No dependency on a foreign API for inference

### Where the accuracy headroom is

Every figure in this repository was reached on public data held to a commercial
licence standard. ISRO and NRSC hold Cartosat-2S optical and RISAT SAR archives
with annotations — imagery that is operationally native rather than a research
proxy, and carries no licence question at all. The properties those sensors have
are already handled in shipped, tested code: band gating for a missing SWIR
channel, a panchromatic branch, a single-polarisation SAR path with polarisation
dropout, windowed reads and a tile scorer for scenes of 10,000 pixels and up.

Corpora are generated from annotations rather than scraped, so the generator is
the transferable asset: pointed at ISRO's annotations it emits the same corpus
shape, the same composites and the same answer contracts. Fine-tuning is a
re-run, not a rebuild — about **3 GPU-hours per adapter**.

## Results

Every number measured on held-out public benchmarks, each beside the score a
system gets **without looking at the image**.

| capability | benchmark | ours | blind baseline | published comparison |
|---|---|---:|---:|---|
| Single-image VQA | RSVQA-HR | **85.06** | 62.6 | dataset authors' own model **83.12** |
| Single-image VQA | RSVQA-LR | **83.08** | 55.8 | dataset authors' own model **81.49** |
| Multi-label VQA | BEN binary | **76.78** | 52.6 | RS-InternVL (fine-tuned) 73.29 |
| Multi-label MCQ | BEN MCQ | **73.62** | 29.3 | RS-InternVL 51.49 |
| Change VQA | CDVQA Val | **68.0** | 45.0 | same backbone, fine-tuned 67.86 |
| Referring grounding | VRSBench | **62.7%** acc@0.5 | — | GeoChat (fine-tuned) 60.6% |
| SAR land cover | reBEN held-out | **74.95%** | 50.1 | — |

**Grounding beat a fine-tuned baseline with no training at all.** The full
comparison landscape — every model compared against, per benchmark — is in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

> Scored with the in-repo comparator, not the official scorers — stated on every
> report.

## How it is verified

Three layers, because the first two are not enough:

| layer | cost | catches |
|---|---|---|
| **451 unit tests** | ~20 s | logic, contracts, formatters |
| **50 route checks** | ~6 s offline | every task reachable, every planned tool executable, one crossmodal query *executed* |
| **200-row held-out replay** | GPU | answer quality through the router, on real imagery, against published gold |

That third layer found **nine defects the first two missed** — including a radar
classifier that scored 74.95% in the lab and had never run on a single deployed
request. Fixing the routing moved grounding **12% → 80%** and SAR **46% → 74%**,
with no retraining.

**Component benchmarks do not measure a system.**

## Licence

Every model, dataset, library and method is recorded in
[`CREDITS.md`](CREDITS.md) with its citation, licence and status — and the
barrier is **enforced by `tests/test_license_blocklist.py`, which fails the
build**, not by convention.

The backbone is Apache 2.0. Nothing encumbered enters the shipped weights.

---

## Important

This repository contains no passwords, API keys, access tokens or `.env` files.
Credentials are loaded from `~/.satquery/credentials.env`, **outside the
repository tree**, by `satquery/credentials.py` — which refuses to read a
credential file from inside the repo. The reasoning is in that module's
docstring: `.gitignore` only governs git, and zipping a folder for submission
takes ignored files with it.
