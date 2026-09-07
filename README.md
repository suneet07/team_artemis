<div align="center">

# SatQuery AI

**Ask satellite imagery a question. Get an answer, the evidence, and the reasoning.**

`SIH26167` · ISRO / Space Applications Centre · Space Technology

[![tests](https://img.shields.io/badge/tests-440%20passing-brightgreen)]()
[![routes](https://img.shields.io/badge/route%20checks-50%2F50-brightgreen)]()
[![base](https://img.shields.io/badge/base-Qwen3--VL--4B-blue)]()
[![licence](https://img.shields.io/badge/sources-all%20cleared-blue)](CREDITS.md)

</div>

---

An agentic vision–language system for remote sensing. It answers questions about
**optical, SAR, bi-temporal and cross-modal** satellite scenes — and every answer
ships a trace showing which tool ran, on what parameters, which threshold was
chosen and *why*.

```
"Use the optical and SAR images together to identify built-up regions"

  → crossmodal_vqa
  → coreg_check          pair co-registered to 1.58 px (mutual_information)
  → lulc_classifier      Inland waters 93% · Inland wetlands 90%
  → texture_seg          by optical, builtup covers 19.0% (surface texture)
  → sar_backscatter      by radar, builtup covers 0.3%
  → D1 fusion            IoU 0.01 — no physical rule explains this disagreement;
                         reporting the agreed extent, confidence × 0.6
```

That last line is the point: **the system says when it does not know.**

---

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
| Captioning | VRSBench | 25.2 ROUGE-L | 24.4 | LLaVA-1.5 (fine-tuned) **36.9** |

**Grounding beat a fine-tuned baseline with no training at all.** Captioning is
our one loss and is reported as such.

> Scored with the in-repo comparator, not the official scorers — stated on every
> report. See [`docs/paper/09-evaluation-method.md`](docs/paper/09-evaluation-method.md).

---

## How it works

```
question + imagery (GeoTIFF: optical / SAR / pair)
                    │
          ROUTER  (rules, not an LLM)         100% on 285 cases
          8 task branches                     15 of 16 need no LLM
                    │
          PARAMETER GATE                      refuses before running
          manifest + band inventory           a plan the gate would reject
                    │                         is never planned
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
DETERMINISTIC   LEARNED        RADAR
spectral_index  rs_vqa         lulc_classifier
texture_seg     change_vqa     (BIFOLD, 74.95%)
sar_backscatter rs_ground_caption
coreg_check     (base + prompt)
change_stats
     └──────────────┼──────────────┘
                    ▼
          D1 DECISION FUSION                  5 physical rules
          optical vs SAR                      no rule → agreed extent only
                    ▼
     ANSWER + EVIDENCE (GeoTIFF) + TRACE
```

**One 4B backbone. Two LoRA adapters at 40.3M parameters each (0.899%). Four
capabilities. One L4 GPU.**

---

## Quick start

```bash
# install
pip install -e .

# the whole suite -- 440 tests, ~20 s, no GPU
python -m pytest -q

# every task branch + one executed crossmodal query -- ~6 s, no GPU, no server
python scripts/verify_routes.py --offline-only --stub-adapters

# the console
cd frontend && npm install && npm run dev
```

Ask a question headlessly:

```bash
python -m satquery.evalcli \
  --images path/to/scene.tif \
  --question "Is there water in this image?" \
  --evidence
```

---

## Layout

```
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

docs/
├── paper/              the full account: method, results, failure atlas, limits
├── segments/           per-capability engineering record
├── ppt/                SIH idea-submission deck, one file per slide
├── 01-plan/            master plan
├── 02-data/            dataset specs, question generation
└── 03-compute/         cost model, Modal runbook

configs/                preprocessing.yaml · per-tool manifests · trace schema
tests/                  440 tests, incl. the licence blocklist gate
logs/                   curated measurement reports -- every number traces here
```

---

## Documentation

| you want | read |
|---|---|
| **the whole story, properly** | [`docs/paper/`](docs/paper/) — 15 files |
| the headline argument | [`docs/paper/14-the-case.md`](docs/paper/14-the-case.md) |
| every number with its baseline | [`docs/paper/12-results.md`](docs/paper/12-results.md) |
| **what we cannot claim** | [`docs/paper/13-limitations.md`](docs/paper/13-limitations.md) |
| nine bugs and what each proved | [`docs/paper/11-failure-atlas.md`](docs/paper/11-failure-atlas.md) |
| which question routes where | [`docs/segments/06-routing.md`](docs/segments/06-routing.md) |
| *"what's yours and what isn't?"* | [`CREDITS.md`](CREDITS.md) |
| onboarding as a teammate | [`TEAM_CONTEXT.md`](TEAM_CONTEXT.md) |

---

## How it is verified

Three layers, because the first two are not enough:

| layer | cost | catches |
|---|---|---|
| **440 unit tests** | ~20 s | logic, contracts, formatters |
| **50 route checks** | ~6 s offline | every task reachable, every planned tool executable, one crossmodal query *executed* |
| **200-row held-out replay** | GPU | answer quality through the router, on real imagery, against published gold |

That third layer found **nine defects the first two missed** — including a radar
classifier that scored 74.95% in the lab and had never run on a single deployed
request. Fixing the routing moved grounding **12% → 80%** and SAR **46% → 74%**,
with no retraining.

**Component benchmarks do not measure a system.**

---

## Ground rules

Each of these has been violated at least once, and each cost real time.

1. **Never quote a score without its blind baseline.** RSVQA presence is 76.3%
   "yes" — a model that never opens the image scores that.
2. **Trace a dataset's provenance to the imagery programme**, not to the paper
   that published it. A permissive badge on a repo says nothing about the pixels.
3. **Manifest first.** Annotations → manifest → only the patches it names.
4. **Train on exactly what deployment sees.** Parity is asserted by test, not
   assumed — served views are checked pixel-for-pixel against training views.
5. **A falling loss is not evidence of training.** Check the opening loss against
   `ln(vocab)`. Check your images are not black.
6. **Treat every ✅ as a claim to re-verify**, not as authority.

---

## Licence

Every model, dataset, library and method is recorded in
[`CREDITS.md`](CREDITS.md) with its citation, licence and status — and the
barrier is **enforced by `tests/test_license_blocklist.py`, which fails the
build**, not by convention.

The backbone is Apache 2.0. Nothing encumbered enters the shipped weights.
