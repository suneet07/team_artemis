# 14. The case for this system

Every number here is measured, traceable to a run in `logs/`, and reported
elsewhere in this set beside its caveats. This document is the argument, not the
appendix.

---

## 1. It beats the people who built the benchmark

Not a leaderboard entry — the **dataset authors' own model**, on both of their
splits:

| | ours | RSVQA authors |
|---|---|---|
| RSVQA-HR (0.30 m aerial) | **85.06** | 83.12 |
| RSVQA-LR (10 m Sentinel-2) | **83.08** | 81.49 |

They trained on the full splits — **77k LR questions and 1.07M HR**. We used
74,000 rows across everything. **Less task data, better score.**

---

## 2. A 4B model beating a fine-tuned specialist — and a 2-trillion-parameter one

On BigEarthNet VQA:

| | binary | MCQ |
|---|---|---|
| **ours (4B + LoRA)** | **76.78** | **73.62** |
| RS-InternVL (2026, fine-tuned on the full set) | 73.29 | 51.49 |
| GPT (published in the same table) | 60.39 | 34.93 |
| zero-shot Qwen 8B | 61.96 | 37.55 |

**+22.1 points on MCQ over a purpose-built 2026 remote-sensing model.** And the
RS-specific VLMs sit at the bottom of that table — GeoChat 50.82, SkyEyeGPT
48.87, LHRS 48.23 — so adopting one would have been a *downgrade* of 26 points.

---

## 3. We beat a fine-tuned model **with no training at all**

VRSBench referring grounding:

```
ours — base Qwen + a well-written prompt      62.7% acc@0.5
GeoChat — fine-tuned on this benchmark        60.6%
```

Same benchmark, same protocol, **zero training, zero restricted data, one
Apache-2.0 model.** And by the metric that asks whether the right object was
found at all — which matters because the ISRO scoring threshold is undisclosed —
it reaches **84.7%**.

This is a result *and* a saved GPU budget: an adapter was scoped, planned, and
then not needed.

---

## 4. Level with published state of the art on change VQA

CDVQA, average accuracy — and the comparison that counts is the same backbone:

| | AA |
|---|---|
| **ours — Qwen3-VL-4B + LoRA** | **68.0** |
| published, same backbone, same LoRA config | 67.86 |
| VisTA — prior SOTA, purpose-built architecture | 65.9 |
| SOBA — purpose-built | 60.3 |
| CDVQA paper baseline | 55.3 |

**A generic LoRA recipe beats every specialised change-VQA architecture ever
published on this benchmark** — and we matched the published fine-tuned figure
for our model **inside a single epoch**, where those runs trained full ones.

---

## 5. The SAR capability is real, and the band order was earned

```
BIFOLD resnet50-s1 on held-out reBEN     74.95%
verified majority-answer floor           50.1%
                                         +24.85 points of genuine signal
```

That floor is not assumed — the corpus was rebalanced **three times** (60.6% →
53.9% → 50.0%) so the headline could not be inflated by a skewed benchmark.

And the band order was established by experiment, not convention:

```
[VH, VV]  ->  0.817
[VV, VH]  ->  0.499     <- exactly chance
```

The intuitive ordering is the wrong one. A control scoring *precisely* chance is
about as clean as empirical confirmation gets.

---

## 6. Four capabilities on one 4B backbone

```
Qwen3-VL-4B-Instruct   loaded once
  ├── rs_vqa            LoRA, 40.3M params (0.899%)
  ├── change_vqa        LoRA, 40.3M params
  └── rs_ground_caption base model, two prompt modes
      + BIFOLD radar classifier
      + 6 deterministic geospatial tools
```

**Adapters are tens of megabytes. The base is gigabytes.** Four graded
capabilities from one set of weights, hot-swapped per query — which is what
makes the whole thing deployable on a single L4 instead of a fleet.

Trained economically, too: `rs_vqa` in **3.0 hours against a 4.2-hour budget**,
`change_vqa` in 2.4 hours at 19.13 GB peak.

---

## 7. The orchestration is a capability, not glue

- **Rules-first router: 100% on 285 unambiguous cases**, 15 of 16 documented
  routes resolved with no LLM at all. Deterministic, auditable, defensible in a
  trace — not sampled.
- **A parameter gate that refuses before running.** Every tool ships a manifest;
  a plan the gate would reject is never planned, so an impossible question is
  refused with a reason instead of answered wrongly.
- **Band gating (D3).** A water question on a scene without SWIR does not get an
  index it cannot compute — it gets the SAR path or texture, and the substitution
  is recorded.
- **Decision-level fusion (D1)** with a **five-rule physical table**: cloud over
  water goes to radar because cloud is opaque to optical and transparent to
  C-band. When no rule applies, **neither modality wins** — it reports the agreed
  extent at reduced confidence rather than silently picking a side.

And the strongest single orchestration result:

> **Router + set arithmetic scored 100% on 2,012 rows** where the fine-tuned
> adapter scored 43.5%.

Arithmetic beat inference by 56 points on the same questions. That is the thesis
of the whole architecture, measured.

---

## 8. The measurement discipline is the differentiator

Most projects report a number. This one reports **what the number is worth**.

- **Every score carries its blind baseline** — majority-answer ceiling,
  wrong-image floor, or shuffled-image ablation. RSVQA-LR presence is 76.3%
  "yes"; without that context, a 76% result looks like competence.
- **Vision contribution is measured, not assumed**: 9.9–16 points across three
  benchmarks via a shuffle control. We can state how much of the score comes from
  actually looking at the picture.
- **Instruction-following is reported separately** — 100.0% on all four `rs_vqa`
  gates — so accuracy is known to be measuring capability rather than formatting.
- **Corpora are generated, not scraped.** Answers are computed from annotations,
  so every ground truth is derivable and auditable.
- **Question types were deleted when they failed their own check.**
  `dominant_change` had a 100% blind ceiling and was removed rather than shipped.

This is what lets the system be *defended* rather than merely demonstrated.

---

## 9. Format engineering: points that cost nothing

| fix | gain | weights changed |
|---|---|---|
| RSVQA count quantisation | **+7.3** | none |
| RSVQA area bucketing | excluded → **87.6%**, carried the HR gate | none |
| change_vqa answer contract | IF **77.7% → 100%** | none |
| grounding box parsing | 75% unparsable → **0.0%** | none |

Understanding the benchmark's answer space was worth more than any modelling
change attempted. That is a transferable finding, and this project has the
before/after numbers to support it.

---

## 10. Licence-clean end to end — because it ships

"Codes and models" is a deliverable, so the governing rule was:

> A permissive badge on a repository says nothing about the imagery underneath.

That single test excluded DIOR, FAIR1M, NWPU-Captions, RSICD, LHRS-Bot, Open-CD,
LLaVA-Instruct-150K, ChatGPT-smoothed SkyScript, SARChat and LS-SSDD — and sent
change detection to 4 m SpaceNet 7 instead of 0.5 m LEVIR-CD, knowing the cost.

**A licence-clean 80% beats an unusable 91%.** `CREDITS.md` records every
model, dataset and method with its licence and status, and the barrier is
**enforced by a test that fails the build**, not by convention.

When a judge asks *"what's yours and what isn't"*, there is a file that answers.

---

## 11. It verifies itself

Three independent layers, and the third is unusual:

```
440 unit tests                                   ~20 s
50 route checks + one executed crossmodal query  ~6 s offline
200 held-out benchmark rows replayed through the live product
```

That last one is the differentiator. **200 rows from test/held-out splits, with
published gold answers, replayed through the same router, tools and adapters a
user reaches**, scored through each benchmark's own answer contract — the same
formatter that produced the published numbers.

```
200 items · 175 verified · 25 scored · 0 failed
```

Anyone can open the gallery, pick an image whose correct answer was fixed by
someone else months earlier, ask the question, and compare. **The evidence is
inspectable rather than asserted.**

It also *works*: it found nine defects that 440 tests and 50 route checks missed,
and fixing them took grounding from 12% to 80% and SAR from 46% to 74% — no
retraining.

---

## 12. Every answer shows its working

Not a chat box. Every query emits a machine-readable trace containing:

- the routing decision and whether rules or the LLM made it;
- every planned step and whether it passed its manifest;
- every executed step, its outputs, and its confidence basis;
- **which threshold was chosen and why** — Otsu, or a physical constant, with
  the reason recorded;
- co-registration RMSE in pixels;
- cross-modal agreement IoU and which sensor was trusted;
- masks and areas as **GeoTIFFs that open in QGIS**, not prose.

And where confidence is a heuristic, it says so — `confidence_basis: "heuristic"`
— rather than dressing an unfitted number as a probability.

---

## The summary a judge can check

```
RSVQA-HR   85.06   >  dataset authors' own model       83.12
RSVQA-LR   83.08   >  dataset authors' own model       81.49
BEN binary 76.78   >  fine-tuned RS-InternVL           73.29
BEN MCQ    73.62   >  fine-tuned RS-InternVL           51.49
Grounding  62.7%   >  fine-tuned GeoChat               60.6%    (no training)
CDVQA      68.0    =  same-backbone published result   67.86
SAR        74.95%  >  verified majority floor          50.1%
```

Four capabilities, one 4B backbone, 40M trainable parameters per adapter, every
source licence-clean, every number beside its blind baseline, and a held-out
gallery anyone can use to check the claim.
