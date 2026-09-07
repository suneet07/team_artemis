# 1. The problem, and the three constraints that shaped every decision

## 1.1 What was asked

SIH26167, ISRO / Space Applications Centre: a system that takes satellite
imagery and a natural-language question, and answers it — across single images,
image pairs over time, and optical/SAR pairs of the same place. The problem
statement's own first representative query is:

> *"Describe the land-cover and major objects visible in this image."*

The graded capabilities, and where each ended up:

| gate | capability | ships as |
|---|---|---|
| G1/G2 | single-image VQA | `rs_vqa` — trained LoRA |
| G3a | referring grounding | base model + a prompt |
| G3b | captioning | base model + a prompt |
| G4 | change VQA / description | `change_vqa` — trained LoRA |
| G5 | optical–SAR | BIFOLD classifier + decision-level fusion |
| G6 | routing and orchestration | rules-first router |

Two of four planned adapters were never trained. Both times that was decided by
measurement rather than by budget, and both are argued in full later
([05](05-grounding.md), [07](07-sar-and-fusion.md)).

## 1.2 Constraint one: the graded set is hidden, and it is not ours

Scoring happens on **Cartosat-2S optical and RISAT SAR** — Indian sensors,
never released to us. Every training source available is Western: Sentinel-2,
Sentinel-1, WorldView, Planet, GoogleEarth-derived aerial.

This cannot be closed with public data. It can only be *measured*, so the
project measures transfer instead of pretending the gap is shut:

- **Scale conditioning.** Every prompt carries the scene's ground sample
  distance, so the model is told it is looking at 10 m pixels or 0.3 m pixels
  rather than inferring it. This is `scale_prefix`, and it is the same
  implementation in training, evaluation and serving — one function, not three.
- **Three spectral composites** rather than RGB, so the model sees NIR and SWIR
  where the sensor provides them.
- **Domain-transfer honesty.** The RSVQA paper's own test1→test2 drop is
  **5.6 points**; that is the yardstick any claim about transfer is held to.

The one structural match to the deployment sensors — SpaceNet 6, Capella X-band
SAR co-registered with Maxar optical at 0.5 m — is staged only on its optical
half. Its `SAR-Intensity` subset is a 41 GB fetch that was never made. That
remains the single largest open item ([13](13-limitations.md)).

## 1.3 Constraint two: "codes and models" is a deliverable

The system is handed over. That makes **licence a hard functional requirement**,
not paperwork — and it disqualified more good data than any technical problem
did.

The governing rule, written early and never relaxed:

> A permissive badge on a repository says nothing about the imagery underneath
> it.

That single sentence rejected:

| source | why |
|---|---|
| **DIOR · FAIR1M · NWPU-Captions · RSICD** | all GoogleEarth-derived; no shippable licence on the imagery |
| **LHRS-Bot** | Apache-2.0 *code*, weights trained on the imagery above — permissive wrapper, encumbered weights |
| **Open-CD** | Apache-2.0, but published weights come from LEVIR-CD (academic-only) or S2Looking (no stated licence anywhere) |
| **LLaVA-Instruct-150K** | OpenAI output; terms forbid developing competing models |
| **SkyScript** (ChatGPT-smoothed) | same defect, one step removed |
| **Gemini-generated captions** | Google API terms, identical class — and we had never measured that Gemini is even good at 0.3 m aerial imagery, only that Qwen is not |
| **SARChat** | non-commercial |
| **LS-SSDD** | staged-then-withdrawn; recorded as NOT STAGED in `CREDITS.md` |

The trade was stated once and then simply obeyed:

> **A licence-clean 80% beats an unusable 91%.**

That is not rhetoric. It was the actual decision on change detection: published
Siamese networks reach F1 0.91 on LEVIR-CD at 0.5 m, and we trained on
SpaceNet 7 at 4 m instead, expecting less, because LEVIR-CD is academic-only.

One rejection was later judged too harsh and is recorded as such: a 20,000-pair
0.5 m dataset was excluded purely because its GitHub repository had no licence
file — the weakest evidence in the whole assessment.

## 1.4 Constraint three: one base model, several adapters

Section C1 of the plan commits to a **single shared base** with LoRA adapters
swapped over it. This is a serving constraint that reached back into training:
every adapter must be a LoRA on `Qwen3-VL-4B-Instruct`, because three separate
4B bases do not fit on one L4 — a fact learned the hard way when they were
loaded per-adapter and the GPU ran out of memory mid-query
([11](11-failure-atlas.md)).

Why 4B is enough is settled by the literature rather than by hope: the BEN.txt
paper's own **RS-InternVL is 1B** and beats every zero-shot model in its table
after fine-tuning — including a 2T-parameter GPT. The task does not need scale.
It needs vocabulary and in-domain supervision.

## 1.5 What "agentic" had to mean here

Not a model with tools bolted on. The problem statement grades orchestration
directly (G6), so the system had to be able to say *why* it answered the way it
did:

- a **rules-first router** — 15 of 16 documented cases resolve without an LLM,
  so the routing decision is auditable rather than sampled;
- a **parameter gate** that refuses a tool whose manifest the inputs cannot
  satisfy, before it runs;
- a **trace** carrying every planned step, every executed step, the thresholds
  chosen and why, and the co-registration error in pixels;
- **deterministic evidence** — masks, areas, boxes — written as files that can
  be opened in QGIS, not described in prose.

The full design is [08](08-orchestration.md). The short version is that the
router is the component the rest of the system is arranged around, and it was
also the component with the most undetected bugs — because every published
number was produced by calling a component *directly*, and nothing exercised the
path a user actually takes until very late ([11](11-failure-atlas.md)).
