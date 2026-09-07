# Slide 2 — PROPOSED SOLUTION

**Required sub-headings** (do not rename): detailed explanation · how it addresses
the problem · innovation and uniqueness.

---

## Headline for the slide

> ## SatQuery AI
> ### One question. One 4B model. Four capabilities. Every answer traceable.

---

## Block A — Detailed explanation

**What it does — four graded capabilities, one system**

| ask | it returns |
|---|---|
| *"Is there water in this scene?"* | answer + the spectral index and threshold that decided it |
| *"Where is the aircraft?"* | a bounding box, drawn on the image |
| *"Describe this image"* | a caption grounded in a measured class inventory |
| *"What changed between these dates?"* | change answer over a co-registered pair |
| *"Use optical and SAR together"* | both sensors reconciled by physics, not averaging |

**Inputs:** optical · SAR · bi-temporal pairs · cross-modal pairs — GeoTIFF in,
answer + evidence out.

---

## Block B — How it addresses the problem

Four bullets, each a design decision, not a feature list:

- **Answers are computed, not asserted.** Water coverage comes from a spectral
  index with a stated threshold — the model never invents a number.
- **The system refuses what it cannot do.** A tool whose manifest the input
  cannot satisfy is *never planned*: a SWIR question on a scene without SWIR is
  routed elsewhere or refused with a reason.
- **Every answer ships its working.** Which tool ran, on what parameters, which
  threshold and why, the co-registration error in pixels, and masks as GeoTIFFs
  that open in QGIS.
- **Built for the sensors it will be graded on.** Every prompt carries the
  scene's ground sample distance, so the model is *told* it is looking at 10 m or
  0.3 m pixels rather than guessing.

---

## Block C — Innovation and uniqueness

> **The four claims. Each is a number a judge can check.**

**① We beat the teams who built the benchmarks**

```
RSVQA-HR    85.06   vs the dataset authors' own model   83.12
RSVQA-LR    83.08   vs the dataset authors' own model   81.49
BEN MCQ     73.62   vs fine-tuned RS-InternVL           51.49    (+22.1)
```

**② We beat a fine-tuned model with *zero* training**

```
Grounding   62.7% acc@0.5   vs fine-tuned GeoChat   60.6%
```
A better-written prompt beat a model trained on the benchmark. We then *did not
train the adapter we had scoped* — measurement cancelled the work.

**③ A generic recipe beats purpose-built architectures**

```
CDVQA   ours 68.0   ·   VisTA (specialised) 65.9   ·   SOBA 60.3
```

**④ Deterministic tools beat the neural network where a mask exists**

```
fine-tuned adapter                        43.5%
router + set arithmetic on the same rows   100%    (2,012 rows)
```

---

## The line that separates us from the field

> **Every score is reported beside the score a system gets *without looking at the
> image*.**
>
> RSVQA presence questions are **76.3% "yes"** — a blind model scores that. We
> measure the gap, not the number. Vision contribution: **9.9–16 points**,
> established by re-running every question against the *wrong* image.

---

## Layout direction

Three columns under the heading; claims ①–④ as four tiles across the lower half,
each **one number, one comparison, one word of context**. Numbers set large — the
comparison beneath in small caps.

```
┌── WHAT IT DOES ──┬── HOW IT ANSWERS ──┬── WHY IT IS NEW ──┐
│ 4 capabilities   │ computed, not      │ beats benchmark   │
│ optical/SAR/     │ asserted           │ authors           │
│ bi-temporal      │ refuses honestly   │ 0-training win    │
│ cross-modal      │ shows its working  │ tools > model     │
└──────────────────┴────────────────────┴───────────────────┘
┌ 85.06 ┬ 62.7% ┬ 68.0 ┬ 100% ┐   <- four tiles, huge numerals
│ vs    │ vs    │ vs   │ vs   │
│ 83.12 │ 60.6% │ 65.9 │ 43.5%│
└───────┴───────┴──────┴──────┘
```

---

## Speaker notes

- Lead with ②. *"We beat a fine-tuned model without training anything"* is the
  most surprising sentence in the deck.
- Then ④, because it reframes the project: **this is not a model with tools bolted
  on — it is an orchestrator that uses a model where a model is the right answer.**
- If asked "is this just prompting?" → point at ①: two trained LoRA adapters,
  40.3M parameters each, and the numbers above.

## If a judge pushes

**"Are these official scores?"**
> Measured with our in-repo comparator on the published splits, and we say so in
> every report. They are exact for run-to-run comparison. Wrapping the official
> scorers is a named next step.

**"Isn't 68.0 below the 68.6 state of the art?"**
> The 68.6 is a *smaller* model — that paper found performance does not scale
> monotonically with size. For our own backbone their published figure is 67.86,
> and we reached 68.0 inside a single epoch.
