# G3 grounding — what we measured, and why we are not fine-tuning it

**Decision: ship the base model with a written prompt (`qwen_precise`). No LoRA
for the grounding half of `rs_ground_caption`.** Measured 2026-09-05.

This reverses the plan's assumption that G3 grounding needs a trained adapter.
It does not reverse anything about the **captioning** half, which still needs
one — see [Captioning is untouched](#captioning-is-untouched).

---

## The headline

`Qwen3-VL-4B-Instruct`, **no fine-tuning**, scores **62.7% acc@0.5** on VRSBench
referring. Published fine-tuned GeoChat scores **60.6%** on the same split.

Everything below was measured on **150 questions drawn at random** from
`VRSBench_EVAL_referring.json`, with **every variant scored on the identical
rows** (`/data/eval/pipeline_sample.json`, seed 7). Paired rows matter more than
sample size here: a 0.7-point gap at n=150 is one question, and only a paired
comparison makes small differences readable at all.

## Every approach tried

| approach | acc@0.5 | vs plain | acc@0.7 | found the object |
|---|---|---|---|---|
| Model + SAM trim | **64.7%** | +4.0 | 38.0% | 83.3% |
| Model + SAM, 3 masks | 64.0% | +3.3 | 38.7% | 83.3% |
| Precise prompt + SAM | 63.3% | +2.6 | 38.7% | 83.3% |
| **Precise prompt** ← shipped | 62.7% | +2.0 | **42.0%** | 84.7% |
| Roomy prompt + SAM | 62.0% | +1.3 | 39.3% | **85.3%** |
| Model alone | 60.7% | — | 40.7% | 84.0% |
| Zoom in and re-ask | 58.7% | −2.0 | **44.7%** | 81.3% |
| Point + SAM | 53.3% | −7.3 | 33.3% | 84.0% |
| Grounding DINO + pick + trim | 36.0% | −24.7 | 22.7% | 68.0% |
| Grounding DINO + model picks | 34.7% | −26.0 | 24.0% | 68.7% |
| Grounding DINO alone | 21.3% | −39.3 | 14.0% | 60.0% |

**Of seven ideas tested against plain prompting, three helped, one was neutral,
three actively hurt.** Several that sounded compelling in argument lost badly.

## Why `qwen_precise` rather than the top score

`qwen_sam` scores 2 points higher at IoU 0.5. We ship `qwen_precise` anyway:

1. **It costs nothing.** A prompt string versus loading SAM and running a second
   model on every query.
2. **It is the only variant better than baseline at *both* thresholds.** Every
   SAM variant buys acc@0.5 by giving back acc@0.7 — it trims past the
   annotation's real extent. The ISRO hidden set's threshold is undisclosed, so
   a change tuned to one cut-off is a bet on something we do not know.
3. **One fewer model in the execution trace**, which the PS grades directly.

Head-to-head against the nearest rival (`qwen_generous_sam`): net +1 row at
IoU 0.5, +4 at 0.7, −1 on "found the object". Statistically a tie; decided on
cost.

## The prompt, and why it says what it says

Derived by measuring the annotations, not by intuition. On 150 rows of this
split:

* the reference box is the **axis-aligned bounds of DOTA's rotated
  quadrilateral** — 99/150 agree within 0.02, median gap 0.010;
* **42/150 targets are elongated** (aspect ratio >2 or <0.5) — diagonal piers,
  bridges, ships, where the enclosing rectangle is much larger than the object;
* the median target covers **2.5% of the image**; 55/150 are under 1%;
* the base model draws **too big rather than too small by 61 to 32**.

Hence the instruction pushes *tight*, and names the one case where tight still
means large. It lives in `scripts/eval_grounding_pipelines.py` as
`PRECISE_PROMPT`.

## The metric finding — read this before optimising anything

**acc@0.5 understates the model by 22 points.** Of 150 questions under
`qwen_precise`:

```
 94   scored correct
 33   marked wrong, but pointing at the right object
 23   genuinely looking somewhere else
```

The model **locates the correct object 84.7% of the time**. IoU at a threshold
counts loose draughtsmanship as blindness. We added a "found the object" metric —
the prediction's centre inside the true box, or the true centre inside the
prediction; the *pointing game* from weakly-supervised localisation — and it
should be reported alongside acc@0.5 permanently.

Two consequences:

* **Only 23 rows are genuine failures.** Effort spent on box precision is
  chasing 22 points that three separate tightening tricks failed to convert;
  effort spent on the 23 is chasing something real.
* **For the product, "found it" may be the truer metric.** The PS asks for
  *visual evidence* returned to the user, who cares whether the right object is
  highlighted, not whether the rectangle scored 0.52 or 0.48.

## Why the detector stack lost

Grounding DINO proposes candidates, the VLM picks one by number
(Set-of-Mark). It measured **34.7%** against the base model's 60.7%.

* **Its ceiling was never high enough.** Recall of the correct box among the
  detector's top-N was 62.5% frequency-weighted — and that is a *ceiling*
  assuming a perfect selector.
* **The selector works, and it is not enough.** 21.3% (top confidence, no
  language) → 34.7% (VLM picks) shows the model genuinely understands "the large
  vehicle at the bottom-left". It recovers only about half the available ceiling.
* **It is wildly uneven by class** — 100% recall on aircraft, 8% on tennis
  courts. Trained on ground-level photographs, it has no idea what a tennis court
  looks like from above. The base VLM is far more uniform, which is a better
  property for an unseen hidden set.

## Coordinate conventions — three bugs, same shape

This project has now hit the same class of bug three times. **Never infer a
coordinate scale from magnitude, and always print raw model output before
trusting a score.**

| where | convention | the bug |
|---|---|---|
| VRSBench references | `{<25><40><33><60>}`, **0–100** | parser defaulted to ÷1000; every reference would have collapsed into the top-left corner and every grounding score read near zero |
| Qwen predictions | JSON, **0–1000** | normalised by image size instead; coordinates above 512 clamped to 1.0 and collapsed the box. Reported **75% "unparsable"** — indistinguishable from a model that cannot ground |
| our generator | 0–1 floats | — |

Plus two identifier-digit bugs: a bare number scan takes the **`2` out of
`bbox_2d`** (and `point_2d`) as the first coordinate and shifts everything one
place. The second occurrence scored **0.0%** and looked like total model failure.
`satquery.evalcli.formatter._BOX_NUMBER` now guards against digits welded to
letters or underscores, and **should be imported rather than re-derived**.

## Sampling — stratified is for diagnosis, never for a score

VRSBench referring is **28% `vehicle`**. A class-balanced draw gives it 4% and
over-weights classes with a handful of rows.

* For **Grounding DINO** this mattered a lot — 65.3% stratified vs **62.5%**
  frequency-weighted, because its per-class performance is wildly uneven.
* For **Qwen** it barely mattered — 61.3% stratified vs 60.7% random — because
  it is uniform across classes.

`--sampling random` is the default and the only mode comparable to a published
number. `--sampling stratified` exists for per-class diagnosis on small samples.

## Captioning is untouched

Everything here concerns **grounding**. `rs_ground_caption` is two tasks, and a
detector cannot caption at all. `VRSBench_EVAL_Cap.json` is staged and graded,
our corpus contains **zero captions**, and BEN.txt's **19,983** are the only
source we hold. That half still needs a trained adapter and is unmeasured.

## Open questions

1. **Does fine-tuning beat 62.7%?** Untested. The corpus covers 9% of VRSBench
   referring classes, and narrowing a model that already grounds 26 classes to 3
   risks losing the other 23. There is now a number to beat.
2. **What threshold does the ISRO set use?** Undisclosed. Decides whether
   `qwen_precise` (better at 0.7) or `qwen_sam` (better at 0.5) is right, and
   whether "found the object" is closer to the truth than either.
3. **The 23 genuine failures.** Unexamined. That is where real headroom is.
4. **Official scorer.** Every number here uses the in-repo comparator, which the
   master plan flags as a P0 gap (C8). Not quotable as a benchmark result.

## Reproducing

```bash
modal run scripts/modal_phase0.py::pipeline_bakeoff --pipeline qwen_precise --samples 150
```

Shared row list at `/data/eval/pipeline_sample.json`; delete it to draw a new
sample, keep it to stay comparable. Per-variant reports land in
`logs/pipeline_<variant>.json` with every prediction retained, so metrics can be
recomputed without re-running the GPU.

Viewers published 2026-09-05: the eleven-way comparison, the base model's own
answers, and the Grounding DINO recall study.
