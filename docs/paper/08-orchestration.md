# 8. The agent — router, planner, gate, executor, composer

Orchestration is graded directly (G6), so this is a deliverable rather than
plumbing. It is also the component that carried the most undetected bugs, for a
reason worth stating up front: **every published number in this project was
produced by calling a component directly.** Nothing exercised the path a user
actually takes until the held-out gallery was built, months later
([11](11-failure-atlas.md)).

## 8.1 Design commitment: rules first, LLM last

The router is deterministic. Measured on `phase0_item11_router`: **100% on 285
unambiguous cases**, and **15 of 16** documented routing cases resolve without an
LLM at all.

There is exactly one LLM tie-breaker — a question carrying both caption and
question phrasing (*"Describe what you see — how many buildings?"*).

The reasoning against a larger LLM role is in the problem statement itself:

> *"Internal reasoning text is neither required nor evaluated."*

An LLM rewriting step is pure added risk and latency against a sub-20-second
target, and it makes the routing decision sampled rather than auditable. The
trace has to defend the decision; a deterministic rule can be defended.

## 8.2 The pipeline

```
question + scenes
   |
   route()          task + targets, from words and input shape
   |
   plan_for()       smallest sufficient tool plan (D3 band gating applies here)
   |
   check_plan()     parameter gate: manifest, band inventory, modalities
   |
   execute_plan()   dependency waves, parallel within a wave
   |
   D1 fusion        when an optical and a SAR mask both exist
   |
   compose()        the answer
   |
   TraceBuilder     every step, threshold, and reason
```

### Task selection — precedence *is* the algorithm

Checked top to bottom, first match wins:

```
1. cross-modal?    optical + SAR present            -> CROSSMODAL_*
2. temporal?       2 same-modality images, or
                   change words + >=2 images        -> CHANGE_*
3. change words, one image, not a referring expr    -> CHANGE_VQA, empty plan (refusal)
4. grounding words, mask words, or a referring expr -> SINGLE_GROUNDING
5. caption words, not ending in "?"                 -> SINGLE_CAPTION
6. caption words AND ending in "?"                  -> SINGLE_VQA (the one LLM tie-break)
7. everything else                                  -> SINGLE_VQA
```

**Inputs beat words at the top; words decide at the bottom.** A co-registered
optical+SAR pair is unambiguous on input type alone, so no phrasing can route it
elsewhere. By step 4 there is one image and the text is all there is.

Full worked examples for all eight branches:
[`docs/segments/06-routing.md`](../segments/06-routing.md).

## 8.3 The parameter gate — refusing before running

Every tool ships a YAML manifest declaring `required_modalities`, permitted
parameters with ranges, and declared outputs. `check_plan` validates the plan
against those manifests *and* against the scene's band inventory before anything
executes.

The design rule this enforces:

> **Planning a step the parameter gate would refuse yields a failed trace rather
> than an answer.**

So the planner and the gate must agree by construction. `add_lulc()` checks
`has_sar` before planning `lulc_classifier`, because the manifest says
`required_modalities: [sar]` and a plan the gate rejects is worse than no plan —
the executor records the failure as a warning and answers from whatever else
ran, which looks healthy from outside.

That last clause is not hypothetical. It is exactly how BIFOLD came to score
74.95% standalone and **run on zero deployed requests** without anything
reporting a problem.

D3 lives here too: a target whose index needs SWIR on a source without SWIR gets
`texture_seg` instead of `spectral_index`, and the substitution is recorded as a
routing note rather than silently made.

## 8.4 The executor

Steps are grouped into **dependency waves** and run in parallel within a wave.
A dependency absent from the plan does not block — waiting for a tool the gate
dropped would hang forever — and a genuine cycle raises rather than running
partially.

One behaviour cost hours of debugging and is now called out in the code: **a tool
that raises is appended to `failures` and `warnings` but *not* to
`execution.steps`.** It vanishes from the trace. A crashing tool and a tool that
was never planned look identical from outside, which is why `lulc_classifier`'s
absence was invisible until someone read the raw warnings.

## 8.5 The composer — whose voice answers

The rule, arrived at late and argued in full:

> **When a learned adapter answered, its answer is the whole answer.**

The benchmark numbers — RSVQA-HR 85.06, CDVQA AA 68.0 — were all measured with
the adapter's own text as the entire answer string. The composer used to append
deterministic sentences to it, producing:

```
"yes. Smooth covers 67.6% of the valid pixels."
```

Three problems, in ascending order: it is not the string that was scored; the
reader cannot tell which half the model said; and the halves can contradict. The
last is not theoretical — on one chip the adapter correctly answered that there
was no water while an appended MNDWI sentence reported **68.6% water**, in the
same paragraph, in the model's voice.

Deterministic measurements did not disappear. They remain in the trace steps and
the evidence panel, attributed to the tool that produced them. What changed is
*which of the two is the answer*.

When no adapter speaks — a cross-modal extraction, say — the deterministic
sentences *are* the answer, and each now names its sensor (*"By optical…"*,
*"By radar…"*), because two tools describing the same class from different
physics read as a self-contradiction otherwise.

## 8.6 The trace

Every query emits: planned steps with `within_manifest`, executed steps with
outputs, chosen thresholds **and the reason they were chosen**, co-registration
RMSE in pixels, compatibility checks, agreement/IoU when fusion ran, and
warnings.

Two honest omissions, both found and fixed late:

- the emitted trace carried only *executed* steps, so a planned-but-dropped tool
  was indistinguishable from one never planned — dropped tools now warn
  explicitly;
- confidence is labelled `confidence_basis: "heuristic"` and the system page says
  plainly that no calibrator is fitted, so it is *an ordering, not a
  probability*. Inventing a calibrated number at the tool would put a figure in
  the graded trace that nothing measured.

## 8.7 The orchestration result worth reporting

The clearest demonstration that orchestration is a capability rather than glue:

```
trained change_vqa adapter, SpaceNet 7 benchmark      AA 43.5%
router + change_stats, given ground-truth footprints  AA 100%   (2,012 rows)
```

**Set arithmetic over a mask beats a fine-tuned VLM by 56 points**, on the same
questions. It has never shipped, because nothing produces the mask well enough —
our detector reached F1 0.2931 — and that is the honest reading: the pipeline was
never the bottleneck, perception was.

It is also why counting is routed to deterministic tools wherever a mask exists.
The shuffle control showed `change_count` gains only **1.2 points** from the
image — the VLM is not using the picture to count, which matches its near-floor
accuracy and argues directly for arithmetic over inference.

## 8.8 Tool inventory

| tool | modality | what it is |
|---|---|---|
| `spectral_index` | optical | NDVI / NDWI / MNDWI / NDBI, thresholded — the real optical opinion |
| `texture_seg` | optical | local-variance fallback when no index is computable; morphological, *not* a material class |
| `sar_backscatter` | sar | sigma-nought dB thresholding |
| `lulc_classifier` | sar | BIFOLD `resnet50-s1`, 19 reBEN classes, 74.95% measured |
| `coreg_check` | any pair | mutual-information co-registration, reports RMSE in px |
| `change_stats` | pair | set arithmetic on class maps — 100% given ground truth |
| `centroid_prior` | any | mask centroid as a point prior; trace artifact, not injected into prompts |
| `rs_vqa` | optical | trained LoRA |
| `change_vqa` | pair | trained LoRA |
| `rs_ground_caption` | optical | base model, two prompt modes |
| `change_map` | pair | **absent by decision** — F1 0.2931, wrong task |
| `object_box_fallback` | optical | **removed** — blob proposals beside a learned box are worse than nothing |

## 8.9 Thresholding, and a lesson about adaptivity

Otsu is used behind a **bimodality gate**: if the histogram is not genuinely
bimodal, or the smaller class holds under 5% of valid pixels, the physical
constant is used instead and the trace says so.

A second bound was added after MNDWI chose **−0.224** on a scene with no water
and reported **68.6% water** — far below the physical MNDWI > 0 boundary for
water. The first version of that bound rejected *any* crossing, which also killed
a perfectly good NDVI cut at 0.2813 against a 0.3 boundary.

The final rule is a **tolerance, not an edge**: 0.1 on indices running −1..1.
Marginal adaptation is kept; a sign flip is rejected. Both cases are in the test
suite, because the difference between them is the whole point.
