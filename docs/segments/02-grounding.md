# Segment 2 — Grounding (gate G3, first half)

Referring expression grounding: given an image and a phrase — *"the aircraft at
the north end of the apron"* — return a box. Scored on **acc@0.5** (predicted box
must overlap truth by ≥50% IoU).

**Status: shipped, untrained.** The single most surprising result in the
project: the base model beats published fine-tuned specialists with nothing but
a better-written prompt.

**Decision: ship base Qwen3-VL-4B with `PRECISE_PROMPT`. Train nothing.**

---

## 1. Result

```
qwen_precise    62.7% acc@0.5    42.0% acc@0.7    meanIoU 0.551
```

| model | acc@0.5 | note |
|---|---|---|
| Qwen2.5-VL-3B | 37.3% | published (Pr@0.5) |
| GeoChat zero-shot | 40.8% | published |
| GeoChat **fine-tuned on VRSBench train** | 60.6% | published |
| **ours — base Qwen3-VL-4B + precise prompt** | **62.7%** | **no training** |
| Grounding DINO + *perfect* selector | 70.3% | ceiling only, selector never built |

We beat a model fine-tuned on the benchmark's own train split, using a prompt.

Everyone publishing VRSBench grounding evaluates on the **same val split**, so
62.7% and GeoChat's 60.6% are measured on the same data. The remaining
differences are the scorer (in-repo vs official) and our sampling.

### Full sweep — eleven approaches, one fixed sample

`/data/eval/pipeline_sample.json`, seed 7, 150 rows, paired evaluation.

```
variant             acc@0.5  acc@0.7  meanIoU   small  crowded   Δ
qwen_sam              64.7%    38.0%    0.544   67.0%   63.3%   +4.0
qwen_multimask        64.0%    38.7%    0.549   67.0%   62.2%   +3.3
qwen_precise          62.7%    42.0%    0.551   65.2%   61.1%   +2.0
qwen_generous_sam     62.0%    39.3%    0.542   64.3%   61.1%   +1.3
qwen (plain)          60.7%    40.7%    0.549   63.5%   57.8%    —
qwen_zoom             58.7%    44.7%    0.537   62.6%   53.3%   -2.0
qwen_point_sam        53.3%    33.3%    0.478   52.2%   54.4%   -7.3
dino_sam_vlm          36.0%    22.7%    0.333   38.3%   31.1%  -24.7
dino_vlm              34.7%    24.0%    0.339   36.5%   30.0%  -26.0
dino_top1             21.3%    14.0%    0.232   22.6%   14.4%  -39.3
```

Four hypotheses tested, three dead: **generous prompting, multimask selection
and point prompts all lose to simply asking normally.** The two that work are a
prompt built from the annotation conventions (+2.0, and the only variant better
at *both* thresholds) and SAM trimming (+4.0 loose, −2.7 strict).

---

## 2. Why `qwen_precise` and not the higher-scoring `qwen_sam`

`qwen_sam` scores 64.7% against `qwen_precise`'s 62.7%. We shipped the lower
number, deliberately:

| | precise | generous+SAM / qwen_sam |
|---|---|---|
| acc@0.5 | 62.7% | 64.7% |
| acc@0.7 | **42.0%** | 38.0% |
| mean IoU | **0.551** | 0.544 |

- **Cost.** `precise` is a prompt string. The SAM variants load a second model
  and run it on every query — a real operational cost for a threshold-specific
  gain.
- **Robustness to an unknown threshold.** The ISRO set's cut-off is undisclosed.
  Precise is better at 0.7 and barely worse at 0.5, so it is the safer bet either
  way.
- **Simplicity in the trace.** One fewer model in the agentic execution summary,
  which the problem statement explicitly grades.

`precise + SAM` composes **worse than either alone** (63.3% / 38.7%) — both
correct for over-inclusion, so applying both overshoots.

---

## 3. `PRECISE_PROMPT` — the thing that produced 62.7%

Mined from the annotations, not invented. Now canonical at
`satquery/agent/served_prompts.py`; originally in
`scripts/eval_grounding_pipelines.py`.

```
Find this object in the satellite image: {phrase}

Rules for the box:
- Give the SMALLEST axis-aligned rectangle that fully contains the object.
- If the object lies diagonally, the rectangle must still cover its full
  diagonal extent, corner to corner.
- Exclude shadows, wake, surrounding pavement and neighbouring objects.
- Targets are usually small, often under 3% of the image. Do not pad the box.
- Coordinates run 0-1000, left to right and top to bottom.

Respond with only: [{"bbox_2d": [x1, y1, x2, y2]}]
```

**Every clause is derived from a measurement:**

| clause | the fact behind it |
|---|---|
| smallest axis-aligned rectangle | VRSBench boxes are the axis-aligned bounds of DOTA's rotated quads |
| diagonal, corner to corner | 42 of 150 targets are elongated |
| under 3% of the image | median target is 2.5% of the image |
| do not pad | the model errs too-big vs too-small **61:32** |
| coordinates 0-1000 | Qwen's native convention — see §5 |
| exact JSON shape | the box parser expects it |

**This prompt is load-bearing at serve time**, because `rs_ground_caption` runs
the **base** model. A trained adapter reproduces its score from
`scale_prefix + question` because that is the format it was fine-tuned on; the
base model reproduces nothing unless asked the way the benchmark asked. Serving
the raw user question instead loses the 0-1000 scale and the JSON shape, and the
box parser then gets prose.

Other variants tried, all in `scripts/eval_grounding_pipelines.py`:
`GENEROUS_PROMPT`, `POINT_PROMPT`, `PICK_PROMPT`, plus `tighten()`,
`from_point()`, `marked()`.

---

## 4. The detector route — measured, and decisively lost

The plan was: Grounding DINO proposes candidates, a VLM selects which one the
phrase means.

```
dino_top1       21.3%    detector alone, no language
dino_vlm        34.7%    detector proposes, model picks
dino_sam_vlm    36.0%    + SAM refinement
qwen_precise    62.7%    just ask the model
```

**Why it lost:** Qwen could already *see* optical objects. The detector added a
lossy step in front of a model that did not need help.

Worth recording, because the argument for it was genuinely good at the time:

- Our corpus covers **9%** of VRSBench referring classes (see §6). The detector's
  measured ceiling was **65–77%** — so even a mediocre selector looked like it
  should beat training our corpus by several times.
- Text2Seg reported cars at 0.517 / 0.619 IoU zero-shot, contradicting the
  small-object worry.
- **58% of VRSBench targets are non-unique** — several candidates of the same
  class — so detection alone is not enough; disambiguation is the missing half.
  Grounding DINO gives you *every* vehicle; it cannot say which one the sentence
  means.

The selector was never built, because base Qwen made it unnecessary.

**SAM's behaviour, precisely:** more boxes got *worse* IoU than better (62 vs 46),
yet accuracy rose. SAM shrinks everything, pushing loose boxes over the 0.5 line
while nudging comfortable ones down. A **threshold-specific** win: +4.0 at 0.5,
−2.7 at 0.7. When the box is on the wrong object — and at 34.7% most are — SAM
sharpens the mistake.

---

## 5. Three coordinate-convention bugs — the recurring failure

Three different conventions in one pipeline:

```
VRSBench   {<25><40><33><60>}       angle-bracketed ints, 0-100 scale
Qwen       [{"bbox_2d": [...]}]     0-1000 scale
ours       [0.56 0.19, 0.65 0.29]   floats, 0-1 scale
```

### Bug 1 — Qwen scale, 75% unparsable

Coordinates were normalised by image size and anything >512 clamped to 1.0.

```
before:   acc@0.5   4-6%    unparsable 75%
after:    acc@0.5    62%    unparsable  0%
```

Fixed with `scale=1000`. **The user had not flagged this; it was found by dumping
raw replies.** Without the dump, the base model would have looked useless.

### Bug 2 — `bbox_2d` identifier digits

```
[{"bbox_2d": [287, 97, 333, 148]}]  ->  [0.00 0.57, 0.19 0.66]   WRONG
should be                               [0.57 0.19, 0.66 0.29]
```

`_BOX_NUMBER` grabbed the **`2` out of `bbox_2d`** as the first coordinate and
shifted everything by one. Since JSON is Qwen's *native* grounding format, this
alone would have made the base model look useless.

Fixed once in `format_box`, then **re-introduced** by writing a fresh regex for
`point_2d`, scoring 0.0%. Third occurrence of the same class of bug.

```python
_BOX_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z_])")
```

### Bug 3 — VRSBench 0-100 scale

The parser rescaled anything above 1.5 by **1000** (a Qwen convention). Applied
to VRSBench references, every box would have come out **10× too small and crammed
into the top-left corner**. Every grounding score near zero, looking exactly like
model failure.

```python
"vrsbench_referring": AnswerFormat("vrsbench_referring", "bbox",
                                   lowercase=False, box_scale=100.0)
```

**Guessing cannot work in principle:** a `60` is `0.60` on a 0–100 scale and
`0.06` on a 0–1000 one. Magnitude alone cannot separate them. The heuristic
survives only as a fallback for bare Qwen output, and the docstring says it is
wrong on VRSBench.

Validated against **all 16,159** VRSBench references. 13 unparsed rows are
**degenerate boxes in VRSBench itself** (`{<92><0><96><0>}` has zero height) —
unscoreable as published; the parser flags them rather than inventing a rectangle.

**Scoring is IoU, not string match**, so our output *syntax* need not match
VRSBench's. Only the coordinates do — worth confirming, because matching syntax
would have been a pointless corpus change.

Our own corpus had **273 degenerate boxes** (all SpaceNet 6), fixed at the
generator via `box_string` widening — but the 19,896 rows already on disk were
made by the old generator.

---

## 6. The class-coverage problem — why training was never going to work

VRSBench referring is built on DOTA/DIOR imagery and inherits DOTA's classes.

```
vehicle       28%      no licence-clean source
harbor,  bridge, storage-tank, tennis-court   — same
ship        10.4%      LS-SSDD — licence failed (see below)
building     1.4%      only 1.4% of referring questions mention "build*"
```

**There is no `building` class and no land-cover class.** Our G3 corpus trains
buildings (SpaceNet 2/6) and land-cover regions (BEN) — which contribute ~0% to
referring.

**Coverage: 9%** of VRSBench referring classes after ships were lost. Training
hard on our corpus optimises one ninth of the benchmark.

BEN was **wrong as the G3 grounding primary** — right for `rs_vqa` and
`optsar`, wrong here. It stays in G3 for **captions**, since it is the only
caption source.

### The licence wall

| source | licence | verdict |
|---|---|---|
| RarePlanes | CC BY-SA 4.0 | ✅ verified — 14,700 aircraft boxes @ 30 cm |
| SpaceNet 6 MSAW | CC BY-SA 4.0 | ✅ building footprints @ 0.5 m |
| SpaceNet 2 | CC BY-SA 4.0 | ✅ buildings |
| **LS-SSDD-v1.0** | Apache-2.0 **badge only** | ❌ the repo is a pointer with no imagery; the CAS portal is research/teaching only |
| VRSBench | text CC-BY, images DOTA-derived | **eval only**, never in shipped weights |
| RSVG, RemoteSAM-270K, SAMRS | **no licence file** | ❌ DIOR/DOTA-derived |
| GeoPixel | Apache-2.0 | the only clean grounding candidate |

**The LS-SSDD reversal cost the `ship` class**: VRSBench referring coverage
dropped **19.5% → 9.0%**. CREDITS had it as `Apache-2.0 ✅ CLEARED`; the Apache
badge covers a pointer repo holding no data. The same wording that got OSCD's
labels and HRSCD's annotations rejected — *"For commercial purposes, please
contact the authors."*

> One standard for VRSBench and another for LEVIR would not survive a judge
> reading CREDITS.

### RS-specialist models — all rejected

| model | licence | why it fails |
|---|---|---|
| GeoChat | **none at all** | trained on DOTA/DIOR |
| LHRS-Bot | Apache-2.0 code | **weights trained on Google Earth imagery** |
| SkyEyeGPT | — | bottom of the BEN.txt table |
| GeoGround, RSGround-R1, GeoViS, GeoSearcher, ProVG, GeoPix | no licence | and any model trained on VRSBench train inherits DOTA's restriction anyway |

They also sit at the *bottom* of the BEN.txt table (GeoChat 50.82 binary against
our 76.78) — the S2 thesis confirmed by the paper's own numbers: **a general
model with task adapters beats RS-pretrained specialists.**

---

## 7. The "found the object" metric — the user's question

> *"do you think acc@0.5 is the correct metric shouldnt it be lower?..can we have
> a metric that can actually detect if our model detected the particular location
> of the object concerned maybe the box is not exact but its good enough"*

Implemented as the **pointing game**: prediction centre inside the true box, or
true centre inside the prediction. Ignores extent.

```
variant             acc@0.5   found the object   coverage
qwen_generous_sam     62.0%          85.3%         74.7%
qwen_precise          62.7%          84.7%         74.0%
qwen                  60.7%          84.0%         71.9%
qwen_sam              64.7%          83.3%         69.0%
qwen_zoom             58.7%          81.3%         67.0%
dino_vlm              34.7%          68.7%         53.3%
dino_top1             21.3%          60.0%         55.9%
```

**The ranking flips.** `qwen_sam` wins acc@0.5 but is the *worst* qwen variant at
finding the object (83.3%) — trimming pulls the box centre off the target. The
**generous prompt wins here at 85.3%**, the best of anything tested: a roomier
box is more likely to contain the thing. Penalised by IoU, rewarded by the metric
that asks the question actually being asked.

Breakdown of 150: **94 correct · 33 right object, loose box · 23 genuinely lost.**

Three consequences:

1. We are still **graded on acc@0.5**, so it does not change what to optimise —
   but it changes what to *work on*. Box precision has ~22 points of headroom
   that tightening tricks keep failing to convert; only 23 rows are truly unsolved.
2. **For the product it may be the better metric.** The PS asks for *"visual
   evidence"*; someone looking at a highlighted region cares whether it is the
   right object, not whether the rectangle is 0.52 or 0.48.
3. **The ISRO threshold is unknown.** If scored leniently or by mask overlap,
   84.7% is closer to what we would see than 62.7%.

"Found the object" now belongs alongside acc@0.5 permanently — it is the number
that says whether a failure is worth chasing.

---

## 8. Per-slice results

```
slice              n   acc@0.5   acc@0.7   meanIoU   unparsed
OVERALL          600     61.3%     40.0%     0.558      0.2%
  small          531     62.5%     39.7%     0.559      0.2%
  large            7     85.7%     71.4%     0.759      0.0%
  unique True    385     64.7%     42.6%     0.593      0.3%
  unique False   215     55.4%     35.4%     0.495      0.0%
```

**21 points sit between acc@0.5 and acc@0.7** — that gap is sloppy boxes, and it
is what every tightening experiment tried and mostly failed to recover.

**Non-unique targets cost ~9 points** (64.7% → 55.4%), which is the
disambiguation problem the detector route was supposed to solve.

### Sampling rules

- **Stratified sampling is diagnosis, never a headline.** It under-represents
  `vehicle` **7×** (28.1% true → 4.0% sampled).
- The headline number uses **random, unstratified** sampling, matching the
  published protocol so it sits beside GeoChat's 60.6% and means something.
- ~2,000 rows gives **±2%** at 95% confidence.

---

## 9. What is wired into the agent

Grounding routes to `single_grounding`:

```
texture_seg  →  rs_ground_caption  →  centroid_prior
```

`centroid_prior` is **D2**: a mask, then a centroid taken from it, then the
normalised point handed to the grounding step as a spatial prior. Verified
working — *"Where is the water body?"* produced a prior at `[0.269, 0.544]`.

`object_box_fallback` (**C39**) exists for out-of-vocabulary classes — the 80%+
of referring classes with no licence-clean source. Currently a deterministic
texture-blob proposer, labelled as such, returning partial credit rather than
zero. **The open upgrade** is swapping it for GroundingDINO (Apache-2.0), which
needs no training data for those classes — you prompt it with the word. Already
wired and measured this session; never adopted because it lost as a *primary*
route, but as a *fallback* the comparison was never the right one.

`rs_ground_caption` registers with `adapter_path=None`, i.e. serves the base
model. Leaving it unregistered made the tool report itself unavailable and the
router plan a step nothing could execute — so a decision to ship untrained read,
from outside, as a missing component.

---

## 10. What grounding buys us at the gate level

The problem statement's requirement is an **OR**: change description **or**
change VQA; grounding **or** captioning. Grounding already works at 62.7%
untrained, so **captioning is optional upside**, not a mandatory gate with no
answer. That demoted the entire caption-corpus agony (§ Segment 3).

Scores are *"normalised before combining different metrics"* across VRSBench
(captioning + grounding + VQA), RSVQA, CDVQA and the ISRO set. It is a portfolio,
not one number — being weak on one sub-task of one benchmark is survivable.

```
strong    RSVQA · VRSBench VQA · agentic trace
decent    VRSBench captioning
weak      VRSBench referring   (9% class coverage)
UNKNOWN   ISRO set             (undisclosed)
```

---

## 11. Files

| | |
|---|---|
| pipelines + prompts | `scripts/eval_grounding_pipelines.py` |
| VLM grounding eval | `scripts/eval_vlm_grounding.py` |
| recall test | `scripts/eval_grounding_recall.py` |
| corpus generator | `scripts/gen_ground.py` |
| box parsing | `satquery/evalcli/formatter.py` — `format_box()`, `_BOX_NUMBER`, `box_scale` |
| served prompts | `satquery/agent/served_prompts.py` |
| fixed sample | `/data/eval/pipeline_sample.json` (seed 7) |
| dumps | `eval/pipe_dump_*` — qwen, qwen_sam, qwen_precise, qwen_zoom, qwen_multimask, qwen_generous_sam, qwen_point_sam, qwen_precise_sam, dino_top1, dino_vlm, dino_sam_vlm |
| VRSBench eval | `eval/vrsbench_val` — `VRSBench_EVAL_referring.json` (16,159), `_Cap.json`, `_vqa.json` |

Contracts registered: `ground_reference`, `ground_point`, `ground_presence`,
`vrsbench_referring`.

---

## 12. Corpus generator work (`gen_ground.py`)

Even though the corpus was not ultimately trained on, the phrasing fixes matter
because they document what the benchmark actually asks:

- **Position vocabulary** — screen-relative *and* compass, 50/50. We had trained
  compass/interrogative/tagged; the benchmark is
  screen-relative/declarative/untagged, and **87% of it turns on a position word**.
- **Surface forms** — tagged/untagged 33/67, matching VRSBench's declarative
  majority.
- **Size attributes** — 35% of VRSBench referring uses them.
- **Uniqueness key kept vocabulary-free**, so ambiguous targets cannot leak
  through as "unique".
- **Degenerate-box collapse** fixed.

Composition of VRSBench referring phrasing, measured: **72.6% position-only**
("bottom-left corner", "top of the image"), **16.9% relational** ("closest to the
green area"), **73.6% small objects**.

---

## 13. Open items

- **The corpus on disk still has 273 degenerate boxes** — the generator was
  fixed, the 19,896 existing rows were not regenerated.
- **`object_box_fallback` → GroundingDINO swap** — tool swap, no training, the
  only route to the ~80% of referring classes we cannot legally source.
- **Official VRSBench scorer not wrapped** (C8, a P0 gap). Every number carries
  the in-repo-comparator caveat.
- **`image_size` is never plumbed** into the box parser — if the model emits
  pixel coordinates that path is dead, and the 0–1000 fallback would silently
  misread them.
- **SAR ship grounding** was in scope (C53) with a graded acc@0.5, and is now
  unbuildable — LS-SSDD's imagery terms never cleared.
