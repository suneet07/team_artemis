# Segment 6 — Routing: which question goes where

Read off `satquery/agent/router.py` (669 lines) and verified against
`scripts/verify_routes.py`, which drives all eight branches on every run.

Two stages, and the order matters:

1. **`route(context)`** picks the **task** — from the question text *and* the
   inputs supplied. Rules first; the LLM tie-breaker sees exactly one case.
2. **`plan_for(task, context, targets)`** picks the **tools** — gated by
   modality and by what bands the scene actually carries (D3).

**Nothing the model produces feeds stage 1.** `QueryContext` carries only the
question text, the modalities, the band inventories, the image count and the
dates — *"What the router is allowed to see. Nothing here is model output."*

---

## 1. Task selection — the decision order

Checked top to bottom. **The first match wins**, so precedence *is* the
algorithm.

```
1. cross-modal?     optical + SAR present        -> CROSSMODAL_*
2. temporal?        2 same-modality images, OR
                    change words + >=2 images    -> CHANGE_*
3. change words, only 1 image, NOT a referring
   expression                                    -> CHANGE_VQA with an EMPTY plan
4. grounding words, mask words, OR a referring
   expression                                    -> SINGLE_GROUNDING
5. caption words, not ending in "?"              -> SINGLE_CAPTION
6. caption words AND ending in "?"               -> SINGLE_VQA  (LLM tie-breaker)
7. everything else                               -> SINGLE_VQA
```

**Inputs beat words at the top; words decide at the bottom.** A co-registered
optical + SAR pair is unambiguous on input types alone, so no phrasing can route
it elsewhere. By the time we reach step 4 there is one image and the question
text is all there is.

### The five word lists that drive it

| flag | triggers on |
|---|---|
| `mentions_change` | change · changed · changes · difference(s) · before · after · over time · between the two · new construction · deforestation · expanded · grown · since |
| `wants_grounding` | where is · where are · locate · find the/all/any/every · **detect the/all/any/every** · show me the · mark the · highlight · bounding box · bbox · point to · identify the location · segment the |
| `wants_caption` | describe · description of · write a short · caption · summarise · summarize · what does this · overview of |
| `wants_mask` | mask · map of · raster · segmentation · footprint · delineate · extract · shapefile |
| `wants_numbers` | how much · how many · what fraction · what percentage · what proportion · ratio · area of · count · largest · smallest · most · least |

Plain substring matching, lower-cased. No stemming, no embeddings — the router
is deterministic and auditable, which is what the trace has to defend.

`detect` is **scoped** rather than bare. As a substring it also matched "Can
coniferous forest be **detect**ed in the satellite image?" — a verbatim BEN
binary question whose gold answer is "yes" — and sent it to grounding, which
returned a box covering the whole frame. Presence is a VQA question; only an
imperative asking for a location is grounding.

### The sixth trigger: a referring expression

None of the five lists fire on this, which is the whole problem:

> "The relatively larger airplane positioned near the right edge of the image
> can be found in the bottom-right corner."

That is a **VRSBench referring row**, and the entire referring benchmark is
written this way — declarative, never interrogative. Every one of them fell to
SINGLE_VQA, where the VQA adapter answered "yes." and no box was ever produced.
On the held-out gallery that was **16 of 22 grounding failures**, and grounding
scored 12% against a measured 62.7%.

`_is_referring_expression` requires all three, because any one alone over-fires:

| condition | why |
|---|---|
| does **not** end in `?` | "Is there a ship on the left?" is presence — VQA's |
| names a known object | the 26 DOTA classes VRSBench is built from, taken from the benchmark's own `obj_cls`, in hyphenated *and* spaced spellings |
| **places** that object | `located` · `positioned` · `situated` · `can be found` · `top-left` … · `corner` · `adjacent to` · `section of` · `crossing the` · `centered in` |

It also **beats the single-image change refusal** at step 3. "The ship is
centered in the image, positioned on the water *between the two* small harbors"
trips `mentions_change`, where that phrase is spatial rather than bi-temporal —
so a single image was being refused for a question it answers perfectly well.

After this: **25 of 25** gallery referring rows route to grounding, and the six
guard cases (presence, count, caption, real change) all stay put.

---

## 2. The eight branches, with worked examples

### CROSSMODAL_EXTRACTION / CROSSMODAL_VQA — optical + SAR present

Split on **`wants_mask or wants_numbers`**: asking for a *thing* (a mask, a
count) is extraction; asking a *question* is VQA.

```
"Use the optical and SAR images together to identify built-up
 and water-covered regions."                        -> CROSSMODAL_VQA
"Extract built-up regions from the optical and SAR pair."  -> CROSSMODAL_EXTRACTION
```

Targets default to `["water"]` when the question names none — this is the
problem statement's own representative query and water is its default subject.

```
coreg_check -> lulc_classifier -> [per target] spectral_index|texture_seg -> sar_backscatter
```

**Two independent decisions, then reconcile** — that is D1. `lulc_classifier`
(BIFOLD, 74.95%) adds a named class inventory *beside* the thresholded masks, and
never replaces either decision, so the disagreement table still has two masks to
work with.

### CHANGE_MAP / CHANGE_VQA / CHANGE_DESCRIPTION — two same-modality images

Entered when `is_temporal` **or** (`mentions_change` and ≥2 images). Then:

| phrasing | task |
|---|---|
| `wants_mask` | **CHANGE_MAP** |
| `wants_numbers` | **CHANGE_VQA** |
| `wants_caption` or the word `describe` | **CHANGE_DESCRIPTION** |
| anything else | **CHANGE_VQA** (the default) |

```
CHANGE_VQA / CHANGE_DESCRIPTION   coreg_check -> change_vqa
CHANGE_MAP                        coreg_check -> change_map   (absent; see below)
```

**The split is artefact versus answer.** `_MASK_TERMS` (mask · map of · raster ·
segmentation · footprint · delineate · extract · shapefile) means the user wants
a georeferenced file they could open in QGIS. Everything else wants text.

```
"Produce a change mask for this pair."          -> CHANGE_MAP
"Give me a change map of the area."             -> CHANGE_MAP
"Export a shapefile of the new construction."   -> CHANGE_MAP
"Delineate the areas that changed."             -> CHANGE_MAP

"What changed between the two dates?"           -> CHANGE_VQA
"How many buildings appeared?"                  -> CHANGE_VQA
"Which land cover changed the most?"            -> CHANGE_VQA
"Has the forest shrunk since 2018?"             -> CHANGE_VQA
"Describe the changes between these images."    -> CHANGE_DESCRIPTION
```

**A phrasing trap worth knowing:** *"Extract the changed regions"* and *"give me
the building footprints that appeared"* both route to **CHANGE_MAP**, because
`extract` and `footprint` are mask words. Someone who meant "tell me what
changed" gets the absent-tool path instead of an answer. If that phrasing turns
up in a demo, the fix is to narrow `_MASK_TERMS`, not to widen the task.

#### Why CHANGE_MAP plans a tool that cannot run

`change_map` has a manifest and **no implementation**. The Siamese U-Net trained
for it reached **F1 0.2931**, and more decisively it detects *buildings* while
the graded benchmark (CDVQA) asks about *land cover* — so it could not answer
CDVQA at any quality. Closed decision; see segment 5.

It stays in the CHANGE_MAP plan on purpose. That task is a request for a raster,
so the honest response when nothing can produce one is a trace saying a mask was
attempted and no tool was available. Substituting prose for a GeoTIFF would be
worse. The problem statement makes the change map **optional** for G4 — change
description *or* change VQA is what is mandatory — so this costs no gate.

#### Why the graded paths no longer plan it

They used to, and it was wrong twice over:

```
before   coreg_check -> change_map(absent) -> change_stats(no inputs) -> [nothing]
after    coreg_check -> change_vqa
```

1. **`change_stats` could never run.** It needs `class_map_before` /
   `class_map_after`, which come from `change_map` or from two thresholded
   masks, and **nothing in the router produces either**. Every bi-temporal query
   carried two steps that could only emit a warning.
2. **The trained adapter was missing entirely.** `change_vqa` — AA **68.0%** on
   official CDVQA Val against a 45.0% blind ceiling — was in no plan at all. So
   the mandatory G4 gate ran the deterministic half, found it empty, and fell
   through to *"no learned adapter is loaded"*. Nothing errored;
   `/meta/health` correctly reported the adapter loaded, because it **was**
   loaded. It was simply never called.

`change_stats` itself is sound — **100% AA on 2,012 rows** when handed
ground-truth footprints. Perception is what is missing, not reasoning. Restore
both steps the day a mask source lands; `tests/test_change_routing.py` asserts
the current contract and is meant to change with it.

### The refusal branch — change language, one image

```
"What changed in this image?"  with a single scene
    -> CHANGE_VQA,  plan = []          <- deliberately empty
```

From the code:

> routed as change so the validator can refuse with an explanation rather than
> answering the wrong question

**An empty plan is the point.** Routing it to `SINGLE_VQA` would produce a
fluent answer to a question the inputs cannot support. The problem statement
grades refusals — *"that refusal is a graded deliverable, not an error"* — so
this branch exists to reach the validator, not to answer.

### SINGLE_GROUNDING — locate/where/mask phrasing

```
"Where is the water in this image?"        -> SINGLE_GROUNDING
"Find all the aircraft."                   -> SINGLE_GROUNDING
"Highlight the built-up area."             -> SINGLE_GROUNDING
"Extract the building footprints."         -> SINGLE_GROUNDING   (wants_mask)
```

```
texture_seg|spectral_index -> rs_ground_caption -> centroid_prior
```

with `rs_ground_caption` declared `depends_on: ["centroid_prior"]` when the
prior is available — that is **D2**: a mask, a centroid taken from it, then the
normalised point handed to the grounding step as a spatial prior.

**The noun extraction matters more than it looks.** `_grounding_noun` first scans
for any word in `GROUNDING_VOCABULARY` (36 entries: buildings, aircraft, ships,
water, forest, urban, region…). If none matches it falls back to a regex that
captures **two words**, not one:

> "locate the storage tanks" must not become "storage", which matches no shape
> prior and loses the only word that carries the class

Trailing prepositions (`in`, `on`, `at`, `of`, `from`) are dropped.

That noun becomes the `phrase` parameter, and **that is what reaches
`PRECISE_PROMPT`** — the prompt substitutes into
`"Find this object in the satellite image: {phrase}"`. Passing the whole
interrogative would build *"Find this object in the satellite image: Where is the
water in this image?"*, which is not the input 62.7% was measured on.

An out-of-vocabulary noun routes to `object_box_fallback` instead — C39's
deliberate slot for the ~80% of VRSBench referring classes with no licence-clean
training source.

### SINGLE_CAPTION — describe, and not a question

```
"Describe the land-cover and major objects visible in this image."  -> SINGLE_CAPTION
"Write a short overview of this scene."                             -> SINGLE_CAPTION
```

```
optical only:  rs_ground_caption
SAR present:   lulc_classifier -> rs_ground_caption   (depends_on lulc_classifier)
```

**`rs_ground_caption`, not `rs_vqa`** — the corpus assigns 19,983 caption rows and
39,996 grounding rows to `rs_ground_caption`, and 57,044 VQA rows to `rs_vqa`.
Routing a description to the VQA adapter answers with a model tuned to emit short
factual answers, which is the opposite register.

The `lulc_classifier` step is the **attribute prior** from the JTTS paper,
delivered as a tool rather than a training change: 19 named classes with learned
confidences, so the description is grounded in a measurement rather than only in
the model's impression of the scene. Gated on `has_sar`, because the classifier
is scoped to radar (0.7525 vs 0.4968 optical — chance).

### The one ambiguous case — the only LLM tie-breaker

```
"Describe what you can see -- how many buildings?"
    -> SINGLE_VQA,  router_path = LLM
       ambiguous_between = (SINGLE_CAPTION, SINGLE_VQA)
```

**Caption words present *and* the text ends in `?`.** That reads both ways, and
it is the *only* branch that leaves `router_path=RULES`. Everything else is
resolved deterministically. `SINGLE_VQA` is the provisional answer so the query
still executes if the tie-breaker is unavailable.

### SINGLE_VQA — the default

```
"What is the vegetation cover in this scene?"   -> SINGLE_VQA
"Is there a water body here?"                   -> SINGLE_VQA
"What land cover is in this radar scene?"       -> SINGLE_VQA
```

```
[per named target]  spectral_index|texture_seg|sar_backscatter
SAR present:        lulc_classifier
otherwise:          rs_vqa
```

A deterministic tool is added **only when the question names a target a tool can
measure**. Otherwise one learned step is the whole plan.

### On radar, the classifier answers and the VLM does not

`rs_vqa` is **not** planned when all three hold: the input is radar-only, the
classifier is in the plan, and the question names one of the 19 reBEN classes it
predicts. Then `lulc_classifier` is the last step and its answer is the answer.

The measurement is not close. On 4,000 held-out cross-modal rows, both models
answering the same questions:

```
radar classifier   0.7525
optical classifier 0.4968      <- chance
                   radar wins 87% of their disagreements
```

Planning both produced two learned answers, and the composer takes the learned
reply — so the component measured at **0.4968 on this exact task** was speaking
over the one measured at **0.7525**. Held-out SAR sat at 23/50, below its own
50.1% majority floor. With the classifier answering: **37/50**, against a
benchmarked 74.95%.

Scoped tightly, and the scope is the point. An open question on a radar scene
("What is happening here?") still reaches `rs_vqa`, because the classifier can
only answer about classes it predicts — `answer_for` returns `None` otherwise,
which is the signal to let something else speak rather than to guess.

Two ingest facts make this work at all, both learned the hard way:

- **Sentinel-1 GeoTIFFs name no bands.** The inventory came back empty and the
  tool refused with `needs band 'VH'`, so BIFOLD never ran on a single real
  patch. `bands.assumed_sar_orders` supplies the names.
- **The order is VH then VV**, and it is *measured*: `logs/bifold_probe_sar.json`
  scores `vh_vv` at **0.817** and `vv_vh` at **0.499**. The intuitive reading is
  the wrong one.

---

## 3. Target extraction — three buckets, not free text

`_targets_in` maps the question onto exactly three measurable targets. A question
can carry several.

| target | triggers on |
|---|---|
| `water` | water · lake · river · reservoir · flood · pond · sea · coast · wetland |
| `builtup` | building · urban · built-up · builtup · settlement · city · construction · house |
| `vegetation` | vegetation · forest · crop · farm · green cover · tree · plantation |

**Anything else yields no target**, which is why a plan can be a lone learned
step. That is honest: these three are what `spectral_index` and `sar_backscatter`
can actually threshold.

---

## 4. Modality and band gating (D3)

Tools do not enter a plan they cannot execute. **Planning a step the parameter
gate would refuse yields a failed trace rather than an answer.**

```python
add_optical(target)   # spectral_index if the index is computable from the bands
                      # present; else texture_seg; else nothing
add_sar(target)       # only when context.has_sar
add_lulc()            # only when context.has_sar  (manifest: required_modalities [sar])
                      # and on radar-only input it also *replaces* rs_vqa when
                      # the question names a class it predicts
```

`_index_for` consults `context.computable_indices()`, which comes from the
scene's real `BandInventory`:

> A target whose index needs SWIR on a source with no SWIR does not get a
> `spectral_index` step that would fail the parameter gate — it gets the SAR
> path, or `texture_seg`, and **the substitution is a routing note**.

So a Sentinel-2 scene with SWIR gets MNDWI for water; a plain RGB scene gets
`texture_seg` with `target: smooth`; a SAR scene gets `sar_backscatter`. Same
question, three different plans, each recorded.

`QueryContext` also exposes `optical_is_spectrally_blind` and `is_single_pol`,
which the validator uses to explain a refusal in physical terms rather than as a
generic failure.

---

## 5. Which model answers, and with which prompt

The last hop, and the one that is easy to get wrong because it is invisible in
the plan.

| tool | weights | prompt it receives |
|---|---|---|
| `rs_vqa` | trained LoRA | `scale_prefix + question` |
| `change_vqa` | trained LoRA | `scale_prefix + question` |
| `rs_ground_caption` mode=`grounding` | **base Qwen** | `scale_prefix + PRECISE_PROMPT.format(phrase=target)` |
| `rs_ground_caption` mode=`caption` | **base Qwen** | `scale_prefix + STRONG_PROMPT` |

**The asymmetry is the whole point.** A trained adapter reproduces its score from
`scale_prefix + question` because that is the format it was fine-tuned on. The
base model reproduces nothing unless it is asked the way the benchmark asked —
62.7% grounding and 0.252 captioning came from those two exact strings, and the
0–1000 coordinate scale in `PRECISE_PROMPT` is what the box parser depends on.

**The router declares which**, via `params: {"mode": ...}`, because only the
planner knows which of the two capabilities the manifest is serving. `mode` is a
declared enum on `configs/tools/rs_ground_caption.yaml`, so the parameter gate
validates it like any other parameter.

Verified automatically on every run — the stub runner reports which prompt it
received:

```
single_caption    -> STRONG_PROMPT
single_grounding  -> PRECISE_PROMPT
single_vqa        -> scale_prefix+question
```

---

## 6. Quick reference

| you ask | task | tools |
|---|---|---|
| "What is the vegetation cover?" | `single_vqa` | texture_seg → rs_vqa |
| "How many buildings are visible?" | `single_vqa` | texture_seg → rs_vqa |
| "What land cover is in this radar scene?" | `single_vqa` | lulc_classifier → rs_vqa |
| "Is there forest in this image?" *(radar only)* | `single_vqa` | sar_backscatter → lulc_classifier — **no `rs_vqa`**; the classifier answers |
| "Is there forest in this image?" *(optical)* | `single_vqa` | spectral_index → rs_vqa |
| "Can coniferous forest be detected in the image?" | `single_vqa` | *presence, not location — `detect` alone no longer grabs it* |
| "Describe the land-cover and major objects." | `single_caption` | rs_ground_caption |
| "Describe this radar scene." | `single_caption` | lulc_classifier → rs_ground_caption |
| "Where is the water?" | `single_grounding` | texture_seg → rs_ground_caption → centroid_prior |
| "Find all the aircraft." | `single_grounding` | texture_seg → rs_ground_caption → centroid_prior |
| "The ship is located at the middle-right of the image." | `single_grounding` | *referring expression — a statement, no grounding verb* |
| "The ship … between the two small harbors." | `single_grounding` | *referring beats the single-image change refusal* |
| "Is there a ship on the left side?" | `single_vqa` | *presence, despite naming and placing an object* |
| "Extract the building footprints." | `single_grounding` | texture_seg → rs_ground_caption → centroid_prior |
| "What changed between the two dates?" | `change_vqa` | coreg_check → change_vqa |
| "How much of the area changed?" | `change_vqa` | coreg_check → change_vqa |
| "Describe the changes." | `change_description` | coreg_check → change_vqa |
| "Produce a change mask." | `change_map` | coreg_check → change_map *(absent — honest refusal)* |
| "What changed?" *(one image)* | `change_vqa` | **empty — routed to be refused** |
| "…optical and SAR together… built-up and water" | `crossmodal_vqa` | coreg_check → lulc_classifier → (texture_seg → sar_backscatter) **per target** — two targets here, so both pairs appear |
| "Extract built-up from the optical+SAR pair." | `crossmodal_extraction` | coreg_check → lulc_classifier → texture_seg → sar_backscatter |
| "Describe what you see — how many buildings?" | `single_vqa` | **LLM tie-breaker**, the only non-rules path |

---

## 7. Checking it

```bash
python scripts/verify_routes.py --offline-only --stub-adapters
```

0.7 s, no server, no GPU. Drives all eight branches, asserts every planned tool
exists **and** is executable, and fails if any `Task` is unreachable. Absent
tools are annotated (`change_map(absent)`) rather than failed, because absence is
a decision here — but a plan where *everything* is absent is a failure, since
nothing would execute.

`tests/test_lulc_routing.py` (7) and `tests/test_served_prompts.py` (11) pin the
specific behaviours: SAR gating on `lulc_classifier`, captions not reaching
`rs_vqa`, the declared modes, and the referring phrase substitution.

Added after the held-out gallery found the router's blind spots — every one of
these is a bug that shipped, not a hypothetical:

- `tests/test_referring_expression_routing.py` (15) — seven verbatim VRSBench
  rows reach grounding, five guards do not over-fire, and the whole sentence
  (not the extracted noun) reaches the prompt.
- `tests/test_radar_landcover_path.py` (7) — VH before VV, the classifier
  answers on radar, the VLM still gets open questions, and `eval_bifold.py`
  imports the scoring rule rather than keeping a second copy.
- `tests/test_crossmodal_fusion_context.py` (3) — **runs** a co-registered pair
  through `answer_query` and requires an answer plus an `agreement` block in the
  trace. Asserting the second is what makes the first mean anything: without it
  the pair could route crossmodal, execute cleanly, and still never touch the
  broken line.

### Planning is not executing

The bug above is the reason this section grew a second half. Everything up to
here checks that the router *plans* correctly — every Task reachable, every
planned tool known and executable. None of it runs the plan.

That gap shipped an outage. The fusion branch read `context.scene_of("optical")`
from the router's `QueryContext`, which has no such method, so **every**
crossmodal query failed with an `AttributeError`. All 411 tests passed. All 44
route checks passed. The branch is reached only when an optical mask *and* a SAR
mask both materialise, and nothing in the suite built a pair that produced two.

So `verify_routes.py` now ends with an execution probe:

```
ok    crossmodal executes end to end  -- fused: IoU 0.0 verdict conflict
```

It synthesises a co-registered optical/SAR pair — one CRS, one transform, named
bands, and content chosen so both thresholds clear their floors — runs
`answer_query`, and fails on any exception or refusal. Deterministic tools only:
no GPU, no server, ~6 s.

Verified the way a regression test should be: **reintroduce the bug and watch it
fail.**

```
$ # with context.scene_of restored
  FAIL  crossmodal executes end to end -- AttributeError: 'QueryContext' object
        has no attribute 'scene_of'
  10 passed, 1 failed

$ # fixed
  ok    crossmodal executes end to end -- fused: IoU 0.0 verdict conflict
  11 passed, 0 failed
```

A probe that reports `no fusion (masks did not both materialise)` has **not**
covered the branch, and says so rather than passing quietly — a test that skips
the thing it exists to cover is worse than no test.

What this still does not cover is answer *quality*: stub adapters return a fixed
string. Quality needs real weights against held-out rows with published answers,
which is what the testing-corpus gallery is for
([08-testing-corpora.md](08-testing-corpora.md)).
