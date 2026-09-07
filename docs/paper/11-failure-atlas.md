# 11. Failure atlas — every bug that mattered, and what each proved

This is the chapter with the most transferable content. The claim it supports:

> **Component-level benchmarks do not measure a system.** Every published number
> in this project was produced by calling a component *directly*. The first
> artefact that exercised all of them *through the router* found nine bugs, seven
> of which were invisible to 440 unit tests and 50 route checks.

Bugs are grouped by what they teach, not by when they happened.

---

## 11.1 Class A — a format bug wearing a model bug's clothes

Every large score movement in this project belongs here. **Not one changed a
weight.**

| bug | looked like | actually |
|---|---|---|
| RSVQA `count` scored as integers | the model cannot count (27%) | the benchmark scores *ranges*; quantising gave **+7.3 points** |
| RSVQA `area` excluded as unusable | a broken question type | the protocol has five buckets; bucketing gave **87.6%**, and it carries the HR gate |
| `change_vqa` free-text answers | a weak adapter | instruction-following **77.7% → 100%** |
| box coordinates divided by 512 | the model cannot localise | anything >512 clamped to 1.0 — **75% "unparsable"** |
| `_NUMBER` read `,` as a decimal point | Qwen emits unparseable boxes | `(560,190)` parsed as `560.190` — **every** Qwen-format box rejected |
| `bbox_2d` key | boxes systematically shifted | the `2` in the key name read as a coordinate |

**The lesson.** Before concluding a model is weak, print what it actually emitted
and what the scorer actually compared. Three separate box-scale bugs each
produced numbers indistinguishable from a model that cannot see.

**The fix that generalises.** The served path now parses boxes with *the same
formatter the benchmark scores with*, so what a user sees is what acc@0.5
measured. One implementation, not two.

---

## 11.2 Class B — train/serve divergence

The benchmark harness and the product are different code paths. Every difference
is a silent error.

**Channels reversed.** Sentinel-2 stores B02, B03, B04 — so "first three bands in
file order" is *blue, green, red*, fed straight into the R, G, B slots. Every
true-colour scene reached the adapter with red and blue swapped.

**Two thirds of the input duplicated.** One scene produced one view, repeated to
fill three slots. The corpus fills those slots with three *different* composites
— true-colour, false-colour, short-wave — so **NIR and SWIR reached the adapter
in training and never at serving time.** Questions whose answers live outside
visible light were being asked of a model that could not see the evidence.

**A hardcoded view count.** `_VIEWS = 3` for every adapter, while the corpus uses
3 / 6 / 2. The two-image tasks were served a duplicated date they were never
scored with.

**The wrong prompt input.** `PRECISE_PROMPT` was measured on the *whole referring
sentence*; serving passed an extracted noun, discarding the half that
disambiguates one aircraft from several in frame.

**The lesson.** Parity must be *asserted*, not assumed. The composite fix is now
tested by **pixel equality against the corpus PNGs the adapter trained on** — a
shape check would have passed while the content drifted.

---

## 11.3 Class C — the router never reached the component

The most serious class, and the one that only the held-out gallery could find.

**BIFOLD never ran.** Sentinel-1 GeoTIFFs carry no band descriptions, so the
inventory came back empty and `lulc_classifier` refused with
`needs band 'VH'`. The component holding the **74.95%** was absent from every
deployed SAR answer, and the VLM — measured at **0.4968, chance**, on the same
task — answered in its place.

**Referring expressions were unroutable.** VRSBench referring rows are
*statements*: *"The relatively larger airplane positioned near the right edge of
the image can be found in the bottom-right corner."* No `where is`, no `locate`,
no `find the` — so they fell to the VQA adapter, which answered `"yes."` and
produced no box. **16 of 22 grounding failures.**

**Vocabulary gaps.** Eight of VRSBench's 26 DOTA classes — `roundabout`,
`storage-tank`, `ground track field` — were missing from the grounding
vocabulary, so a sentence naming one read as ordinary prose.

**`detect` was a bare substring.** It matched *"Can coniferous forest be
**detect**ed in the satellite image?"* — a verbatim binary question whose gold
answer is `yes` — and returned a box covering the whole frame.

**Results.** Grounding **12% → 80%**. SAR **46% → 74%**, against a benchmarked
74.95%.

**The lesson.** A capability is only as good as the path that reaches it. Testing
a component in isolation measures the component; it says nothing about whether
the system ever calls it.

---

## 11.4 Class D — failures that hide themselves

These are the reason the class above went undetected for so long.

**A raising tool vanishes from the trace.** `absorb()` appends a throwing tool to
`failures` and `warnings` but **not** to `execution.steps`. A crashing tool and a
tool that was never planned look identical from outside. This is precisely why
BIFOLD's absence was invisible — health reported `available: true`, the answer
looked fine, and only the raw warnings said `lulc_classifier failed`.

**The trace carried only executed steps.** No `plan` key, so "never planned" and
"planned then dropped" were indistinguishable. Dropped tools now warn explicitly.

**`produced_by` was a filename token.** `path.stem.split("_")[0]`, so
`texture_seg_mask.tif` reported `"texture"` — no such tool. The contract types
that field as the tool name and the disagreement panel matches on it exactly, so
that panel rendered **NO MASK on every cross-modal conflict since it was
written**. It is now resolved against the tool registry.

**A stale Volume mount.** Modal snapshots a Volume at container start; anything
written afterwards is invisible until reload. A freshly published gallery
reported counts from hours earlier. Quietly wrong, not broken.

**The lesson.** Observability failures are worse than the failures they hide,
because they convert a loud error into a plausible answer.

---

## 11.5 Class E — the contract is not the code

**`lulc_classifier` had a live optical fallback.** The decision to use radar only
was recorded in the manifest (`required_modalities: [sar]`), in the router
(`add_lulc` requires `has_sar`), and in the plan. The *tool itself* still
contained `elif optical is not None:` loading the chance-level model behind a
warning.

Two guards made it unreachable. **Unreachable is not safe** — relax either and
the system serves a 0.4968 classifier as a measurement. Now removed, with tests
at all three layers plus the tool refusing when called directly, which was the
missing one.

**A gallery endpoint invented its own bundle shape.** It wrote `kind` where the
contract says `pair_type`, and omitted `label`, `prep_ms`, `provenance`,
`supported_tasks`, `blocked_tasks` and `bounds_wgs84`. All required in
`types.ts`. A missing required field **blanks** the console. The fix was to
*call* the existing endpoint rather than duplicate it.

**The lesson.** "The manifest says so" is not the same as "the code cannot do
otherwise". Assume every recorded decision has at least one live path
contradicting it, and test the decision rather than the documentation.

---

## 11.6 Class F — a crash on a path nothing exercised

`pipeline.py` read `context.scene_of("optical")` where `context` is the router's
`QueryContext`, which has no such method — only `ToolContext` carries scenes.

The branch is reached **only when an optical mask and a SAR mask both exist**.
Nothing in the suite built a pair that produced two. So **every crossmodal query
failed outright** with an `AttributeError` while **411 tests and 44 route checks
passed**.

The fix is one word. The interesting part is the test: `verify_routes.py` now
synthesises a co-registered optical/SAR pair — one CRS, one transform, named
bands, content chosen so both thresholds clear their floors — and actually runs
`answer_query`. Validated the only way that means anything:

```
with the bug     FAIL  AttributeError: 'QueryContext' has no attribute 'scene_of'
                       10 passed, 1 failed
fixed            ok    fused: IoU 0.0 verdict conflict
                       11 passed, 0 failed
```

A probe reporting *"no fusion (masks did not both materialise)"* has **not**
covered the branch and says so, rather than passing quietly.

**The lesson.** Route coverage is not execution coverage. A plan that is checked
but never run leaves the entire executor untested on that path.

---

## 11.7 Class G — data that lies

**A staged archive: 42.3% of files unreadable.** p7zip could not read the RAR5 leg and left
9,871 damaged files in place; `unar` extracted good copies into a separate tree
beside them. A dict keyed on filename lets whichever path `rglob` yields last
win — a coin flip between a valid image and a truncated one. The fix collects
candidates per name and keeps **the first that PIL can decode**.

A 40-pair spot check reporting "0 unreadable" against that rate was **luck, not
evidence**.

**Blank SpaceNet 7 months.** Removing them was obviously right and took the
`change_presence` blind ceiling from **53.9% to 74.3%** — most "nothing changed"
examples *were* blank months. Deleting bad data made the benchmark easier.

**`dominant_change` had a 100% blind ceiling.** It always answered NDBI, because
forest showed mean |ΔNDBI| 0.222 against |ΔNDVI| 0.046 — backwards for forest.
The question type was deleted rather than shipped.

**The lesson.** Measure the blind ceiling *after* every corpus change, not once
at the start.

---

## 11.8 Class H — my own reasoning errors, recorded

Kept because they were as costly as the code bugs.

**A false leakage alarm at 97.2%**, from comparing validation against a manifest
that embeds validation. Nearly triggered a corpus rebuild.

**A `--limit 160` slice read as a result** (0.727). `--limit` takes the *first* N
rows, not a sample. The balanced 1,200-row answer was 68.0%.

**A blunt physical threshold bound.** Written to stop MNDWI choosing −0.224 on a
scene with no water, it also rejected a perfectly good NDVI cut at 0.2813 against
a 0.3 boundary. The correct rule is a **tolerance, not an edge**.

**A top-up that replaced each failed item with itself.** The exclusion set was
built from the manifest *after* pruning, so the failures were no longer in it and
the balanced pool returned the same rows in the same order. The docstring claimed
this could not happen.

**A flaky test misdiagnosed twice.** `test_execute_sar_backscatter` builds two
Gaussians with an **unseeded** RNG and asserts Otsu was accepted; the bimodality
gate rejects **37 of 400 seeds**, so it fails about one run in eleven. Twice it
went red and twice it was blamed on a concurrent edit — once mid-deploy. Fixed by
seeding, plus an autouse fixture reseeding numpy per test.

**Claiming a fix without checking it live.** The `NO MASK` panel was "fixed" by
widening a filter; the real cause was one layer deeper (`produced_by` never
emitted tool names), and the first fix would have changed nothing.

**The lesson.** State the diagnosis, then verify it against the running system
before claiming it. Twice here the second layer was the real one.

---

## 11.9 Summary table

| # | bug | invisible to | found by |
|---|---|---|---|
| 1 | `context.scene_of` on the wrong object | 411 tests, 44 route checks | held-out gallery |
| 2 | referring expressions unrouted | all tests | held-out gallery |
| 3 | grounding vocabulary missing 8 classes | all tests | held-out gallery |
| 4 | prompt fed a noun, not the sentence | all tests | reading the eval script |
| 5 | Sentinel-1 bands unnamed → BIFOLD never ran | all tests, health said "available" | held-out gallery |
| 6 | BIFOLD emitted labels but no answer | all tests | held-out gallery |
| 7 | raising tools vanish from `steps` | — | manual trace reading |
| 8 | stale Volume mount | all tests | comparing published vs served counts |
| 9 | `produced_by` never emitted tool names | all tests, 50 route checks | a user screenshot |

Seven of nine required real imagery through the real router. That is the case for
the held-out gallery as a permanent verification layer, and it is the strongest
methodological finding this project has.
