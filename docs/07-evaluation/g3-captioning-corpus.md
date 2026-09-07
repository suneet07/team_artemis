# G3 captioning — the corpus we fine-tune on, and everything rejected

**Decision: train on captions generated from RarePlanes and SpaceNet 2/6, in
the VRSBench register mined from its own references. BEN.txt enters only as a
second training stage behind a saved checkpoint, so its effect is measured
rather than assumed.** Decided 2026-09-05.

Companion to [`g3-grounding-sweep.md`](g3-grounding-sweep.md), which settled the
*grounding* half. The conclusions are opposite and that is not a contradiction —
see [Why the two halves differ](#why-the-two-halves-differ).

---

## Why we fine-tune this half at all

Published VRSBench captioning, against our prompt-only base model:

| model | BLEU-1 | ROUGE-L | CIDEr |
|---|---|---|---|
| GeoChat, zero-shot | 13.9 | 13.2 | 0.4 |
| GPT-4V | 37.2 | 30.1 | 19.1 |
| GeoChat, fine-tuned | 46.7 | 35.2 | 28.2 |
| LLaVA-1.5, fine-tuned | **48.1** | **36.9** | **33.9** |
| **ours, prompt only** | **36.5** | **25.2** | — |

Fine-tuned models reach ROUGE-L 35–37. Prompting took us 19.5 → 22.8 → **25.2**
across three iterations and is flattening. An 11.7-point gap will not close on
wording.

**This is the opposite of grounding**, where the base model *beat* published
fine-tuned GeoChat (62.7% vs 60.6%) and training would have narrowed a model
that already saw 26 classes down to the 3 our corpus covers.

## What we train on

| source | images | GSD | contributes |
|---|---|---|---|
| RarePlanes (real) | 5,815 | 0.3 m | aircraft, by role, with counts and positions |
| SpaceNet 2 | 6,000 | 0.3 m | buildings across Vegas, Paris, Shanghai, Khartoum |
| SpaceNet 6 | 3,401 | 0.5 m | buildings, Rotterdam |
| | **~15,200** | | |

All three are **CC BY-SA 4.0**, staged and verified on 2026-09-05, and already
carry the annotations a caption needs: object class, count, box (hence position
and relative size). Captions are generated from those annotations by the
recipe the master plan already specifies at line 1131 — template from the
reference map, then linguistic augmentation.

## The register, mined not invented

Measured over all 9,350 VRSBench references. Every figure below is what the
generator targets and what the prompt states:

* **Opening.** "The image, sourced from GoogleEarth" 17.3%, "The
  high-resolution image from GoogleEarth" 9.4%, "The image from GoogleEarth"
  6.9%. Verbs: shows, features, captures. **86% contain the literal token
  "GoogleEarth"** — worth real points and carrying no information.
* **Shape.** Median 47 words, 3 sentences, ~16 words each. 10th percentile 29,
  90th 70.
* **Attributes.** "small" in **53.8%**, "large" 15.8%, "green" 12.3%.
* **Counts as words.** one 25.8%, two 25.9%, three 9.4%, several 9.1%.
* **Position.** right 41.8%, bottom 40.0%, left 32.3%, top 31.4%, middle 21.5%,
  side 21.9%, edge 12.0%, corner 11.3%; compounds like bottom-right 9.4%.
* **Positional phrasing.** "in the" 37.8%, "on the" 27.1%, "at the" 24.7%,
  "towards the" 19.1%, "near the" 12.6%.

## Rejected, and why

### BEN.txt captions — wrong register, wrong resolution, wrong subject

| | BEN.txt | VRSBench |
|---|---|---|
| resolution | 10 m Sentinel-2 | 0.3 m |
| length | **110 words** median | 47 |
| subject | country, season, climate zone, land-cover shares | objects, counts, frame positions |

> "This satellite image, captured during the fall season in Portugal, showcases
> a diverse landscape within the 'temperate, dry summer, hot summer' climate
> zone. The dominant feature is complex cultivation patterns..."

VRSBench never names a country, a season or a climate zone. Training on this
teaches a 110-word climate report — the wrong register applied harder.

**It is not dropped, it is made testable.** Per the agreed design, training runs
RarePlanes+SpaceNet first, **saves the adapter**, then continues on BEN. Both
checkpoints are scored on the same 300 rows.

> **Caveat on that design.** Sequential is not mixed. BEN landing last gets the
> strongest pull, so this answers "does adding BEN at the end help?" If BEN
> *hurts* here it might still help in a mixed corpus; if it *helps* despite
> being last, that is strong evidence.

### Public caption corpora — none writes in this register

| corpus | licence | why not |
|---|---|---|
| SkyScript | MIT | 5.2M pairs, but OSM-tag alt-text, not prose. High-res coverage "mainly US and Europe" |
| RS5M | MIT | aggregated from other datasets — licence inherits from whatever it absorbed — and machine-captioned with quality "not guaranteed" |
| ChatEarthNet | **none** | 10 m Sentinel-2, same resolution mismatch as BEN, and no licence at all |
| UnigeoCLIP | MIT | agentic captions built for CLIP alignment, not detailed description |
| RSICD / UCM / Sydney / NWPU-Captions | — | **Google Earth-derived, rejected under C32**; captions are one short sentence, a different and easier task |

All four large corpora are built for **CLIP-style alignment** — short tags. The
VRSBench register exists because GPT-4V generated it over DOTA imagery; no
public corpus imitates it.

### Trained RS captioners — six of seven unlicensed

`CapFormer`, `MLAT`, `WordSent`, Region-Driven, mask-guided-with-topic-token and
multilabel+transformer all carry **no licence**. `MetaCaptioning` is MIT but
trains on RSICD, which C32 rejects. They also target one-sentence RSICD-style
captions, and the VRSBench table shows no CNN-plus-transformer captioner near
the top — every leading score is a fine-tuned modern VLM.

**One thing was taken from them.** Region-Driven, multilabel-then-caption and
mask-guided all *extract explicit structure before generating*. That pattern is
why captions are generated from annotations rather than free-form, and why a
two-pass inventory-then-narrate variant was tested.

## Why the two halves differ

| | grounding | captioning |
|---|---|---|
| what training would teach | **sight** — the model already has it | **register** — the model lacks it |
| our corpus vs the benchmark | 9% class overlap | register generatable exactly |
| base vs published fine-tuned | **62.7 vs 60.6 — we win** | **25.2 vs 36.9 — we lose** |
| decision | ship the base model | **train** |

The same question, asked honestly of two tasks, gives two answers. Neither
generalises to the other.

## The known limitation

Our sources are **aircraft and buildings**. VRSBench captions mention vehicles
(31%), roads (21%), buildings (21%), fields (19%), water (17%), trees (14%).
So the generated corpus teaches the **form** exactly and covers perhaps a third
of the **content**.

That is a real gap with no licence-clean fix — vehicles and courts exist only in
Google Earth-derived sets. The bet is that register transfers further than
vocabulary: a model taught to write "two small vehicles towards the bottom-right"
about aircraft should carry the construction to objects it already recognises,
since the base model demonstrably sees 26 classes.

**Untested.** The first fine-tuned checkpoint measures it directly, and if
register does not transfer this document is wrong and should be revised, not
worked around.

## Open questions

1. **Does register transfer across object classes?** The whole bet. Measured by
   the first checkpoint.
2. **Does BEN help or hurt?** The saved-checkpoint ablation answers it.
3. **Do tools plus a fine-tuned model beat the fine-tuned model alone?** Today's
   evidence says combinations help when the parts fix *different* failures and
   hurt when they fix the same one — `precise prompt + SAM` scored below either
   alone. Counting is computable and belongs to a tool; phrasing is judgment and
   belongs to the model. Testable once a checkpoint exists.
4. **Official scorer.** Every number here uses the in-repo comparator (C8).

## Revision: invented surroundings removed (2026-09-05)

The first generator asserted scenery no annotation supports. RarePlanes captions
shuffled seven airport nouns -- "a runway", "taxiways", "an apron", "marked
parking stands", "paved ground", "terminal buildings", "service roads" -- into
every caption; SpaceNet captions sampled one to three of "roads", "vegetation",
"open ground", "parking areas", "water" from a building-footprint file that
records none of them. The justification written into the code at the time was
that this bought length against the brevity penalty. That is a metric argument,
and it was the wrong one: it trades a scoring gain for training the adapter to
hallucinate context, which is the single failure mode a captioning model must
not learn. Most RarePlanes tiles are not airfield centres, so a meaningful share
of those assertions were also simply false.

Both sites now pass an empty surroundings list. The **parameter stays** -- the
OpenStreetMap join fills it with tags read at each tile's own coordinates, which
is the same mechanism LHRS-Bot and SkyScript use and is the point of that work.

Removing it exposed three defects the padding had hidden:

| defect | symptom | fix |
| --- | --- | --- |
| appositive comma eaten | "The satellite image, sourced from WorldView-3 features ..." | dropped `.rstrip(",")` at both sites |
| every mode one length | concise and detailed both 23 words, so the 50/100/150-word conditioning was noise the model would learn to ignore | `_inventory` now keeps the per-quadrant box counts it was discarding; longer modes spend them on a spread sentence |
| verbless clause | "One small military bomber near the left side." | all three `OBJECT_FORMS` carry a verb; the place-fronted form drops the predicative verbs, since "Towards the left edge is visible a bomber" is not English |

Measured after the change, on a synthetic four-box tile over six seeds:
concise **25 words** mean, detailed **40 words**. The references average ~50, so
concise still runs short. That shortfall is now honest -- it is what the
annotations actually support -- and closing it is the OSM join's job, not a word
list's.

**The length lever is annotation-derived.** A group whose members fall in more
than one quadrant occupies more of the frame than one clause admits, and naming
where they fall is reading the label file. No fact enters a caption that a
label does not carry.

## Attribute supervision, after Ye et al. (TGRS 2022)

*A Joint-Training Two-Stage Method for Remote Sensing Image Captioning* adds a
multilabel classification stage whose predicted attributes condition caption
generation. Three of its four contributions are **not portable**: the
margin-based sampling operator, the dynamic contrast loss and the
attribute-guided decoder are all internals of a two-layer LSTM, and the MSO
exists only to make a non-differentiable sampling step differentiable so
gradients reach a CNN classifier. A LoRA'd transformer VLM has no such step and
no hidden state to gate.

The fourth -- the attribute prior itself -- is architecture-independent, and
their ablation makes it the largest single submodule: SPICE +3.9 / +1.8 / +3.6
on UCM-Captions / Sydney-Captions / RSICD, without joint training.

**The inference-time version is already measured, and it is a wash.** The
two-pass prompt (inventory the objects, then narrate) is exactly "predict
attributes, then condition on them":

| strategy | ROUGE-L | blind floor | gain |
| --- | --- | --- | --- |
| strong prompt, one pass | 0.252 | 0.222 | +0.030 |
| two-pass, inventory first | 0.258 | 0.232 | **+0.025** |

Six thousandths of a point, and a *smaller* margin over describing the wrong
image. Prompting the prior does not help.

What has not been tried is putting the attribute signal in the **loss**, which
is where Ye et al. put it. `_inventory` already derives that set to write each
caption, so the corpus emits a second row per tile -- ``task:
attribute_list``, "Which object types are visible in this image?", answered
with the sorted type list. Types only: no counts, no positions, or it becomes a
second captioning task rather than a prior.

Three limits, stated rather than discovered later:

- **`--attr-share` defaults to 0.5**, and only multi-type tiles qualify. Every
  tile already emits two caption rows, so attributing all of them would make a
  third of the corpus a task VRSBench never asks about.
- **SpaceNet is excluded.** A building-footprint file yields one type; the row
  would read "building" and teach nothing.
- **The transfer is an assumption.** Ye et al. measure an LSTM captioner on
  RSICD. That a LoRA adapter over Qwen3-VL gains anything from the same signal
  is untested, which is what the staged-adapter comparison is for. Set
  `--attr-share 0` to drop the task entirely.
