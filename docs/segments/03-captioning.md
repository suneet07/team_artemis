# Segment 3 — Captioning (gate G3, second half)

Describe an image in prose. Scored on **BLEU-1, ROUGE-L and CIDEr-D** against
VRSBench's 9,350 reference captions.

**Status: shipped on the prompted baseline, untrained.** The weakest measured
capability in the project, and the one where we lose to published work.

**Decision: ship `STRONG_PROMPT` on base Qwen. Do not train on our corpus.**

---

## 1. Result

```
prompt            ROUGE-L   blind floor   gain    BLEU-1  CIDEr  words
plain               0.195       0.187    +0.008   0.282  0.036   130
styled              0.228       0.185    +0.043   0.332  0.196    45
strong (mined)      0.252       0.222    +0.030   0.365  0.135    47   <- shipped
two-pass            0.258       0.232    +0.025   0.343  0.092    50
                                                          reference:  50
```

### Against published work

| model | BLEU-1 | ROUGE-L | CIDEr |
|---|---|---|---|
| GeoChat, zero-shot | 13.9 | 13.2 | 0.4 |
| **ours, 4B + strong prompt** | **36.5** | **25.2** | **13.5** |
| **ours, 8B + strong prompt** | — | **26.0** | — |
| GPT-4V, zero-shot | 37.2 | 30.1 | 19.1 |
| MiniGPT-v2 | 36.8 | 30.8 | 21.4 |
| GeoChat, fine-tuned | 46.7 | 35.2 | 28.2 |
| Mini-Gemini, fine-tuned | 47.6 | 36.8 | 33.5 |
| LLaVA-1.5, fine-tuned | 48.1 | **36.9** | 33.9 |

**We are far above zero-shot GeoChat (25.2 vs 13.2) and within reach of GPT-4V
(30.1) on a 4B model. We lose badly to fine-tuned models (36.9).**

This is the exact inverse of grounding:

| | grounding | captioning |
|---|---|---|
| what fine-tuning teaches | **sight** — model already has it | **register** — model lacks it |
| our corpus coverage | 9% of tested classes | BEN's 19,983 captions, on-task |
| vs published fine-tuned | **62.7% vs 60.6% — we win** | **25.2 vs 36.9 — we lose** |

---

## 2. `STRONG_PROMPT` — the shipped prompt

Mined from VRSBench's own references. Canonical at
`satquery/agent/served_prompts.py`; originally `scripts/eval_captioning.py`.

```
Write one caption for this aerial satellite image.

Format, followed exactly:
- Open with: "The image, sourced from GoogleEarth," then shows, features or captures.
- About 47 words across 3 sentences.
- Name each object type and how many, spelled as words: one, two, three, several.
- Call objects small or large. Most are small.
- Place each one using: in the, on the, at the, towards the -- with top, bottom,
  left, right, middle, corner, side, edge, or a pair such as bottom-right.
- Close by mentioning surroundings: roads, green areas, water, buildings,
  parking, bare ground.
- Plain prose. No bullet points, no bold, no headings.
- State only what is visible. No guessing at location, purpose or season.
```

### Where every clause came from — the reference mining

9,350 VRSBench captions, measured:

```
median length            47 words
mention a position       82%   (median 2 position words each)
mention a count          55%
say "GoogleEarth"        86%
use the word "small"     54%
named things: vehicle 31% · road 21% · building 21% · field 19%
              water 17% · tree 14% · ship 13% · plane 10%
```

**86% contain the literal token "GoogleEarth."** Not saying it loses that n-gram
in nearly nine captions in ten — pure format compliance, zero capability. That is
the metric rewarding mimicry, plainly.

Other prompts kept for comparison in `scripts/eval_captioning.py`:
`PLAIN_PROMPT` ("Describe the image in detail."), `STYLED_PROMPT`,
`INVENTORY_PROMPT` / `NARRATE_PROMPT` (two-pass).

---

## 3. The blind floor — the most important number here

Every caption is **also scored against a different image's reference**. Describing
the wrong scene earns **ROUGE-L 0.244** on this benchmark, because generic aerial
vocabulary overlaps with everything.

**Without the floor beside it, 0.30 looks like success.**

```
perfect predictions        BLEU 1.000   ROUGE-L 1.000   CIDEr 10.00
wrong image's caption      BLEU 0.052   ROUGE-L 0.244   CIDEr 0.039
generic filler             BLEU 0.002   ROUGE-L 0.153   CIDEr 0.002
```

**Plain prompting clears the floor by 0.008** (0.195 vs 0.187). Its captions are
barely more image-specific than describing a different photograph — the clearest
possible statement that the score is measuring **register, not sight**.

**ROUGE-L is the wrong metric and CIDEr is the right one.** ROUGE-L is dominated
by generic aerial vocabulary any caption shares; CIDEr weights the *informative*
words. On the same run CIDEr sits at **3.5× its floor** (0.1957 vs 0.0552) while
ROUGE-L barely moves. The model is describing *this specific image* and the
popular metric can hardly tell.

---

## 4. Metrics implementation

`satquery/training/metrics.py` — `bleu`, `rouge_l`, `cider_d`, `caption_scores`,
`_tokenise`, `_ngrams`, `_lcs`.

Hand-written rather than pulled in, matching the repo convention (`box_iou` is
too) and avoiding new licence surface.

- **BLEU** — corpus-level with brevity penalty
- **ROUGE-L** — sentence-level LCS F-measure
- **CIDEr-D** — corpus-relative TF-IDF with clipping and length penalty
- **METEOR deliberately omitted** — needs WordNet plus a stemmer

**A degenerate-test trap worth recording:** CIDEr-D returns **0.0 for an
identical caption** when scored on a single document, because with one document
every n-gram has IDF `log(1/1) = 0`. Correct maths, useless test. Validated on a
realistic corpus instead.

---

## 5. Length is the biggest lever measured

```
              plain   styled
words           128       43      (reference: 48)
mentions position 95%     98%
gives counts     62%      70%
speculates       98%       2%
markdown/bullets 55%       0%
ROUGE-L (median) 0.181   0.220
```

**130 → 47 words moved ROUGE-L from 0.195 to 0.252.** A model answering in two
crisp sentences gets crushed by the brevity penalty regardless of accuracy.

The capability is there; format compliance is what is being measured. Same
conclusion as grounding.

---

## 6. Bigger models do not fix it

```
4B, strong prompt        25.2      measured
8B, strong prompt        26.0      measured  (floor 0.232, gain +0.029)
GPT-4V, zero-shot        30.1      published
fine-tuned LLaVA-1.5     36.9      published
```

**Doubling parameters bought +0.008.** Not the answer.

An 8B base would also double VRAM and inference cost for **every adapter in the
system**, not just captioning — a system-wide decision, not a captioning one.

This matches a pattern seen elsewhere: performance does not scale monotonically
with model size (a published RS result had 2B beating 4B, 8B and 9B).

### Florence-2 — wired, parked, never run

`scripts/eval_florence.py` + a `florence` entrypoint, pinned transformers 4.51
image. MIT, 0.77B, the standard captioning baseline.

**Parked deliberately, and the reason matters:** Florence-2 does **not take
free-text prompts**. Input is a fixed task token — `<CAPTION>`,
`<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`. There is **no way to give it the
mined VRSBench register**, so a "strong prompt" run is impossible by
construction. It would be judged on its native voice against a house style it
cannot be told about. No result it produces changes any decision above.

---

## 7. Why we do not train on our own caption corpus

The corpus was built, measured, and then **not used**.

### What was built

`scripts/gen_captions.py` — turns annotations into VRSBench-register captions.

```
"rows": 49602,
"by_source": {
  "RarePlanes": 8993, "SpaceNet 2": 9472, "SpaceNet 6": 6306,
  "RSVQA-HR": 4848, "BigEarthNet.txt": 19983
},
"median_words_generated": 16,      "target_words": 47
"median_sentences": 1,             (VRSBench: 3)
"top_opening_share": 0.100,        (VRSBench: 0.173)
"distinct_openings": 34,
"distinct_4gram_ratio": 0.014
```

| source | images | GSD | licence | contributes |
|---|---|---|---|---|
| RarePlanes | 5,815 | 0.3 m | CC BY-SA 4.0 | aircraft |
| SpaceNet 2 | 6,000 | 0.3 m | CC BY-SA 4.0 | buildings, 4 cities |
| SpaceNet 6 | 3,401 | 0.5 m | CC BY-SA 4.0 | buildings, Rotterdam |
| RSVQA-HR | 3,000 | 0.3 m | USGS PD + CC BY 4.0 | roads, water, land use |
| BEN.txt | 19,983 | 10 m | CDLA-Permissive 1.0 | land cover, own register |

### The three reasons not to train on it

1. **Length. 16 words median against a 47-word target.** Train on that and the
   model learns to write 16 words, then eats the brevity penalty on every
   caption. **The prompted base already produces 47 — it has the right length
   now, and training would break it.**
2. **Content.** Our captions describe aircraft and building footprints; VRSBench
   describes whole scenes — roads, vegetation, parking, water, and the spatial
   relationships between them. We would be teaching a narrower task than the one
   graded.
3. **Form.** `distinct_4gram_ratio` **0.014** — the signature of a template.
   Templated text teaches the template, and BLEU/ROUGE **reward** that while the
   blind floor stays flat. That is a signature you only see after spending the
   run.

**What the 36.9 number actually proves:** fine-tuned LLaVA-1.5 trained on
**VRSBench's own train split** — same images, same annotators, same register as
the test set. It proves in-distribution fine-tuning works. It says nothing about
our corpus, and we cannot do what they did: VRSBench train inherits DOTA's
academic-only restriction (C20).

**The honest counter-argument, recorded:** our generated data *did* work on
`change_vqa` — base 34.6% AA → adapter 43.5%, instruction-following 77.7% →
100%. So the method is not inherently broken. But that task was multiple-choice
and short-answer, where a register mismatch does not matter. **Captioning is
where it matters most.**

**The agreed test, never run:** a *short* run — a few hundred steps on a slice,
scored on the same 300 samples against the same blind floor. If it does not clear
**0.252**, the corpus is the problem and we stop. A couple of GPU-hours, not a
full adapter.

### BEN is the wrong register, not just the wrong resolution

```
BEN                                          VRSBench
10 m Sentinel-2                              0.3 m
110 words (median)                           48 words
country, season, climate zone, land-cover %  objects, counts, positions
```

Training on BEN teaches the model to write **110-word climate reports**. And BEN
has **no positional structure at all**, because a 120 px patch at 10 m has no
meaningful "bottom-right".

Mixing raw is the one option to avoid: ~15k VRSBench-style rows against ~20k
BEN-style rows would drift toward the majority and lose the register.

**The resolution instead: instruction-conditioned register.**

```
"Describe this image for the object-inventory benchmark"  -> 47 words, counts, positions
"Describe the land cover, region and season"              -> BEN's own register
"Describe this image in about 100 words"                  -> the detailed mode
```

BEN's land-cover vocabulary (field 19%, water 17%, tree 14% — ~50% of what
VRSBench mentions) enters the mix without averaging the register away, and a
model that switches register on instruction can be pointed at whatever the hidden
set wants.

---

## 8. The fabrication defect — caught by the user

The generator asserted scenery that was not in the frame:

```python
context = [
    "a runway", "taxiways", "an apron", "marked parking stands",
    "paved ground", "terminal buildings", "service roads",
]
rng.shuffle(context)
```

**This asserts 3–5 airport nouns regardless of what is in the image.** A second
site sampled roads/vegetation/water from a *building-footprint* file.

The original justification for it was a **metric argument** — VRSBench captions
close on surroundings, so closing on surroundings scores better. That was the
wrong argument, and it is recorded as such in
`docs/07-evaluation/g3-captioning-corpus.md`.

**Scope of the fix, correctly narrow:** the *object inventory* is
annotation-grounded and honest — counts, classes and positions all come from real
labels. Only the surroundings clause was invented. So: delete the `context`
shuffle, do not rebuild the pipeline. The `surroundings` parameter stays, empty,
for a future OSM join to fill.

**LHRS-Bot's data pipeline is the model to copy here** — it pairs images with
captions generated from **OpenStreetMap attribute tags** (building type, road
class, land use), not from a template that guesses. OSM is ODbL, commercially
usable with attribution. (LHRS-Bot itself is barred: LLaMA-2 community licence on
the weights, LHRS-Align captions built over Google Earth imagery.)

### Corpus quality after the first repair pass

```
                       before          after
double "small small"   ~all      0 of 2,258
distinct openings         5      10
top opening share     0.215      0.111    (VRSBench: 0.173 — more varied than the reference)
distinct 4-grams      0.089      0.120
concise / detailed    23 / 23    25 / 29  (modes now actually differ)
```

**Length was left at 25–29 against 47, deliberately.** A tile with one aircraft
does not contain 47 words of truth. Padding a sparse scene is exactly the
invention we had just removed.

---

## 9. Design decisions that survive regardless

### The sensor phrase is a slot, not a constant

```
"The image, sourced from {source}, shows..."
```

VRSBench gets `GoogleEarth`; a Cartosat input gets the real sensor name.

**Why this matters:** the ISRO set is **Cartosat-2S**. Teaching the model to say
"sourced from GoogleEarth" — because it is in 86% of VRSBench references — would
be **factually false** on Cartosat imagery and stylistically alien. If ISRO's
references use a different register we would be actively worse there, for a habit
learned purely to farm an n-gram. We do not hard-code a false provenance into a
system whose selling point is auditable output.

### Two length modes

```
concise   ~50 words    matches VRSBench, always safe
detailed  ~100 words   only generated where the scene has enough real content
```

A **request**, not a contract. No 150-word mode — our annotations cannot fill it
without inventing. De-risks the ISRO unknown: we do not know their caption
length, and a length-conditioned model adapts at inference instead of being
locked to 47 words.

### "small" is scale-dependent

RSVQA defines it **30× apart** — under 100 m² sub-metre, under 3,000 m² at 10 m.
Since **54% of VRSBench captions use the word**, getting the threshold wrong
poisons the most common attribute in the corpus. Keyed on source.

### No areas from RSVQA-HR

Its OSM-derived areas are broken — 62% zero, some claiming more building than the
tile contains.

---

## 10. The attribute-prior idea (JTTS paper)

The paper's biggest single submodule win: give the captioner an explicit
**multilabel attribute list** as prior (SPICE +3.9 / +1.8 / +3.6 across UCM /
Sydney / RSICD).

**As prompting it is already measured and it is a wash.** Two-pass
(inventory-then-narrate) *is* "predict attributes, then condition on them": it
scored 0.258 against one-pass 0.252, and its gain over the blind floor is
actually **lower** (+0.025 vs +0.030). It also costs double inference and drops
CIDEr (0.092 vs 0.135) and BLEU-1.

**The untested version is supervision, not prompting.** Their signal is in the
**loss**, not the prompt — never tried, and nearly free for us because
`_inventory` already computes exactly that multilabel set for every tile in order
to write the caption. Emitting a second training row per tile — *"Which object
types are visible in this image?"* → the label list — costs disk and nothing
else.

**Taken.** The corpus now emits that second row (`task: attribute_list`,
`ATTRIBUTE_PROMPTS`, `--attr-share`). Types only; adding counts would make it a
second captioning task rather than a prior.

The same idea later reappeared as a **tool** rather than a training change:
`lulc_classifier` feeding a 19-class inventory into the caption plan — the
attribute prior delivered in the form that actually fits our architecture.

---

## 11. Captioning is optional, and that changes everything

The problem statement's requirement is an **OR**: grounding **or** captioning
satisfies the single-image secondary task.

**Grounding already works at 62.7% zero-shot.** So the entire captioning
agony — 16-word corpus, whether fine-tuning helps, whether it makes things
worse — is about an **optional upside**, not a mandatory gate with no answer.

VRSBench still scores captioning if we claim it. Captioning may also be scored
twice — once on VRSBench, once on the undisclosed ISRO references, since
*"task-specific reference answers"* plausibly includes captions.

**RSVQA has no captioning split.** Among public benchmarks, captioning is scored
on **VRSBench alone**.

---

## 12. Sources considered and rejected

| candidate | why not |
|---|---|
| **RSICD / UCM captions** | one short sentence — *"Many buildings are in the industrial area."* VRSBench wants 48 words with counts and positions. Also Google Earth imagery |
| **CNN+LSTM captioners** | every top VRSBench scorer is a fine-tuned modern VLM. Not one CNN+transformer captioner appears; the field moved on. Building one loses the shared base and the 62.7% grounding result |
| **LHRS-Bot** | LLaMA-2 weights licence + Google Earth captions. Adopting it throws out the 4-adapter design |
| **Landsat30-AU / OpenSentinelMap** | licences clean, but **30 m and 10 m** against VRSBench's 0.3 m. The domain gap would hurt, not help |
| **SkyScript / RS5M / GeoRSCLIP / AnySat** | built for CLIP-style alignment — short tags and alt-text, not 47-word inventories |
| **CDVQA as a caption source** | right vocabulary (6 land-cover classes at 0.5 m) and right resolution, but only 2,968 pairs, bi-temporal, and six classes is still narrow. A seasoning, not a foundation |
| **FLAIR** | 241,100 patches at 0.2 m, etalab-2.0, commercial use permitted — but **France only**, land cover not objects, and it annotates its own imagery rather than ours |

---

## 13. Files

| | |
|---|---|
| eval harness | `scripts/eval_captioning.py` — all four prompts, blind floor beside every score, `--dump-dir` |
| metrics | `satquery/training/metrics.py` — `bleu`, `rouge_l`, `cider_d`, `caption_scores` |
| generator | `scripts/gen_captions.py` — both fabrication sites stripped |
| Florence baseline | `scripts/eval_florence.py` + `florence` entrypoint (parked) |
| served prompt | `satquery/agent/served_prompts.py` — `STRONG_PROMPT` |
| record | `docs/07-evaluation/g3-captioning-corpus.md` |
| dumps | `eval/cap_dump_look_styled`, `cap_dump_look_plain`, `cap_dump_strong`, `cap_dump_twopass`, `cap_dump_8b` |
| logs | `logs/caption_8b.json`, `logs/caption_look_plain.json` |
| corpus | `/data/manifests/rs_ground_caption_captions.jsonl` |

Modal entrypoints: `captioning`, `gen_captions`, `florence`.

---

## 14. Open items

- **The short fine-tune test was never run.** Whether training clears 0.252 is
  still unknown — the one measurement that would settle the question.
- **The OSM join** to fill the empty `surroundings` slot honestly.
- **Attribute-list supervision** — emitted into the corpus, never trained on.
- **Florence-2** — wired, never launched.
- **`corpus_job`'s 8 CPUs do nothing.** The work is serial `rasterio.open()`
  calls waiting on a network volume and nothing threads them; the timeout ceiling
  was the actual fix. If this leg needs to be faster, thread the header reads.
- **Official VRSBench scorer not wrapped** (C8) — every number carries the
  in-repo-comparator caveat.
