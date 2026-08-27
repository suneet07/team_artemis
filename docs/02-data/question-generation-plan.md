# SatQuery — Question Generation Plan (QGP v1)

**Scope.** How every (image, question, answer) training sample is produced, for all
training sources, across the four adapters. Covers schema, generators, template bank,
negative/balance policy, augmentation, and the validation gates a sample must pass
before it enters a manifest.

**Status.** Proposal for Phase 0 freeze. Nothing here contradicts the master plan v3.8;
where it makes a v3.8 rule operational, the rule is cited.

---

## 1. Four principles

These are the whole plan. Everything below is machinery for enforcing them.

1. **Answers are computed, never authored.** Every answer string is derived
   programmatically from the source annotation (geometry, class list, ID set, mask
   arithmetic). No LLM ever produces a training answer. Consequence: correctness is a
   property of ~2k lines of generator code you can audit once, not of 300k samples you
   cannot.
2. **Questions are templated, then paraphrased — never invented per sample.** Cost scales
   with template count (hundreds), not sample count (hundreds of thousands). This is
   v3.8's "template-plus-augmentation recipe" (§5.3b.3, §7 item 29) made concrete.
3. **The manifest is built from annotations, before imagery is downloaded.** Annotation
   files (GeoJSON, COCO JSON, XML, CSV) are small. Generate, balance, filter and audit
   the entire QA set from them; download only the chips the surviving samples reference.
   This is the natural completion of the manifest-first staging rule (C13/C29).
4. **Every sample is traceable to a template ID and a source annotation ID.** Free at
   generation time. Buys per-template accuracy breakdowns at eval, licence provenance for
   §2.2a, and single-command retraction of any template or source later found bad.

---

## 2. Canonical sample schema

One JSONL record per sample. Frozen in Phase 0 alongside the task enum (§4.5.1).

```json
{
  "sample_id": "sn7_chgvqa_count_000481",
  "adapter": "change_vqa",
  "task": "change_count",
  "images": ["sn7/L15-0331E-1257N/2018_01.png", "sn7/L15-0331E-1257N/2019_01.png"],
  "image_roles": ["t0", "t1"],
  "modality": ["optical", "optical"],
  "effective_gsd_m": [4.0, 4.0],
  "question": "How many new buildings appeared between the two acquisitions?",
  "answer": "17",
  "answer_type": "count",
  "answer_vocab": null,
  "template_id": "CHG.COUNT.NEW.v1",
  "paraphrase_id": 3,
  "source": "spacenet7",
  "source_ann_ids": ["L15-0331E-1257N#2018_01", "L15-0331E-1257N#2019_01"],
  "licence": "CC-BY-SA-4.0",
  "provenance_chain": ["spacenet7", "planet-labs-mosaic"],
  "generator_version": "qgen-1.0.0",
  "split": "train",
  "balance_key": "change_count|bucket_11-20"
}
```

Field notes that matter:

- `effective_gsd_m` is the **post-resample** GSD, per the resolution policy — RSVQA-HR
  records 0.30, not 0.15; OEM-SAR records 0.50, not its native 0.15–0.5. The prompt
  builder reads this field, so the two can never drift apart.
- `answer_vocab` names the closed vocabulary when one exists (`rsvqa_presence`,
  `ben_clc19`, `cdvqa_yesno`). Generator asserts `answer ∈ vocab`. This makes §5.4's
  answer-format discipline mechanical rather than aspirational.
- `image_roles` disambiguates bi-temporal (`t0`/`t1`) from cross-modal (`opt`/`sar`).
  Never rely on list order alone.
- `balance_key` is what the sampler balances on (§6). Written by the generator, not
  inferred later.

**Prompt assembly** happens at collation time from the record, not at generation time.
One assembler, one place to change the GSD-conditioning format:

```
<image>{t0}</image><image>{t1}</image>
[sensor: optical | GSD: 4.0 m] [sensor: optical | GSD: 4.0 m]
{question}
```

Storing the assembled string in the record would freeze a formatting decision into
300k files. Store the parts.

---

## 3. Six annotation primitives

The 14 datasets are not 14 problems. They are six annotation shapes. Write one generator
module per shape; each dataset contributes a thin loader that emits the primitive.

| # | Primitive | Datasets emitting it | Generator module |
|---|---|---|---|
| P1 | Multi-label class vector (+ optional pixel map) | BEN.txt / reBEN | `gen_multilabel.py` |
| P2 | Bounding boxes with class + attributes | RarePlanes, LS-SSDD | `gen_bbox.py` |
| P3 | Polygon footprints, optionally with persistent IDs | SpaceNet 6, SpaceNet 7 | `gen_polygon.py` |
| P4 | Semantic segmentation mask (class per pixel) | OEM-SAR, HRSCD, BEN CLC map | `gen_segmask.py` |
| P5 | Bi-temporal pair + change mask / ID delta | SpaceNet 7, HRSCD, OSCD, self-gen Sentinel | `gen_change.py` |
| P6 | Co-registered modality pair (+ any of P1–P4 as labels) | BEN S1+S2, SpaceNet 6, OEM-SAR | `gen_crossmodal.py` |
| — | *Pre-existing QA text* (pass-through + filter) | RSVQA-LR/HR, SARLANG-1M, BEN.txt QA | `gen_passthrough.py` |

A dataset routes into more than one primitive when its annotations support more than one.
SpaceNet 6 emits P3 (footprints) and P6 (optical–SAR pair). OEM-SAR emits P4 and P6.
BEN.txt emits P1, P4 (CLC map), P6, plus pass-through QA. **That is the mechanism by
which one dataset serves several adapters with different questions** — different
primitive, different generator, different template family.

Loader contract (keeps datasets from leaking their quirks into generators):

```python
class Loader(Protocol):
    source: str
    licence: str
    provenance_chain: list[str]
    def emit(self) -> Iterator[Primitive]: ...   # yields P1..P6 records
```

---

## 4. Template bank

### 4.1 Structure

One YAML file per template family, versioned, under `configs/qgen/templates/`.

```yaml
- id: GRD.REFER.SINGLE.v1
  primitives: [P2, P3, P4]
  adapters: [rs_ground_caption]
  task: refer_box
  requires:                       # generator refuses the sample if unmet
    instance_count: 1             # exactly one instance of target class
    min_box_px: 12                # the 12-16 px legibility floor
  question: "Highlight the {class} in this image."
  answer_fn: box_xyxy_normalised
  answer_type: box
  paraphrases: 8
```

`answer_fn` names a function in the generator module. Template YAML never contains an
answer string — only the name of the code that computes one.

### 4.2 Template ID scheme

`{FAMILY}.{TYPE}.{VARIANT}.v{N}` — e.g. `VQA.PRESENCE.BINARY.v1`,
`CHG.RATIO.CLASS.v2`, `XM.AGREE.WATER.v1`. Immutable once shipped; a changed question
means a new `v{N}`, so eval numbers stay comparable across manifest rebuilds.

### 4.3 Size target

~180–240 hand-written seed templates total. Rough split: VQA 60, grounding 45, change 55,
cross-modal 30, captioning 20. Hand-written, not generated — every sample inherits the
phrasing distribution of its seed, so this is the one place to spend human attention.

---

## 5. Per-adapter question programs

### 5.1 `rs_vqa` (G2)

| Source | Primitive | Question types | Answer space |
|---|---|---|---|
| BEN.txt | P1, pass-through | presence, multi-label class, count-of-classes, comparison, MCQ | `ben_clc19`, yes/no, int |
| BEN.txt (modality dropout, ~20%) | P6 minus optical | same question set, S2 withheld | same |
| RSVQA-LR / HR | pass-through | presence, count, area, comparison, rural/urban | RSVQA closed vocab |
| SARLANG-1M | pass-through (filtered) | SAR object/scene QA | free short-form |

Rules specific to this adapter:

- **Drop Country / Climate Zone / Season MCQ from BEN entirely** (§5.4). Europe-derived
  priors are actively harmful for Indian imagery. Enforce as a hard filter in the loader,
  not a sampling weight.
- **SARLANG-1M enters filtered to the SpaceNet 6 + OEM-SAR portions only** (C47). Filter
  at load, and assert on `provenance_chain` that no DFC2023 / SARDet-100K record survives.
- **RSVQA language-prior guard.** Its OSM-templated questions are the documented
  answer-centric exploit. Mitigation is in §6.3, and it is not optional.
- Modality-dropout samples must be *generated as their own records*, not produced by a
  runtime augmentation that hides the fact from the manifest. A dropped-modality sample
  has `modality: ["sar"]` and one image. It must be countable and auditable.

### 5.2 `rs_ground_caption` (G3)

| Source | Primitive | Question types |
|---|---|---|
| BEN.txt referring LULC + point detection | P1, P4 | refer-region, point-to-class, region caption |
| RarePlanes (real + synthetic crops) | P2 | refer-box single, count, attribute-conditioned refer, exhaustive detect |
| LS-SSDD | P2 | refer-box ship, count, **absence (negative)** |
| SpaceNet 6 | P3 | refer-building, count, spatial-relation refer |
| OEM-SAR (manual subset) | P4 | refer-segment (building/water/road), class-presence |

Rules:

- **Vocabulary is closed and stated: land-cover regions, buildings, aircraft, ships.**
  No template may name a class outside it. A generator that sees `vehicle`, `bridge`,
  `harbour` raises — that path belongs to `object_box_fallback` (§4.6.9), not to training.
- **Box legibility floor.** Reject any refer-box sample whose target is under ~12 px on
  its longest side at effective GSD. Ships at LS-SSDD's 10 m are ~3 px — those images
  supply *counting and absence* questions, and refer-box only where a vessel is genuinely
  resolvable. Do not train "highlight the ship" against 3 px of speckle.
- **RarePlanes attribute questions are cheap and high-value.** The 10 fine-grained
  attributes (wingspan, propulsion, wing-shape, canards) are exactly the "major objects"
  detail the PS wants and cost nothing to template. Strip the rest of the metadata at load
  (specs doc flags the RAM cost).
- **Synthetic RarePlanes enters only as 512² annotation-centred crops** (resolution policy
  §4). A whole 1920×1080 synthetic frame silently becomes 0.84 m — 2.4× off the real half.
  Generator asserts crop size before emitting.
- **LS-SSDD pure-background images stay in, as absence questions.** "Are there any ships
  in this image?" → "no". Deleting them destroys the PBHT property the dataset was built
  around, and absence is a question type, not an empty sample.
- **OEM-SAR: manual subset only for anything the answer depends on.** Pseudo-labels
  (~68% mean agreement, Bareland 0.02 IoU) are weak supervision at best. Hard rule:
  bareland never appears as a refer target from this source; pseudo-labelled records
  carry `label_confidence: weak` and are capped at a fixed fraction of the mix.

### 5.3 `change_vqa` (G4)

| Source | Primitive | Question types |
|---|---|---|
| SpaceNet 7 (primary) | P5 + persistent IDs | new/demolished count, "what changed", presence-of-change, ratio, direction-of-development |
| HRSCD | P5 semantic | class-transition ("what did this parcel become"), semantic change presence |
| OSCD (+S1) | P5, P6 | binary change presence, cross-modal change agreement |
| Self-generated Sentinel (India) | P5 index-diff | change presence, index-driven change type, region caption |

Rules:

- **SpaceNet 7's persistent building IDs are the highest-value annotation in the entire
  change inventory.** Set arithmetic on ID sets between timestamps gives exact,
  expert-grounded counts — appeared, disappeared, persisted, ratio. Those are precisely
  the CDVQA question types where published models sit at 32–37%. Generate them densely.
- **Quantitative answers train the adapter *and* validate `change_stats` (D4).** Same
  computation, two consumers. Emit the ground-truth counts into the D4 validation set from
  the same pass.
- **`POLYGON EMPTY` is a valid scene, not a parse error.** It generates "no buildings" and
  "no change" answers. Loader handles it; specs doc flags it as a jump scare.
- **RGBA alpha is nodata.** Any tile whose alpha-invalid fraction exceeds a threshold
  (start at 5%) is rejected before question generation — otherwise counts are computed
  over pixels the model cannot see.
- **Index-differenced labels are weaker than expert ones. Mark them.** Self-generated
  samples carry `label_confidence: derived`. Keep them capped and keep CDVQA as the
  *evaluation* set (PS-nominated, so evaluating is compliance). Never let derived labels
  become the majority of the change mix now that SpaceNet 7 is primary.
- **HRSCD: change-stratified tile selection is a generation-time decision.** ~220k
  candidate tiles, ~10–20k needed. Select change-containing tiles plus stratified
  no-change; the no-change tiles are the negatives that stop the adapter answering "yes,
  something changed" unconditionally.

### 5.4 `optsar_fusion` (G5)

| Source | Primitive | Question types |
|---|---|---|
| BEN.txt S1+S2 (primary) | P6 + P1 | joint-extraction VQA, cross-modal class agreement, "which modality shows X" |
| SpaceNet 6 | P6 + P3 | generated cross-modal building captions, agreement questions |
| OEM-SAR | P6 + P4 | water/building agreement, single-pol land-cover |
| SARLANG-1M (filtered) | pass-through | SAR-side QA |

Rules:

- **Disagreement is a first-class question type.** D1's whole thesis is that optical and
  SAR disagree in physically explicable ways. Generate samples where the answer *is* the
  disagreement ("optical suggests water, backscatter does not — cloud shadow"). Ground
  truth for these comes from SpaceNet 6's real footprints and OEM-SAR's water masks, which
  is exactly what makes cross-modal agreement a measured number rather than a reported one.
- **Pol-dropout samples generated explicitly**, mirroring modality dropout: synthesise
  single-pol from SpaceNet 6 quad-pol, emit as its own record with `modality: ["sar"]` and
  a pol field. Inference must never be the first time the model meets a configuration.
- **Sub-metre floor of ~15% of the mix** (C27 GSD stratification), or the 10 m Sentinel
  majority dilutes the sub-metre sources to noise.

---

## 6. Negatives, distractors, balance — where accuracy actually comes from

Template quality sets the ceiling. This section sets the floor.

### 6.1 Negatives are generated, not collected

For every presence/absence family, the generator emits matched negatives from the *same*
imagery distribution:

- **Class negatives**: ask about a class absent from this tile but common in the source.
  Not a random class — a *plausible* one, drawn from the source's own class frequency
  distribution. "Is there a harbour?" over farmland teaches nothing; "is there water?"
  over farmland teaches a decision.
- **Spatial negatives**: "highlight the building in the north-west" where buildings exist
  elsewhere but not there → refusal / empty answer.
- **Temporal negatives**: no-change pairs (HRSCD stratified no-change, SpaceNet 7
  consecutive months with zero ID delta).
- **Modality negatives**: object visible in optical, absent in SAR.

Target: **30–40% negatives** in every presence-style family. Below ~25%, the adapter
learns "yes" as a prior.

### 6.2 Balance keys

Balance is enforced at manifest-build, on `balance_key`, per source per family:

| Family | Balance on |
|---|---|
| presence / binary | answer yes:no ≈ 1:1 |
| count | log-spaced buckets (0, 1, 2–3, 4–10, 11–30, 31+), flat over buckets |
| class / MCQ | inverse-frequency capped — BEN CLC is heavily skewed |
| refer-box | target class × object-size bucket |
| change | change:no-change ≈ 1:1; change-type flat |
| GSD | sub-metre floor 15% of each relevant mix |

Counting is where balance matters most and is most often skipped: an unbalanced count
distribution teaches the modal count, and modal-count guessing is exactly the 32–37%
plateau.

### 6.3 The language-prior guard (mandatory)

The RSVQA jump scare — procedurally generated questions with rigid syntax let a model
answer without looking. Three controls, all cheap:

1. **Blind baseline, run before training.** Fit a text-only model (question → answer) on
   the generated manifest. Per-template accuracy above chance by a wide margin means that
   template is answerable from language alone. Fix or drop it. This single check is the
   highest-value item in the whole plan and costs an afternoon.
2. **Answer-balance per question surface form**, not just per family. If "Is there a
   *road*?" is 80% yes across the corpus, the phrase itself is a shortcut.
3. **Adversarial pairing**: where possible emit the same question over two images with
   opposite answers. Forces the visual path.

Same three controls catch the mirror failure — image-only shortcuts — if you also run an
image-only baseline on the grounding families.

---

## 7. Paraphrase augmentation protocol

Constrained, offline, batched, cached. Never per-sample, never touching answers.

1. Input: one seed template question string + its `requires` block + example rendered
   instances.
2. Ask for **8 paraphrases** preserving: the answer-determining semantics, every slot
   token (`{class}`, `{n}`, `{direction}`), and the answer type. Vary register, word
   order, question form (interrogative / imperative / elliptical).
3. **Automatic rejection filters** before a paraphrase enters the bank:
   - any slot token missing or duplicated → reject
   - semantic-drift check: paraphrase must not add or remove a qualifier (a second model
     answers "do these two questions have the same answer for any image?" — disagreement
     rejects)
   - near-duplicate of an existing paraphrase (edit distance / embedding threshold)
   - length outside 0.5–2.0× the seed
4. **Human sign-off on the bank, once.** ~200 templates × 8 = ~1,600 strings. One
   afternoon of reading. This is a *tiny* review surface compared to auditing samples, and
   it is the last point where a phrasing error is cheap to fix.
5. Frozen bank is committed and versioned. Manifest builds sample `paraphrase_id` from it
   deterministically (seeded), so a rebuild reproduces byte-identically.

Cost: a few dollars of API calls. Reused across every sample the template touches.

---

## 8. Validation gates

A sample enters a manifest only after passing all of V0–V5. Gates are asserts in the
build, not a checklist someone remembers.

| Gate | Check | On failure |
|---|---|---|
| **V0 — licence** | `provenance_chain` traced to an imagery programme; source on the cleared list; no excluded ancestor (SECOND, LEVIR, DOTA, DFC2023, SARDet-100K, Google Earth) | hard fail build |
| **V1 — schema** | all required fields; `answer ∈ answer_vocab` where declared; image files exist; `image_roles` length matches `images` | drop sample, log |
| **V2 — geometry** | boxes inside image bounds; target ≥ legibility floor at effective GSD; crop size matches resolution policy; alpha/nodata fraction under threshold | drop sample, log |
| **V3 — semantics** | answer recomputed by an *independent* second implementation of `answer_fn` and compared | drop + alert (a mismatch is a generator bug, not a data issue) |
| **V4 — balance** | per-source per-family distribution within tolerance of §6.2 targets; negative fraction in range; sub-metre floor met | fail build, resample |
| **V5 — shortcut** | text-only baseline per template not above threshold; image-only baseline on grounding families | quarantine template, require fix |

V3 is worth the duplicated effort on the ~6 highest-volume `answer_fn`s only (count,
presence, refer-box, change-count, class-list). Full duplication everywhere is waste.

---

## 9. Human audit protocol

Automated gates catch what you thought of. The audit catches what you did not.

- **100 samples per source**, stratified across that source's template families, rendered
  as an HTML contact sheet: image(s), assembled prompt, answer, template ID, overlaid
  geometry.
- Reviewer marks each: correct / wrong-answer / bad-question / bad-crop / ambiguous.
- **Gate: ≥97% correct and ≥95% unambiguous** before the source enters a training
  manifest. Below that, fix the generator and re-audit — never hand-patch samples.
- Re-audit 30 samples per source after any `generator_version` bump.
- Second reviewer on 20% of the audit set; report agreement. Two people who disagree about
  what the answer should be have found a real ambiguity in the template.

Budget: ~1,400 samples total across all sources. Two days of one person, once.

---

## 10. Build order

Ordered so the critical path unblocks first and each step de-risks the next.

| Step | Work | Output | Gate |
|---|---|---|---|
| 1 | Freeze sample schema + task enum + template ID scheme | `configs/qgen/schema.json` | reviewed |
| 2 | Write `gen_passthrough` + BEN.txt loader | BEN QA/caption/referring samples | V0–V3 |
| 3 | **Blind-baseline harness** (V5) — build it early, run it on step 2 | per-template shortcut report | — |
| 4 | Seed template bank, hand-written, all families | ~200 templates | human review |
| 5 | `gen_bbox` + RarePlanes, LS-SSDD loaders | grounding samples, objects | V0–V4, audit |
| 6 | `gen_polygon` + SpaceNet 6/7 loaders | grounding + change samples | V0–V4, audit |
| 7 | `gen_change` + SpaceNet 7 ID arithmetic, HRSCD, OSCD | change samples + D4 validation set | V0–V5, audit |
| 8 | `gen_segmask` + OEM-SAR, HRSCD semantic | region grounding | V0–V4, audit |
| 9 | `gen_crossmodal` + BEN S1+S2, SpaceNet 6 pairs, pol-dropout | fusion samples | V0–V4, audit |
| 10 | Paraphrase bank generation + filters + sign-off | frozen bank v1 | human review |
| 11 | Manifest builder: balance, split, dedup, emit download list | 4 adapter manifests | V4 |
| 12 | Download only manifest-referenced chips | staged datasets | — |

Steps 2–4 are the critical path for `rs_vqa` and `rs_ground_caption`, matching the
staging order in §2.2. Steps 5–9 parallelise across people cleanly — one primitive each,
shared schema, no coupling.

---

## 11. Per-dataset generation hazards

Every item is drawn from `DATASET_SPECS.md` or the resolution policy, restated as a rule
the generator enforces. This table is the checklist a loader PR is reviewed against.

| Source | Hazard | Generation-time rule |
|---|---|---|
| BEN.txt | 10/20/60 m bands differ in array size | upsample to 120² before compositing; assert shape |
| BEN.txt | multi-label, not single-label | never emit softmax-style "what is the class"; use presence/multi-label |
| BEN.txt | CLC class skew | inverse-frequency cap on class questions |
| BEN.txt | Country/Climate/Season MCQ | hard-filtered at load |
| SpaceNet 7 | RGBA alpha = nodata | reject tiles over nodata threshold before counting |
| SpaceNet 7 | `POLYGON EMPTY` | valid → "none" answers, not a parse error |
| SpaceNet 7 | persistent IDs | ID set arithmetic for all count/ratio answers |
| SpaceNet 6 | MAG-POL is 6-band | extract VV/HH for single-pol synthesis; never feed 6 bands as RGB |
| SpaceNet 6 | no image–text annotations | P3 → template captions; nothing pass-through |
| OEM-SAR | already 8-bit stretched | no dB thresholds; no reverse-calibration assumptions in answers |
| OEM-SAR | pseudo-labels, Bareland 0.02 IoU | manual subset for answer-bearing samples; bareland never a refer target |
| RarePlanes | 1920×1080 synthetic frames | 512² annotation-centred crops only; assert before emit |
| RarePlanes | 10 dense attributes | keep as attribute-question source; strip remainder at load |
| LS-SSDD | 3-channel JPG, no calibration | pure CV source; no radiometric questions |
| LS-SSDD | pure-background images | retained as absence negatives; never filtered |
| LS-SSDD | ships ~3 px at 10 m | count/absence questions; refer-box only above legibility floor |
| HRSCD | coarse raw labels | HRSCD-Clean subset only |
| HRSCD | 2006 imagery non-redistributable | train-only flag; excluded from demo/artifact manifests |
| HRSCD | 100 MP scenes, 220k candidate tiles | change-stratified selection at generation time |
| HRSCD | aerial relief displacement | do not generate questions depending on nadir geometry |
| OSCD | SAR already in dB, negative floats | normalisation expects dB domain; no linear-DN assumptions |
| SARLANG-1M | 0.1–25 m spread | GSD injection mandatory; assert `effective_gsd_m` present |
| SARLANG-1M | tainted sub-datasets | filter to SpaceNet 6 + OEM-SAR portions; assert on provenance |
| RSVQA-HR | 15 cm native | resample to 0.30 m at staging; record effective, not native |
| RSVQA-HR/LR | template-driven language priors | V5 blind baseline mandatory; per-surface-form balance |

---

## 12. What this plan does not decide

Named so they are not silently assumed:

1. **Caption target length.** The cost model flags that captioning targets are longer than
   the 24-token synthetic answers used in the timing harness. Set a target length before
   the budget is trusted.
2. **Precompute vs online preprocessing.** Out of scope here, but it determines whether
   the generator writes tensors or image paths. Decide before step 11.
3. **Exact per-adapter sample counts.** The cost model says the reserve is large and the
   right use is more `rs_ground_caption` samples. This plan produces as many as the
   balance targets allow; the cap is a training decision.
4. **Whether `rs_vqa` and `optsar_fusion` merge** (triage rung 2). If they do, their
   manifests concatenate cleanly — the schema is shared. No generation work is wasted
   either way.
