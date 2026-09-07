# Segment 5 — `change_vqa` (gate G4)

Bi-temporal change VQA: two co-registered images of the same place at different
dates, plus a question. **Mandatory** — the problem statement says *"change
description or change-based visual question answering from a bi-temporal image
pair shall be mandatory"* and names **CDVQA** as the evaluation benchmark.

**Status: trained twice.** SpaceNet 7 first, then retrained on CDVQA. The CDVQA
adapter is what ships.

**Result: AA 68.0% on official CDVQA Val.**

---

## 1. Result — the shipped adapter

**1,200 rows of official CDVQA Val, 150 per question type:**

```
AA 68.0%   ·   overall 68.0%   ·   instruction-following 1.000   ·   0 unparsable
```

(AA and overall are identical because the split is exactly balanced.)

| type | n | acc |
|---|---|---|
| change_or_not | 150 | 85.3% |
| increase_or_not | 150 | 83.3% |
| decrease_or_not | 150 | 81.3% |
| change_ratio_types | 150 | 80.7% |
| largest_change | 150 | 66.7% |
| change_to_what | 150 | 65.3% |
| change_ratio | 150 | 52.0% |
| **smallest_change** | **150** | **29.3%** |

### The three numbers to read it against

```
45.0%   mean blind ceiling  (majority answer per type, no vision)
55.3%   published CDVQA baseline        (Test1)
68.6%   CDVQA SOTA — Qwen3.5-2B + LoRA r=16 α=32, the same config we use (Test1)
```

**+23 points over the blind ceiling. That is the defensible claim**, not the
absolute number.

### What this number is NOT

Published CDVQA figures are **Test1** through the **official scorer**. Ours is
**Val** through the **in-repo comparator**. Two uncontrolled differences, so
68.0 is *"we are in that range under our own measurement"*, not *"we matched
SOTA"*. Closing it needs a Test1/Test2 run through the official scorer.

For reference, the published field:

| method | AA (Test1) | OA (Test1) |
|---|---|---|
| original CDVQA baseline | 55.3% | 65.9% |
| SOBA | 60.3% | 69.2% |
| VisTA (prior SOTA) | 65.9% | 73.1% |
| **Qwen3.5-2B + LoRA (r=16, α=32)** | **68.59%** | **74.74%** |

**Every published CDVQA number is a fine-tuned number** — all trained on CDVQA's
train split. Evaluating a zero-shot model against them is not a fair comparison.

---

## 2. Training run (CDVQA)

```
--adapter change_vqa
--manifest /data/manifests/change_vqa_cdvqa_combined.jsonl
--image-root /data --split train --out /data/checkpoints/change_vqa
--micro-batch 8 --grad-accum 1 --steps 4000
--snapshot-every 1000 --val-every 250 --val-batches 20
--smoke-passed /data/checkpoints/change_vqa_smoke/run.json
```

```
4,000 steps in 2.41 h (2.167 s/step)
loss 6.314 -> 0.172
peak 19.13 GB
LoRA r=16 α=32 all-linear, 40,271,872 trainable (0.899%)
```

**val_loss fell to the last check** — no overfitting:

| step | val_loss | train_loss |
|---|---|---|
| 250 | 0.2163 | 0.2952 |
| 1750 | 0.1909 | 0.2069 |
| 2250 | 0.1638 | 0.0893 |
| 2750 | 0.1763 | 0.1580 |
| 3250 | 0.1637 | 0.1661 |
| 3500 | 0.1615 | 0.1565 |
| 3750 | 0.1615 | 0.1556 |
| **4000** | **0.1608** | 0.1718 |

Snapshots at `adapter_step1000/2000/3000/4000`, final at `adapter/`. All five
hashes distinct; `adapter/` differs from `adapter_step4000` because the final
save happens after the step-4000 snapshot.

### Two false overfitting alarms

The trainer fired *"val_loss has risen for three consecutive checks"* at step
2750. **It was noise — val set four new bests afterwards.**

**Root cause:** `--val-batches 20` is only **160 samples** per check, and the
printed `train_loss` is a **single batch**. Do not trust that warning at this
setting.

A related trap: **val loss is the wrong signal here.** With a 19-answer closed
vocabulary, loss saturates fast while *accuracy* keeps moving — especially on
`change_ratio` and `smallest_change`, the types with the lowest ceilings. The
real read comes from scoring the adapters, not from the curve.

---

## 3. The corpus — CDVQA

### Staging

`scripts/stage_cdvqa.py` joins CDVQA's **three separate JSONs** by id —
`*_questions.json`, `*_answers.json`, `*_images.json` — since questions carry no
answer and answers carry no image.

`index_pairs()` verifies **each PNG actually decodes** and prefers readable
duplicates. `cap_per_type()` **samples rather than truncates**.

### Type distribution and per-type blind ceilings

```
type                 rows    blind ceiling
change_or_not      22,668       57.0%
change_ratio_types  9,585       47.3%
increase_or_not     7,580       66.6%   <- most gameable
decrease_or_not     7,524       69.7%   <- most gameable
change_to_what      4,926       37.4%
largest_change      4,722       41.6%
smallest_change     4,722       24.0%
change_ratio        3,148       16.4%   <- hardest type, 7.2x least data
                              mean 45.0%
```

**AA is the mean over types**, so `change_ratio` counts as much toward the score
as `change_or_not` while getting a seventh of the training signal — *and* it is
simultaneously the hardest type. Published work shows the same pattern: the 2026
paper says *"fine-grained ranking and quantitative change estimation remain
difficult"*, which is exactly `change_ratio` and `largest/smallest_change`.

### Capping

```
change_or_not       22,668 -> 5,000     change_to_what      4,926 -> 4,926
change_ratio_types   9,585 -> 5,000     largest_change      4,722 -> 4,722
increase_or_not      7,580 -> 5,000     smallest_change     4,722 -> 4,722
decrease_or_not      7,524 -> 5,000     change_ratio        3,148 -> 3,148
                                                    ≈ 37,518 rows
```

### Manifests

```
change_vqa_cdvqa_combined.jsonl   37,518 train + 1,200 stratified val   <- trained on
change_vqa_cdvqa_capped.jsonl     37,518  (train only)
change_vqa_cdvqa_train.jsonl      64,875  (uncapped)
change_vqa_cdvqa_val.jsonl        16,162  (full Val)
cdvqa_heldout.jsonl                1,200  (balanced, 150/type)   <- scored on
change_vqa_cdvqa_test.jsonl       38,770  (official Test, ceiling 44.7%)
change_vqa_cdvqa_test2.jsonl      30,350  (official Test2)
```

### Verification

| check | result |
|---|---|
| joins re-verified independently, 400 random rows | **0 mismatches** |
| gold answers surviving their own contract | **16,162 / 16,162** |
| rows with no contract | **0** |
| train/val leakage | **zero** |
| official Train vs Val image pairs | 1,574 vs 393, **intersection empty** |

---

## 4. Licence

**CDVQA is Apache-2.0** — `YZHJessica/CDVQA`, with questions, answers and image
lists directly in the repository, no download portal. It is also PS-nominated.

### Archive recovery

- The imagery ships as **RAR5**, which **p7zip cannot read** — 9,871 errors,
  **42.3% of files unreadable**.
- Fixed with **`unar`** (Debian main).
- **55 pairs remain unrecoverable**; 279 val rows lost to them.
- A 40-pair integrity spot-check reported "0 unreadable" against 42% corruption —
  the sample was **under 1%**, pure luck. Recorded as a lesson about sample size.

---

## 5. The formatter — 8 CDVQA contracts

`satquery/evalcli/formatter.py`.

```python
CDVQA_CLASSES = ("NVG_surface", "buildings", "low_vegetation",
                 "trees", "water", "playgrounds")
CDVQA_RATIO_BUCKETS = (... "0", "0_to_10", "10_to_20", ...)
```

Eight contracts registered, with **`ratio_types` ordered before `ratio`** — a bare
prefix match would otherwise let `change_ratio` shadow `change_ratio_types`.

### `_vocab_key` — the bug that manufactured wrong answers

```python
def _vocab_key(value: str) -> str:
    return re.sub(r"[\s_\-]+", " ", value.strip().lower())
```

CDVQA spells classes `NVG_surface` and `low_vegetation`. A model answering
*"low vegetation"* or *"NVG surface"* **has answered correctly**, and matching raw
strings marked both wrong.

**Worse: the answer was then abstained to the first vocabulary entry**, so *"low
vegetation"* scored as `NVG_surface` — **a wrong answer manufactured by the
scorer.**

The abstention path still exists (`vocabulary[0]`, which is `NVG_surface`, the
most common `smallest_change` gold answer), so it can still produce
correct-looking answers by accident. **It did not fire on our run** —
`instruction_following` is `recovered / total` where `recovered = sum(f.matched)`,
and we measured **IF 1.0000, unparsable 0** across all 1,200 rows. Zero
abstentions.

Longest-candidate-first matching handles the ratio buckets: `"0 to 10"` must win
over the bare `"0"`, which is also a valid answer meaning no change at all.

---

## 6. `composites = 6` — the trap

```python
_COMPOSITES = {"rs_vqa": 3, "change_vqa": 6, "optsar_fusion": 3, "rs_ground_caption": 2}
```

**Six = 3 composites × 2 dates.** The plan's 1,536-vision-token budget depends
on it.

**`--composites 6` is mandatory when scoring this adapter.** `bakeoff` defaults
to leaving rows as-is, which would score the model *on an input shape it never
saw*.

### Truncation was made to raise

Silently keeping the first three views of a six-view sample hands the model a
**single date**, trains without error, converges on the answer prior, and
surfaces only as a mediocre score after a 15.7-hour run.

| case | before | now |
|---|---|---|
| `change_vqa`, 6 views | **3** — second date destroyed | **6** |
| `rs_vqa`, 1 view | 3 (repeat) | 3 — unchanged |
| `change_vqa`, 4 views | 6, wrapped, duplicating date 0 | **passes through as 4** |
| `rs_vqa`, 6 views | 3, silently | passes through |
| beyond `MAX_VIEWS = 8` | — | **raises** |
| any, `resize_views=True` | resizes | resizes — opt-in preserved |

Repetition is only safe from a **single** image — the RSVQA case, where it is
in-distribution because training did it. The current contract lets a genuine
multi-view row through with its own count (SpaceNet 7 gives 2, Sentinel gives 6);
only the budgeting ceiling raises.

---

## 7. The first adapter — SpaceNet 7

Trained before CDVQA was staged, on a self-generated corpus. **Superseded, but
the corpus design work is the most methodologically careful thing in the
project.**

```
change_vqa adapter_step1000:  AA 43.5%, IF 100%, shuffled 33.6% (floor 35.2%)
  presence 65.2 (+15.2), compare 66.0 (+15.2), where 38.2 (+10.1),
  magnitude 39.8 (+8.2), count 18.2 (+0.8), direction n=36
adapter_step4000: AA 45.7, but 41.5 on n>=100 types — step1000 wins by 4
base model:       AA 34.6%,  IF 77.7%
```

**Training earned +8.9 points, and the single biggest win was instruction-
following: 77.7% → 100%.** The base model knew things and answered in the wrong
format.

### Corpus design — complementary pairs

**8,678 rows, 60 AOIs, 555 distinct images**, validated against real imagery.
None missing, none blank, **no train/val leakage, no twins straddling the split**
(the split is **AOI-level**, because a pair-level split leaks validation scenes
into training).

Every question ships with a **twin carrying the opposite answer** (Goyal et al.
2017), keyed on the answer so marginal balance survives capping. That is what
removes the language prior.

Result: **every closed-vocabulary question type sits at its chance blind
ceiling.**

| SpaceNet 7 type | rows | ceiling |
|---|---|---|
| change_compare | 15,200 | 50.0% |
| change_count | 14,478 | 19.5% |
| change_where | 13,408 | 25.4% |
| change_magnitude | 6,380 | 28.6% |
| change_presence | 1,896 | 50.0% |
| change_direction | 594 | 33.3% |

`dominant_change` was **disabled** — it came back with a **100% blind ceiling**,
always answering NDBI. The index needs low-signal pixel masking before it is
trustworthy.

A caught flaw: `change_count` was **57% of the entire corpus**, because the cap
is per-answer and a type with 21 answers gets 21×400 slots while yes/no gets
2×400. The adapter would have spent most of its training on counting.

### The shuffle control — IO_AdTest

```
IO_AdTest = (real − shuffled) / (100 − shuffled)
step 1000: (43.46 − 33.60) / 66.40 = 14.8%
```

**9.9-point vision contribution.** The model is genuinely looking at the imagery,
not just learning answer priors. Motivated by Chappuis et al. 2023 (RS VQA
language bias — models did not change answers when forest was erased).

**This is why the SpaceNet 7 work was not wasted**, and why the same control must
be run on the CDVQA adapter. **It has not been.**

### Why 43.5% and 68.0% are not comparable

- **Different corpora, different taxonomies.** 43.5% is
  `change_compare/count/direction/magnitude/presence/where` on SpaceNet 7;
  68.0% is CDVQA's eight land-cover types.
- **Our val set was deliberately built so guessing scores 35.2%** — every type at
  chance by construction. CDVQA is not balanced that way; zero-shot Qwen scored
  **AA 67.86** on it against 34.6% on ours. **Same model, same weights, 33 points
  apart**, purely because our corpus removed the priors CDVQA leaves in.

### Why retraining on CDVQA rather than mixing

1. **Every published CDVQA number is CDVQA-trained.** Mixing makes ours
   incomparable to the only reference points that exist.
2. **The vocabularies are disjoint.** SN7 answers are counts and compass
   directions; CDVQA's are six land-cover classes and ratio buckets. Different
   answer spaces with no overlap is **interference risk, not transfer**.
3. **The format-compliance argument does not hold.** SN7 took IF from 77.7% to
   100%, but that is what *any* fine-tune on a fixed answer vocabulary does.
   CDVQA teaches it just as well.
4. **65,967 rows is ample** for a LoRA at 0.899% trainable parameters.

### The mismatch that made retraining necessary

Our corpus is SpaceNet 7 — **building instances**, appeared/demolished/counted.
CDVQA asks about **land-cover change** across six categories.

| axis | problem |
|---|---|
| **Taxonomy** | of the CDVQA types, only `change_presence` → `change_or_not` maps cleanly. `change_direction/compare/count/where` have **no CDVQA equivalent at all** |
| **Vocabulary** | our magnitude bands are `none / a few / dozens / many` — **counts**. CDVQA's are `very small … very large` — **ratios**. The adapter was trained to emit the wrong vocabulary |
| **Semantics** | `cdvqa_largest_change` needs land-cover categories. Our corpus is **buildings only** — nothing in training ever mentions vegetation or water |
| **Resolution** | CDVQA is **0.5 m**; our training imagery is **4 m Planet**. An 8× gap — the same objection we raised against LEVIR-CD, now pointing at us |

---

## 8. `smallest_change` at 29.3% — investigated, genuine

The one weak type. Diagnosed rather than guessed.

**Not a scoring artifact.** IF 1.0000 / unparsable 0 means the abstention path
never fired.

**Not superlative confusion.** Hypothesis: the model answers "what changed"
rather than "what changed *least*". Tested by checking how often it gives the
same answer to both superlatives on the same scene:

```
model gave SAME answer to both:  4/44 = 9.1%
gold has same answer to both:    8/44 = 18.2%
```

It differentiates **more** than the truth does. Hypothesis dead.

**It is a genuinely hard type.** The gold distribution is nearly flat —
34/32/32/30/15/7 across six classes, largest class 22.7%, which *is* the 22.4%
blind ceiling. **No majority to exploit**, unlike `largest_change` where one class
holds 40% (and we score 66.7%).

Predicted vs gold distributions:

```
smallest_change  PRED: buildings 57, trees 42, NVG 18, low_veg 16, water 12, playgrounds 5
                 GOLD: NVG 34, buildings 32, low_veg 32, trees 30, water 15, playgrounds 7

largest_change   PRED: NVG 75, buildings 47, low_veg 26, trees 2
                 GOLD: NVG 60, buildings 50, low_veg 35, trees 4, water 1
```

On `largest_change` the prediction tracks gold closely; on `smallest_change` it is
near-inverted.

**Verdict: genuine class confusion on a known-hard type.** Published models manage
32–37%; we are at 29.3%. Slightly under the field, not broken.

Raw generations: `logs/dump_heldout/Qwen_Qwen3-VL-4B-Instruct_cdvqa.jsonl`.

---

## 9. The leakage scare — and the correction

An early check reported **97.2% image overlap** between the val manifest and the
combined manifest, read as contamination.

**It was wrong.** The comparison was against the *entire* combined manifest, which
**embeds the 1,200 Val rows**. Val overlapped itself.

**Compared against `split == "train"` rows only, the overlap is zero.** Official
Train (1,574 image pairs) and official Val (393 pairs) share **no images**.

The measurement that matters: `--limit 160` **slices** the first 160 rows rather
than sampling. That slice was **59% yes/no questions** with four types on n=8–12,
and scored **0.727** — above published SOTA. **An artifact of which rows the slice
contained.** Balanced at 150/type, the real number is 68.0%.

**Always score the balanced 1,200 (`cdvqa_heldout.jsonl`), never a bare
`--limit`.**

`change_vqa_cdvqa_val.jsonl` (16,162 rows) is the full official Val and is fine;
the 1,200-row file is a balanced subset of it.

---

## 10. `change_map` — the abandoned measurement route

Fully documented in `TEAM_CONTEXT.md` §5. Summary:

**Commissioned, not rejected.** The pitch was measurement over inference: CDVQA
`smallest_change` sits at 32–37% for every published model, and the plan targeted
**>60% via `change_stats` arithmetic** on a mask.

**Trained and failed:** Siamese U-Net on 12,004 SpaceNet 7 pairs, best **F1
0.2931 / IoU 0.1717** at step 2750, then drifted down.

**Alternatives all died:** TinyCD non-commercial · Open-CD Apache code but weights
trained on LEVIR-CD (academic-only), binary masks are non-directional, 8×
resolution gap · SpaceNet winners' weights are Apache-2.0 and genuinely published
but wrong task and resolution.

**The decisive reason is task mismatch, not weight quality.** CDVQA asks about
**land cover**; our detector finds **buildings**, and the measurement pipeline
needs footprints at inference that CDVQA does not have. At F1 0.95 it still could
not answer a CDVQA question.

**What survives:** the arithmetic half is **finished and verified** — router plus
`change_stats` scored **100% AA on 2,012 rows**, given SpaceNet 7's own
ground-truth footprints. **Perception is empty, not reasoning.** If a land-cover
segmenter ever lands on a clearable licence, the reasoning layer is already built.

---

## 11. Files

| | |
|---|---|
| adapter | `/data/checkpoints/change_vqa/adapter` on **both** `suneetsharan14` and `suneet-sharan-ug25` |
| snapshots | `adapter_step1000/2000/3000/4000` |
| stager | `scripts/stage_cdvqa.py` |
| SpaceNet 7 generator | `scripts/gen_change.py` |
| formatter | `satquery/evalcli/formatter.py` — `CDVQA_CLASSES`, `CDVQA_RATIO_BUCKETS`, `_vocab_key` |
| results | `logs/cdvqa_heldout_step4000.md` / `.json` |
| raw generations | `logs/dump_heldout/Qwen_Qwen3-VL-4B-Instruct_cdvqa.jsonl` |
| slice artifact (do not quote) | `logs/cdvqa_step4000.md` — the 0.727 |
| corpus summary | `logs/cdvqa_corpus.json` |
| tests | `tests/test_cdvqa_formats.py` (20) |

The benchmark command, with the composites trap accounted for:

```bash
MODAL_PROFILE=suneet-sharan-ug25 modal run --detach scripts/modal_phase0.py::bakeoff \
  --benchmarks cdvqa=/data/manifests/cdvqa_heldout.jsonl \
  --adapter /data/checkpoints/change_vqa/adapter \
  --composites 6 --out logs/cdvqa_heldout_step4000
```

---

## 12. Open items

- **The shuffle control has never been run on the CDVQA adapter.** It is the
  decisive check and the harness exists — one eval run. Without it, we cannot say
  how much of 68.0% is vision. The SpaceNet 7 adapter measured 9.9 points.
- **Test1 and Test2 are staged but never scored.** Those are the graded splits;
  the 55.3% and 68.6% reference numbers are Test1. `cdvqa_test_balanced.jsonl`
  (500/type, 4,000 rows) is already uploaded and ready.
- **Official CDVQA scorer not wrapped** — and note the repo ships **no evaluation
  code at all**: 12 JSON files, a LICENSE and a citation. There is no official
  scorer to defer to. Our AA (mean of per-type accuracies) matches the paper's
  definition; what differs is the split and the answer matching.
- **`smallest_change`** — 29.3%, below the published 32–37% band. Cause understood,
  no fix.
- **916 Test rows skipped** for missing imagery; 55 pairs unrecoverable.
