# 4. `change_vqa` — trained twice, on two different datasets

This is the most instructive capability in the project, because the first
version was finished, measured, and then thrown away.

## 4.1 Result (the second training, the one that ships)

**CDVQA Val, 1,200 balanced rows, average accuracy — the metric CDVQA reports.**

| | AA |
|---|---|
| **ours — Qwen3-VL-4B + LoRA** | **68.0%** (Val) |
| Qwen3.5-2B + LoRA — best in the published paper | 68.59% (Test1) |
| **Qwen3-VL-4B + LoRA — the same paper, our backbone** | **67.86%** (Test1) |
| VisTA (prior SOTA, specialised) | 65.9% |
| SOBA | 60.3% |
| CDVQA paper baseline | 55.3% |
| **blind ceiling** (majority answer per type) | **45.0%** |

Per type, and the spread is the story:

```
change_or_not        85.33%      largest_change       66.67%
increase_or_not      83.33%      change_to_what       65.33%
decrease_or_not      81.33%      change_ratio         52.00%
change_ratio_types   80.67%      smallest_change      29.33%
instruction-following 100%
```

**Level with the published result for our own backbone**, and 0.6 behind the
paper's best — which is a *smaller* model, because performance on this task does
not scale monotonically with size. VisTA and SOBA are purpose-built change-VQA
architectures and both sit below a generic LoRA recipe.

## 4.2 The comparison that matters: same base model, same recipe

The published table is a **size sweep of the same method**, and the paper states
that *"performance does not scale monotonically with model size"* — 2B beat 4B,
8B and 9B. Every entry is a fine-tuned LoRA result on CDVQA's train split:

| model | AA (Test1) |
|---|---|
| Qwen3.5-2B + LoRA (r=16, alpha=32) | **68.59** — best in paper |
| **Qwen3-VL-4B + LoRA** — *our exact base* | **67.86** |
| Qwen 8B + LoRA | 66.94 |
| VisTA (prior SOTA, specialised architecture) | 65.9 |
| SOBA | 60.3 |
| original CDVQA baseline | 55.3 |

**Ours: 68.0 on Val with Qwen3-VL-4B + LoRA r=16 alpha=32.**

So against the published result *for our own backbone and essentially our own
configuration* — **67.86** — we are level. Against the paper's best, a 2B model,
we are 0.6 behind. And we reached it **inside a single epoch at effective batch
8**, where the published runs trained full epochs.

The second finding is the one that validated the whole approach: **plain LoRA on
a Qwen beats every specialised architecture on this task** — 68.6 against VisTA's
65.9. There was never a reason to adopt SOBA or VisTA, both of which carry the
same architecture-swap and licence problems as LHRS-Bot.

### Caveats that do apply

**Ours is Val; the published numbers are Test1.** Not the same split. CDVQA's
Test1 and Test2 are staged and were never scored, so the comparison above is
indicative rather than exact.

**Scored with the in-repo comparator**, not an official script — CDVQA ships no
evaluation code at all, and our AA definition matches the paper's, but the
caveat stands.

**We never ran the base model on CDVQA.** That is a genuine gap in the ablation
table — we can say the adapter matches the published fine-tuned result for its
backbone, but we cannot state the delta over our own zero-shot baseline. One
eval run would close it.

## 4.3 The first training, and why it was discarded

The original `change_vqa` was trained on **SpaceNet 7 / MUDS**: 51,956 generated
rows over 1,237 distinct mosaics and 11,983 image sets.

```
base model            AA 34.6%    instruction-following 77.7%
adapter (step 1,000)  AA 43.5%    instruction-following 100%
shuffled images       AA 33.6%    against a 35.2% guessing floor
vision contribution   9.9 points
step 4,000            AA 45.67%
```

The corpus was built so that **guessing scores 35.2%** — every type at chance by
construction. That is why 43.5% is not comparable to a CDVQA number: CDVQA is
*not* balanced that way, and the same weights score 33 points apart on the two
sets purely because our corpus removed the priors CDVQA leaves in.

### Why it was thrown away

It answered the wrong questions. Our SpaceNet 7 corpus is built on **annotated
building footprints**, so it asks about buildings appearing and disappearing.
**CDVQA — the graded benchmark — asks about land cover.** The vocabulary the
adapter learned was not the vocabulary it would be scored on.

Two structural defects in that corpus made the decision easier:

- **`change_count` was 57% of the entire corpus**, and 93% of it unpaired,
  because in real SpaceNet 7 one direction routinely blows past the 20 cap: 200
  buildings appeared forward means 200 demolished backward, so the reverse
  question is never asked.
- **`change_direction` had 594 rows**, which is why its evaluation landed on
  n=36 and produced a meaningless 0.333 → 0.667 swing.

### One artefact of that run is worth keeping

**The best checkpoint was step 1,000 of 10,976**, and it survived only because a
monitored checkpoint existed — the rolling one had been overwritten long before
anyone knew it was wanted. Validation every 100 steps is what made the peak
visible at all.

## 4.4 The second training

```
base            Qwen3-VL-4B-Instruct
LoRA            r=16, alpha=32, dropout=0.1, all-linear
trainable       40,271,872  (0.899%)
corpus          37,518 samples  (CDVQA, capped ~5,000 per type)
composites      6   -- the budget for 3 composites x 2 dates
steps           4,000        wall clock 2.4 h      peak VRAM 19.13 GB
loss            6.314 -> 0.172
```

The cap matters. **Average accuracy weights every type equally**, and raw CDVQA
gives `change_or_not` 7.2x the rows of `change_ratio` — which is simultaneously
the *hardest* type and the one with the **lowest blind ceiling (16.5%)**. Capping
each type near 5,000 rows stops the metric being decided by the easiest,
best-supplied class.

Blind ceilings per type, which is what makes the per-type scores readable:
`change_ratio` **16.5%**, `increase_or_not` **66.6%**, `decrease_or_not`
**69.7%**.

## 4.5 `smallest_change` — 29.3%, diagnosed rather than excused

Below the 32–37% band every published model occupies. Investigated and closed as
**genuine class confusion**, not a formatter bug: instruction-following is 100%,
the gold distribution is flat, and the blind ceiling is 22.4%.

The original plan intended to beat this with arithmetic rather than inference —
route the question to `change_stats` and compute the answer from a change mask,
targeting >60%. That plan died with the change detector, below.

## 4.6 What was rejected

### The Siamese change detector — `change_map`

Built, trained, measured at **F1 0.2931**, and cut. Two reasons, and the second
is decisive:

1. **The number is weak.** For scale, the SpaceNet 7 SCOT competition winner
   scored 0.41 and the baseline 0.17 — so ours is not absurd, but it is not
   useful either. The frequently quoted **F1 0.91 belongs to LEVIR-CD at 0.5 m**;
   SpaceNet 7 is 4 m, where a building is 8x smaller in pixels. That number does
   not transfer and was never ours to claim.
2. **It detects the wrong thing.** It finds *buildings*. CDVQA asks about *land
   cover*. It could not answer the graded benchmark at any quality.

LEVIR-CD itself was barred anyway — academic-only, and the deliverable ships.
This is where *"a licence-clean 80% beats an unusable 91%"* was decided.

The arithmetic that would have consumed the mask is not wasted work, and it is
worth recording exactly what it proved:

> **`change_stats` scored 100% on all 2,012 rows given ground-truth footprints.**

Perfect arithmetic, zero perception. The pipeline was never the bottleneck —
perception was. That is the honest epitaph for the whole branch, and it is why
`change_stats` and `change_map` are both absent from the served change plans
today rather than emitting a warning on every query.

### Ideas considered and declined

| option | why not |
|---|---|
| take the SpaceNet 7 winners' weights | right idea, wrong week — 41 GB of staging, a new label space, a units problem our own docs flag, and no way to score the result against CDVQA, which is graded and had zero measurements |
| Open-CD | Apache-2.0 code, weights from LEVIR-CD (academic-only) or S2Looking (no stated licence) |
| per-date building segmentation instead of change | easier supervision (positive class 5–15% instead of 0.5%), but still the wrong task for a land-cover benchmark |

## 4.7 Serving notes

`composites: 6` is a **sequence-length budget, not a forced count**. A CDVQA row
carries two real images and is served with two — the loader repeats a single
view up to the budget and passes multi-view rows through untouched. Reading 6 as
"always send six" pads a duplicate date onto every pair; reading it as "truncate
to six" hands the model one date and it converges on the answer prior. Both
mistakes were made and fixed ([11](11-failure-atlas.md)).

Which checkpoint serves was verified **by hash rather than by name**, because
two Modal accounts hold a directory called `change_vqa/adapter` and only one is
the CDVQA run:

```
ug25 change_vqa/adapter            sha256 46129c61...   <- deployed
s14  change_vqa/adapter            sha256 46129c61...   identical
s14  change_vqa/adapter_step4000   sha256 3b21a4b4...   different (SpaceNet 7 era)
```

`46129c61` is the file the 68.0% was measured on, and it is what the deployment
loads.
