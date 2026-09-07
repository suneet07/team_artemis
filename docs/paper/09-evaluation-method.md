# 9. How we measured — the methodology, and why it changed conclusions

If one part of this project generalises beyond it, it is this chapter. Most of
the large score movements were caused by measurement decisions, not modelling
ones, and three of them reversed a conclusion we had already acted on.

## 9.1 Never quote a score without its blind baseline

Every number in this project is reported beside the score a system gets **without
looking at the image**. The gap is the claim; the raw number is not.

Three forms, one per task shape:

| task | blind baseline | how |
|---|---|---|
| VQA | **majority-answer ceiling**, per question type | answer the most common class for that type |
| captioning | **wrong-image floor** | caption a *different* photograph and score it against the reference |
| any | **shuffled-image ablation** | pair each question with another row's images |

Why this is not optional, in one line: **RSVQA-LR presence questions are 76.3%
"yes"** on the official test split. Water presence is 86.2% yes, residential
83.5%. A model that never opens the image scores that.

The blind baselines actually used:

```
BEN binary        52.6        RSVQA-LR      55.8
BEN MCQ           29.3        RSVQA-HR      62.6
CDVQA (AA)        45.0        VRSBench captioning   0.244 ROUGE-L
optsar (all arms) 50.1        change_vqa (SpaceNet 7, by design)  35.2
```

Getting the optsar corpus to an **exact 50.0%** floor took three iterations
(60.6% → 53.9% → 50.0%). A benchmark whose majority answer scores 60% makes a
75% result look far better than it is.

## 9.2 The shuffle control

```
IO_AdTest = (real - shuffled) / (100 - shuffled)
```

Pair each question with another row's images. This keeps the input distribution
exactly as the model expects and removes **only** the correspondence — unlike a
blank-image ablation, which also takes the input off-distribution, so a drop
there is weaker evidence (the model may be reacting to the strangeness rather
than the missing content). Shuffle is preferred; blank is the sanity check on
the sanity check.

Measured vision contribution:

```
rs_vqa, across three benchmarks     9.9 - 16 points
RSVQA-HR                            uniform 8.7 - 11.0 across all four types
change_vqa (SpaceNet 7)             9.9 points   (0.435 -> 0.336)
rs_vqa count                        2.9 points
change_count                        1.2 points
```

The last two are the useful ones. **Counting barely uses the image**, on both
adapters, which is consistent with `count` being the one failing gate and is the
direct argument for routing counting through deterministic arithmetic.

The uncomfortable result is reported too: shown a **wrong** image, `rs_vqa` still
scores **59.0 against a 39.7 floor**, purely from knowing which answers go with
which question phrasings. That is a property of VQA benchmarks generally, and it
is why every number here carries its floor.

One caveat kept with the data: the shuffle run **predates** the LR count
quantisation, so LR's 61.78 shuffled figure pairs with 75.82, not with 83.08.

## 9.3 Answer contracts — the formatter is part of the measurement

A benchmark scores a *string*. If the model emits `"about 12"` and the benchmark
expects `"between 11 and 100"`, that is a formatting failure being recorded as a
perception failure.

`contract_for(benchmark, question_type)` resolves a contract, and
`format_answer` applies it. The contracts:

```
ben_binary_vqa · ben_mcq · ben_caption · ben_ref_detection
rsvqa_lr_presence · rsvqa_lr_comparison · rsvqa_lr_count · rsvqa_lr_rural_urban
rsvqa_hr_presence · rsvqa_hr_comparison · rsvqa_hr_count · rsvqa_hr_area
cdvqa_change_or_not · cdvqa_change_ratio · cdvqa_change_ratio_types · ...
```

What this was worth:

```
RSVQA-LR count quantisation      75.82 -> 83.10     +7.3 points
RSVQA-HR area bucketing          excluded -> 87.6%  carried the gate at 85.06
change_vqa instruction-following 77.7% -> 100%      no weights changed
```

**Not one of those changed a parameter.** They are the largest single-step
improvements in the project.

Two details that matter more than they look:

- **Order in the contract table is load-bearing.** It returns the *first* prefix
  match, and `change_ratio` is a prefix of `change_ratio_types`. Listed the wrong
  way round, every ratio-by-type row scores against the plain ratio contract —
  same vocabulary today, so it would look fine, and would break silently the day
  the two diverge.
- **Instruction-following is reported separately.** If IF is ~100%, format
  learning worked and the accuracy number means what it says. If IF is low, the
  accuracy is measuring formatting and the report says so instead of hiding it.
  All four `rs_vqa` gates report **IF = 100.0%**, which is why constrained
  decoding was rejected — it targets a problem measured at zero.

## 9.4 Average accuracy, not overall accuracy

RSVQA and CDVQA both report **average accuracy** — the mean over question types,
weighting each type equally. Overall accuracy weights by row count, so a
benchmark with 4,002 `comp` rows and 100 `rural_urban` rows lets one type decide
the score.

This is also why the CDVQA training corpus caps each type near 5,000 rows: AA
weights `change_ratio` as heavily as `change_or_not`, and raw CDVQA gives the
latter 7.2× the data while the former is the hardest type with the lowest blind
ceiling (16.5%).

## 9.5 Metrics can disagree, and that is information

On captioning, ROUGE-L and CIDEr point in different directions and the split *is*
the finding:

- **ROUGE-L** is dominated by generic aerial vocabulary that every caption
  shares. Our best prompt clears the wrong-image floor by **0.008**.
- **CIDEr** weights informative words by inverse document frequency and sits at
  roughly **four times its floor**.

A related trap: **CIDEr-D returns 0.0 even for an identical caption** when there
is a single reference, because every n-gram has IDF `log(1/1) = 0`.

A second metric was added permanently to grounding for the same reason. acc@0.5
is a cliff — a box on the right object but slightly loose scores zero, exactly
like a box on the wrong side of the image. Breaking down 150 rows:

```
94  correct at 0.5          62.7% acc@0.5
33  right object, loose box
23  genuinely lost          84.7% found the object
```

Both are reported, because **the ISRO set's scoring threshold is undisclosed**.

## 9.6 What we do *not* claim

**The official scorers were never wrapped.** RSVQA, VRSBench and CDVQA are all
scored with the in-repo comparator. Every report carries the caveat verbatim:

> Scored with the in-repo comparator, NOT the official benchmark scorer. Valid
> for run-to-run comparison; not quotable as a benchmark result.

CDVQA ships **no evaluation code at all**, so there is nothing to defer to there;
our AA definition matches the paper's. For RSVQA and VRSBench the caveat stands.

**CDVQA Val is not CDVQA Test1.** The 55.3% baseline and 68.6% SOTA are Test1
numbers. Our 68.0% is Val. Test1 and Test2 are staged and were never scored.

**Our BEN MCQ set is not the paper's.** We generated the questions and dropped
the Location, Season and Climate subtypes — which their table shows are Qwen's
*strongest* (47.76, 45.93). Our set is therefore **harder**, and the honest
comparable zero-shot baseline is **35.72, not the headline 37.55**.

## 9.7 Three times measurement reversed a decision

**The false leakage alarm.** An early check reported **97.2%** train/val overlap
and nearly triggered a corpus rebuild. The comparison was wrong — validation was
being compared against a combined manifest that *embeds* validation. Train and
val share zero images.

**The `--limit` slice artefact.** `--limit 160` takes the *first* N rows rather
than sampling, so the slice is not representative. It produced a 0.727 that
looked like a result. A balanced 1,200-row run gave **68.0%**.

**Deleting bad data made the benchmark easier.** Removing blank SpaceNet 7 months
was obviously correct and took the `change_presence` blind ceiling from **53.9%
to 74.3%** — because most "nothing changed" examples *were* the blank months.
Fixed with a twin-safe rebalance after filtering.

A fourth, smaller: a 40-pair spot check reporting "0 unreadable" against a **42.3%
archive corruption rate** was luck, not evidence. Sample sizes are now stated
beside every claim.

## 9.8 The layer this project was missing

Three layers of verification, and it took a long time to notice the gap between
them:

| layer | cost | catches | missed |
|---|---|---|---|
| unit tests (440) | 20 s | logic, contracts, formatters | a crash on the crossmodal path — nothing ran a two-modality pair |
| `verify_routes.py` | 6 s | every task reachable, every planned tool executable, one crossmodal query *executed* | answer quality — stub adapters return a fixed string |
| held-out gallery (200 rows) | ~25 min GPU | quality through the router, on real imagery, against published gold | — |

**Every published number was produced by calling a component directly.** The
gallery was the first thing to exercise all of them *through the router*, and it
found nine bugs, seven of which were invisible to the other two layers
([11](11-failure-atlas.md)). That is the methodological result of this project:
component-level benchmarks do not measure a system.
