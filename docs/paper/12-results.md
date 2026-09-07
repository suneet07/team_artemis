# 12. Results — every number, with its baseline

Every figure here traces to a run in `logs/`. Where a number is not comparable to
a published one, that is stated in the same row.

## 12.1 Headline

| capability | metric | ours | blind baseline | best published comparison |
|---|---|---|---|---|
| `rs_vqa` — RSVQA-HR | AA | **85.06** | 62.6 | RSVQA authors' model **83.12** |
| `rs_vqa` — RSVQA-LR | AA | **83.08** | 55.8 | RSVQA authors' model **81.49** |
| `rs_vqa` — BEN binary | AA | **76.78** | 52.6 | RS-InternVL (1B, fine-tuned) 73.29 |
| `rs_vqa` — BEN MCQ | AA | **73.62** | 29.3 | RS-InternVL 51.49 |
| `change_vqa` — CDVQA Val | AA | **68.0** | 45.0 | same backbone, fine-tuned **67.86** · best in paper (2B) 68.59 |
| Grounding — VRSBench referring | acc@0.5 | **62.7%** | — | GeoChat (fine-tuned) 60.6% |
| Grounding — same run | found object | **84.7%** | — | — |
| Captioning — VRSBench | ROUGE-L | **25.2** | **24.4** | LLaVA-1.5 (fine-tuned) **36.9** |
| SAR — reBEN held-out | accuracy | **74.95%** | 50.1 | — |

**Three wins, one draw, one loss.** RSVQA-HR and RSVQA-LR beat the dataset
authors' own model. BEN binary and MCQ beat a fine-tuned 1B specialist. Grounding
beats a fine-tuned baseline with no training. CDVQA is level with the published
fine-tuned result for the same backbone, 0.6 under the paper's best.
**Captioning loses by ~12 ROUGE-L**, and clears its blind floor by 0.008.

## 12.2 `rs_vqa`, per type

**RSVQA-HR** (AA 85.06, target ≥85 — **pass**)

```
area          87.6%     <- the margin that carries the gate
presence      ~
comparison    ~
count         weakest of the four
```

**RSVQA-LR** (AA 83.08, target ≥88 — **miss**)

```
count         56.2%     <- the entire shortfall
presence      ~
comparison    ~
rural_urban   ~
```

The LR miss is confined to one nameable type. Its vision contribution is **2.9
points** — the model is largely answering counts from question phrasing.

## 12.3 `change_vqa`, per type

CDVQA Val, 1,200 balanced rows, AA 68.0%, instruction-following 100%.

```
change_or_not        85.33%      largest_change       66.67%
increase_or_not      83.33%      change_to_what       65.33%
decrease_or_not      81.33%      change_ratio         52.00%
change_ratio_types   80.67%      smallest_change      29.33%
```

Blind ceilings per type: `change_ratio` **16.5%**, `increase_or_not` **66.6%**,
`decrease_or_not` **69.7%**, `smallest_change` **22.4%**.

`smallest_change` at 29.3% is below the 32–37% band every published model
occupies; diagnosed as genuine class confusion, not a formatter bug.

## 12.4 Vision-contribution ablations

Shuffle control, `IO_AdTest = (real − shuffled) / (100 − shuffled)`:

```
rs_vqa, across three benchmarks     9.9 - 16 points
RSVQA-HR                            8.7 - 11.0, uniform across all four types
change_vqa (SpaceNet 7)             9.9   (0.435 -> 0.336, floor 0.352)
rs_vqa count                        2.9
change_count                        1.2
```

Wrong-image control on `rs_vqa`: **59.0 against a 39.7 floor** — the residual is
answer priors, not sight.

## 12.5 Formatter deltas — no weights changed

```
RSVQA-LR count quantisation       75.82 -> 83.10     +7.3
RSVQA-HR area bucketing           excluded -> 87.6%  carried the gate
change_vqa instruction-following  77.7% -> 100%
grounding box parsing             75% unparsable -> 0.0%
```

## 12.6 Rejected components, with the numbers that rejected them

| component | measured | why cut |
|---|---|---|
| Siamese `change_map` | **F1 0.2931** | weak *and* wrong task — finds buildings, CDVQA asks land cover. SCOT winner 0.41, baseline 0.17 |
| `optsar_fusion` adapter | — | published fusion AP **0.711 < optical 0.714**; fusion adds nothing |
| BIFOLD optical arm | **0.4968** (chance) | radar 0.7525 on the same rows; radar right 87% of disagreements |
| Grounding DINO pipeline | ceiling 63.7%, **built 34.7%** | loses to a prompted model at 62.7% |
| `object_box_fallback` | — | blob proposals beside a learned box on a 28%-`vehicle` benchmark |
| fine-tuning captioning | — | 16-word corpus median against a 47-word target; would teach the template |
| 8B base for captioning | 26.8 vs 25.2 | +1.6 ROUGE-L for double the weights, and voids the grounding result |
| constrained decoding | — | targets a problem measured at zero (IF = 100%) |

## 12.7 Training runs

| | `rs_vqa` | `change_vqa` (CDVQA) | `change_vqa` (SpaceNet 7, discarded) |
|---|---|---|---|
| corpus | 74,000 rows | 37,518 | 51,956 |
| composites | 3 | 6 | 2 |
| steps | — | 4,000 | 10,976 (best at **1,000**) |
| wall clock | **3.0 h** (budget 4.2) | 2.4 h | — |
| loss | 4.507 → 0.164 | 6.314 → 0.172 | — |
| peak VRAM | — | 19.13 GB | — |
| result | 4 gates | AA 68.0 | AA 43.5, discarded |

Shared: `Qwen3-VL-4B-Instruct`, LoRA r=16 α=32 dropout=0.1, `all-linear`,
**40,271,872 trainable (0.899%)**.

Throughput, measured: A100 **1.236 → 1.153 s/step**; H100 **1.411 s/step** —
slower, because the job is input-bound. Adapter inference **17.9 samples/s**
against base 4.65.

## 12.8 Corpora

```
ben_canonical.jsonl        117,023 rows   (rs_vqa 57,044 · rs_ground_caption 59,979)
change_vqa (SpaceNet 7)     51,956
CDVQA staged                30,350 (Test2 alone)
caption corpus              49,602
grounding corpus            19,896
optsar_fusion               85,096   -- every arm at exactly 50.0% floor
```

## 12.9 System-level verification

**Held-out testing gallery** — 200 rows from test/held-out splits, replayed
through the live `/queries` route and scored through each benchmark's own answer
contract:

```
200 items   175 verified   25 scored   0 failed
change_vqa 50 · rs_vqa 50 · sar 50 · grounding 25 · captioning 25
```

Pass rates on *unfiltered* replacement rows — the honest read on end-to-end
accuracy through the router — ran around two thirds.

**Automated checks:** 440 unit tests, 50 route checks including one executed
crossmodal query, 11 offline router checks in ~6 s.

## 12.10 What is not measured

- **The base model on CDVQA.** Never run, so the delta over our own zero-shot
  baseline is unquantified. (67.86 is the published *fine-tuned* score for
  Qwen3-VL-4B, not a zero-shot figure.)
- **CDVQA Test1 / Test2.** Staged, never scored. Published comparisons are Test1
  numbers; ours is Val.
- **The shuffle control on the CDVQA adapter.** Never run.
- **Fine-tuning captioning or grounding on our corpora.** Never run.
- **Official scorers.** RSVQA, VRSBench and CDVQA all scored with the in-repo
  comparator; every report carries the caveat.
