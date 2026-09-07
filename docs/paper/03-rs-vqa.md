# 3. `rs_vqa` — beating the dataset authors on their own benchmark

## 3.1 Result

| benchmark | ours | for comparison | blind baseline |
|---|---|---|---|
| **RSVQA-HR** (0.30 m aerial) | **85.06 AA** | RSVQA authors' own model **83.12** | majority class 62.6 |
| **RSVQA-LR** (10 m Sentinel-2) | **83.08 AA** | RSVQA authors' own model **81.49** | majority class 55.8 |
| BEN binary | **76.78** | RS-InternVL (1B, fine-tuned) **73.29** · zero-shot Qwen 61.96 · GPT 60.39 | 52.6 |
| BEN MCQ | **73.62** | RS-InternVL **51.49** · zero-shot 37.55 · GPT 34.93 | 29.3 |

Both graded splits clear. **RSVQA-LR misses its internal target of 88**, and the
shortfall is confined to one type that can be named: `count` at **56.2%**.

The RS-specific VLMs sit at the bottom of the same table — **GeoChat 50.82,
SkyEyeGPT 48.87, LHRS 48.23** on binary VQA. Adopting one would have been a
downgrade, which is why none was.

## 3.2 The honest form of the claim

The temptation is "our 4B beats their 8B". That is not what was measured. The
defensible sentence is:

> A 4B model, fine-tuned on 74,000 in-domain examples and given three spectral
> composites plus scale conditioning, scores **76.78** where an 8B zero-shot RGB
> baseline scores **61.96** on the same questions and the same split.

Every clause does work. Remove the fine-tuning, the composites, or the scale
prefix and the number is not the same number.

The genuinely favourable comparison is the other one: **RS-InternVL is 1B, is a
2026 model, and was fine-tuned on the full BEN.txt** — roughly 50× our data per
task — and we beat it 76.78 to 73.29. The BEN.txt paper settles the scale
question on our behalf: its 1B model beats every zero-shot entry in its table,
including a 2T-parameter GPT. **This task does not need scale. It needs
vocabulary and in-domain supervision.**

One caveat is carried permanently: **our MCQ set is not the paper's.** The
questions are generated from BEN.txt by our own pipeline, and we dropped the
Location, Season and Climate subtypes — which their table shows are Qwen's
*strongest* (47.76, 45.93). So our restricted set is **harder**, and the honest
comparable zero-shot baseline is **35.72, not the headline 37.55**.

## 3.3 Training

```
base            Qwen3-VL-4B-Instruct           (Apache 2.0)
LoRA            r=16, α=32, dropout=0.1, targets = all-linear
trainable       40,271,872 parameters  (0.899%)
composites      3   (true_colour, false_colour, short_wave)
corpus          74,000 rows
wall clock      3.0 h  (budgeted 4.2 h)
loss            4.507 → 0.164
```

`all-linear` rather than attention-only is a deliberate divergence from the
published Qwen change-VQA recipe, which uses Q/K/V/O only. That choice is
**4.6× more adaptation**: attention-only LoRA on the comparison model gives
8.7M trainable against our 40.3M, on a base half the size and at 384 px rather
than 512.

Measured throughput, because it decided the hardware:

```
A100   1.236 → 1.153 s/step
H100   1.411 s/step        ← slower: the job is input-bound, not compute-bound
adapter inference  17.9 samples/s   vs base 4.65
```

The H100 result is worth keeping. A faster accelerator made the run *slower*,
because the bottleneck was image loading. Buying compute would not have helped.

## 3.4 The two formatter fixes that carried the gates

Neither of these changed a weight. Both moved the headline number more than any
modelling decision did.

### `count` quantisation — worth +7.3 points

RSVQA does not score exact integers. It scores **ranges** — `0`, `between 1 and
10`, `between 11 and 100`, `between 101 and 1000`, `more than 1000`. Scoring the
model's integer against a range answer gave **27%**. Quantising into the
benchmark's own answer space took **RSVQA-LR from 75.82 to 83.10**.

The deeper option was considered and deliberately deferred: bucket the
*training* manifest too, so the model learns to emit `between 101 and 1000`
directly rather than guessing an integer that is then binned. Step 2 usually
beats Step 1 — the binning becomes the model's job instead of a post-processing
step — but Step 1 was sufficient to clear the gate.

### `area` bucketing — the margin that carries HR

`area` was initially **excluded as unusable**. Reading the protocol properly,
bucketing into the paper's five classes and retraining turned it into **87.6%**,
and that is the margin that carries HR at **85.06 against a target of 85**.

It is also why the restricted score (84.20) is *lower* than the official one
(85.06): the official protocol's buckets are kinder than our restricted set's.
Both are reported.

## 3.5 Does it actually look at the image?

The shuffle control pairs every question with **another row's images**, keeping
the input distribution identical and removing only the correspondence:

```
IO_AdTest = (real − shuffled) / (100 − shuffled)
```

Measured vision contribution: **9.9 to 16 points** across three benchmarks.
**RSVQA-HR is the healthiest** — uniform 8.7 to 11.0 point drops across all four
types.

The uncomfortable half of the result is reported too. **Shown a wrong image, the
model still scores 59.0 against a 39.7 floor** — purely by knowing which answers
go with which question phrasings. That is a real property of VQA benchmarks, not
a defect we introduced, and it is exactly why the blind baseline travels with
every number in this document.

`count` has a **2.9-point** vision contribution. That is the smallest of any
type, and it is consistent with `count` being the failing gate: the model is
substantially answering counts from the question's phrasing rather than from
pixels. A measured, named limit rather than a surprise.

One caveat, measured rather than assumed: the shuffle run **predates** the LR
count quantisation, so LR's 61.78 shuffled figure pairs with 75.82, not with the
current 83.08.

## 3.6 What was rejected

| option | why |
|---|---|
| **Constrained decoding** | targets a problem measured at zero — instruction-following is **100.0%** on all four gates |
| **A second architecture** (RS-InternVL's recipe) | breaks the one-shared-base commitment (C1); kept as stretch-only |
| **Falling back to a large general LLM** | the BEN.txt paper measures GPT at **60.39 / 34.93** on exactly the tasks our adapter scores 76.78 / 73.62 |
| **Adopting an RS-specific VLM** | GeoChat 50.82 · SkyEyeGPT 48.87 · LHRS 48.23 — all below our number, and LHRS is licence-barred anyway |

## 3.7 Where it is weak, stated plainly

**Counting.** LR `count` is 56.2%, the vision contribution there is 2.9 points,
and it is the sole reason the LR gate misses 88. The fix is known (bucket the
training manifest) and was not run.

**The two BEN gates are self-imposed.** BEN binary and BEN MCQ are our own
sanity checks on our own generated questions. The **PS-scored** numbers are
RSVQA-LR, RSVQA-HR and VRSBench. Both are reported, and which is which is
stated.
