# Segment 1 — `rs_vqa` (gates G1 + G2)

Single-image VQA. The first adapter trained, the only one that beat its
published comparators, and the source of most conventions the later adapters
inherited.

**Status: trained, evaluated, shipped.** 3 of 4 gates pass.

---

## 1. Result

| gate | score | target | verdict |
|---|---|---|---|
| BEN.txt binary VQA | **76.78** | ≥70 | **PASS** |
| BEN.txt MCQ | **73.62** | ≥45 | **PASS** |
| RSVQA-HR *(official, all four types)* | **85.06** | ≥85 | **PASS** by 0.06 |
| RSVQA-HR *(restricted, area excluded)* | 84.20 | ≥85 | FAIL |
| RSVQA-LR | **83.08** | ≥88 | FAIL by 4.9 — entirely `count` |

Instruction-following **100% on 20,447 answers**, 0 unparsable.

### Against published work

| | ours | best published |
|---|---|---|
| BEN binary | **76.78** | 73.29 (RS-InternVL 1B, fine-tuned) · 61.96 (Qwen-8B zero-shot) · 50.82 (GeoChat) |
| BEN MCQ | **73.62** | 51.49 · 37.55 |
| RSVQA-HR AA | **85.06** | 83.12 — *the dataset authors' own model* |
| RSVQA-LR AA | **83.08** | 81.49 — *the dataset authors' own model* |

First of fourteen on both BEN.txt tasks. Beats the RSVQA authors on **both**
splits, winning every question type except `count`.

Per-type against the RSVQA paper:

| RSVQA-HR type | paper | ours |
|---|---|---|
| Presence | 90.43 | **92.5** |
| Comparison | 88.19 | **89.5** |
| Area | 85.24 | **87.6** |
| Count | 68.63 | **70.6** |
| **AA** | 83.12 | **85.06** |

| RSVQA-LR type | paper | ours |
|---|---|---|
| Presence | 87.46 | **91.4** |
| Comparison | 81.50 | **91.6** |
| Rural/Urban | 90.00 | **93.0** |
| Count | **67.01** | 56.2 |
| **AA** | 81.49 | **83.08** |

**Honest framing.** Beating a 2020 CNN+RNN is expected. The fair comparison is
RS-InternVL: a 2026 model fine-tuned on full BEN.txt at **1B** parameters, which
we beat 76.78 to 73.29 with **4B**. Four times the parameters for 3.5 points is
not a triumph.

Nobody scores 95% on these benchmarks. If that was the expectation, the
expectation was wrong, not the adapter.

---

## 2. Training run

```
NVIDIA A100-SXM4-80GB
--manifest /data/manifests/rs_vqa_train.jsonl --image-root /data
--split train --out /data/checkpoints/rs_vqa
--micro-batch 8 --grad-accum 1
--smoke-passed /data/checkpoints/rs_vqa_smoke/run.json
```

```
9,250 steps in 2.96 h (1.153 s/step)
loss 5.045 -> 0.172
peak 14.66 GB
Model: flash_attention_2  trainable=40,271,872 (0.899%)  checkpointing 60/60
Adapter written to /data/checkpoints/rs_vqa/adapter          (154 MB)
```

Cost **~$7.40**. Checkpoints every 500 steps.

**Mixed run, not sequential** — the user's call (*"Okay we will do a mixed
run"*). Sequential training across sources causes catastrophic forgetting; the
same reason this adapter is never continue-trained on a new dataset. Doing so
would drift it toward the new distribution and quietly degrade BEN and RSVQA,
discoverable only by re-running the full eval — which costs as much as the
retrain it was meant to avoid.

### H100 was tried and lost

`train_job_h100` exists and reached real `NVIDIA H100 80GB HBM3` hardware. The
H100 came out **slower** than the A100 because the run is partly
**input-bound** — CPU-side image decode is the constraint, so a faster
accelerator does not help. A100 measured 1.236 s/step in smoke, 1.153 s/step in
the real run.

Smoke gate (required before any real run):

```
rs_vqa: 200 steps in 0.07 h (1.236 s/step), loss 4.507 -> 0.194, peak 14.6 GB
Adapter written to /data/checkpoints/rs_vqa_smoke/adapter
```

---

## 3. Corpus

**74,000 rows**, 13 question types, 0 validation warnings.

| source | rows | GSD | note |
|---|---|---|---|
| BEN.txt | 40,000 | 10 m | genuinely 3 composites (true colour, false colour, short-wave) |
| RSVQA-HR | 24,000 | 0.30 m | **`area` included** after the bucketing fix |
| RSVQA-LR | 10,000 | 10 m | size questions kept, thresholds stated in the prompt |

An earlier build was 70,529 rows with `area` excluded. Including it required a
restart and was worth it — see §5.

Ratio (BEN 40k / HR 24k / LR 10k) was kept deliberately: *"i dont think we
should readjust the ratio"*.

Tile geometry: RSVQA-HR is 512×512 px at 0.1524 m = 78 m × 78 m = **6,088 m²**
(downsampled 2× LANCZOS to 0.3048 m during staging).

### Licence position

RSVQA **train** splits are used. The dataset is CC BY 4.0 and the problem
statement does not forbid it, but it is disclosed explicitly in `CREDITS.md`
under *"Note — RSVQA train splits are used, and disclosed"* (20,529 rows at the
time of writing).

> The authors' **evaluation-script repository** (`syvlo/RSVQA`) is
> **GPL-3.0**. We use the data, not their code. **Do not vendor those scripts.**

---

## 4. `composites=3` — the convention everything else inherited

```python
_COMPOSITES = {"rs_vqa": 3, "change_vqa": 6, "optsar_fusion": 3, "rs_ground_caption": 2}
```

`RealChipDataset` pads every `rs_vqa` sample to exactly three views. BEN has
three real composites; **RSVQA has one image, sent three times**.

**This was the convenient choice, not the right one.** It triples vision tokens
(~334 × 3 ≈ 1,000 per sample) for zero extra information on 13,604 of 20,447
eval rows. For the adapter, **prefill is 86% of inference time** (~0.77 s per
batch of 16 against 0.13 s decoding), so this is the dominant serving cost.
Sending one image would cut prefill roughly 3× on two-thirds of the rows. It is
locked in only because eval must match training.

### The near-miss it caused

The eval path defaulted `composites=None` and did no padding:

| benchmark | images/row | training saw | eval would have sent |
|---|---|---|---|
| BEN bench | 3 | 3 | 3 ✓ |
| RSVQA-HR | 1 | **3** (repeated) | **1** ✗ |
| RSVQA-LR | 1 | **3** (repeated) | **1** ✗ |

It would not have errored. It would have scored both RSVQA benchmarks under a
condition the model was never fine-tuned on, and the drop would have been
reported as the adapter's ability. Caught before the run.

Fix: `_load` takes `composites` and threads it into `RealChipDataset`; `gates`
passes `--composites 3`, tied by comment to `adapter_composites("rs_vqa")` so it
cannot silently drift; the report records `composites`, so a run cannot be
compared against one measured at a different width.

### Train/eval parity checklist (all verified)

| aspect | training | eval | |
|---|---|---|---|
| composites | 3 | was `None` | **fixed** |
| max_pixels | 262144 | 262144 (same frozen YAML) | ✓ |
| gsd_conditioning | True | True | ✓ |
| size-rule source match | `RSVQA-HR` / `RSVQA-LR` | identical strings | ✓ |
| image path resolution | `/data` + prefix | manifest parent | ✓ |
| area answer format | bucketed | bucketed | ✓ |

---

## 5. The `area` reversal — the decision that carries the HR gate

**Morning position: `area` is broken, exclude it.** Of 3,471 HR rows: 62% are
`0m2` (OpenStreetMap gaps), 275 (8%) exceed the 6,088 m² tile (max 21,264 m²),
39 are implausibly tiny (a 6 m² "school"). Only ~29% looked usable.

**Afternoon correction: we were scoring it wrong.** RSVQA's paper quantises area
into **five buckets**:

```
0m2 · 1-10m2 · 11-100m2 · 101-1000m2 · more than 1000m2
```

Our manifest carried raw values (`10631m2`), so we had been treating a 5-class
classification problem as exact-integer regression.

**Result after bucketing and retraining: `area` scores 87.6%** — the
third-strongest HR type, and the margin that carries the gate at 85.06 against a
target of 85. Without this decision, HR fails too.

This is why the **restricted** score (84.20) is *lower* than the **official** one
(85.06): excluding `area` now hurts. The caveat predicting otherwise was written
when `area` was untrained and never caught up.

---

## 6. The `count` problem — the one gate that fails

`count` is the entire RSVQA-LR shortfall.

**RSVQA counts are OpenStreetMap *record* counts, not visible objects.** Max
14,536 buildings in a 256×256 image at 10 m; 33.5% exceed 20. Reference answers
reach **2,392 buildings in a 256×256 tile** — you could not see 2,392 of
anything there.

**The reversal that matters:** RSVQA-LR's *test* split is 29% `count` — the same
noisy type. Dropping it from training would cap LR at ~71% against a ≥88 target.
**The noise is in the benchmark you are graded on.**

### The answer-space fix (+7.3 points, no retraining)

RSVQA is standardly evaluated as **classification over a fixed 94-answer
vocabulary**, not open-ended generation. Scoring `count` against the benchmark's
own answer space:

```
count:        27.2%  ->  56.2%     (+29.0)
RSVQA-LR AA:  75.82  ->  83.08     (+7.3)
rural_urban 93.0% · comp 91.6% · presence 91.4%   (unchanged)
ben AA 0.750 unchanged — confirms the fix touched only LR
```

Adapter weights untouched. Report: `logs/rs_vqa_gates_counts.md`.

**Still fails, 83.08 against ≥88.** To clear 88 with the other three types where
they are, `count` needs roughly **78%**; we are at 56.2 and the paper's own model
gets 67.01. The target was set against modern specialists at ~92, not against the
2020 paper.

**HR `count` at 70.6% against LR's 56.2%** is resolution doing exactly what you
would expect: 0.3 m supports counting, 10 m does not.

**LR's three answerable types average 92.0%** — specialist parity. The gate fails
only because `count` was kept.

Better training data does **not** fix this: the test set is fixed, and it asks
for counts that are not visually recoverable. SpaceNet 7 could give `rs_vqa` a
genuine counting capability (hand-annotated, visible, controllable crop density)
— but it would show as *"the model can count sparse buildings"* in a demo, not
as a higher RSVQA-LR number.

---

## 7. Size-word conditioning

RSVQA's two scales define the same words **30× apart** (paper Table I):

```python
_SIZE_RULES = {
    "RSVQA-HR": "small <100 m2, medium <500 m2, large >=500 m2",
    "RSVQA-LR": "small <3000 m2, medium <10000 m2, large >=10000 m2",
}
```

The rows were **not** deleted. The rule is stated in the prompt instead, keyed on
the **source string**, not on GSD.

**Why source and not GSD:** BEN.txt is 10 m like RSVQA-LR but defines **no** size
categories at all. A resolution-based guess (`if gsd < 1`) would assert RSVQA
thresholds over a BEN question — stating a rule that is simply false for it. The
demo had exactly this bug; both paths now use the same lookup.

A user-typed word like *"big"* is outside the trained vocabulary
(`small|medium|large`) and gets no threshold line. That matches training, so it
is not a prompt mismatch — but user-phrased size words fall outside the
conditioning.

### Scale conditioning generally

`scale_prefix` emits GSD, scene extent (computed from the **loaded** image rather
than the manifest, so it cannot drift), and the size rule when applicable:

```
[ground sample distance: 10 m; scene 2560 x 2560 m, 6,553,600 m2] is there water in the image
views 3   adapter /data/checkpoints/rs_vqa/adapter
```

Scene area spans **1,000×** across the corpus: HR 6,088 m² · BEN 1,440,000 m² ·
LR 6,553,600 m². GSD alone does not distinguish them, which is why extent is
stated too. It returns an empty prefix when no GSD is known — a wrong scale is
worse than no scale, because the model learns to trust the field.

---

## 8. Formatter

`satquery/evalcli/formatter.py` — **one implementation shared by eval and the
headless CLI**, so a reported score predicts what a judge scores.

### Three wrong class strings, found and fixed

| serving formatter (wrong) | actual RSVQA class |
|---|---|
| `between 0m2 and 10m2` | `between 1m2 and 10m2` |
| `between 10m2 and 100m2` | `between 11m2 and 100m2` |
| `between 100m2 and 1000m2` | `between 101m2 and 1000m2` |

| | before | after |
|---|---|---|
| HR area classes | wrong wording | the paper's wording |
| `ben_mcq` | **not registered** — a graded gate with no formatter | registered, letter + option-text recovery |
| `rsvqa_hr_comparison` | not registered | registered |
| raw area value `10631m2` | abstained to `0m2` | bucketed to `more than 1000m2` |
| test coverage | **none** | 38 tests |

### Why a formatter and not constrained decoding

- **The raw output survives.** Prediction *and* formatted value both land in the
  dump, so a bad rule is auditable after the fact. Constrained decoding destroys
  that evidence — you can never tell whether the model knew the answer or was
  forced into a legal one.
- **The IF rate stays meaningful.** Constrained decoding makes
  instruction-following 100% by construction, deleting exactly the signal the
  BEN.txt paper tracks and that tells you whether an accuracy number is real.
- **Model-agnostic** — the same code path for base and adapter.

### AA vs overall accuracy

AA (mean of per-type accuracies) is the metric the gates are set against, because
it stops a model looking good by nailing one large easy type while failing a
small hard one. They are close on BEN (nine roughly equal types, ~4,500 rows
each) and diverge on RSVQA-LR (4,002 `comp` rows against 100 `rural_urban`).

### Outstanding

The **official RSVQA scorer is still not wrapped**; we compute the metric
ourselves, which §6.4 calls a P0 bug. Every report therefore carries
`OFFICIAL_SCORER_CAVEAT`: *valid for run-to-run comparison, not quotable as a
benchmark result.* The formatter is tested against shapes derived from our
manifests, not against RSVQA's published worked examples.

---

## 9. Baselines and the vision ablation

### Majority-class (blind) baselines

| gate | majority-class AA | target | zero-shot Qwen |
|---|---|---|---|
| BEN binary | **52.6** | 70 | 61.96 |
| BEN MCQ | **29.3** | 45 | 37.55 |
| RSVQA-LR | **55.8** | 88 | — |
| RSVQA-HR | **62.6** | 85 | — |

BEN binary and MCQ are the informative gates — their priors are near chance, so
70 / 45 genuinely means the model is reading the image. **RSVQA-HR is the weak
one**: a 62.6 prior means a model at 70 has barely beaten guessing.

### Shuffle ablation — pair every question with the *wrong* image

| | real | shuffled | vision contribution |
|---|---|---|---|
| BEN | 75.0 | 59.0 | **16.0** |
| RSVQA-LR | 75.8 | 61.8 | **14.0** |
| RSVQA-HR | 85.1 | 75.2 | **9.9** |

Report: `logs/rs_vqa_gates_shuffle.md` / `.json`. Same rows, same everything, one
variable changed.

**What it means, stated exactly:**

- ✅ "76.78 on BEN.txt binary, first of 14 in the paper's table" — stands.
- ✅ "The model uses the imagery" — supported, 16-point drop.
- ❌ "76.78 measures image understanding" — **not supported.** A majority of the
  margin is answer-distribution learning. Question-conditional priors alone are
  worth **19.3 points** on BEN: shown a *wrong* image the model still scores 59.0
  against a 39.7 floor, purely by knowing which answers go with which phrasings.
- On **RSVQA-LR, 70% of the margin over the prior is vision**; on BEN, 45%.

BEN.txt's questions are template-generated from CORINE class pairs, so *"does
arable land touch broad-leaved forest"* carries real statistical signal about the
answer. RSVQA's are about object presence in a specific tile — far less
predictable from phrasing. **This is a property of BEN.txt's generation, not a
flaw in our model** — every model in that table is presumably doing some version
of it. We just measured it.

**RSVQA-HR is the healthiest benchmark**: uniform 8.7–11.0 point drops across all
four types, no type coasting on priors.

Publishing an ablation that partially undercuts your own headline is the
strongest card here — it pre-empts exactly the question a sharp judge asks.

### Claims that must never be made

1. "The model counts objects" — BEN count is 0–1.3 points of vision. Dead.
2. "75.82 on RSVQA-LR shows image understanding" — mostly it does not. Report the
   score with the ablation column beside it.
3. Any general "our model reads satellite imagery" without naming *which*
   capabilities.

### Why the adapter beats the base model — all of it, honestly

1. **Answer priors** — trained on RSVQA's own train split, so it knows 75% of LR
   presence answers are `yes`, 66% of HR areas are `0m2`, 67% of comparisons are
   `no`. Worth 52–63 points depending on the gate, and free to acquire. Nobody
   should call this unfair; it is the thing being measured.
2. **Vocabulary grounding** — 40,000 CORINE class names seen. The base model has
   to map *"land principally occupied by agriculture with significant areas of
   natural vegetation"* to pixels cold.
3. **Input convention** — `composites=3`; the base model never saw it.
4. **Scale conditioning** — the GSD prefix is a learned format; to the base model
   it is just unusual text.
5. **Format** — removed from the comparison by applying the same formatter to
   both, so the measured delta is smaller and more honest than a naive one.

---

## 10. Domain transfer — the untested risk

Every number is on 120×120 Sentinel-2 at 10 m and USGS aerial at 0.3 m. The
hidden set is **Cartosat/RISAT** — Indian, metre-class, different sensors.

> Hitting all the gates says nothing about the hidden set. Benchmark performance
> and domain transfer are two independent risks.

**RSVQA-HR is the Cartosat proxy** — 0.3 m aerial is the closest available scale.
So **85.06 is the strongest domain-gap evidence available**, and it is the gate
the `area` fix rescued.

The ablation predicts transfer: the hidden set has ISRO's questions, not BEN.txt
templates, so **whatever the model gets from answer priors will not transfer**.
High-vision capabilities (LR `rural_urban` 44 points, BEN `mcq_presence` 32.7,
all of HR) should survive. Both count types will collapse.

The adapter does handle a **33× GSD range and 1,000× scene-area range** (85.06
and 83.08), which is real evidence the scale conditioning generalises.

**Off-template phrasing is untested.** The model saw 74,000 questions in RSVQA's
and BEN.txt's exact wordings (*"Is a water area present?"*). Asked *"is there a
big waterbody in the image"* it kept the answer format — genuine generalisation —
but reliability off-template is almost certainly lower, and benchmarks only ever
ask in-template.

### RSVQA test 2 vs CDVQA test 2 — different experiments

RSVQA's second test split shifts **geography and sensor** (Philadelphia); CDVQA's
shifts the **answer distribution** on purpose. So: *RSVQA test 2 + ablation tests
whether vision transfers; CDVQA test1−test2 tests whether the model learned
priors.* You need both. The yardstick for the Philadelphia run is the RSVQA
paper's own **5.6-point** test1→test2 drop, against our measured 9.9-point vision
contribution on HR test 1.

---

## 10b. Serving parity — the adapter's answer is the whole answer

85.06 and 83.08 were measured with the adapter's own text as the entire answer
string. The served composer used to append deterministic sentences to it, so a
question that scored as `"yes"` was delivered as:

```
yes. Smooth covers 67.6% of the valid pixels.
```

Three problems, in ascending order of seriousness:

1. **It is not the string that was scored.** No benchmark number covers the
   concatenation, and any comparison against 85.06 silently changes the subject.
2. **The reader cannot tell which half the model said.** One sentence attributed
   to nothing reads as the adapter elaborating on itself.
3. **The halves can contradict.** `_29_57` was the demonstration — the adapter
   correctly answered that there is no water, and the appended MNDWI sentence
   reported **68.6% water** on a scene with none, in the same paragraph, in the
   model's voice.

`_compose_answer` in `satquery/agent/pipeline.py` and `_summarise` in
`graph.py` now return the learned text alone whenever a learned tool ran. The
deterministic sentences are unchanged and still produced — they are the answer
only when no adapter spoke. Every measurement stays in the trace steps and the
evidence panel, attributed to the tool that produced it. This changed which of
the two is *the answer*, not what is available.

---

## 11. Files

| | |
|---|---|
| adapter | `/data/checkpoints/rs_vqa/adapter` (154 MB, on `suneet-sharan-ug25`) |
| corpus | `/data/manifests/rs_vqa_train.jsonl` |
| gate results | `logs/rs_vqa_gates.md` / `.json` |
| count fix | `logs/rs_vqa_gates_counts.md` |
| ablation | `logs/rs_vqa_gates_shuffle.md` / `.json` |
| zero-shot baseline | `logs/rs_vqa_gates_zeroshot.md` |
| stagers | `scripts/stage_rsvqa_hr.py`, `scripts/stage_rsvqa_lr.py`, `scripts/stage_ben_txt.py` |
| formatter | `satquery/evalcli/formatter.py` |
| corpus browser | `browse` — `@modal.asgi_app()`, `https://suneet-sharan-ug25--satquery-phase0-ml-browse.modal.run` |

Eval rows: BEN 6,843 (binary 5,531 + MCQ 1,312) · RSVQA-LR 10,004 · RSVQA-HR
3,600 = **20,447**.

---

## 12. Mistakes worth not repeating

- **`\b` became a backspace character (0x08), three separate times** — non-raw
  strings written through a heredoc, in `build_ben_manifest.py`,
  `modal_phase0.py` and `dataset.py`. Greps clean, matches nothing. Fixed with
  `chr(92)+"b"` or an `r` prefix.
- **Both runs writing to the same `logs/rs_vqa_gates.md`** — the adapter run
  would have silently overwritten the baseline, losing exactly the comparison
  that was wanted.
- **`timeout 900` wrapper** killed the HR fetch at 1,160/3,000 images.
- **`stage_ben_txt.py` hardcoded `"split": "train"`** — would have labelled 887
  human-verified bench patches as training data.
- **False 26.6% contradiction alarm** — a subject matcher that collapsed
  small/medium/large into one bucket. Re-run with exact phrase matching: **0
  contradictions of 73 pairs.**
- **`| tail -N` buffering** made working jobs look hung, three times. Switched to
  writing files and polling.
- **Chained `ruff && python`** — a ruff failure silently prevented a code
  insertion, and the deploy showed no `browse` function.
- **HR fetch died overnight** because the laptop slept. Moved to Modal for
  unattended work.
