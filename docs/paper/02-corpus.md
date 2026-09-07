# 2. The corpus — what was staged, what was refused, and how it was kept honest

## 2.1 The licence position, stated once

`CREDITS.md` records a **dual-character determination**, and it is the reason
some benchmarks appear in results but never in weights:

- **Anything that ships** — training corpora, adapter weights, deployed
  inference — is judged as **commercial use**. Non-commercial and academic-only
  sources are excluded with no exception.
- **Benchmark evaluation** is judged as **academic research use**, which is what
  academic-only terms grant. Read-only evaluation does not enter the weights and
  is not redistributed.

The barrier is **enforced mechanically, not by convention**:
`scripts/stage_benchmarks.py` carries `eval_only` and `train_forbidden` flags,
and `tests/test_license_blocklist.py` fails the build if a forbidden path
appears in training code.

The limit is stated in `CREDITS.md` too, and it matters: the position holds only
while the evaluation stays academic. Quote VRSBench or CDVQA in a procurement
deck and the academic-only sources no longer back the claim.

## 2.2 What was built

| corpus | rows | sources | feeds |
|---|---|---|---|
| `ben_canonical.jsonl` | **117,023** | BigEarthNet.txt | `rs_vqa` 57,044 · `rs_ground_caption` 59,979 |
| `change_vqa_train.jsonl` | **51,956** | SpaceNet 7 / MUDS | the *first* change adapter |
| CDVQA (staged) | **30,350** Test2 alone | CDVQA (Apache-2.0) | the *second* change adapter — the one that ships |
| caption corpus | **49,602** | RarePlanes 8,993 · SpaceNet 2 9,472 · SpaceNet 6 6,306 · RSVQA-HR 4,848 · BEN.txt 19,983 | captioning experiments |
| `rs_ground_caption` grounding | **19,896** | RarePlanes 13,923 · SpaceNet 6 5,973 | grounding experiments |
| `optsar_fusion.jsonl` | **85,096** | reBEN (BigEarthNet v2.0) | SAR/fusion evaluation |

The `rs_vqa` training run consumed **74,000 rows** — BEN.txt at 40,000 of
70,529 (57%) plus the RSVQA splits — and took **3.0 h against a 4.2 h budget**.

Two facts about this table are worth naming. First, **the corpora are
generated, not scraped**: questions are computed from annotations, so the answer
is derivable and auditable rather than authored. Second, **the same imagery
serves several adapters**, which is safe precisely because each adapter is an
independent 40M-parameter LoRA over a frozen base — nothing is shared but the
pixels.

## 2.3 Balance is a corpus property, and it decides what a score means

Every generated corpus was measured for its **blind ceiling** — the score a
model gets by answering the majority class without looking at the image. This
was not a formality; it changed the corpus twice.

**The blank-month regression.** SpaceNet 7 mosaics include months where the
imagery is effectively blank. Removing those pairs was obviously correct, and it
took the `change_presence` blind ceiling from **53.9% to 74.3%** — because most
of the corpus's "nothing changed" examples *were* the blank months. Deleting bad
data made the benchmark easier and the resulting score meaningless. Fixed with a
twin-safe `_balance()` that rebalances after filtering, not before.

**`dominant_change` was deleted outright.** It came back with a **100% blind
ceiling** — the generator always answered NDBI. An index that produces one
answer for every input is not a question; it needs low-signal pixel masking
before it can be trusted, and until then the type does not exist.

**Uncapped ceilings are stated, not hidden.** `change_presence` is **93.4%
"yes"** before balancing. RSVQA-LR presence is **76.3% "yes"** on the official
test split — a fact that later explained a serving answer that looked like a
model failure and was actually the benchmark's own prior ([11](11-failure-atlas.md)).

## 2.4 Split hygiene

**Train and validation share zero images.** This was checked, and the first
check was wrong in an instructive way: an early leakage alarm reported **97.2%
overlap**, because validation was compared against a combined manifest that
*embeds* validation. The corrected comparison found no shared images. The lesson
is recorded because the false alarm was more dangerous than a real one would
have been — it nearly caused a good corpus to be rebuilt.

**Splits come from the dataset's own metadata where it exists.** One CDVQA chip
was found mislabelled `train` by `ben-micro-split` and corrected against reBEN's
`metadata.parquet`; the manifest records `split_source` so the provenance of the
correction survives.

## 2.5 The imagery fought back

**Archive damage in a staged source.** One corpus shipped as RAR5, which p7zip
could not read — it left 9,871 damaged files in place, and `unar` extracted good
copies into a separate tree beside them. A dict keyed on file name would let
whichever path `rglob` yielded last win, which is a coin flip between a valid
image and a truncated one. The indexer therefore collects candidates per name and
keeps **the first that PIL can actually decode**. 55 pairs remain rejected
because neither copy decodes.

That defect resurfaced twice more while building the testing gallery, months
later, and both times the fix was the same: re-fetch from the intact tree and
verify the bytes decode ([11](11-failure-atlas.md)).

**Download bandwidth decided the staging order.** Measured from Modal:

```
spacenet-dataset.s3   59–62 MB/s
huggingface.co        39.6 MB/s
rareplanes-public.s3  26 MB/s
zenodo.org            0.95–5.28 MB/s   (erratic)
```

Corpora were built smallest-first for the same reason: *"if the corpus generator
has a defect, we find it on a 3 GB download instead of after a 118 GB one."*

## 2.6 What the corpus deliberately does not contain

- **Counts of objects on 10 m imagery.** India's generated questions ask only
  about *direction and magnitude of a surface property* across a 2.56 km chip —
  never counts, never objects — because at 10 m the count is not recoverable and
  a generated question with an unanswerable ground truth teaches the model to
  guess.
- **Any GoogleEarth-derived source**, per [1.3](01-problem-and-constraints.md).
- **Captions from another model.** Gemini and GPT output were both rejected on
  terms, and the second reason was measurement: we had never established that
  either is good at 0.3 m aerial imagery — only that Qwen is not.
