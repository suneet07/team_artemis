# Segment records

One file per capability, reconstructed from the full session transcript so
nothing is lost to a compaction. Each covers what was trained, what was
measured, which formatter and prompt produced the number, which sources were
rejected and why, and what is still open.

| # | segment | ships | headline |
|---|---|---|---|
| [01](01-rs-vqa.md) | **`rs_vqa`** (G1+G2) | trained LoRA | BEN binary **76.78** · BEN MCQ **73.62** · RSVQA-HR **85.06** · RSVQA-LR 83.08 |
| [02](02-grounding.md) | **Grounding** (G3a) | base model + `PRECISE_PROMPT` | **62.7% acc@0.5**, beating published fine-tuned GeoChat (60.6%) with no training |
| [03](03-captioning.md) | **Captioning** (G3b) | base model + `STRONG_PROMPT` | ROUGE-L **0.252** against a 0.222 blind floor; fine-tuned SOTA is 0.369 |
| [04](04-sar-and-optsar.md) | **SAR / optical–SAR** (G5) | BIFOLD tool + D1 floor | `resnet50-s1` **74.95%** against a 50.1% floor, zero training |
| [05](05-change-vqa.md) | **`change_vqa`** (G4) | trained LoRA | **AA 68.0%** on CDVQA Val · blind ceiling 45.0% · published baseline 55.3% · SOTA 68.6% |
| [06](06-routing.md) | **Routing** (G6) | rules-first router | which question reaches which task and tools — all 8 branches, 15 of 16 cases resolved without the LLM |
| [08](08-testing-corpora.md) | **Testing corpora** | — | which Modal account holds which test split, which `change_vqa` checkpoint ships, and why `composites` is a budget |
| [07](07-future-changes.md) | **Deferred work** | — | considered deferrals with the context to pick each up cold — not bugs in flight |

## What ships as what

Only **two** of the four planned adapters were trained. That was the right
answer twice, and both times it was decided by measurement:

```
rs_vqa             trained LoRA            beat the dataset authors on both splits
change_vqa         trained LoRA            retrained on CDVQA after a taxonomy mismatch
rs_ground_caption  NOT an adapter          base Qwen beats fine-tuned GeoChat at grounding
optsar_fusion      NOT an adapter          BIFOLD (MIT) measured 74.95%, and fusion adds nothing
```

## Rules that recur across all five

- **Never quote a score without its blind baseline.** Majority-answer ceiling for
  VQA, wrong-image reference for captioning, shuffled-image ablation for vision
  contribution. The gap is the claim; the raw number is not.
- **Format bugs look exactly like model failure.** Every large score change in
  this project was a formatter fix, not a modelling one — RSVQA `count`
  quantisation (+7.3), RSVQA `area` bucketing (the margin carrying HR),
  `change_vqa` instruction-following (77.7% → 100%), and three separate box-scale
  bugs that would each have read as a broken model.
- **Train/eval parity is load-bearing.** `composites` must match what the adapter
  saw. A mismatch does not error; it silently scores the model on an input shape
  it never trained on.
- **A prompt is load-bearing when the model is not fine-tuned.** Trained adapters
  reproduce their score from `scale_prefix + question` because that is what they
  saw. The base model reproduces nothing unless asked the way the benchmark asked.
- **Licence clears the weights, not just the data.** A permissive badge on a repo
  says nothing about the imagery underneath it — the trap that caught LS-SSDD,
  LHRS-Bot, Open-CD, SOMA-1M and VRSBench.
- **The hidden ISRO set is Cartosat-2S + RISAT, and every training source is
  Western.** Unclosable with public data. The mitigation is measuring transfer,
  not pretending it is closed.
