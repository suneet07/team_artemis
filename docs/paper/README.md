# SatQuery AI — the full record

SIH26167 (ISRO / Space Applications Centre): an agentic system that answers
natural-language questions about satellite imagery, and shows its working.

This directory is the complete account: what was built, what was measured, what
was rejected and why. It is written to be checkable — every number here traces
to a run in `logs/`, and where a number is not comparable to a published one,
that is said in the same sentence.

`docs/segments/` holds the per-capability engineering records (3,400 lines).
This set is the layer above: the argument, the evidence, and the parts of the
journey those records do not carry.

| # | file | what it covers |
|---|---|---|
| [00](00-abstract.md) | **Abstract** | the system, the three transferable findings, and the four defining decisions |
| [01](01-problem-and-constraints.md) | **The problem** | the graded task, the hidden Cartosat/RISAT set, and the licence constraint that shaped every dataset decision |
| [02](02-corpus.md) | **The corpus** | what was staged, what was refused, split hygiene, leakage checks |
| [03](03-rs-vqa.md) | **`rs_vqa`** | the adapter that beat the dataset authors' own model on both splits |
| [04](04-change-vqa.md) | **`change_vqa`** | trained twice, on two different datasets, and why the first was thrown away |
| [05](05-grounding.md) | **Grounding** | why the *base* model beat published fine-tuned baselines, and the three coordinate bugs |
| [06](06-captioning.md) | **Captioning** | the one capability that plateaued, and the honest reason |
| [07](07-sar-and-fusion.md) | **SAR + fusion** | BIFOLD, the radar-over-optical decision, and D1/D2/D3 |
| [08](08-orchestration.md) | **The agent** | router, planner, parameter gate, executor, composer, trace |
| [09](09-evaluation-method.md) | **How we measured** | blind ceilings, shuffle controls, answer contracts, scorer caveats |
| [10](10-serving.md) | **Serving** | one base, three adapters, and a deploy that verifies itself |
| [11](11-failure-atlas.md) | **Failure atlas** | every bug that mattered, and what each one proved |
| [12](12-results.md) | **Results** | every number in one table, each with its baseline |
| [13](13-limitations.md) | **Limitations** | what this system cannot claim |
| [14](14-the-case.md) | **The case** | the argument for the system, in twelve measured points |

## The one-paragraph version

A 4B vision-language model (Qwen3-VL-4B-Instruct), two LoRA adapters totalling
40.3M trainable parameters each, one published radar classifier, and a
rules-first router that decides which of them answers. On the graded splits it
scores **RSVQA-HR 85.06** and **RSVQA-LR 83.08** — both above the RSVQA authors'
own model (83.12 / 81.49) — **68.0% AA** on CDVQA Val against a 55.3% published
baseline, **62.7% acc@0.5** on VRSBench referring with *no training at all*, and
**74.95%** on held-out reBEN radar. Every one of those is reported beside the
blind baseline that says what the number is worth.

## The three rules everything here obeys

**Never quote a score without its blind baseline.** A majority-answer ceiling
for VQA, a wrong-image reference for captioning, a shuffled-image ablation for
vision contribution. The gap is the claim; the raw number is not. RSVQA-LR
presence is 76.3% "yes" — a model that never looks at an image scores that.

**Format bugs look exactly like model failure.** Every large score movement in
this project was a formatter fix, not a modelling one: RSVQA `count`
quantisation was worth **+7.3 points**, `area` bucketing carried the HR gate,
and `change_vqa` instruction-following went 77.7% → 100% without the weights
changing.

**Measurement beats inference.** Where a component could be measured, it was,
and the measurement decided the design — including four times when it killed
something we had already built.
