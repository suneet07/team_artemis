# Phase 0 item 9 — base model bake-off

**Date:** 2026-08-30T17:55:19.278320+00:00

**Adapter:** `/data/checkpoints/rs_vqa/adapter`

> Scored with the in-repo comparator, NOT the official benchmark scorer. Valid for run-to-run comparison; not quotable as a benchmark result (master plan section 6.4, C8).

| model | benchmark | AA / acc@0.5 | overall / mIoU | n |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | ben | 0.710 | 0.792 | 48 |
| `Qwen/Qwen3-VL-4B-Instruct` | rsvqa_lr | 0.777 | 0.729 | 48 |
| `Qwen/Qwen3-VL-4B-Instruct` | rsvqa_hr | 0.884 | 0.896 | 48 |

## Section 6.1 gates — `Qwen/Qwen3-VL-4B-Instruct`

Adapter: `/data/checkpoints/rs_vqa/adapter`

| gate | score | target | verdict |
|---|---|---|---|
| BEN.txt binary VQA | 86.55 | 70 | **PASS** (n=42, IF=100.0%) |
| BEN.txt MCQ | 40.00 | 45 | FAIL (n=6, IF=100.0%) |
| RSVQA-LR | 77.73 | 88 | FAIL (n=48, IF=100.0%) |
| RSVQA-HR *(official, all types)* | 88.37 | 85 | **PASS** (n=48, IF=100.0%) |
| RSVQA-HR *(our restricted set)* | 87.83 | 85 | **PASS** (n=38, IF=100.0%) |

> RSVQA-HR is reported twice. `area` is excluded from our training corpus because its OpenStreetMap-derived answers are unusable (66% of the test split is `0m2`; the rest are near-unique exact integers). The official row scores all four types and so includes a capability the adapter was never trained for; the restricted row scores the three types we train. Neither number alone is honest.

## Deciding

Section 5.1: commit to **one** base. Splitting bases across adapters doubles VRAM and removes the multi-LoRA serving argument (C1).

Grounding carries gate G3 and is scored on the hidden set whichever option we elect (C17), so a grounding win outranks a VQA win of similar margin. If the 2B leads on VQA but trails on grounding, take the 4B.

Record the decision and the numbers behind it in `TEAM_CONTEXT.md`, and tick 'Base model committed' in the Phase 0 checklist.