# Phase 0 item 9 — base model bake-off

**Date:** 2026-08-30T18:30:02.032341+00:00

**Adapter:** `/data/checkpoints/rs_vqa/adapter`

> Scored with the in-repo comparator, NOT the official benchmark scorer. Valid for run-to-run comparison; not quotable as a benchmark result (master plan section 6.4, C8).

| model | benchmark | AA / acc@0.5 | overall / mIoU | n |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | ben | 0.750 | 0.751 | 6843 |
| `Qwen/Qwen3-VL-4B-Instruct` | rsvqa_lr | 0.758 | 0.726 | 10004 |
| `Qwen/Qwen3-VL-4B-Instruct` | rsvqa_hr | 0.851 | 0.850 | 3600 |

## Section 6.1 gates — `Qwen/Qwen3-VL-4B-Instruct`

Adapter: `/data/checkpoints/rs_vqa/adapter`

| gate | score | target | verdict |
|---|---|---|---|
| BEN.txt binary VQA | 76.78 | 70 | **PASS** (n=5531, IF=100.0%) |
| BEN.txt MCQ | 73.62 | 45 | **PASS** (n=1312, IF=100.0%) |
| RSVQA-LR | 75.82 | 88 | FAIL (n=10004, IF=100.0%) |
| RSVQA-HR *(official, all types)* | 85.06 | 85 | **PASS** (n=3600, IF=100.0%) |
| RSVQA-HR *(our restricted set)* | 84.20 | 85 | FAIL (n=3082, IF=100.0%) |

> RSVQA-HR is reported twice. `area` is excluded from our training corpus because its OpenStreetMap-derived answers are unusable (66% of the test split is `0m2`; the rest are near-unique exact integers). The official row scores all four types and so includes a capability the adapter was never trained for; the restricted row scores the three types we train. Neither number alone is honest.

## Deciding

Section 5.1: commit to **one** base. Splitting bases across adapters doubles VRAM and removes the multi-LoRA serving argument (C1).

Grounding carries gate G3 and is scored on the hidden set whichever option we elect (C17), so a grounding win outranks a VQA win of similar margin. If the 2B leads on VQA but trails on grounding, take the 4B.

Record the decision and the numbers behind it in `TEAM_CONTEXT.md`, and tick 'Base model committed' in the Phase 0 checklist.