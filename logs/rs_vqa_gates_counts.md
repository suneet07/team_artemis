# Phase 0 item 9 — base model bake-off

**Date:** 2026-08-30T20:39:48.373251+00:00

**Adapter:** `/data/checkpoints/rs_vqa/adapter`

> Scored with the in-repo comparator, NOT the official benchmark scorer. Valid for run-to-run comparison; not quotable as a benchmark result (master plan section 6.4, C8).

| model | benchmark | AA / acc@0.5 | overall / mIoU | n |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | ben | 0.750 | 0.751 | 6843 |
| `Qwen/Qwen3-VL-4B-Instruct` | rsvqa_lr | 0.831 | 0.812 | 10004 |
| `Qwen/Qwen3-VL-4B-Instruct` | rsvqa_hr | 0.851 | 0.850 | 3600 |

## Section 6.1 gates — `Qwen/Qwen3-VL-4B-Instruct`

Adapter: `/data/checkpoints/rs_vqa/adapter`

| gate | score | target | verdict |
|---|---|---|---|
| BEN.txt binary VQA | 76.78 | 70 | **PASS** (n=5531, IF=100.0%) |
| BEN.txt MCQ | 73.62 | 45 | **PASS** (n=1312, IF=100.0%) |
| RSVQA-LR | 83.08 | 88 | FAIL (n=10004, IF=100.0%) |
| RSVQA-HR *(official, all types)* | 85.06 | 85 | **PASS** (n=3600, IF=100.0%) |
| RSVQA-HR *(our restricted set)* | 84.20 | 85 | FAIL (n=3082, IF=100.0%) |

> RSVQA-HR is reported twice. The official row scores all four question types, as published work does. The restricted row excludes `area`, whose references come from OpenStreetMap polygons and are noisy at the item level (66% of the test split is `0m2`). Area is quantised into the dataset's own five classes for both training and scoring, per the paper; scored as exact integers it would be unanswerable by construction.

## Deciding

Section 5.1: commit to **one** base. Splitting bases across adapters doubles VRAM and removes the multi-LoRA serving argument (C1).

Grounding carries gate G3 and is scored on the hidden set whichever option we elect (C17), so a grounding win outranks a VQA win of similar margin. If the 2B leads on VQA but trails on grounding, take the 4B.

Record the decision and the numbers behind it in `TEAM_CONTEXT.md`, and tick 'Base model committed' in the Phase 0 checklist.