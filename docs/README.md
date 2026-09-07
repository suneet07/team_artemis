# docs/ — the reasoning that is not in the code

The repo holds the build. This folder holds the decisions behind it: why each dataset is
in or out, what a training run costs, what has been measured versus assumed, and how
training data gets generated. Most of it was produced during Phase 0.

Start with [`../TEAM_CONTEXT.md`](../TEAM_CONTEXT.md) — what is built, what is not, and
every mistake already made once. Then read in the order below.

## 00-archive — superseded documents

Nothing current. Kept so citations resolve and reversed decisions stay recoverable.
See [`00-archive/README.md`](00-archive/README.md). **Do not follow guidance from
anything in it.**

## 01-plan — what we are building

| File | What it owns |
|---|---|
| [`01-plan/satquery-master-plan-v3.8.md`](01-plan/satquery-master-plan-v3.8.md) | **The single source of truth.** Thesis, source inventory, architecture, all ten pipelines, adapters, evaluation targets, phase sequence, risk register. Everything else defers to it on architecture |

v3.5 and earlier are superseded and have been removed. If you find a copy, it reverses
decisions this project has since made — most dangerously C33, which permitted LEVIR and
SECOND data that v3.8 purges entirely. Do not read it.

## 02-data — what we train on, and how it becomes samples

| File | What it owns | Read before |
|---|---|---|
| [`02-data/DATASET_SPECS.md`](02-data/DATASET_SPECS.md) | Per-dataset formats, bands, bit depths, label schemas, and the engineering traps in each ("jump scares") | Touching any dataset |
| [`02-data/phase-0-resolution-policy.md`](02-data/phase-0-resolution-policy.md) | Verified chip sizes from primary sources, and the effective-GSD decision per dataset | Staging, tiling or resampling anything |
| [`02-data/question-generation-plan.md`](02-data/question-generation-plan.md) | QGP v1 — the sample schema, six annotation primitives, template bank, negatives and balance policy, validation gates | Writing any data-prep code |
| [`02-data/data_prep_brief_rareplanes.md`](02-data/data_prep_brief_rareplanes.md) | The RarePlanes assignment as handed to a teammate | Working on RarePlanes |

> **Known conflict.** The RarePlanes brief and QGP v1 specify different record shapes and
> coordinate conventions. Both are currently live. See `../TEAM_CONTEXT.md` §10 — reconcile
> before a second person generates data.

## 03-compute — what it costs and how to measure it

| File | What it owns | Read before |
|---|---|---|
| [`03-compute/phase-0-cost-model.md`](03-compute/phase-0-cost-model.md) | How to price a run. `max_pixels` is a cap not a target; ~0.35 ms/token; per-adapter hours; the Lightning `.train()` bug | Any GPU run, or any argument about the budget |
| [`03-compute/phase-0-timing-review.md`](03-compute/phase-0-timing-review.md) | Why the first 200-step run did **not** clear the Phase 0 gate, and the bugs it exposed | Trusting any earlier timing number |
| [`03-compute/modal-runbook.md`](03-compute/modal-runbook.md) | Running A100-80GB sweeps on Modal, with the gotchas that cost money | Spending GPU credit |

## 04-open — what nobody knows yet

| File | What it owns |
|---|---|
| [`04-open/SPECULATIONS.md`](04-open/SPECULATIONS.md) | Open hypotheses with post-production review checklists. Includes the BigEarthNet imagery-extraction blocker, which gates all real training |

## 07-evaluation — what the models actually score

| File | What it owns |
|---|---|
| [`07-evaluation/g3-grounding-sweep.md`](07-evaluation/g3-grounding-sweep.md) | **Supersedes the plan's assumption that G3 grounding needs a trained adapter.** Eleven approaches measured on identical VRSBench rows; the shipped choice and why; the three coordinate-convention bugs; why stratified sampling must never produce a headline score. Says nothing about the captioning half, which still needs training |

## Where everything else lives

| Path | Contents |
|---|---|
| `../CREDITS.md` | Every model, dataset, library and method with citation, licence and status. The judge-facing answer. Kept at the root deliberately |
| `../TEAM_CONTEXT.md` | Build state, findings, mistakes, reversed decisions, open questions |
| `../logs/` | Raw measurement output from timing runs |
| `../configs/` | `preprocessing.yaml`, `trace_schema.json`, `tool_manifest_schema.json` |
| `../frontend/FRONTEND_PLAN.md` | Frontend plan, colocated with its contracts and mocks |

## Precedence when two documents disagree

1. `02-data/phase-0-resolution-policy.md` overrides the cost model's §3 chip sizes.
2. `03-compute/phase-0-cost-model.md` overrides `../logs/phase_0_timing_sweep_a100_80gb.md`
   on hours. That sweep's **cost per token is still correct**; only its per-adapter hours
   are wrong, because it measured synthetic tiles that always hit the cap.
3. Anything in `docs/` overrides the master plan on **Phase 0 measurements**.
4. Nothing in `docs/` overrides the master plan on **architecture**.
5. `../CREDITS.md` is final on licence status. Where the plan and CREDITS disagree, CREDITS
   was updated more recently — but per the v3.7 standing instruction, treat every ✅ in
   either as a claim to re-verify rather than as authority.

## Adding a document here

Put it in the folder matching its subject, link it from the table above, and say in its
first paragraph what it supersedes. A document that quietly contradicts an older one is
worse than no document — this project has already been bitten by exactly that.
