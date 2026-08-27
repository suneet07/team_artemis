# SatQuery AI

**SIH26167 · ISRO / SAC · Space Technology**

A vision–language assistant for remote-sensing imagery: ask questions of optical, SAR,
bi-temporal and cross-modal satellite scenes and get answers, captions, grounded boxes and
change maps back — each with a trace showing which tool ran, on what parameters, and how
confident the system is.

Four LoRA adapters over Qwen3-VL-4B (`rs_vqa`, `rs_ground_caption`, `change_vqa`,
`optsar_fusion`) sit behind a deterministic geospatial pipeline that validates inputs,
routes queries, enforces tool parameters and fuses optical with SAR decisions.

---

## Start here

| If you are… | Read |
|---|---|
| **New to the project** | [`TEAM_CONTEXT.md`](TEAM_CONTEXT.md) — build state, findings, mistakes already made, open questions. **Read this before writing code.** |
| Looking for the design | [`docs/01-plan/satquery-master-plan-v3.8.md`](docs/01-plan/satquery-master-plan-v3.8.md) |
| Working on data | [`docs/README.md`](docs/README.md) → `02-data/` |
| Running a GPU job | [`docs/README.md`](docs/README.md) → `03-compute/` |
| Asked "what's yours and what isn't?" | [`CREDITS.md`](CREDITS.md) |

[`docs/README.md`](docs/README.md) maps every document and states which one wins when two
disagree.

---

## Layout

```
.
├── README.md                  you are here
├── TEAM_CONTEXT.md            onboarding: state, findings, mistakes, unknowns
├── CREDITS.md                 every source + licence + status  (judge-facing)
│
├── docs/
│   ├── README.md              document map + precedence rules
│   ├── 01-plan/               master plan v3.8 — the source of truth
│   ├── 02-data/               dataset specs, resolution policy, question generation
│   ├── 03-compute/            cost model, timing review, Modal runbook
│   └── 04-open/               speculations and unresolved hypotheses
│
├── satquery/                  the package
│   ├── ingest/                P1  reader, modality, bands, radiometry      [built]
│   ├── agent/                 P5  task enum, trace                         [partial]
│   ├── tools/                 P6  registry, manifest, dummy tool           [partial]
│   ├── evalcli/               headless eval mode                           [partial]
│   ├── sar/ coreg/ tiling/    P2 P3 P4                                     [EMPTY]
│   ├── fusion/ confidence/    P7                                           [EMPTY]
│   └── report/ serving/ api/  P8 P9 + API                                  [EMPTY]
│
├── configs/                   preprocessing.yaml, trace + tool schemas
├── scripts/                   phase-0 timing sweeps (local + Modal)
├── training/                  data extraction, routing eval set
├── tests/                     incl. licence blocklist + manifest gate
├── frontend/                  contracts + mocks (no app yet)
├── logs/                      raw measurement output
└── ben-micro-split/           21 black dummy PNGs — timing fixtures ONLY, never train on these
```

**The plan is well ahead of the build.** Eight pipeline modules are empty directories and
`training/train_lora.py` does not exist yet. `TEAM_CONTEXT.md` §1 has the honest inventory.

---

## Ground rules

These have each already been violated once. Details in `TEAM_CONTEXT.md` §4 and §6.

1. **Trace every dataset's provenance to the imagery programme** before it enters a
   manifest — never to the paper that published it. No licence statement = no permission.
2. **Manifest first.** Never download a full corpus. Annotations → manifest → only the
   patches the manifest names.
3. **Train on exactly what deployment sees.** Spare VRAM is not a reason to change the
   input representation.
4. **Never start a full run without a 200-step smoke on that exact config.**
5. **A falling loss is not evidence of training.** Check opening loss against `ln(vocab)`.
   Check your images are not black.
6. **Put `model.train()` in the training script explicitly.** Lightning does not.
7. **Treat every ✅ in the plan as a claim to re-verify**, not as authority.

---

## Status

Phase 0, incomplete. Licence verification is done and the compute picture is measured.
Blocking everything downstream: BigEarthNet.txt ships annotations only, and its imagery
sits in a 600 GB+ monolithic archive that cannot be streamed selectively
(`docs/04-open/SPECULATIONS.md` item 2). No model on this project has yet seen a real
satellite image.
