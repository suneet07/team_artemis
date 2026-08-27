# SatQuery AI — Team Context & Handover

**SIH26167 · ISRO / SAC.** Everything a new teammate needs that is **not** visible from
reading the code or the master plan: what is actually built versus written down, what we
found the hard way, which mistakes have already been made once, which decisions were
reversed and must not be reopened, and what is still unknown.

Read this **after** `docs/01-plan/satquery-master-plan-v3.8.md` and **before** writing any code.

Companion documents, all authoritative in their own domain:

| File | What it owns |
|---|---|
| `docs/01-plan/satquery-master-plan-v3.8.md` | Architecture, pipelines, adapters, phases, risk register |
| `docs/02-data/DATASET_SPECS.md` | Per-dataset formats, bands, bit depths, engineering traps |
| `CREDITS.md` | Every source + licence + status. The judge-facing answer |
| `docs/04-open/SPECULATIONS.md` | Open hypotheses to review post-production |
| `logs/` | Raw measurement output |
| `docs/02-data/question-generation-plan.md` | How training QA is generated (QGP v1) |
| `docs/03-compute/phase-0-cost-model.md` | How to price a training run |
| `docs/02-data/phase-0-resolution-policy.md` | Per-source GSD / chip decisions |
| `docs/03-compute/phase-0-timing-review.md` | Why the first timing run did not clear the gate |
| `docs/03-compute/modal-runbook.md` | Running A100 sweeps on Modal |

All of these now live in this repo. `docs/README.md` maps them and states which
document wins when two disagree.

---

## 1. Where the project actually stands

The master plan reads like a system. The repo is a skeleton with a few load-bearing
bones. Both statements are true and the gap between them is the single most important
thing a new person needs to understand.

### Built and real

| Area | State |
|---|---|
| `satquery/ingest/` | Real — reader, modality detection, band inventory, radiometry, compatibility |
| `satquery/agent/task_enum.py`, `trace.py` | Real — enum and trace scaffolding |
| `satquery/tools/` | Registry, manifest, one dummy tool |
| `satquery/evalcli/` | Smoke path exists |
| `configs/` | `preprocessing.yaml`, `trace_schema.json`, `tool_manifest_schema.json`, two tool configs |
| `tests/` | 8 test modules incl. licence blocklist and manifest gate — the compliance rails are real |
| `scripts/phase0_timing_sweep.py`, `modal_timing_sweep.py` | Working measurement harness |
| `training/data/extract_ben_micro.py`, `training/eval/generate_routing_300.py` | Exist |
| `frontend/` | Contracts (`openapi.yaml`, `types.ts`) + mock fixtures. No app yet |

### Empty stubs — directory exists, zero lines of code

`satquery/sar/` · `satquery/coreg/` · `satquery/tiling/` · `satquery/fusion/` ·
`satquery/confidence/` · `satquery/report/` · `satquery/serving/` · `satquery/api/`

That is **P2, P3, P4, P7, P8, P9 and the whole API** — most of the pipeline the plan
describes. `training/train_lora.py` does not exist at all. Nobody should read the Part 7
repository tree as a description of the present.

Total: ~1,340 lines of Python across the package.

### Not started, and on the critical path

- Question/manifest generation code (QGP v1 is a plan, not an implementation)
- Any real training run (see §3 — no real imagery has ever been through the model)
- vLLM serving smoke test (Phase 0 item 7, unrun — it can force an architecture change)
- India holdout v0

---

## 2. The single biggest unsolved blocker

**BigEarthNet.txt ships annotations only. The pixels are locked in a 600 GB+ monolithic
LMDB archive that cannot be streamed selectively.**

BEN.txt is the primary training source for *all four adapters* — the clear majority of
every mix. The manifest-first staging rule assumes we can download 80k specific patches.
We currently cannot.

Two known routes, neither tried:
1. Download the full archive, extract the manifest patches, discard the rest. Costs disk
   and time we have not budgeted, and free-tier disk cannot hold it.
2. Write a Sentinel-2 / Sentinel-1 fetcher that pulls the tiles by lat/long from
   Copernicus and reproduces reBEN's patch geometry ourselves.

Route 2 is likely correct — we need the same acquisition machinery anyway for India
holdout v0 and for the self-generated change pairs (C46). One tool, three deliverables.
But it has to reproduce reBEN's exact patch boundaries or the annotations do not line up.

**Nothing about training is real until this is solved.** Recorded in `docs/04-open/SPECULATIONS.md`
item 2; flagged as blocking in the cost model's action list.

---

## 3. Findings — things we learned by measuring, not by reading

These cost real time to discover. They are not in the code and are easy to lose.

### 3.1 Lightning never calls `model.train()`

`transformers.from_pretrained` returns a model in **eval** mode; PEFT preserves it;
Lightning only *warns* ("Found N module(s) in eval mode at the start of training") and
its only `.train()` calls are in validation/test hooks to restore mode afterwards.

Consequences, both silent and both in the wrong direction:
- **Gradient checkpointing no-ops.** Transformers gates it on
  `gradient_checkpointing AND self.training`. Measured **0 of 60** layers active;
  `--grad-checkpointing` and `--no-grad-checkpointing` produced byte-identical 15.9 GB of
  activations.
- **Every `nn.Dropout` is inert**, so the §5.2 LoRA dropout of 0.1 does nothing.

Fix: explicit `model.train()` in the training script. Effect: activations 15.9 GB → 2.6 GB
(6.1×) at a 58% step-time cost.

**Rule: put `model.train()` in the real training script explicitly. Never rely on the
framework. Never scroll past that warning.**

### 3.2 `max_pixels` is a cap, not a target

Qwen3-VL uses native dynamic resolution and **never upsamples**. Measured against the real
processor at `max_pixels = 262,144`:

| source image | vision tokens |
|---|---|
| 120×120 (BigEarthNet patch) | **16** |
| 256×256 | 64 |
| 512×512 | 256 |
| 1024×1024 | 256 (cap binds) |
| 2048×2048 | 256 (cap binds) |

`vision_tokens_per_image = min(source_pixels, max_pixels) / 1024` (patch 16, merge 2, both
verified against the processor).

Three consequences people get wrong:
1. Raising `max_pixels` above a source's native size **costs nothing and buys nothing**.
   It is completely inert on BigEarthNet.
2. `262,144` is exactly a 512×512 view — the benchmark chip size. Preserves chips exactly.
3. The budget is driven almost entirely by the **sub-metre adapters**, not by BEN.

### 3.3 Cost per token — the transferable number

From the A100-80GB sweep, micro_batch=4, bf16-mixed, ~25% MFU: roughly **0.35 ms/token**
above short sequences.

    epoch_hours ≈ 80,000 × tokens_per_sample × 0.35e-3 / 3600 ≈ tokens_per_sample × 0.0078

Sanity check: 575 tokens → 4.5 h predicted, 4.3 h measured.

**This is pessimistic** — 25% MFU at micro_batch=4 leaves ~70 GB of an 80 GB card unused.
At 40% MFU the constant drops to ~0.22 ms/token, i.e. ×0.63 on every hours figure.

### 3.4 The C22 three-composite ablation is nearly free

v3.5 priced it at 5–8 of the 50 hours, based on the Modal sweep's "+8.1 h (+47%)". That
sweep used **synthetic 2048² tiles that always hit the cap**. On real BEN imagery, a
composite is 16 tokens, so three-vs-two is a 16-token delta on the majority source — a
**sub-hour** experiment. Cut-list item 3 ("third composite → two") frees almost nothing.
Re-order or drop it.

### 3.5 Umbra is X-band (9.8 GHz nominal)

Closes the v3.4 checklist line and its risk row. OEM-SAR is therefore a **second
independent X-band source** alongside SpaceNet 6 Capella, so RISAT-2B σ⁰ tuning is not
single-sourced on Rotterdam — *provided* OEM-SAR is cropped/resampled per the resolution
policy rather than cap-downsampled.

### 3.6 The synthetic timing sweep's hours are wrong; its cost-per-token is right

`logs/phase_0_timing_sweep_a100_80gb.md` measured synthetic 2048×2048 tiles that always
hit the cap. Real imagery mostly does not. **Its per-adapter hours are wrong for any
adapter whose sources are smaller than the cap. Its ms/token is correct and transferable.**
The cost model is built on the second and supersedes the first.

Also: `logs/ckpt_audit.md` currently has **one filled cell** and a budget table marked
*(incomplete)*. It is not a completed sweep. Do not cite it as one.

---

## 4. Mistakes already made — do not repeat

Each of these actually happened on this project.

| # | Mistake | What it looked like | The rule now |
|---|---|---|---|
| 1 | **Label masking bug** | `labels = batch["input_ids"]` with `padding="max_length"` scored the model on pad tokens and image placeholders | Mask pads and image placeholders; verify the supervised span decodes to the answer *only* |
| 2 | **Believing a loss curve** | Loss fell 11.32 → ~4.4 and flatlined. Looked like learning. Opening loss was 95% of `ln(151669)=11.93` — a pretrained model at near-uniform-random on its own chat template. And all 21 micro-split images were **constant black** (`getextrema() = ((0,0),(0,0),(0,0))`) | A falling loss is not evidence of training. Check opening loss against `ln(vocab)`. Check your images are not black |
| 3 | **Comparing the wrong budget rows** | A single-adapter epoch estimate (2.69 h) was declared SUCCESS against §5.5's **31 h**, which is the four-adapter scheduled total plus smoke/debug. Correct comparison was the `rs_vqa` row: **4 h**. Real headroom 1.5×, not 11.5× | Compare like with like. Name which row you are comparing against |
| 4 | **Measuring a config that will never run** | 128-token hard cap, ~16 vision tokens, no checkpointing, no accumulation, `precision` unset, 40 GB card against an 80 GB plan | Smoke the *exact* config you will run. Guard rail 1 |
| 5 | **Trusting the framework** | See §3.1 | Explicit `model.train()`; assert checkpointing is actually active |
| 6 | **Relicensing derived annotations** | `ben-micro-split/dataset-metadata.json` declares **CC0-1.0** for Kaggle upload; BEN.txt's annotation layer is **CDLA-Permissive 1.0**. Derived annotations cannot be relicensed | Every staged artifact carries the licence of its most restrictive ancestor. **Caught by our own C45 rule — check this file is fixed before any upload** |
| 7 | **Assuming a licence through six revisions** | BEN.txt's licence was inferred from BigEarthNet's for six plan versions. That is the exact inference pattern that failed for CDVQA-from-SECOND and VRSBench-from-DOTA | C45: trace to the **imagery programme**, never to the publishing paper |
| 8 | **Searching by category instead of by root** | The first licence sweep searched for "object detection datasets", found only Google Earth-derived ones, and concluded object grounding was impossible. A second sweep working backwards from *licence-clean imagery bases* found RarePlanes, OEM-SAR and LS-SSDD | When a category comes back empty, invert the search: start from clean imagery programmes and see what was built on them |
| 9 | **Hyperparameters drifting from the plan** | LoRA dropout 0.05 in code vs 0.1 in §5.2; flat LR 2e-5 (a full-finetune LR, 5–10× low for LoRA) vs the specified warmup 1e-6 → 1e-4 over 1% then cosine | Training configs are code-reviewed against §5.2 |
| 10 | **Batch-size arithmetic under DDP** | `steps_for_epoch = 80000 // batch_size` treats per-device batch as global — wrong by N× | Global batch = micro × accum × devices. Write it out |
| 11 | **Optimising frozen parameters** | `AdamW(self.model.parameters())` includes the frozen base | Pass only trainable params |
| 12 | **No peak VRAM reported** | The one number needed to choose a batch size was missing from the first run | Every timing run reports peak VRAM per cell |

---

## 5. Reversed decisions — closed, do not reopen

Reopening these wastes days and re-imports risk. Each was reversed with a stated reason.

| Decision | Was | Now | Why it flipped |
|---|---|---|---|
| **C28 → C31** | "No sub-metre co-registered optical–SAR dataset exists; the gap is not fixable by finding another dataset" | **Wrong.** SpaceNet 6 MSAW is ~0.5 m Capella X-band quad-pol co-registered with ~0.5 m Maxar optical, ~48k footprints, CC BY-SA 4.0 | A confident negative claim that was never verified |
| **C32 → C36/C37/C39/C53** | "Object grounding impossible on a shippable licence" | G3 ships **regions + buildings + aircraft + ships**, all on CC BY / CC BY-SA / Apache-2.0 | Search method was wrong (see mistake 8) |
| **C33 → C56** | LEVIR/SECOND/QAG permitted for `change_map` because "the output is a mask, not weights" | **Dropped entirely.** `change_map` is a fine-tuned Siamese model, and that model *is* a shipped weight under "codes and models" | C55 killed the "fall back to the pretrained backbone" escape hatch, which made the flaw visible |
| **C52 → C54** | RSVQA-HR imagery source DISPUTED (possibly Sentinel-2 over the Netherlands) | **USGS HRO 15 cm, US northeast, public domain.** C27 stands | Blogs conflate HR and LR because they share a paper |
| **Compute as the binding constraint** | T4 free tier, 200–320 GPU-hours, triage ladder live | ~50 h on a single A100-80GB; ladder is paper-only fallback | C41 |
| **Change data** | SECOND-family (CDVQA / SECOND-CC / QAG-360K) | SpaceNet 7 primary + HRSCD + OSCD + self-generated Indian Sentinel pairs | SECOND has **no licence statement of any kind**, and all three "independent" sources descend from it |

**Permanently excluded — with reasons, in `CREDITS.md`:** xView3-SAR · DIOR · FAIR1M ·
NWPU-Captions · RSICD · VRSBench (training use) · TinyCD and ChangeFormer weights ·
LEVIR-CD · LEVIR-MCI · SECOND · QAG-360K · DFC2023 and SARDet-100K portions of SARLANG-1M.
DynamicEarthNet is flagged ⚠ and **not staged** — commercial Planet Fusion at its core.

---

## 6. Rules that are load-bearing but invisible in the code

A new person will violate these without knowing they exist.

1. **Provenance traceability (C45).** Before any dataset enters a training manifest, trace
   it to the *imagery programme*, not the publishing paper. A derived dataset inherits the
   most restrictive licence in its ancestry. **No licence statement = no permission**;
   silence grants fewer rights than an explicit academic-only clause. Record the chain in
   `CREDITS.md`. This is a hard gate on manifest build.
2. **Manifest first.** Never download a full corpus. Pull annotation indices → select the
   patch-ID manifest → download only manifest patches + official benchmark splits → stage
   as private Kaggle Datasets. Full-corpus downloads are banned; they do not fit free-tier
   disk and we do not need them.
3. **Train on exactly what deployment sees.** Extra VRAM is never a reason to relax
   `max_pixels` or change the input representation. Violating this reintroduces the S3
   failure mode by the back door.
4. **Never start a full run without a 200-step smoke on that exact config.** Ten minutes to
   protect twelve hours. And `rs_vqa` runs first — if the data format or LoRA wiring is
   broken, find out at hour 4, not hour 16.
5. **Checkpoint every ~500 steps** (adapter + optimizer state) to the private HF repo.
6. **One adapter per person, on that person's own account.** Checkpoint relay is for
   resuming *your own* interrupted sessions. Relaying one run across several accounts to
   defeat quotas violates Kaggle/Colab terms and risks bans mid-project.
7. **Treat every ✅ in the master plan as a claim to re-verify, not as authority.** Seven
   revisions of licence review each found something the previous one missed, and two
   confident conclusions were later reversed.
8. **Official scorer scripts only.** Our harness wraps them; it never reimplements a
   metric. Any gap between our number and the official script's is a P0 bug.
9. **Grounding and mask outputs are graded on the hidden set and are never cut** — whatever
   the triage ladder says. The PS scores "bounding boxes, or masks, as applicable".
10. **SNAP is GPL-3.0 and is invoked strictly as an external process** — never imported,
    never linked. AROSICS must be pinned `>=1.0.0` (pre-1.0 was GPL-3.0).
11. **Work discipline:** if anyone spends more than one day debugging a stranger's
    `requirements.txt`, pull them off it and reimplement the method instead.

---

## 7. Numbers: which to trust

| Number | Trust? | Note |
|---|---|---|
| ~0.35 ms/token at 25% MFU | ✅ | Measured, transferable, cost model is built on it |
| Vision-token counts per source size | ✅ | Verified against the real processor |
| Peak VRAM per config (80 GB sweep) | ✅ | Measured |
| "Checkpointing was inert" | ✅ | Byte-identical activations proved it |
| Umbra = X-band | ✅ | Confirmed |
| Per-adapter hours in `logs/phase_0_timing_sweep_a100_80gb.md` | ❌ | Synthetic 2048² tiles, cap always bound. Superseded by the cost model |
| `logs/ckpt_audit.md` budget table | ⚠ | One cell filled, table marked *(incomplete)* |
| 40% MFU reachable | ⚠ | Assumed. Needs a batch-size sweep — 2–3 cells, under $1 |
| Chip sizes for RarePlanes / SpaceNet 6-7 / HRSCD / OEM-SAR | ⚠ | Taken from published tiling, **not verified against downloaded files** |
| Dataset mix weights | ⚠ | Assumed |
| Data-loading cost | ❌ **Not modelled at all** | See §8.1 — the dominant remaining risk |
| Captioning target length | ⚠ | Longer than the 24-token synthetic answers used in the harness. Budget is unvalidated for it |
| Checkpoint I/O cost | ❌ | Unmodelled |

---

## 8. Open questions — nobody knows these yet

### 8.1 Precompute vs online preprocessing — the dominant unmodelled risk

Every timing measurement used synthetic tiles from a memory cache. If the §4.2 SAR chain
(orbit file, calibrate σ⁰, multilook, Refined Lee, terrain correction, dB, stretch) runs
online per sample, **it can exceed GPU time entirely**.

Precompute preprocessed tensors offline and the cost model holds. Run the chain online and
it does not. This is a design decision nobody has made, and it also determines whether the
question generator writes tensors or image paths.

### 8.2 `preprocessing.yaml` is not frozen

The plan calls it "FROZEN Phase 0". In the file: `tiling.max_pixels: null` with an explicit
"must be measured, not guessed" TODO, and the X- and L-band threshold entries are `null`
("deliberately null — nothing is tuned yet, and inventing thresholds here" would be worse).

Freezing `max_pixels: 262144` needs it to clear **two** constraints, and only one has ever
been measured: the compute budget (measured) and the **5 min/scene prep SLA on the demo
machine** (never measured, and named by the YAML's own TODO as the other binding
constraint).

### 8.3 Everything else still open

| Question | Owner per plan | Note |
|---|---|---|
| How is the hidden set executed? Container we submit / API we host / organisers run our code? | Integration lead | Until answered, headless eval assumes the most restrictive case: one container, one command, no network, CPU-capable deterministic path |
| Which base model — Qwen3-VL-4B vs Qwen3.5-2B | ML | Bake-off unrun |
| Which card — is 80 GB access confirmed, or does §5.5 need re-deriving for 40 GB? | Integration lead | The first run used a 40 GB card against an 80 GB plan |
| Does the hidden set contain bi-temporal pairs at all? | Integration lead | Never asked. Changes how much `change_vqa` matters |
| Judging weights | Integration lead | PS placeholder still unfilled — optimise for breadth meanwhile |
| Bhoonidhi access | Integration lead | Apply Day 1. If no grant by end of Week 3, ship on holdout v0 permanently and stop chasing |
| Demo machine — named and specced? | Backend | The vLLM smoke test is meaningless without a named machine |
| DynamicEarthNet licence | Integration lead | Do not stage until it clears |
| SpaceNet 7 licence external re-verification | Integration lead | Verified from papers only; has had one round, not the C53–C58 external job |

---

## 9. Repo-level gaps a collaborator hits immediately

1. **The repo is not under version control.** No `.git`. Before sharing with anyone:
   `git init`, add a `.gitignore` that excludes staged imagery and any HRSCD 2006 data,
   and commit. Sharing an un-versioned folder means no history, no blame, no branches, and
   no way to merge two people's work.
2. ~~The project docs are not in the repo.~~ **Resolved.** The cost model, resolution
   policy, timing review, Modal runbook and question-generation plan are now under
   `docs/`, indexed by `docs/README.md`.
3. **`ben-micro-split/dataset-metadata.json` declares CC0-1.0** and must not be uploaded
   anywhere until corrected to CDLA-Permissive 1.0 (see mistake 6).
4. **No `requirements.txt` / lockfile pin for AROSICS `>=1.0.0`**, which is a licence
   control, not a preference.
5. **No CI yet** for the checks the plan treats as gates: mask conformance, restricted-
   checkpoint blocklist, GeoTIFF fixtures, manifest provenance.
6. **`ben-micro-split/images/` contains 21 constant-black PNGs.** They are dummies for
   timing only. Delete or clearly quarantine them so nobody trains on them by accident.

---

## 10. Known inconsistency between two live specs

`docs/02-data/data_prep_brief_rareplanes.md` (already handed to a teammate) and the question-generation
plan (QGP v1) specify **different output formats**, and both are currently "the spec":

| | RarePlanes brief | QGP v1 |
|---|---|---|
| Record shape | ShareGPT-style `conversations` array | Flat JSONL with typed fields |
| Coordinates | `[ymin, xmin, ymax, xmax]` scaled 0–1000 | `answer_fn: box_xyxy_normalised` |
| Question variety | 10–15 templates sampled randomly in-script | Central template bank + filtered paraphrase bank, sampled deterministically at manifest build |
| GSD conditioning | Not mentioned | Mandatory `effective_gsd_m` field |
| Provenance | Not mentioned | Mandatory `template_id`, `source_ann_ids`, `provenance_chain` |
| Negatives | "empty backgrounds to prevent false positives" | Explicit 30–40% negative target with typed negatives |

**Neither is wrong; they were written for different moments.** But they must be reconciled
before more than one person generates data, or the four adapters will train on
incompatible manifests. Decide once: which record shape, which coordinate convention, who
owns the template bank. The coordinate convention in particular must match what the base
model was trained to emit — verify against Qwen3-VL's own documented format rather than
assuming.

---

## 11. What a new teammate should do on day one

1. Read `docs/01-plan/satquery-master-plan-v3.8.md` Parts 1, 2 and 5. Skim the rest.
2. Read `docs/02-data/DATASET_SPECS.md` warnings — every one is a bug someone will otherwise write.
3. Read this file's §4 (mistakes) and §6 (invisible rules).
4. `git init` if nobody has yet.
5. Ask which of §8's open questions is theirs.
6. Before touching any dataset: trace its provenance chain and check it against
   `CREDITS.md`. If the chain is not recorded, record it. That is a gate, not a courtesy.

---

## 12. The honest summary

Strong: licensing discipline, measurement discipline, a plan that states its own limits,
and a data inventory with two independent sub-metre cross-modal sources that most teams
will not have.

Weak: most of the pipeline is unwritten, no model has ever seen a real satellite image on
this project, the primary dataset's imagery cannot currently be obtained, the biggest cost
variable (data loading) is unmeasured, `preprocessing.yaml` is not frozen despite being
called frozen, and the repo has no version control.

The plan is ahead of the build by a wide margin. That is recoverable, but only if
newcomers are told — which is what this file is for.
