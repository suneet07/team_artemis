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

`satquery/serving/` · `satquery/api/` — that is P9 and the whole FastAPI layer.
(P2, P3, P4, P7 and P8 are no longer on this list; see §3.7.)

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

## 3.7 The audit of 2026-08-28 — what was stubbed, and what it cost

A pass over Phases 0-4 against the code found that most of P5 existed as
*shape* rather than as behaviour, and that several stubs were not merely
incomplete but actively false in the graded artifact. Recorded here because the
pattern matters more than the individual fixes.

**The controller could not run.** `route_node` handed a string to
`set_routing`, which does `self._task.value` at build time, so `TraceBuilder`
raised on every non-refusal path. `gate_node` handed a `ToolManifest` dataclass
to something that called `.get()` on it. Neither had ever been executed
end to end; both were covered by tests that exercised the pieces separately.

**Stubs that lie are worse than stubs that fail.** `executor_node` wired
`lambda p: {"result": "ok", "confidence": 0.9}` for every tool; `confidence_node`
returned a hardcoded `0.95`; `evalcli/batch.py` wrote `"Mocked answer for ..."`
with `parameter_check.passed = True`. Each of those writes a confident, false
value into the file the problem statement says is the graded artifact, and a
reader cannot tell it from a real run. **A stub must fail loudly or say it is a
stub in its own output.**

**The tools and the parameter gate were mutually incompatible.** The tools took
their rasters through `params` (`index_array`, `sigma0_array`, `image_array`),
but `params` is exactly what section 4.5.4 validates against the manifest, and
the manifests permit `index` and `threshold_method`, not arrays. So no real tool
could pass its own gate — which is *why* the controller ran stubs. Pixels now
travel on `ToolContext`; only manifest-declared settings travel in `params`.
**If a design forces you to bypass a gate, the design is wrong, not the gate.**

**Thresholds had drifted back into Python.** `0.2`, `0.3`, `-18.0`, `-3.0`,
`5.0` were hardcoded in `tools/deterministic.py`, and `(sar_band, polarisation)`
keying did not exist, so an X-band RISAT-2B scene was silently scored against
C-band physics — the Critical risk row the plan opens with. All of it is in
`preprocessing.yaml` now, and an untuned band warns and lowers confidence.

**Two performance choices would have missed the SLA outright.**
`compute_texture` was `generic_filter(np.std, size=3)` — a Python callback per
pixel, minutes on a benchmark chip and hours on a full scene against a 5 min
prep SLA. `GeoTiler` defaulted to 1024 px tiles, four times the frozen
`max_pixels`, so every tile it produced would have been downsampled inside the
processor with no record of it.

**Things that were quietly wrong in ways tests would not catch.**
`format_yes_no` matched `"no"` as a substring, so "a road to the **no**rth",
"**no**ne of the fields" and "we can**no**t tell" all scored `no` on binary VQA.
`create_refusal` returned a `remedy` key the trace schema forbids, so every
graceful refusal failed validation. `change_stats` returned `float("inf")` for
new construction, which is not valid JSON. The ECE binning used `< upper` on
every bin, so a prediction of exactly 1.0 fell in no bin and was dropped —
precisely the samples a calibration number is most often wrong about. The
disagreement table returned the *water* row `wet_smooth_soil` for a built-up
conflict, and picked `sar` by default when no rule fired, against the plan's
explicit "never silently pick one".

**The routing eval set measured nothing.** It sampled the expected task and the
expected router path *at random*, so its ground truth was noise; it was
unseeded, so the file changed on every run; it wrote to a hard-coded path on one
developer's machine; and every `expected_tools` entry was `["dummy_tool"]`. It
is now authored case by case, deterministic and byte-stable, and
`training/eval/score_routing.py` reports the number the risk register triggers
on. Current: **100% task accuracy on unambiguous cases, 100% invalid-config
catch rate, 95% routed by rules.**

**The licence gate cried wolf.** It scanned for bare substrings, so the ordinary
English words "second" and "ground" tripped it the day real prose was written.
It now matches word-boundary identifiers, exempts `CREDITS.md` (naming an
exclusion is that file's job), checks that every shipped weight maps to a
CREDITS entry marked CLEAR, checks the AROSICS `>=1.0.0` pin, and asserts SNAP
is never imported.

**CI existed as an empty directory.** `.github/workflows/` had no workflow in
it, so none of the gates the plan calls non-negotiable had ever run
automatically. There is now a workflow with mask conformance, the parameter
gate, the trace schema, the licence rails and routing accuracy as separately
named checks.

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
| 13 | **Inferring a coordinate scale from magnitude** | Three times. VRSBench references are **0–100**, Qwen predictions **0–1000**, our generator **0–1 floats**. Normalising Qwen by image size clamped every value above 512 to 1.0, collapsed the box, and reported **75% "unparsable"** — indistinguishable from a model that cannot ground at all. Only dumping raw replies showed the answers were fine | Coordinate scale is **declared per contract**, never guessed. `AnswerFormat.box_scale`. Print raw model output before trusting any score |
| 14 | **Re-deriving a fixed regex** | A bare number scan takes the **`2` out of `bbox_2d`** as the first coordinate and shifts every value one place. Fixed once in `format_box`, then re-introduced by writing a fresh pattern for `point_2d` — which scored **0.0%** and read as total model failure | Import `formatter._BOX_NUMBER`; do not write a new number pattern |
| 15 | **Reporting a stratified sample as a score** | Class-balanced sampling gives `vehicle` 4% weight where the benchmark gives it 28%. It moved Grounding DINO's number by 3–7 points | `--sampling random` for any headline. Stratified is for per-class diagnosis only, and must be labelled as such |
| 16 | **Comparing models on different samples** | The detector and the base VLM were each measured on their own draw for a full day before anyone scored them on the same questions | Fixed shared row list (`/data/eval/pipeline_sample.json`). Paired comparison or no comparison |
| 17 | **Two `modal run` calls of one app** | The second silently displaces the first — seen as "Webhook label stolen", then as a fetch that simply stopped. Cost a SpaceNet 6 run and a completed sweep variant | One app, one run at a time. Chain sequentially, or use a single entrypoint that does both. `--detach` for anything that must outlive the terminal |

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

**`change_map` — commissioned, trained, abandoned. Closed 2026-09-06; re-litigated twice, do not reopen.**

It was never rejected on licence grounds, which is the version everyone
half-remembers. C33 -> C56 dropped *training it on LEVIR/SECOND/QAG*; the
SpaceNet 7 route that replaced them is permitted, and that is what we trained.
The full arc:

| step | what happened |
|---|---|
| pitched | Measurement over inference. CDVQA `smallest_change` sits at 32-37% for every published model; the plan targeted >60% via `change_stats` arithmetic on a mask |
| trained | Siamese U-Net on 12,004 SpaceNet 7 pairs. Best **F1 0.2931 / IoU 0.1717** at step 2750, then drifted down. Dead end |
| alternatives | TinyCD non-commercial · Open-CD Apache code but weights inherit LEVIR academic-only, masks are non-directional, 8x resolution gap · SpaceNet winners' weights are Apache-2.0 and genuinely published, but wrong task and resolution · awesome-list has no weights |
| closed | The problem statement names **CDVQA** as the graded benchmark, so the trained VLM adapter is what serves change VQA |

**The decisive reason is task mismatch, not weight quality.** CDVQA asks about
*land cover* -- buildings, low vegetation, trees, water, playground. Our detector
finds *buildings*, and the measurement pipeline needs footprints at inference
that CDVQA does not have and we cannot produce. At F1 0.95 it still could not
answer a CDVQA question.

**What survives and is worth keeping.** The arithmetic half is finished and
verified: router plus `change_stats` scored **100% AA on 2,012 rows**. It was
scored by handing it SpaceNet 7's own ground-truth footprints, so what is empty
is *perception*, not reasoning. If a land-cover segmenter ever lands on a
clearable licence, the reasoning layer is already built and tested.

**Do not compare 43.5% to 68.0%.** The old adapter's 43.5% was our SpaceNet 7
corpus (`change_compare/count/direction/magnitude/presence/where`); the new
adapter's 68.0% is CDVQA's eight land-cover types. Different corpora, different
taxonomies. The claim that holds is "the shipped adapter is measured good on the
benchmark we are graded on", not a 24-point gain.

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
| change_vqa AA 68.0% on CDVQA Val | ✅ | 1,200 balanced rows, IF 1.000, 0 unparsable. See §7.1 |
| change_vqa "beats 68.6% SOTA" | ❌ | Val + in-repo scorer vs Test1/Test2 + official scorer. Not comparable |
| The 0.727 from the first bakeoff run | ❌ | `--limit 160` slice, 59% yes/no. Superseded by the balanced 1,200 |
| +23 points over the 45.0% blind ceiling | ✅ | The defensible claim about vision contribution |

---

## 7.1 The change_vqa adapter — trained, scored, done (2026-09-06)

First adapter in this project to generate a token. Everything before this was
loss curves.

**Training.** 4,000 steps, 2.41 h on A100, 2.167 s/step, peak 19.13 GB.
Loss 6.314 -> 0.172. LoRA r=16 alpha=32 all-linear, 40,271,872 trainable
(0.899%). Corpus `change_vqa_cdvqa_combined.jsonl`, 37,518 train rows, six
composite views per sample.

val_loss fell to the last check -- 0.1637 (3250), 0.1615 (3500), 0.1615 (3750),
**0.1608 (4000)**. No overfitting. The step-2750 "val has risen three times"
warning was noise; val set four new bests after it fired. Do not trust that
warning at `--val-batches 20`: 160 samples per check is too few, and train_loss
in the same line is a single batch.

**Result.** 1,200 rows of official CDVQA Val, 150 per question type:

| metric | value |
|---|---|
| AA (macro over 8 types) | **68.0%** |
| overall | 68.0% (identical -- split is balanced) |
| instruction-following | 1.000 |
| unparsable | 0 |

| type | acc | | type | acc |
|---|---|---|---|---|
| change_or_not | 85.3% | | largest_change | 66.7% |
| increase_or_not | 83.3% | | change_to_what | 65.3% |
| decrease_or_not | 81.3% | | change_ratio | 52.0% |
| change_ratio_types | 80.7% | | **smallest_change** | **29.3%** |

Blind ceiling is 45.0%, so **+23 points come from actually seeing the imagery**.
That gap is the defensible claim, not the absolute number.

**What this number is NOT.** Published CDVQA figures (55.3% baseline, 68.6%
SOTA) are Test1/Test2 through the official scorer. This is Val through the
in-repo comparator. Two uncontrolled differences, so 68.0 is not "we matched
SOTA" -- it is "we are in that range under our own measurement". Closing it
needs a Test1/Test2 run through the official scorer.

**`smallest_change` at 29.3% is the outlier.** Every other type clears 52%. It
costs roughly 5 points of AA alone. Unresolved whether that is a real capability
gap or a formatter/vocabulary mismatch -- it is the type most likely to have
near-tied class ratios, where a small ranking error flips the answer.

**A sampling trap, recorded because it nearly became a false claim.** The first
run used `--limit 160`, which *slices* the first 160 rows rather than sampling.
That slice was 59% yes/no questions and left four types on n=8-12; it scored
**0.727**, above published SOTA. The number was an artifact of which rows the
slice happened to contain. Always score the balanced 1,200
(`/data/manifests/cdvqa_heldout.jsonl`), never a bare `--limit`.

**`--composites 6` is mandatory** when scoring this adapter. It trained on six
views per sample and `bakeoff` leaves rows as-is by default, which would score
the model on an input shape it never saw.

**Leakage was checked and there is none.** Official Train (1,574 image pairs)
and official Val (393 pairs) share zero images. An earlier scare came from
comparing Val against the *combined* manifest, which embeds the 1,200 Val rows --
so Val overlapped itself. Compare against `split == "train"` rows only.

**Adapter inventory.** `change_vqa` is the only trained LoRA adapter that exists.
`rs_vqa` has no checkpoint on the Volume, so `Api` reports it absent -- correct
behaviour, not a bug. `checkpoints/change_map/` holds the Siamese U-Net (`.pt`,
not a LoRA). Final weights: `/data/checkpoints/change_vqa/adapter`, with
snapshots at `adapter_step1000/2000/3000/4000`.

**Housekeeping left.** `checkpoints/change_vqa/` still holds `step=4500.ckpt`,
`best-v1.ckpt` and `last-v1.ckpt` from the earlier SpaceNet 7 run. They are
stale and misleading beside a 4,000-step run. Not deleted -- removal needs a
decision, not an assumption.

---

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
