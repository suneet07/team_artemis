# Testing corpora — which Modal account holds what

Three Modal workspaces, not one. Each adapter was trained on a different
account, and **the test split lives on the same account as the training run that
produced it**. This is the map, plus the traps that come with it.

`modal profile list` shows all three; `MODAL_PROFILE=<name>` selects one.

| account | test corpora on `satquery-data` | checkpoints | segment |
|---|---|---|---|
| **suneet-sharan-ug25** | `eval/vrsbench_val` (`Images_val`, `VRSBench_EVAL_referring.json`, `_Cap.json`, `_vqa.json`), `eval/rsvqa_hr`, `eval/rsvqa_lr`, `eval/sn2`, `eval/rareplanes`, `eval/sn6` | `rs_vqa/adapter`, `change_vqa/adapter` | **grounding, captioning, `rs_vqa`** — and this is the **serving** account |
| **suneetsharan14** | `eval/cdvqa_qa` (`Val_*`, `Test_*`, `Test2_*`), `eval/sn7`, `eval/india`, `eval/change_vqa_val.jsonl` | `change_vqa/` with `run.json` + step snapshots, `change_map` | **`change_vqa`** |
| **revanshu2473** | `eval/reben` (`patches`, `verify.json`, `stage_summary.json`), `eval/sn6`, `eval/rareplanes` | — | **SAR / BIFOLD** |

## Traps

**`change_vqa` was trained twice, on different datasets.** The first run used
**SpaceNet 7**; the second used **CDVQA**, and the second is the one that ships.
They are easy to confuse because both are bi-temporal and both are called
`change_vqa`:

```
eval/change_vqa_val.jsonl      sample_id sn7_*            FIRST run. Not ours.
eval/cdvqa_qa/Val_*.json       sample_id cdvqa_val_*      CDVQA run. Ours.
logs/dumps/*_change_vqa.jsonl        240 rows, sn7_*      FIRST run's dump.
logs/dump_heldout/*_cdvqa.jsonl     1200 rows, cdvqa_*    CDVQA run's dump.
```

**Which checkpoint is served.** Verified by hash rather than by name, because
the directory names do not distinguish the runs:

```
ug25 checkpoints/change_vqa/adapter          sha256 46129c61...
s14  checkpoints/change_vqa/adapter          sha256 46129c61...   same file
s14  checkpoints/change_vqa/adapter_step4000 sha256 3b21a4b4...   different
```

The **AA 68.0%** figure comes from `logs/cdvqa_heldout_dump.json`, which records
`adapter: /data/checkpoints/change_vqa/adapter` and `composites: 6` over 1,200
rows. That is the `46129c61` file, and it is what the deployment loads. Served
and benchmarked are the same weights.

`adapter_step4000` appears only in `logs/eval_step4000.json`, scored **41.6%**
at `composites: 2` against the *SpaceNet 7* benchmark. Different run, different
data, correctly not served. Do not let the `step4000` in the CDVQA log filenames
suggest otherwise — those logs evaluate `adapter`.

**CDVQA image pairs.** `stage_cdvqa.py` joins
`Val_questions.json` / `Val_answers.json` / `Val_images.json` against image pairs
under the CDVQA image tree (`im1` / `im2`). A row carries exactly two images,
`["first", "second"]`, RGB.

## `composites` is a budget, not a count

The number in `_COMPOSITES` (`rs_vqa` 3, `change_vqa` 6, `rs_ground_caption` 2)
is a **sequence-length budget**. `satquery/training/dataset.py` applies it as:

- one view in the row → repeated up to the budget (RSVQA's single image → 3);
- several *real* views → passed through with the views it has (CDVQA's 2 stay 2);
- more than `MAX_VIEWS` → raises.

So `change_vqa` is scored on **two** views, not six, whenever its rows come from
a two-image source. Reading 6 as "always send six" pads a duplicate date onto
every pair. Reading it as "truncate to six" is the dangerous direction the
dataset comment calls out: keeping the first three of a six-view row hands the
model a single date and it converges on the answer prior.

## Per-item benchmark evidence that exists today

What can be turned into a gallery without paying for another GPU run:

| segment | file | rows | correct |
|---|---|---|---|
| `change_vqa` | `logs/dump_heldout/Qwen_Qwen3-VL-4B-Instruct_cdvqa.jsonl` | 1200 | **816** |
| grounding | `logs/pipeline_qwen_precise.json` → `records` | 150 | **94 hits** |
| captioning | `logs/caption_strong.json` → `examples` | 30 | ROUGE-L per item, no binary |
| `rs_vqa` | — | — | **none dumped** |
| SAR / BIFOLD | — | — | **none dumped** |

`rs_vqa` is recoverable: the harness takes `--dump-predictions <dir>` and writes
one `{model}_{benchmark}.jsonl` per pair. `eval_bifold.py` has no per-item dump
and would need one added.

---

## The gallery, and the seven bugs it found

`data/gallery/manifest.json` holds **200 held-out rows** — 50 `rs_vqa`, 50
`change_vqa`, 50 SAR, 25 grounding, 25 captioning — each with its question, its
published gold answer, its imagery, and the Modal account it came from. Built by
`scripts/build_gallery.py`, verified by `scripts/verify_gallery.py`, served at
`/api/v1/gallery` and shown at `/gallery` (S8).

**175 verified, 25 scored, 0 failed.** Verified means the item was replayed
through the live `/queries` route and its answer matched gold *through that
benchmark's own contract* — `contract_for(benchmark, question_type)`, the same
formatter that produced the published number. Captioning is `scored`, never
verified: a caption has no right answer, and ROUGE-L is not a verdict.

Anything the system got wrong was dropped and replaced from the same pool, then
re-verified. Three top-up rounds; the pass rate on *unfiltered* replacements
(~2/3) is the honest read on real accuracy.

### Why it mattered

Every published number was measured by calling a component **directly**. The
gallery was the first thing to exercise all of them **through the router**, and
the seam is where everything was broken:

| bug | effect | fix |
|---|---|---|
| `context.scene_of` read from `QueryContext` | **every** crossmodal query failed | use `tool_context` |
| referring expressions matched no grounding term | grounding 12%; the VQA adapter answered "yes." | `_is_referring_expression` |
| grounding vocabulary missing 8 DOTA classes | roundabout, storage-tank, ground-track-field unroutable | taken from the benchmark's `obj_cls` |
| `PRECISE_PROMPT` got an extracted noun | not the input 62.7% was measured on | pass the whole sentence |
| Sentinel-1 names no bands | `lulc_classifier needs band 'VH'` — BIFOLD never ran | `assumed_sar_orders: {2: [vh, vv]}` |
| BIFOLD emitted labels, no answer | the 0.4968 component spoke over the 0.7525 one | `answer_for` shared with the eval |
| a raising tool vanishes from `steps` | all of the above were invisible | dropped tools now warn |
| route checks planned but never executed | the crossmodal crash passed 44/44 | execution probe in `verify_routes.py` |
| Volume snapshotted at container start | published gallery updates never appeared | `reload()`, `?refresh=1` |

Results: grounding **12% → 80%**, SAR **46% → 74%** against a benchmarked
**74.95%**.

### Two traps worth remembering

**`assumed_sar_orders` is measured, not assumed.** reBEN S1 is **VH then VV**.
The probe in `logs/bifold_probe_sar.json` scored `vh_vv` at **0.817** and
`vv_vh` at **0.499** — chance. The intuitive reading is the wrong one.

**A top-up must exclude what already failed.** The first implementation built
its exclusion set from the manifest *after* pruning, so the failures were no
longer in it and the balanced pool returned the same rows in the same order —
replacing all 31 dropped items with themselves. It now reads every
`verified*.json` and excludes anything that has ever failed.

### What now catches each class

Three layers, and they cover different things. Naming which is which matters,
because each of today's bugs slipped past the layers above it:

| layer | cost | catches | missed today |
|---|---|---|---|
| `pytest` (436) | 20 s | logic, contracts, formatters | the crossmodal crash — nothing ran a two-modality pair |
| `verify_routes.py --offline-only` | 6 s | every Task reachable, every planned tool executable, **and now** a crossmodal query executed end to end | answer *quality* — stub adapters return a fixed string |
| the gallery (200 held-out rows) | ~25 min GPU | quality against published gold, through the router, on real imagery | nothing found so far that the other two could have |

The middle layer grew its execution probe *because* of the crossmodal bug, and
the probe was validated the only way that means anything — the bug was
reintroduced and the probe failed with the exact original error, then the fix
restored and it passed.

The gallery is the only layer that can catch a wrong *answer*, and it earned its
place: seven of the eight bugs above were invisible to the other two, because a
component measured by calling it directly says nothing about whether the router
ever reaches it.

---

## reBEN's S2 band order — closed as unresolvable, deliberately

Sentinel-1 named no bands and that was fixable: the order is VH then VV, and
`logs/bifold_probe_sar.json` settles it at **0.817 against 0.499**. The S2 side
looks like the same problem and is not.

**What was tried.** Two candidate orders, tested against reBEN's own published
per-band means over 34 staged patches. The test is ratio consistency: if the
order is right, observed/published should be near-constant across bands, that
constant being regional brightness.

```
wavelength  B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12   ratio CV 0.408
configilm   B02,B03,B04,B08,B05,B06,B07,B11,B12,B8A   ratio CV 0.331
```

Neither is clean. Wavelength puts 9 of 10 ratios in a tight band (0.89-1.32)
with channel 10 a wild outlier at 2.93; configilm has a lower CV but no
consistent core at all (0.51, 2.42, 1.53 …). One fits well except for a band
that cannot be explained; the other fits nothing well.

**Why the model cannot break the tie.** `eval_bifold.py probe` ran the optical
model at every candidate order:

```
configilm 0.5913 · wavelength 0.5760 · string_sorted 0.5533   (floor 0.5013)
optical_alone across 4,000 rows: 0.4968      <- chance
```

The optical classifier is **at chance on this data whatever the order**, so its
accuracy carries no information about the ordering. The one experiment that
could decide this is uninformative by construction.

**The decision.** No band map is written. `configs/preprocessing.yaml` warns
that a wrong one silently corrupts every index, and that is exactly the risk
here: a plausible-looking NDVI computed from the wrong channel is worse than no
NDVI, because nothing downstream can tell. The two orders agree only on
B02/B03/B04, and blue/green/red alone computes no index, so a partial map buys
nothing either.

### What that means for cross-modal queries on reBEN

The optical opinion comes from `texture_seg`, and the answer now says so.
`texture_seg` measures **surface roughness**; `sar_backscatter` measures **radar
return**. They are different quantities, so a low IoU between them is the
expected outcome and not evidence that the sensors disagree about the ground --
but D1's rule table is written for *spectral*-vs-radar, so it finds no rule,
reports `cause: unclassified`, and the panel reads as a caught conflict.

The fusion still runs and the agreed extent is still the honest answer. What
changed is that a texture-backed comparison now carries a warning saying the
optical side is a proxy and the low agreement is expected. Two related fixes
landed with it:

- **`produced_by` emitted a filename token, not a tool name.** It was
  `path.stem.split("_")[0]`, so `texture_seg_mask.tif` reported `"texture"`.
  The contract types the field as the tool name and the disagreement panel
  matches on it exactly, so that panel had rendered "NO MASK" on **every**
  cross-modal conflict since it was written. Now resolved against the registry.
- **Coverage sentences were unattributed.** "Builtup covers 19.0% … Builtup
  covers 0.3%" was optical texture and radar backscatter describing the same
  class, reading as a self-contradiction. Now "By optical, …" and "By radar, …".

### To actually close it

A source whose optical bands are named, over a footprint with radar. Our named
S2 chips are T33UUP; reBEN is T34TCR/TEQ/TFN/TCS, so there is no overlap today.
SpaceNet 6 is the right answer -- 0.5 m Capella X-band co-registered with Maxar
optical -- and only its optical half (`PS-RGB`) is staged; `SAR-Intensity` is a
41 GB fetch. That is the open item, and it is a data-acquisition task rather
than a code one.
