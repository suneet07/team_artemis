# P5 — Agentic Controller: Complete Build Specification

**SatQuery AI · SIH26167 · ISRO/SAC**
**Version:** 2.0 · supersedes v1.0 (see §32 for what changed and why)
**Component:** P5 (Agentic Controller) + P6 (Tool Registry) + P7 (Fusion & Confidence) wiring
**Owner:** one engineer, full ownership
**Estimated effort:** ~3 weeks
**Graded by:** G6 — *the judges score the trace file this component produces*

---

## 0. How to read this document

This is a **build specification**, not a tutorial. It tells you exactly what to build, in what
order, what each piece takes in, what it gives out, and how you know it works.

**You do not need to have used LangGraph before.** §3 explains every term. The design is
deliberately boring: a fixed pipeline with a few branches, not a clever self-directing AI. If you
understand a flowchart, you understand this.

**You will be working with an AI coding assistant.** That is expected. §30 has ready-made
prompts. Three rules:

1. **Never let the assistant invent a function that already exists.** §6 lists every function
   already written. Paste it into your assistant at the start of every session. The single most
   common failure is an assistant writing a second, subtly different parameter validator when
   `check_parameters()` already exists.
2. **Never let the assistant loosen a rule.** §5 lists rules that look relaxable and are not. If
   it suggests "we could just clamp the value instead of rejecting it" — that is precisely the
   thing this component exists to *not* do.
3. **Build one node at a time and test it.** Do not ask for the whole graph at once. You will get
   800 lines of plausible code that fails somewhere nobody can find.

**Read before starting:** `docs/01-plan/satquery-master-plan-v3.8.md` §4.5–4.7,
`frontend/FRONTEND_PLAN.md` §4, and `frontend/contracts/types.ts`. That last one is not optional —
see §6.10.

---

## 1. What this component is, in plain English

A user uploads one or two satellite images and types a question, e.g. *"How much water is in this
scene?"* or *"What changed between these two dates?"*

Something has to:

1. Work out **what kind of question that is** (there are exactly 8 kinds).
2. Check the question is **answerable with the images given** — you cannot answer "what changed"
   with one image, nor a colour question about radar.
3. Decide **which tools to run** and with what settings.
4. Check those settings are **legal** before anything runs.
5. **Run the tools**, some in parallel, streaming progress to the UI as it goes.
6. **Combine** their answers and work out **how confident** to be.
7. Write a **trace file** recording every one of the above decisions.

That something is P5. It is the brain between the user and the fourteen tools.

### The two things that matter most

**The trace file is the deliverable.** Not the answer. The problem statement says only the
observable execution trace is evaluated. A correct answer with a missing trace scores zero. A
graceful refusal with a complete trace scores points. Everything here that feels like bureaucracy —
the parameter gate, the recorded router path, the rejected-parameter list — exists because a judge
reads that file.

**The frontend API contract is frozen and you build to it.** `frontend/contracts/openapi.yaml` and
`types.ts` are marked `1.0.0-frozen`. Your teammate is already building against mocks of those
shapes. Every event name, every enum value, every field name in this document comes from that
contract. Where you are tempted to name something differently — don't. Changing it requires a
version bump and messaging both sides the same day.

---

## 2. Where P5 sits

Ten pipelines, P1–P10, split into two phases with **two different speed targets**:

```
  ═══ PREPARATION (async job per uploaded scene, target ≤ 5 min) ══════════════════

     P1 Ingest → P2 SAR Normalise → P3 Co-register → P4 Tiling
                                          │
                                          ▼
                            ImageBundle (cached, reusable)
                            + supported_tasks / blocked_tasks   ← §21, partly yours

  ═══ QUERY TIME (per question, target < 20 s) ════════════════════════════════════

                       ImageBundle + question text
                                  │
                                  ▼
         ┌──────────────────────────────────────────────────┐
         │  ►►►  P5  AGENTIC CONTROLLER   (YOUR JOB)  ◄◄◄    │
         │  route → validate → plan → gate → execute →      │
         │  fuse → confidence → emit trace                  │
         │  …emitting SSE events throughout (§19)           │
         └──────────────────────┬───────────────────────────┘
                                │ calls
                                ▼
         ┌──────────────────────────────────────────────────┐
         │  P6 TOOL REGISTRY (14 tools) · P7 FUSION         │
         │  P9 MODEL SERVING (vLLM multi-LoRA)              │
         └──────────────────────┬───────────────────────────┘
                                ▼
              answer + AssetRefs + trace.json → P8 report, P10 frontend
```

You never touch a raw `.tif`. By the time a question reaches you, P1–P4 have produced an
`ImageBundle`. You consume it.

| You own | You do not own |
|---|---|
| `satquery/agent/` — the whole graph | `satquery/ingest/` (done) |
| `satquery/tools/` wiring + 6 deterministic tools | Tool internals marked "ML" in §14 |
| `satquery/fusion/`, `satquery/confidence/` | Model training |
| `satquery/serving/client.py` (the *client*) | The vLLM deployment itself |
| Refusal logic + trace emission | Frontend rendering; PDF rendering |
| SSE event emission | The FastAPI route handlers (but you define what they stream) |
| `satquery/evalcli/` headless mode | Tiling *algorithm* (P4) — but see §12 |

---

## 3. Vocabulary

| Term | Meaning |
|---|---|
| **Agent** | Here: **not** an autonomous AI. A controller following written rules, calling an LLM for one narrow tie-break. §7. |
| **LangGraph** | MIT-licensed Python library for programs shaped like flowcharts. Boxes (nodes), arrows (edges), one shared object passed along. That is all it is. |
| **Node** | One box = one Python function. Takes state, does one job, returns updated state. |
| **Edge** | Arrow to the next node. A **conditional edge** forks: a function reads state, returns the next node's name. |
| **State** | One dict passed through every node, accumulating results. Ours is `AgentState`, §8. |
| **Tool** | One capability. Fourteen of them, §14. |
| **Tool manifest** | YAML declaring what a tool may be given. `configs/tools/`. A contract enforced by code, not documentation. |
| **Trace** | The JSON recording every decision. The graded artifact. §17. |
| **Task** | One of exactly 8 question types, frozen in `task_enum.py`. |
| **Adapter / LoRA** | Small trained add-on specialising the base VLM for one job. Four of them. You *call* them. |
| **BandInventory** | Which spectral bands exist and which indices are computable. Drives routing. |
| **Modality** | `optical` \| `sar` \| `unknown`. Radar has no colour; that constrains questions. |
| **σ⁰** | Radar backscatter, dB. Water dark (≈ < −18 dB), buildings bright. |
| **Refusal** | A deliberate, explained "I cannot answer that, here is why, here is what to do." A first-class output. |
| **SSE** | Server-Sent Events — a one-way stream of named events from server to browser. How the UI watches you work. §19. |
| **AssetRef** | The frozen object describing any produced file (mask, overlay, boxes). §18. |

---

## 4. Your four framing questions, answered directly

**"What tools will each agent have?"** — There is **one** agent, not several. It can reach all 14
tools, but the router narrows to a per-task permitted set (§15), and `BandInventory` narrows
further.

**"What is the orchestration agent?"** — A LangGraph `StateGraph` in `satquery/agent/graph.py`.
It is **not** an LLM. It is deterministic code. An LLM is consulted at exactly one point (routing
tie-break) with output constrained to a fixed JSON schema.

**"What input does it take?"** — An `ImageBundle` (1–2 prepared images + BandInventory + coreg
report + tile index) and a question string. §9.

**"What output does it give?"** — An answer, a set of `AssetRef`s (masks, boxes, overlays), a
confidence with its basis, and the trace JSON — plus a live SSE event stream while it works.
§18, §19.

---

## 5. The fourteen rules that cannot be relaxed

1. **Out-of-range parameters are REJECTED, never clamped.** Clamping hides the planner's mistake
   and writes a wrong-but-plausible number into the graded file.
2. **Unknown parameter names are rejected, not ignored.** Silently dropping `thresh` when the
   manifest says `threshold_value` hides a bug.
3. **No node executes until the parameter gate has passed for that step.** The gate is not
   advisory.
4. **Never parse free-form LLM output for control flow.** The tie-breaker uses constrained JSON
   decoding. If the constrained decoder is unavailable, fall back to rules and record it.
5. **Rules first, LLM second.** A mostly-rules trace is *more* defensible to a judge, not less.
6. **A refusal beats a hallucination**, and every refusal carries a `remedy` (§20).
7. **Every mask is a full-scene, geo-referenced GeoTIFF in the source CRS**, mosaicked back from
   tiles. Never tile-resolution, never overlay-only.
8. **Smallest sufficient plan.** Explicit step-count penalty in the planner.
9. **Adapter names always include the base model** — `change_vqa@qwen3vl-4b-v2`.
10. **Never silently pick a modality when optical and SAR disagree.** Classify the cause, name the
    winner, lower confidence.
11. **The trace is written on every path** — success, refusal, crash. A missing trace is a zero.
12. **[v2] Inference prompt assembly must byte-match training prompt assembly.** One shared
    assembler, imported by both. §13. Violating this silently degrades every adapter.
13. **[v2] Every name, enum value and event in the frontend contract is frozen.** Build to
    `types.ts`. Do not invent adjacent vocabulary.
14. **[v2] Never send more than the configured tile budget to a learned tool.** Tile selection is
    explicit and recorded, never "feed it everything and hope". §12.

---

## 6. What already exists — do not rewrite any of this

**Paste this section into your AI assistant at the start of every session.**

### 6.1 `satquery/agent/task_enum.py` — DONE, frozen

```python
class Task(StrEnum):
    SINGLE_VQA = "single_vqa"
    SINGLE_CAPTION = "single_caption"
    SINGLE_GROUNDING = "single_grounding"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CHANGE_MAP = "change_map"
    CROSSMODAL_EXTRACTION = "crossmodal_extraction"
    CROSSMODAL_VQA = "crossmodal_vqa"

class RouterPath(StrEnum):
    RULES = "rules"
    LLM = "llm"
```

### 6.2 `satquery/agent/trace.py` — DONE (194 lines)

`SCHEMA_VERSION = 2`; validates against `configs/trace_schema.json` on every build.

`TraceBuilder(query_text, query_id=None, timestamp=None)`, chainable:

| Method | Purpose |
|---|---|
| `.set_routing(task, router_path)` | **Required before `build()`** |
| `.set_inputs(list[dict])` | Per-image metadata |
| `.set_compatibility(dict)` | Coreg / CRS report |
| `.add_routing_note(str)` | Substitutions and reroutes |
| `.add_planned_step(tool, params, within_manifest, defaults_applied=None)` | → `graded.permitted_parameters` |
| `.set_parameter_check(passed, rejected)` | **Required before `build()`** |
| `.set_tools_invoked(list)` | Optional; auto-derived from steps |
| `.add_step(tool, params, outputs, confidence=None, latency_ms=None, param_source="manifest_validated")` | One executed step |
| `.set_agreement(iou, verdict, disagreement_cause)` | **Only 3 fields** — see §6.10 note |
| `.set_fusion(model, answer, confidence)` | |
| `.set_outputs(answer=, masks=, area_km2=, confidence=, extra=, refusal=)` | → `graded.outputs` |
| `.add_evidence(str)` / `.add_warning(str)` | Appendable |
| `.build()` | Validated dict; **raises** if routing or parameter_check unset |
| `.write_json(path)` | Builds + writes, creates parent dirs |

`build()` assembles the `graded` block for you. Never construct it by hand.

### 6.3 `satquery/tools/manifest.py` — DONE

`ToolManifest.from_yaml(path)` / `.from_dict(raw)`, JSON-schema-validated. Fields: `name`,
`description`, `version`, `required_modalities`, `permitted_parameters` (dict of `ParamSpec`),
`outputs`, `confidence_source`, `low_confidence_proposer`, `expected_latency_ms`.

`ParamSpec`: `name`, `type`, `values`, `range`, `optional`, `default`, `has_default`,
`requires_bands`.

### 6.4 `satquery/tools/registry.py` — DONE. **This is the parameter gate.**

```python
check_parameters(manifest, params, band_inventory=None, modalities=None) -> ParameterCheckResult
# ParameterCheckResult(passed: bool, rejected: list[str])
effective_params(manifest, params) -> dict          # merges manifest defaults
ToolRegistry.default() / .register() / .get() / .names()
```

Already checks: required modalities, unknown names, enum membership, numeric type + range,
integer-ness, bool/string types, `requires_bands` against `BandInventory`, missing requireds.

**You do not write a parameter validator. It exists. Call it.**

### 6.5 `satquery/ingest/` — DONE, someone else's

```python
ingest_raster(path, config=None, *, modality_override=None) -> IngestResult
# IngestResult(meta, modality, modality_source, inventory, compatibility)
```

`BandInventory`: `bands`, `has_swir`, `has_nir`, `is_pan_only`, `polarisations`, `sar_band`,
`sensor_hint`, `computable_indices`; method `.has_band(name)` (case-insensitive, understands
`swir`/`nir`).

`CompatibilityReport`: `format_ok`, `crs_valid`, `modality`, `modality_source`, `bands_present`,
`computable_indices`, `nodata_frac`, `bit_depth`, `bit_depth_source`, `pixel_size_m`,
`native_gsd_m`, `warnings`; properties `.is_readable`, `.is_georeferenced`.

### 6.6 `satquery/evalcli/smoke.py` — **your reference implementation**

`run_dummy_query()` is a working miniature of the whole controller for one tool: registry lookup →
`effective_params` → `check_parameters` → TraceBuilder with routing + parameter_check → refusal
branch → execute → outputs. **Read this first. ~60 lines. It is the pattern you scale up.**

### 6.7 `satquery/paths.py`

`CONFIG_DIR` (env `SATQUERY_CONFIG_DIR`), `PREPROCESSING_CONFIG_PATH`, `TRACE_SCHEMA_PATH`,
`TOOL_MANIFEST_SCHEMA_PATH`, `TOOLS_DIR`.

### 6.8 `configs/`

```
preprocessing.yaml          # NOT frozen — tiling.max_pixels still null
trace_schema.json           # frozen, schema_version 2
tool_manifest_schema.json   # frozen
tools/dummy_tool.yaml
tools/spectral_index.yaml   # the model to copy — §16
```

### 6.9 `tests/` — 8 modules, green. Keep them green.

Also: **`training/eval/routing_300.jsonl` exists.** The 300-query routing evaluation set is already
built. You can measure your router in Week 1 with no data-collection detour.

### 6.10 `frontend/contracts/` — **FROZEN. This is a hard constraint on you.**

`openapi.yaml` (`1.0.0-frozen`) and `types.ts` define every shape you must produce. Eleven mock
fixtures exist in `frontend/mocks/fixtures/`, including `trace_param_rejected.json` and
`trace_refusal_missing_input.json`. **Diff your output against those fixtures** — two people have
independently committed to a trace shape and they must agree.

Frozen enums you must use verbatim:

```ts
RefusalCategory = "parameter_gate" | "validator" | "modality_limitation"
                | "missing_input" | "unsupported_class"
ConfidenceBasis = "heuristic" | "calibrated"
QueryState     = "queued" | "running" | "succeeded" | "refused" | "failed" | "cancelled"
AssetKind      = "mask_geotiff" | "overlay_png" | "chart_png" | "report_pdf"
               | "scene_preview" | "bbox_geojson"
ErrorCode      = … "BUNDLE_NOT_READY" | "QUERY_REFUSED" | "PARAM_REJECTED"
                 | "MODEL_UNAVAILABLE" | "JOB_CANCELLED" | "INTERNAL"
```

> **Known contract seam — flag this to the team.** The SSE `agreement` event carries
> `winning_modality` and `explanation`, but `TraceBuilder.set_agreement()` accepts only
> `(iou, verdict, disagreement_cause)`. That is *acceptable* — the event is richer than the trace —
> but compute all four values in the fusion node and pass the extra two to the event emitter only.
> Do not silently drop them, and do not add fields to the trace schema without a version bump.

### 6.11 What is missing — your build list

| Missing | File |
|---|---|
| `langgraph`, `outlines` deps | `pyproject.toml` → new `agent` extra |
| State object | `satquery/agent/state.py` |
| Router | `satquery/agent/router.py` |
| Validator | `satquery/agent/validator.py` |
| Planner | `satquery/agent/planner.py` |
| Tile selection | `satquery/agent/tiling_policy.py` **[v2]** |
| Prompt assembler | `satquery/agent/prompt.py` **[v2, shared with training]** |
| Graph | `satquery/agent/graph.py` |
| Executor | `satquery/agent/executor.py` |
| Refusals | `satquery/agent/refusals.py` |
| SSE emitter | `satquery/agent/events.py` **[v2]** |
| Bundle probe | `satquery/agent/probe.py` **[v2, §21]** |
| Fusion / confidence | `satquery/fusion/`, `satquery/confidence/` (only `__init__.py` today) |
| Serving client | `satquery/serving/client.py` (only `__init__.py` today) |
| Real headless CLI | `satquery/evalcli/__main__.py` |
| 12 tool manifests | `configs/tools/*.yaml` |

First commit, before anything else:

```toml
[project.optional-dependencies]
agent = [
    "langgraph>=0.2",
    "outlines>=0.1",     # constrained JSON decoding for the router tie-break
]
```

Keep it an **extra**. CI runs on a CPU runner and base deps are deliberately thin.

---

## 7. The mental model

If you have read about "AI agents" online you have seen systems where several LLM personas talk to
each other and decide what to do. **Ours is deliberately not that.**

The judges grade the trace. A system where an LLM freely decides produces a different trace every
run, cannot be reproduced, and cannot be defended when a judge asks "why did it do that?" A system
following written rules produces the same trace every time and the rule is on screen.

- **One** graph, not a team of agents.
- **Every decision that can be a rule, is a rule.**
- The LLM is consulted at **exactly one point** — routing tie-break — with constrained decoding.
- The learned models are **tools that answer questions about images**. They do not decide what
  happens next.

When your assistant suggests "let the LLM decide which tools to call" — that is the standard
tutorial pattern and the wrong answer here. Rule 5.

---

## 8. `AgentState`

`satquery/agent/state.py`, a `TypedDict`. Every node takes it and returns a partial update.

| Field | Type | Set by | Meaning |
|---|---|---|---|
| `query_id` | `str` | entry | UUID; matches trace and SSE |
| `query_text` | `str` | entry | Raw question |
| `bundle` | `ImageBundle` | entry | Read-only |
| `band_inventory` | `BandInventory` | entry | Handle into bundle |
| `modalities` | `list[str]` | entry | `["optical"]`, `["optical","sar"]` |
| `pair_type` | `str` | entry | `single`\|`crossmodal`\|`bitemporal` |
| `allow_llm_router` | `bool` | entry | From API `options`; default True |
| `task` | `Task \| None` | N2 | |
| `router_path` | `RouterPath` | N2 | |
| `routing_notes` | `list[str]` | N2, N3, N4 | |
| `validation_ok` | `bool` | N3 | |
| `refusal` | `dict \| None` | N3, N5 | `RefusalWithRemedy` shape, §20 |
| `tile_plan` | `TilePlan \| None` | N3b | **[v2]** which tiles, per tool class, §12 |
| `plan` | `list[PlannedStep]` | N4 | `{tool, params, depends_on}` |
| `gate_passed` | `bool` | N5 | |
| `gate_rejected` | `list[str]` | N5 | |
| `replan_count` | `int` | N5 | Hard cap 1 |
| `results` | `dict[str, ToolResult]` | N6 | Keyed by tool name |
| `agreement` | `dict \| None` | N7 | + `winning_modality`, `explanation` |
| `fused_answer` | `str \| None` | N7 | |
| `confidence` | `float \| None` | N8 | |
| `confidence_basis` | `str` | N8 | `heuristic`\|`calibrated` |
| `assets` | `list[AssetRef]` | N6–N8 | **[v2]** replaces v1's `masks`/`evidence` path lists |
| `warnings` | `list[str]` | any | Appended, never replaced |
| `trace` | `TraceBuilder` | entry | Threaded through |
| `emit` | `Callable` | entry | **[v2]** SSE emitter, §19 |
| `timings` | `dict[str,int]` | every node | ms per node |

**Rules:** nodes **append** to `warnings`, `routing_notes`, `assets` — never overwrite. Nodes never
mutate `bundle`. Only the final node calls `.build()`.

---

## 9. Input contract

```python
run_query(
    bundle: ImageBundle,
    question: str,
    *,
    query_id: str | None = None,
    allow_llm_router: bool = True,
    emit: Callable[[str, dict], None] | None = None,   # SSE sink; no-op in headless
) -> QueryResult
```

`ImageBundle` (master plan Part 3 — **not implemented yet**, see §29):

```python
@dataclass
class ImageBundle:
    images: list[ImageRef]
    band_inventory: BandInventory
    pair_type: Literal["single", "crossmodal", "bitemporal"]
    coreg: CoregReport | None
    tiles: TileIndex | None          # {tile_count, tile_size_px, overlap_frac, tiles[]}
    provenance: list[ProvenanceStep]
```

**You may assume:** images readable, common CRS and grid, `band_inventory` populated,
`pair_type` correct.

**You may NOT assume:** a CRS exists (`crs_valid` can be False), low `nodata_frac`, or that
modality was detected rather than declared (`modality_source` tells you).

---

## 10. The graph — node by node

```
        ┌──────────┐
        │N1 INGEST │ ── emit: accepted
        └────┬─────┘
             ▼
        ┌──────────┐
        │N2 ROUTE  │ ── emit: router
        └────┬─────┘
             ▼
        ┌──────────┐  fail
        │N3 VALIDATE├────────────────────┐  ── emit: validator
        └────┬─────┘                     │
             ▼ pass                      │
        ┌──────────┐                     │
        │N3b TILES │ [v2]                │
        └────┬─────┘                     │
             ▼                           │
        ┌──────────┐                     │
        │N4 PLAN   │◄──── replan (max 1) │
        └────┬─────┘                     │
             ▼                           │
        ┌──────────┐ fail(2nd)           │
        │N5 GATE   ├─────────────────────┤  ── emit: plan
        └────┬─────┘                     │
             ▼ pass                      │
        ┌──────────┐                     │
        │N6 EXECUTE│ (parallel waves)    │  ── emit: step_started, step_completed,
        └────┬─────┘                     │           evidence, token
             ▼                           │
        ┌──────────┐                     │
        │N7 FUSE   │                     │  ── emit: agreement
        └────┬─────┘                     │
             ▼                           ▼
        ┌──────────┐              ┌──────────┐
        │N8 CONFID │              │NR REFUSE │
        └────┬─────┘              └────┬─────┘
             └───────┬─────────────────┘
                     ▼
              ┌─────────────┐
              │N9 EMIT      │ ── emit: fusion, done   (always runs)
              └─────────────┘
```

Template for each: **purpose / input / procedure / output / failure modes / trace writes / SSE /
acceptance test.**

---

### N1 — `ingest_node`

**Purpose.** Normalise entry, open the trace. No real work.

**Procedure.**
1. Generate `query_id` (UUID4) if absent.
2. `TraceBuilder(query_text=question, query_id=query_id)`.
3. Build per-image `inputs` from `CompatibilityReport` + `RasterMeta` — `file`, `modality`,
   `native_gsd_m`, `pixel_size_m`, `crs`, `bands`, `swir_available`, `bit_depth`, `nodata_frac`.
   **Do not re-open rasters.**
4. `.set_inputs(...)`, `.set_compatibility({coregistered, rmse_px, checks_passed})`.
5. Copy `bundle.provenance` warnings into `state.warnings`.
6. Start the wall clock.

**Failure modes.** Bundle missing or `status != ready` → refuse, category `missing_input`,
API error code `BUNDLE_NOT_READY`.

**SSE.** `accepted {query_id, queued_ms}`.

**Test.** Fixture bundle → well-formed `inputs` array, one entry per image; assert no raster opened.

---

### N2 — `route_node`

**Purpose.** Pick one of 8 tasks. Graded as `graded.task_selected`.

#### Stage 1 — deterministic rules (must resolve the majority)

First match wins:

| # | Condition | Task |
|---|---|---|
| R1 | 2 images, same modality, (dates differ OR temporal keyword) **and** question is ratio/area/count/"how much"/"how many" | `CHANGE_MAP` (+`change_stats`) |
| R2 | as R1 but "describe the changes" / "what changed" | `CHANGE_DESCRIPTION` |
| R3 | as R1, any other form | `CHANGE_VQA` |
| R4 | 2 images, different modality, extract/measure a target | `CROSSMODAL_EXTRACTION` |
| R5 | 2 images, different modality, otherwise | `CROSSMODAL_VQA` |
| R6 | 1 image, locate/where/find/highlight/show me + noun phrase | `SINGLE_GROUNDING` |
| R7 | 1 image, describe/caption/summarise, no specific target | `SINGLE_CAPTION` |
| R8 | 1 image, anything else | `SINGLE_VQA` |

Temporal keywords: `before, after, change, changed, between, since, growth, new, removed,
demolished, expanded`, plus any two parseable dates. Set `router_path = RULES`.

#### Stage 2 — LLM tie-break (ambiguous residue only)

Trigger only when two rules match with equal specificity and question form cannot break the tie, or
no keyword matched and `pair_type` alone does not determine the task. **Skip entirely if
`allow_llm_router` is False** (the API exposes this option) — fall back to rules and note it.

Constrained JSON decoding (Outlines / vLLM guided) against:

```json
{"type":"object",
 "properties":{"task":{"enum":["single_vqa","single_caption","single_grounding",
   "change_description","change_vqa","change_map","crossmodal_extraction","crossmodal_vqa"]},
   "reason":{"type":"string","maxLength":200}},
 "required":["task","reason"],"additionalProperties":false}
```

Set `router_path = LLM`, append `reason` to `routing_notes`. **If the decoder is unavailable or
errors:** fall back to the general rule by `pair_type`, note it, keep `router_path = RULES`.
**Never regex free text** (Rule 4).

**SSE.** `router {router_path, task_selected, notes}`.

**Test.** **Week 1 gate.** Extend `tests/test_routing_dataset.py` against the existing
`training/eval/routing_300.jsonl`. Report rules-only, LLM-only and hybrid accuracy separately.
Target: rules resolve ≥ 80%, hybrid ≥ 90%.

---

### N3 — `validator_node`

**Purpose.** Hard, non-LLM compatibility rules before any tool runs. Serves the PS's named
"compatibility checking" deliverable.

**Procedure.** Evaluate every rule; collect all failures, do not stop at the first.

| # | Condition | Action | Category | Remedy action |
|---|---|---|---|---|
| V1 | `change_*` but 1 image | Refuse | `missing_input` | `add_second_image` |
| V2 | CRS/extent mismatch | Try `coreg_check`; if RMSE over threshold, refuse **quoting the RMSE** | `validator` | `reupload` |
| V3 | SAR-only + colour/spectral question | Refuse, state what SAR *can* answer | `modality_limitation` | `add_optical` |
| V4 | Index needs absent band | **Reroute** (D3), log in `routing_notes` | — | — |
| V5 | Pan-only/RGB-only + spectral question | Reroute to SAR/texture if possible, else refuse | `modality_limitation` | `add_sar` |
| V6 | Single-pol SAR + polarimetric question | Degrade, explain, answer what single-pol supports | — | — |
| V7 | `nodata_frac` over threshold | Warn, proceed with masked statistics | — | — |
| V8 | `crs_valid` False | No georeferenced outputs; offer pixel-space answer only | `validator` | `reupload` |
| V9 | **[v2]** grounding noun outside trained vocabulary | Not a refusal — route to `object_box_fallback` (§14). Refuse with `unsupported_class` **only** if the fallback also cannot propose | `unsupported_class` | `ask_different_question` |

**Failure modes.** The node must not raise. A rule that errors becomes a warning and is treated as
passed, so an internal bug never silently blocks a legitimate query.

**SSE.** `validator {passed, refusal?, warnings}`.

**Test.** One case per V1–V9, asserting the exact frozen category and a populated `remedy`.

---

### N3b — `tiling_policy_node` **[v2 — new]**

**Purpose.** Decide which tiles each class of tool sees. Without this, a 10k×10k scene either
crashes the VLM or gets silently downsampled to mush.

See §12 for the full policy. This node's job is to write `state.tile_plan`.

**Procedure.**
1. If `bundle.tiles is None` (scene below the tiling threshold) → `tile_plan = WHOLE_SCENE`; done.
2. Otherwise, for **deterministic raster tools** (`spectral_index`, `sar_backscatter`,
   `texture_seg`, `change_map`) → `ALL_TILES`. They are cheap, run per-tile, and their outputs are
   mosaicked back to full scene (Rule 7).
3. For **learned tools** (`rs_vqa`, `rs_ground_caption`, `change_vqa`, `optsar_fusion`) → run
   `tile_scorer` and select the top *N* tiles, where `N = configs/preprocessing.yaml
   agent.learned_tool_tile_budget` (**start at 4**; it is a config value, not a constant).
4. Record the selection and its reason in `routing_notes`:
   `"tiles: 4 of 63 selected by tile_scorer (coverage 71% of valid pixels)"`.

**This is where `tile_scorer` is used.** In v1 of this document `tile_scorer` appeared in the tool
catalogue but in no task's permitted set, making it unreachable — that was an error, corrected here.

**Failure modes.** `tile_scorer` unavailable → fall back to centre-crop + the 3 highest-variance
tiles, warn, and record `tile_selection: "fallback_variance"`.

**Test.** A 63-tile bundle yields exactly 4 tiles for learned tools and 63 for deterministic ones;
a 1-tile bundle yields `WHOLE_SCENE` and never calls `tile_scorer`.

---

### N4 — `planner_node`

**Purpose.** Turn the task into an ordered list of tool calls with concrete parameters.

**Procedure.**
1. Look up the **permitted tool set** for the task (§15). Never plan outside it.
2. Filter by `BandInventory` — drop tools whose index is not in `computable_indices`, appending a
   `routing_note` per drop (this is D3).
3. Filter by modality — `manifest.required_modalities` must be satisfied.
4. Fill parameters. Defaults come from `effective_params()` — **never hardcode defaults**.
   `threshold_method` defaults to `otsu`; leave `threshold_value` unset so the tool's bimodality
   gate decides. Adapter-backed tools get the fully-qualified `tool@base-model-version`.
5. Compute `depends_on`: `change_stats` ← `change_map`; `optsar_fusion` ← `spectral_index` +
   `sar_backscatter`; `centroid_prior` ← `spectral_index`, and feeds `rs_ground_caption`;
   `object_box_fallback` ← `texture_seg`. Everything else independent.
6. **Step-count penalty (Rule 8).** Between two plans that answer the question, take the shorter.
   Do not add `lulc_classifier` "for extra signal" when a spectral index already answers it.
7. **Grounding vocabulary check.** Trained vocabulary is *land-cover regions, buildings, aircraft,
   ships*. In-vocab → `rs_ground_caption`. Out-of-vocab (vehicle, tank, bridge, harbour) →
   `object_box_fallback` with the low-confidence flag. This is the difference between partial
   credit and a zero.
8. **On replan (`replan_count == 1`) — [v2] do something different.** The rejection strings from
   N5 are in `state.gate_rejected`. Apply, in order:
   - a rejected value with a manifest `default` → drop the value, let the default apply
   - a rejected *optional* parameter → drop it entirely
   - a rejected `requires_bands` value → substitute a computable index from
     `band_inventory.computable_indices`, or drop that tool and note it
   - a rejected **required** parameter with no legal substitute → drop that tool from the plan
   - if dropping empties the plan → refuse
   A replan that produces a byte-identical plan is a bug; assert against it.

**Test.** Golden-file tests: fixed (task, inventory) → exact plan, including a no-SWIR case
asserting NDBI dropped with a routing note. Plus one replan test asserting the second plan differs.

---

### N5 — `parameter_gate_node` — **the highest-value two days in this component**

**Purpose.** Between "the planner decided" and "the tool runs". **Permitted parameters is one of
only four graded fields** — without this gate that is a claim we assert; with it, a claim a judge
can verify by reading the file.

```python
registry = ToolRegistry.default()
all_rejected: list[str] = []

for step in state["plan"]:
    manifest = registry.get(step["tool"])
    merged = effective_params(manifest, step["params"])
    defaults_applied = sorted(set(merged) - set(step["params"]))
    result = check_parameters(
        manifest, merged,
        band_inventory=state["band_inventory"],
        modalities=state["modalities"],
    )
    state["trace"].add_planned_step(
        manifest.name, merged,
        within_manifest=result.passed,
        defaults_applied=defaults_applied or None,
    )
    if not result.passed:
        all_rejected.extend(result.rejected)
    step["params"] = merged

state["trace"].set_parameter_check(passed=not all_rejected, rejected=all_rejected)
```

**Do not write your own checks.** `check_parameters()` covers all of it.

**On rejection:** first failure → back to N4 with `replan_count = 1` and the rejection list.
Second failure → NR, category `parameter_gate`, API code `PARAM_REJECTED`.
**Never clamp** (Rule 1). **Never drop an unknown key** (Rule 2).

**SSE.** `plan {steps: PermittedParameterRecord[], parameter_check: ParameterCheck}` — emit this
**after** the gate, so the UI never shows an unvalidated plan.

**Test.** Extend `test_manifest_gate.py`: `threshold_value: 47` rejected not clamped; unknown key
`thresh` rejected not dropped; NDBI on no-SWIR rejected; rejection strings appear verbatim in the
trace.

---

### N6 — `executor_node`

**Purpose.** Run the plan — parallel where independent, sequential where dependent.

**Procedure.**
1. Topologically sort by `depends_on` into waves.
2. Within a wave, run concurrently — thread pool for CPU/GPU deterministic tools (numpy/rasterio
   release the GIL), async for serving calls.
3. Per tool call: record start; dispatch via the table in `executor.py`; for **learned** tools call
   the serving client with the fully-qualified adapter name **and the assembled prompt from §13**;
   capture outputs matching the manifest's declared `outputs`; record `latency_ms`; call
   `trace.add_step(...)`.
4. **Tile loop.** Deterministic tools iterate `tile_plan.all_tiles` and the executor **mosaics
   results back to full scene in the source CRS before recording** (Rule 7). Learned tools receive
   only `tile_plan.selected_tiles`; where a learned tool returns per-tile answers, aggregate by the
   rule in §12.4.
5. **Per-tool timeout** = `manifest.expected_latency_ms × 5`. On timeout: cancel, warn, continue.
6. On exception: catch, warn, mark that tool failed, continue. Only if **every** tool fails → NR
   (`INTERNAL`).
7. Register every produced file as an `AssetRef` (§18) with `produced_by` set to the tool name.

**SSE.** `step_started {index, tool}` before each; `step_completed {…StepRecord, index}` after;
`evidence {assets}` as assets appear; `token {text}` forwarded from the serving client while a
learned tool streams its answer.

**Test.** Two independent tools finish in ~max(t1,t2), not t1+t2. A deliberately raising tool
leaves the other's results intact.

---

### N7 — `fusion_node`

**Purpose.** Reconcile evidence into one answer. Implements D1.

**Procedure.**
1. **Single-source tasks** — pass the tool answer through.
2. **Cross-modal** — decision-level fusion: two *independent* decisions, then reconcile. Not
   feature-level; naive token concatenation fails on optical–SAR heterogeneity, and decision-level
   survives residual misregistration and yields interpretable confidence.

   | Target | Optical evidence | SAR evidence |
   |---|---|---|
   | Water | MNDWI (or NDWI, or texture_seg) above gate | σ⁰ below threshold |
   | Built-up | NDBI (or texture_seg) above gate | σ⁰ high (double bounce) |

3. Compute **IoU** between the masks. High → `verdict: "consistent"`.
4. **On disagreement, classify the cause — never silently pick one** (Rule 10):

```python
DISAGREEMENT_RULES = [
  ("sar_water_optical_not",  "cloud_over_water",       "sar"),
  ("sar_dark_optical_soil",  "wet_smooth_soil",        "optical"),
  ("sar_dark_terrain_slope", "radar_shadow",           "optical"),
  ("sar_bright_over_water",  "wind_roughened_surface", "optical"),
  ("sar_dark_arid_region",   "dry_smooth_sand",        "optical"),
]
```

   Emit cause, **winning modality**, a plain-English **explanation**, and lowered confidence. The
   cause codes must match `/meta/disagreement-causes` and
   `frontend/mocks/fixtures/disagreement_causes.json`. **This is the pitch's centrepiece** and why
   it ships in Phase 2, not Phase 3.
5. **Change tasks** — `change_stats` consumes the `change_map` output: per-class area
   before/after/delta (m²), change ratio, class change ratio, largest/smallest change by class.
   Pass through the **answer formatter** (master plan §6.4) so the string matches the official
   scorer exactly.

**Trace vs event.** `.set_agreement(iou, verdict, disagreement_cause)` for the trace;
`winning_modality` and `explanation` go to the SSE event only (§6.10).

**SSE.** `agreement {iou, verdict, disagreement_cause, winning_modality, explanation}`.

**Test.** Synthetic mask pairs triggering each of the five rules; correct cause and winner;
confidence lower than the consistent case.

---

### N8 — `confidence_node`

**Purpose.** A calibrated **P(answer correct)** on the final system answer — not per-component
numbers stapled together.

**Procedure.**
1. Features: mask statistics, cross-modal IoU, `threshold_method` flag (`otsu` vs
   `fixed_fallback`), classifier scores, VLM answer log-probability (**a weak signal for free-form
   text — a feature, never alone**), `router_path`, warning count, **[v2]** tile coverage fraction
   from `tile_plan`.
2. **Phases 1–2:** conservative min/product heuristic → `confidence_basis: "heuristic"`.
3. **Phase 3+:** isotonic regression (logistic if data is thin), scikit-learn, fitted on ~500
   labelled end-to-end outputs from the 300-query routing set plus benchmark validation queries.
   One labelling effort, used twice → `confidence_basis: "calibrated"`.
4. **Floor the confidence** whenever `object_box_fallback` contributed — its manifest declares
   `low_confidence_proposer: true` and its boxes carry `"method": "deterministic_fallback"`. It
   must never masquerade as learned grounding.
5. Report ECE on a held-out ~150 and produce a reliability diagram. No system in the landscape
   survey reports calibration — free differentiation, but only if the number is honest.

**Test.** Confidence in [0,1]; fallback-path answer strictly lower than the learned path;
`confidence_basis` always one of the two frozen literals.

---

### NR — `refusal_node`

A refusal is a first-class output, not an error. Shape is frozen (`RefusalWithRemedy`), see §20.

**SSE.** The refusal rides on the `validator` or `plan` event, then `done {state: "refused"}`.

---

### N9 — `emit_node` — always runs

1. `.set_outputs(answer, masks, area_km2, confidence, extra={"confidence_basis": …})`
2. `.add_evidence()` per asset.
3. `.build()` → validates against `configs/trace_schema.json`.
4. `.write_json(out_path)`.
5. Assert elapsed against the 20 s SLA; warn if exceeded.

**Failure modes.** If `.build()` raises (routing or parameter_check unset — a programming bug),
catch, write a minimal trace recording the failure, then re-raise. Rule 11.

**SSE.** `fusion {model, answer, confidence, confidence_basis}` then
`done {query_id, state, total_latency_ms, trace_url}`.

**Test.** Every path — success, each refusal category, partial tool failure — yields a trace that
passes `validate_trace()`.

---

## 11. Routing decision table

| Images | Modalities | Question signal | → Task | → Core tools |
|---|---|---|---|---|
| 1 | optical | what/how many/is there | `single_vqa` | `rs_vqa` |
| 1 | optical | describe/caption | `single_caption` | `rs_ground_caption` |
| 1 | optical | where/locate + **in-vocab** noun | `single_grounding` | `centroid_prior` → `rs_ground_caption` |
| 1 | optical | where/locate + **out-of-vocab** noun | `single_grounding` | `texture_seg` → `object_box_fallback` |
| 1 | sar | any non-spectral | `single_vqa` | `sar_backscatter`, `rs_vqa` |
| 1 | sar | spectral/colour | — | **refuse** (V3) |
| 2 | same | how much/ratio/count + temporal | `change_map` | `change_map` → `change_stats` |
| 2 | same | describe changes | `change_description` | `change_map` → `change_vqa` |
| 2 | same | other + temporal | `change_vqa` | `change_vqa` |
| 2 | optical+sar | extract/measure target | `crossmodal_extraction` | `spectral_index` ∥ `sar_backscatter` → `optsar_fusion` |
| 2 | optical+sar | other | `crossmodal_vqa` | `optsar_fusion` |
| 1 | any | change question | — | **refuse** (V1) |

---

## 12. Tiling policy **[v2 — new]**

A Cartosat scene can be 10,000 × 10,000 px. `max_pixels` in the VLM processor is a **cap of about
512×512** — the model never upsamples, and raising the cap above a source's native size costs
nothing and buys nothing. So a large scene must be tiled, and something must decide what the model
looks at. That decision is explicit, recorded, and yours.

### 12.1 Two classes of tool

| Class | Tools | Tile behaviour |
|---|---|---|
| **Deterministic raster** | `spectral_index`, `sar_backscatter`, `texture_seg`, `change_map`, `object_box_fallback` | Run on **every** tile. Cheap, CPU-parallel. Outputs **mosaicked back to full scene, source CRS** before recording (Rule 7). |
| **Learned (VLM)** | `rs_vqa`, `rs_ground_caption`, `change_vqa`, `optsar_fusion` | Run on **a selected few** tiles, budget from config. Expensive, and the whole scene will not fit. |
| **Scene-level** | `coreg_check`, `tile_scorer`, `lulc_classifier` | Once per scene / per index, not per tile. |

### 12.2 Tile selection for learned tools

`tile_scorer` (P4 wrapper) ranks tiles. Rank by, in order: valid-pixel fraction (reject
mostly-nodata tiles outright), then question-relevant response — if a deterministic mask already
exists for the target (water, built-up, change), tiles containing mask pixels rank first;
otherwise fall back to local variance as a proxy for "something is here".

Take the top `agent.learned_tool_tile_budget` tiles, **default 4**, from
`configs/preprocessing.yaml`. Record the count, the total, and the coverage fraction in
`routing_notes`.

### 12.3 The rule that keeps this honest

**Never send more than the budget to a learned tool** (Rule 14), and **never claim scene-wide
coverage from a tiled sample.** If a learned tool saw 4 of 63 tiles, the answer text must be
qualified and `tile_coverage_frac` must feed the confidence features (§10 N8). A confident
"there are 12 aircraft" derived from 6% of the scene is exactly the kind of wrong a judge will
find.

### 12.4 Aggregating per-tile learned answers

| Answer type | Aggregation |
|---|---|
| Count | Sum across tiles, **de-duplicated in the overlap region** using `overlap_frac` |
| Presence (yes/no) | Any tile yes → yes |
| Free text | Answer from the single highest-scoring tile; do not concatenate |
| Boxes | Union, coordinates translated to scene space, NMS across tile seams |

### 12.5 When there is no tiling

If `bundle.tiles is None`, the scene is below the tiling threshold — everything runs whole-scene
and `tile_scorer` is never called. Most demo scenes will be here. Build this path first; it is
simpler and it is what you will demo.

> ⚠ `configs/preprocessing.yaml` has `tiling.max_pixels: null` with an explicit "must be measured,
> not guessed" TODO. Until that is frozen you cannot know which scenes tile. Build both paths;
> push for the number.

---

## 13. Prompt assembly — the train/serve parity contract **[v2 — new]**

**This is the highest-risk silent failure in the whole component.**

The question-generation plan (QGP v1 §2) specifies that training prompts are assembled **at
collation time from stored parts**, in this format:

```
<image>{t0}</image><image>{t1}</image>
[sensor: optical | GSD: 4.0 m] [sensor: optical | GSD: 4.0 m]
{question}
```

The QGP is explicit that the assembled string is deliberately *not* stored — so that there is one
assembler and one place to change the format.

**If inference assembles prompts differently, every adapter is fed a format it never saw in
training, and quality degrades with no error anywhere.** Nothing crashes. The loss curve looked
fine. You find out at the demo.

**The rule:**

- Write `satquery/agent/prompt.py` with a single function:

```python
def assemble_prompt(
    images: list[ImageRef],
    roles: list[str],            # ["t0","t1"] | ["opt","sar"] | ["single"]
    modalities: list[str],
    effective_gsd_m: list[float],
    question: str,
) -> str
```

- **The training data pipeline imports this same function.** Not a copy. Not a re-implementation.
  Agree this with whoever builds the QGP loaders — it is a shared dependency and it must be in
  `satquery/`, not in `training/`.
- `effective_gsd_m` is the **post-resample** GSD per the resolution policy — RSVQA-HR records
  0.30, not 0.15; OEM-SAR records 0.50, not its native 0.15–0.5. Read it from the bundle, do not
  recompute it.
- For `single_grounding` with a `centroid_prior`, the point prior is injected **inside the
  question segment**, in the format the training templates used. Confirm that format with the QGP
  owner before you write it — if grounding templates never carried a point prior in training,
  injecting one at inference is the same skew bug.
- **Test:** a golden-file test asserting `assemble_prompt()` output byte-matches a fixture taken
  from the training pipeline. When the format changes, exactly one test fails and both sides move
  together.

---

## 14. The tool catalogue — all fourteen

`L` = learned (LoRA on the base VLM, served by vLLM). `D` = deterministic (classical, CPU).

| # | Tool | Kind | Modality | Inputs | Key parameters | Outputs | Built by | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `rs_vqa` | L | optical/sar | 1 image + question | `adapter` | text + logprob | ML | adapter untrained |
| 2 | `rs_ground_caption` | L | optical | 1 image + phrase (+point prior) | `adapter` | bbox / caption | ML | untrained |
| 3 | `change_vqa` | L | same-modality pair | 2 images + question | `adapter` | text + logprob | ML | untrained |
| 4 | `optsar_fusion` | L | optical + SAR | optical + SAR stack + question | `adapter` | text + logprob | ML | untrained |
| 5 | `spectral_index` | D | optical | 1 image | `index`, `threshold_method`, `threshold_value` | `mask_uri`, `area_km2` | **you** | manifest written, code missing |
| 6 | `centroid_prior` | D | optical | a mask | — | normalised (x,y) | **you** | missing |
| 7 | `change_map` | L\* | same-modality pair | 2 images | model version | change mask GeoTIFF | ML | missing |
| 8 | `change_stats` | D | — | change map | class list | areas, ratios, deltas | **you** | missing |
| 9 | `sar_backscatter` | D | sar | 1 SAR image | `pol`, `threshold_db`, `threshold_method` | mask, `area_km2` | **you** | missing |
| 10 | `lulc_classifier` | L | optical | 1 image | model version | multi-label scores | ML | missing, **cut-list** |
| 11 | `coreg_check` | D | pair | 2 images | — | RMSE, checks | P3 owner | missing (wrapper) |
| 12 | `tile_scorer` | D | — | tile index (+ optional mask) | `budget` | ranked tiles | P4 owner | missing (wrapper) |
| 13 | `texture_seg` | D | any | 1 image | window size | texture response | **you** | missing |
| 14 | `object_box_fallback` | D | any | 1 image + noun | class prior key | ranked boxes, flagged | **you** | missing |

\* `change_map` is a fine-tuned Siamese model — Open-CD or TorchGeo **code**, initialised from
TorchGeo MIT-licensed SSL4EO-S12/SeCo weights. **Never TinyCD or ChangeFormer published
checkpoints** (non-commercial terms, and we ship models). Reversal C55/C56; do not reopen.

### 14.1 Details for the tools you own

**`spectral_index` (D2).**

```
NDVI  = (NIR − Red)   / (NIR + Red)
NDWI  = (Green − NIR) / (Green + NIR)     ← works on Cartosat MX
MNDWI = (Green − SWIR)/ (Green + SWIR)    ← needs SWIR
NDBI  = (SWIR − NIR)  / (SWIR + NIR)      ← needs SWIR
```

**Otsu behind a bimodality gate.** Otsu assumes a bimodal histogram; on a tile 95% land with one
pond it returns a confident, meaningless threshold that flows silently into D1, D2 and D4.

1. Compute the Otsu threshold and its between-class variance ratio.
2. Accept **iff** ratio ≥ 0.5 **and** both classes ≥ 5% of valid pixels.
3. Else fall back to fixed physical thresholds from `preprocessing.yaml` (initial: NDWI > 0.2
   water, NDVI > 0.3 vegetation; C-band VV σ⁰ < −18 dB water, > −3 dB dense built-up — provisional,
   tuned on holdout).
4. Record `threshold_method: "otsu" | "fixed_fallback"` **and the value**; lower confidence on the
   fallback path.

*"We detect when our own tools are out of their depth" is itself a judging story.*

**`sar_backscatter`.** Same gate + fallback, thresholds keyed by `(sar_band, polarisation)`. Water
low (specular), built-up high (double bounce). Record the threshold, method, **and the band/pol
assumption**.

**`centroid_prior`.** Connected components on the mask → largest component → centroid → normalised
`(x, y)`, injected into the prompt per §13.

**`object_box_fallback` (~2 days, assembly not new machinery).** Exists so an out-of-vocabulary
grounding query scores **partial credit instead of zero**. Entirely deterministic:

1. Percentile-stretch; on SAR apply the σ⁰ table, on optical use `texture_seg` edge-density and
   local variance.
2. Connected components over the response map, morphological opening to suppress speckle.
3. Filter by area, aspect ratio, solidity against a small prior table keyed to the query noun —
   compact bright high-backscatter blobs for *ship/tank/aircraft*, elongated linear structures for
   *bridge/runway*, dense high-texture clusters for *built-up*.
4. Emit axis-aligned boxes ranked by response strength, **capped at the top few**.

Manifest declares `low_confidence_proposer: true`; every box carries
`"method": "deterministic_fallback"`. It never masquerades as learned grounding.

---

## 15. Task → permitted tools matrix **[v2 — corrected]**

The planner may **only** select from this row.

| Task | Permitted tools |
|---|---|
| `single_vqa` | `rs_vqa`, `spectral_index`, `sar_backscatter`, `lulc_classifier`, `texture_seg`, `tile_scorer` |
| `single_caption` | `rs_ground_caption`, `lulc_classifier`, `tile_scorer` |
| `single_grounding` | `centroid_prior`, `spectral_index`, `texture_seg`, `rs_ground_caption`, `object_box_fallback`, `tile_scorer` |
| `change_description` | `change_map`, `change_vqa`, `coreg_check`, `tile_scorer` |
| `change_vqa` | `change_vqa`, `change_map`, `coreg_check`, `tile_scorer` |
| `change_map` | `change_map`, `change_stats`, `coreg_check`, `tile_scorer` |
| `crossmodal_extraction` | `spectral_index`, `sar_backscatter`, `optsar_fusion`, `coreg_check`, `texture_seg`, `tile_scorer` |
| `crossmodal_vqa` | `optsar_fusion`, `spectral_index`, `sar_backscatter`, `coreg_check`, `tile_scorer` |

`tile_scorer` is present in every row because §12 may invoke it for any task on a tiled bundle. It
is planned **only** when `bundle.tiles is not None` — otherwise the band/modality filters in N4
drop it like any other inapplicable tool.

---

## 16. Writing a tool manifest

You will write 12. Copy `configs/tools/spectral_index.yaml`:

```yaml
name: spectral_index
description: Computes a normalised spectral index over optical imagery and thresholds it into a full-scene mask.
version: 1
required_modalities: [optical]
confidence_source: threshold_statistics
expected_latency_ms: 200
permitted_parameters:
  index:
    type: enum
    values: [NDVI, NDWI, MNDWI, NDBI]
    requires_bands:
      NDVI: [nir, red]
      NDWI: [green, nir]
      MNDWI: [green, swir]
      NDBI: [swir, nir]
  threshold_method:
    type: enum
    values: [otsu, fixed]
    default: otsu
  threshold_value:
    type: float
    range: [-1.0, 1.0]
    optional: true
outputs:
  mask_uri:
    type: geotiff
    crs: source
    resolution: full_scene
  area_km2:
    type: float
```

**Rules:**
- Every accepted parameter must appear. **A parameter not in the manifest cannot be set.**
- Ranges must be physically meaningful, not defensive.
- `requires_bands` is what makes D3 rerouting automatic — the gate rejects NDBI on a no-SWIR source
  without you writing a single `if`.
- `optional: true` = works without it. `default:` = the gate fills it in. Neither = required.
- Bump `version` when permitted values change.
- Drop the file in `configs/tools/`; `ToolRegistry.default()` finds it. No registration code.
- Every manifest is validated by `tests/test_manifest_gate.py`, and served to the frontend at
  `/meta/tools` — so the UI can show "params vs manifest range". Keep `description` user-facing.

---

## 17. The trace — the graded artifact

```json
{
  "schema_version": 2,
  "query_id": "uuid", "timestamp": "ISO8601",
  "query_text": "...",

  "graded": {
    "task_selected": "crossmodal_extraction",
    "tools_invoked": ["spectral_index","sar_backscatter","optsar_fusion@qwen3vl-4b-v3"],
    "permitted_parameters": [
      {"tool":"spectral_index",
       "params":{"index":"NDWI","threshold_method":"otsu","threshold_value":0.14},
       "within_manifest":true},
      {"tool":"sar_backscatter",
       "params":{"pol":"VV","threshold_db":-18.0},
       "within_manifest":true}
    ],
    "parameter_check": {"passed": true, "rejected": []},
    "outputs": {"answer":"...","masks":["water_mask.tif","builtup_mask.tif"],
                "area_km2":3.4,"confidence":0.81}
  },

  "steps": [ … ],
  "router_path": "rules",
  "inputs": [{"file":"cartosat_mx.tif","modality":"optical","native_gsd_m":2.0,
              "pixel_size_m":2.0,"crs":"EPSG:32644","bands":["B","G","R","NIR"],
              "swir_available":false,"bit_depth":12,"nodata_frac":0.02}],
  "compatibility": {"coregistered":true,"rmse_px":0.8,
                    "checks_passed":["crs_match","extent_overlap","gsd_ratio_ok"]},
  "routing_notes": ["SWIR unavailable; NDBI skipped; SAR primary for built-up",
                    "tiles: 4 of 63 selected by tile_scorer (coverage 71% of valid pixels)"],
  "agreement": {"iou":0.86,"verdict":"consistent","disagreement_cause":null},
  "fusion": {"model":"qwen3vl-4b-instruct+optsar_fusion@v3","answer":"...","confidence":0.81},
  "evidence": ["overlay.png","water_mask.tif","builtup_mask.tif"],
  "warnings": ["NDBI unavailable: source lacks SWIR band"]
}
```

You never hand-write this. `TraceBuilder` assembles and validates it.

---

## 18. Output contract **[v2 — rewritten to the frozen shapes]**

```python
@dataclass
class QueryResult:
    query_id: str
    bundle_id: str
    question: str
    state: str                   # queued|running|succeeded|refused|failed|cancelled
    answer: str | None
    confidence: float | None
    confidence_basis: str | None # heuristic|calibrated
    latency_ms: int | None
    refusal: dict | None         # RefusalWithRemedy, §20
    evidence: list[AssetRef]     # NOT paths — see below
    warnings: list[str]
    trace: dict
```

### AssetRef — every file you produce

```python
@dataclass
class AssetRef:
    asset_id: str
    kind: str          # mask_geotiff|overlay_png|chart_png|report_pdf|scene_preview|bbox_geojson
    label: str         # human-readable, shown in the evidence panel
    produced_by: str   # tool name — REQUIRED, the UI groups by it
    media_type: str
    bytes: int
    crs: str | None                 # null ⇒ not georeferenced, use bbox_px
    bounds_wgs84: list[float] | None
    bbox_px: list[list[float]] | None   # grounding boxes, see below
    tile_url_template: str | None
    overlay_url: str | None
    download_url: str
    colour: str | None              # suggested render colour
    stats: dict | None              # {area_km2, pixel_count, class_counts, …}
```

### Bounding box contract **[v2 — was undefined in v1]**

Boxes are graded on the hidden set, so this must be unambiguous.

- **Always** emit `bbox_px` as `[[x0, y0, x1, y1], …]` in **whole-scene pixel coordinates**, origin
  top-left, x right, y down, in the **source raster's** pixel grid — not tile-local, not
  normalised. If the box came from a tile, translate it to scene space before emitting (§12.4).
- **When the scene is georeferenced** (`crs_valid`), *additionally* emit a
  `kind: "bbox_geojson"` asset with the same boxes as polygons in the source CRS, so they load in
  QGIS and overlay on the map.
- When `crs` is null, emit `bbox_px` only and set `crs: null` — the frontend contract explicitly
  handles this case.
- The learned adapter may output normalised coordinates depending on how it was trained.
  **Confirm the convention with the ML owner and convert at the tool boundary**, so nothing
  downstream ever sees two conventions.

### Who makes the overlay PNGs **[v2 — was unassigned in v1]**

**You do not.** The controller emits `mask_geotiff` assets. The API serves
`/assets/{asset_id}/overlay.png` and `/assets/{asset_id}/tiles/{z}/{x}/{y}.png`, rendering from the
GeoTIFF on demand. Your job is to set `overlay_url` and `tile_url_template` on the `AssetRef` to
those endpoints. This keeps one source of truth (the GeoTIFF) and keeps rendering out of the 20 s
query budget.

---

## 19. SSE streaming contract **[v2 — new; frozen in `types.ts`]**

The frontend renders the trace **as it is built**, not after `done`. The event order is frozen:

```
accepted → router → validator → plan → (step_started, step_completed)* →
evidence* → agreement? → token* → fusion → done
```

A refusal ends after `validator` (or `plan`), then `done` with `state: "refused"`.

| Event | Payload | Emitted by |
|---|---|---|
| `accepted` | `{query_id, queued_ms}` | N1 |
| `router` | `{router_path, task_selected, notes}` | N2 |
| `validator` | `{passed, refusal?, warnings}` | N3 |
| `plan` | `{steps: PermittedParameterRecord[], parameter_check}` | N5 (**after** the gate) |
| `step_started` | `{index, tool}` | N6 |
| `step_completed` | `{…StepRecord, index}` | N6 |
| `evidence` | `{assets: AssetRef[]}` | N6 |
| `agreement` | `{iou, verdict, disagreement_cause, winning_modality, explanation}` | N7 |
| `token` | `{text}` | N6, forwarded from serving |
| `fusion` | `{model, answer, confidence, confidence_basis}` | N9 |
| `done` | `{query_id, state, total_latency_ms, trace_url}` | N9 |
| `error` | `ApiErrorEnvelope` | any |

**Implementation.** `satquery/agent/events.py` exposes an `emit(event_name, data)` callable put in
`state["emit"]` at entry. In the web path it publishes to the SSE channel; **in headless mode it is
a no-op**, so the graph code is identical. Every event carries an `id:` so `Last-Event-ID`
resumption works, and the transport sends a `:ping` comment every 15 s (that is the API layer's
job, not yours).

**Do not** emit events from inside tool code. Only nodes emit.

---

## 20. Refusal catalogue **[v2 — aligned to the frozen enum]**

v1 of this document invented categories that do not exist in the contract. The frozen set is five:

```python
RefusalCategory = "parameter_gate" | "validator" | "modality_limitation"
                | "missing_input" | "unsupported_class"
```

Every refusal is a `RefusalWithRemedy`:

```python
{
  "reason": "…what specifically is missing or impossible…",
  "category": <one of the five above>,
  "remedy": {
    "action": "add_second_image" | "add_optical" | "add_sar"
            | "ask_different_question" | "reupload" | "none",
    "label": "Add the second acquisition",           # button text in the UI
    "suggested_questions": ["What land cover types are present?", …]  # optional
  }
}
```

`suggested_questions` is how "here is what I *can* answer" reaches the user — populate it from the
bundle's `supported_tasks` (§21). The pattern already exists in
`evalcli/smoke.py::_refusal_answer()`, which names what the tool *does* accept.

| Situation | category | remedy.action |
|---|---|---|
| Change query, one image | `missing_input` | `add_second_image` |
| Spectral question, SAR only | `modality_limitation` | `add_optical` |
| Coreg failed / no CRS / bad extent | `validator` | `reupload` |
| Parameter gate failed twice | `parameter_gate` | `ask_different_question` |
| Grounding class outside vocabulary *and* fallback found nothing | `unsupported_class` | `ask_different_question` |
| Bundle not ready | `missing_input` | `none` (API returns `BUNDLE_NOT_READY`) |

Map to API error codes: `QUERY_REFUSED` for refusals surfaced as errors, `PARAM_REJECTED`,
`BUNDLE_NOT_READY`, `MODEL_UNAVAILABLE`, `INTERNAL`.

---

## 21. Bundle probing — `supported_tasks` / `blocked_tasks` **[v2 — new]**

The frozen `Bundle` schema carries:

```ts
supported_tasks: string[]
blocked_tasks: { task: string; reason: string }[]
```

This is computed **at preparation time**, not query time, so the UI can grey out impossible
questions before the user types. It is the validator's rules run speculatively.

Write `satquery/agent/probe.py`:

```python
def probe_bundle(bundle: ImageBundle) -> tuple[list[str], list[dict]]:
    """For each of the 8 tasks, decide whether this bundle could support it."""
```

Procedure: for each `Task`, run the §10 N3 rules that depend only on the bundle (V1, V3, V5, V6,
V8 — not the question-dependent ones), and the §15 permitted-tool filter against
`BandInventory`. Supported if at least one tool survives. Blocked otherwise, with the rule's reason
string.

This is cheap, it reuses the validator, and it is what makes `suggested_questions` in a refusal
actually correct rather than generic.

---

## 22. Calling the models — the P9 interface

You do not deploy vLLM; you call it. `satquery/serving/client.py`:

```python
def infer(
    adapter: str,                # ALWAYS fully qualified: change_vqa@qwen3vl-4b-v2
    images: list[Path],
    prompt: str,                 # from assemble_prompt() — §13
    *,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> InferResult                 # (text, logprob | None, latency_ms)
```

- One base model in VRAM, adapters hot-swapped per request. This works **because** all four
  adapters are plain LoRA on the unmodified base with a uniform 3-channel `pixel_values` contract.
- `max_pixels` capped in the processor — the biggest lever on inference cost. Do not raise it to
  "improve quality"; the rule is *train on exactly what deployment sees*.
- **Fallback:** quantised local serving for the offline venue demo. Switch by config, not code
  change. Keep it working throughout.
- **Health.** Expose what `/meta/health` needs: `serving: "vllm"|"local_quantised"|"down"`, `gpu`,
  `offline_mode`, `adapters_loaded: string[]`, `trace_schema_version`.
- **Adapter contention.** Two concurrent queries needing different adapters will contend on swap.
  Serialise learned-tool calls behind a single queue per adapter and measure; if swap latency
  dominates, batch by adapter. Record swap time in `latency_ms` so it is visible.

> ⚠ **Blocking dependency.** The Phase 0 exit test — base model + two dummy-trained LoRA adapters
> hot-swapping under vLLM **on the named demo machine** — has **not been run**, and the plan says
> it can force an architecture change. Push for it before you finalise this interface.

---

## 23. Headless evaluation mode — Week 2, green in CI thereafter

```
python -m satquery.evalcli \
    --images scene_opt.tif scene_sar.tif \
    --question "..." \
    --out answer.json trace.json
```

- **Single container, single command. No frontend, no Redis, no Celery, no network.**
- Deterministic tools **CPU-only** if no GPU; learned tools via the quantised local path.
- `emit` is a no-op, so the graph code is byte-identical to the web path.
- **Batch mode:** consumes a JSONL of (images, question) rows — the shape any benchmark harness or
  hidden-set evaluation takes.
- Simultaneously (a) the artifact we submit if organisers run our code, (b) the venue-demo
  parachute, (c) the CI smoke test.

Replace the current dummy-only `__main__.py`, keeping the dummy path as a fast CI test.

> ⚠ Nobody has confirmed how the hidden set is executed. Until answered, assume the most
> restrictive case — exactly the requirements above.

---

## 24. Invocation, concurrency, jobs **[v2 — new]**

**Web path.** `POST /queries` returns `202` immediately with `{query_id, stream_url}`. The query
runs as a **Celery job** (`JobStatus.kind == "query"`), and the graph's `emit` publishes to the SSE
channel the client is already listening on. `POST /queries/{id}/cancel` sets a cancellation flag
the executor checks between waves — cancellation is cooperative, checked at wave boundaries, not
mid-tool.

**Headless path.** Direct in-process call to `run_query()`. No queue, no broker, `emit` is a no-op.

**One graph invocation = one query.** No shared state between queries. **[Decision]** There is
**no multi-turn conversation state**: the frozen `POST /queries` body accepts only `bundle_id` and
`question`, with no conversation id, and `GET /queries?bundle_id=` lists prior queries. Chat
history is a **client-side rendering of independent queries**. Do not build reference resolution
("what about the northern half?") — it is out of contract. If the team wants it later it is an API
version bump, not a controller change you can sneak in.

**Concurrency.** Deterministic tools → thread pool. Serving calls → async, serialised per adapter
(§22). Never more than one wave in flight per query.

---

## 25. Cross-cutting requirements

**Timeouts.** Per tool `expected_latency_ms × 5`; global query budget 20 s. On global timeout,
return the best partial answer with a warning rather than nothing.

**Caching.** The `ImageBundle` is cached by P1–P4. Additionally cache deterministic tool outputs
keyed by `(tool, version, params, image_hash, tile_id)` — demo reruns are common and this is nearly
free.

**Determinism.** Same bundle + same question ⇒ byte-identical trace apart from `query_id`,
`timestamp`, `latency_ms`. Seed anything stochastic. Test it by running one query twice and diffing
with those three fields masked. This is what makes the trace defensible.

**Logging vs tracing.** Different things. Logs are for you; the trace is for the judges. Never put
debug noise in the trace; never rely on logs for a graded field.

**Config, not constants.** Thresholds and the tile budget live in `preprocessing.yaml`.
`SATQUERY_CONFIG_DIR` overrides the config directory in tests — use it rather than monkey-patching.

---

## 26. Testing requirements

| Test | Asserts |
|---|---|
| `test_routing_dataset.py` (extend) | Rules / LLM / hybrid accuracy on `routing_300.jsonl`, reported separately |
| `test_validator_rules.py` | One case per V1–V9; exact frozen category; populated `remedy` |
| `test_manifest_gate.py` (extend) | 47 rejected not clamped; unknown key rejected not dropped; NDBI without SWIR rejected |
| `test_planner_golden.py` | Fixed (task, inventory) → exact plan, including D3 drops |
| `test_replan.py` | Second plan differs from the first; empty plan → refusal |
| `test_tiling_policy.py` | 63-tile bundle → 4 tiles for learned, 63 for deterministic; 1-tile → WHOLE_SCENE |
| `test_prompt_parity.py` | `assemble_prompt()` byte-matches the training fixture |
| `test_graph_paths.py` | Every path → schema-valid trace |
| `test_fusion_disagreement.py` | Each of 5 rules fires; confidence drops |
| `test_sse_sequence.py` | Emitted event order matches the frozen sequence; refusal ends after validator/plan |
| `test_asset_contract.py` | Every AssetRef has `produced_by`; boxes in scene pixel space; geojson present iff georeferenced |
| `test_determinism.py` | Two runs → identical traces modulo id/timestamp/latency |
| `test_parallel_execution.py` | Independent tools overlap in time |
| `test_evalcli_headless.py` | Runs with no GPU, no network; emits both files |
| **Fixture diff** | Your trace output matches `frontend/mocks/fixtures/trace_*.json` shapes |

All existing tests stay green. CI runs `ruff check . && pytest -q` on a CPU runner — never add a
GPU or heavyweight import to base or `dev` deps.

---

## 27. Build order

| Week | Work | Gate |
|---|---|---|
| **0** | Read §6 and `evalcli/smoke.py`. Add the `agent` extra. Write `state.py` + `events.py`. Two-node graph (ingest → emit) producing a valid trace on the dummy tool. | Trace validates end-to-end through LangGraph |
| **1** | `router.py` — **rules only**. Extend the 300-query eval. | ≥ 80% resolved by rules |
| **1** | `validator.py` + `refusals.py` (all 9 rules, frozen categories) | One test per rule; remedies populated |
| **1** | `probe.py` (§21) — cheap, reuses the validator, unblocks the frontend | `supported_tasks` on a fixture bundle |
| **2** | `planner.py` + gate wiring using existing `check_parameters()`. Manifests for `spectral_index` (done), `sar_backscatter`, `texture_seg`, `centroid_prior`. | Gate rejects 47 and unknown keys |
| **2** | `prompt.py` (§13) — agree the shared import with the QGP owner **this week** | Parity test green |
| **2** | **Headless `evalcli`** on the real graph. Green in CI from here. | CI green, no GPU |
| **3** | `executor.py` — waves, timeouts, partial failure, SSE emission. Deterministic tools you own. | Two independent tools overlap; SSE order test green |
| **3** | `tiling_policy.py` (§12). Whole-scene path first, tiled path second. | Tiling policy test green |
| **3** | LLM tie-break with constrained decoding | ≥ 90% hybrid |
| **4** | `fusion_node` + disagreement rules (D1). **Demoable centrepiece.** | 5 rules fire; matches `disagreement_causes.json` |
| **4** | `change_stats` (D4) + answer formatter | Matches official scorer strings |
| **5** | `object_box_fallback` + box contract (§18) | Out-of-vocab query scores > 0 |
| **5–6** | Confidence heuristic → 500 labelled outputs → calibrator | ECE on held-out 150 |

---

## 28. Definition of done

- [ ] All 8 tasks route correctly; hybrid ≥ 90% on `routing_300.jsonl`
- [ ] All 9 validator rules fire with the **frozen** category and a populated `remedy`
- [ ] Parameter gate rejects (never clamps) out-of-range and unknown parameters
- [ ] Replan produces a materially different plan, capped at one attempt
- [ ] All 14 tools have a schema-valid manifest in `configs/tools/`
- [ ] Tiling policy respects the learned-tool budget and records coverage
- [ ] `assemble_prompt()` is shared with training and parity-tested
- [ ] Independent tools execute in parallel; partial failure survivable
- [ ] Every mask is a full-scene geo-referenced GeoTIFF in the source CRS
- [ ] Boxes emitted in scene pixel space, plus GeoJSON when georeferenced
- [ ] Disagreement classification yields cause + winner, never a silent pick
- [ ] Confidence in [0,1] with an explicit frozen `confidence_basis`
- [ ] SSE events emitted in the frozen order on every path
- [ ] **Every path emits a schema-valid trace**, refusals and crashes included
- [ ] Trace output diffs clean against `frontend/mocks/fixtures/trace_*.json`
- [ ] Headless CLI runs with no GPU and no network
- [ ] Two identical runs → identical traces modulo id/timestamp/latency
- [ ] `ruff check . && pytest -q` green on a CPU runner

---

## 29. Questions to get answered before you go far

1. **Who writes `ImageBundle`?** Specified in master plan Part 3, **implemented nowhere** —
   verified: no `ImageBundle`, `CoregReport`, `TileIndex` or `ProvenanceStep` in `satquery/`.
   Everything you build consumes it. Day-one blocker.
2. **Has the vLLM multi-LoRA smoke test run on the named demo machine?** (§22.) It can force an
   architecture change and would change your executor's interface to learned tools.
3. **What is `tiling.max_pixels`?** Still `null` in `preprocessing.yaml` with a "must be measured,
   not guessed" TODO, despite the plan calling that file frozen. Determines which scenes tile at
   all, and your fixed-fallback thresholds live in the same file (also `null` for X- and L-band).
4. **What coordinate convention do the grounding adapters emit?** (§18.) Convert at the tool
   boundary; you need the answer from the ML owner.
5. **Does the QGP grounding template carry a point prior?** (§13.) If not, do not inject one.
6. **Is 80 GB GPU access confirmed, or 40 GB?** Affects serving, not graph shape.
7. **Which base model — Qwen3-VL-4B or Qwen3.5-2B?** Bake-off unrun; the name is in the graded
   trace.
8. **How is the hidden set executed?** (§23.) Determines how strict headless must be.
9. **Does the hidden set contain bi-temporal pairs at all?** Never asked. Changes how much
   `change_vqa` / `change_map` matter.
10. **Who owns `change_map` training?** You call it; someone must train it.

Also note: `satquery/fusion/`, `confidence/`, `serving/`, `report/`, `api/`, `sar/`, `coreg/`,
`tiling/` each contain only `__init__.py`. The repo **is** under version control (`.git` present) —
an earlier handover note said otherwise.

---

## 30. Prompts for your AI assistant

**Session opener — every time:**

> I'm building the P5 agentic controller for a satellite vision-language system, in Python with
> LangGraph. Existing code I must reuse and must NOT rewrite: [paste §6].
> Non-negotiable rules: [paste §5].
> Today I am building [node] as specified here: [paste that node's subsection from §10].
> Write only that node. Use the existing functions. Do not invent a parameter validator —
> `check_parameters()` already exists. Do not invent field names — the frontend contract in
> `frontend/contracts/types.ts` is frozen. Ask me before adding any dependency.

**When it violates a rule:**

> That violates rule N: [quote it]. The trace is the graded artifact and this component exists to
> surface errors rather than absorb them. Rewrite it so the invalid input is rejected and the
> rejection is recorded.

**When it invents a name:**

> That field/enum is not in `frontend/contracts/types.ts`, which is frozen at 1.0.0. Use the exact
> frozen name, or tell me which one you think is missing so I can raise a version bump.

**For tests:**

> Write pytest tests for [node]. Acceptance criteria: [paste the node's test line].
> Use the fixtures pattern in `tests/fixtures.py`. Must pass on a CPU runner, no GPU, no network.

---

## 31. Glossary of project shorthand

| Term | Meaning |
|---|---|
| **PS** | Problem statement — the SIH26167 brief |
| **G5, G6** | Phase gates; G6 grades the trace |
| **D1–D4** | Differentiators: D1 decision-level opt/SAR fusion, D2 spectral-index priors, D3 band-aware rerouting, D4 change statistics |
| **C13, C19, C22, C41, C45, C55…** | Numbered decisions in the master plan change log. C19 = the parameter gate; C45 = provenance traceability |
| **BEN / BEN.txt / reBEN** | BigEarthNet and variants — the main training corpus |
| **QGP** | Question Generation Plan — `docs/02-data/question-generation-plan.md` |
| **σ⁰** | Radar backscatter coefficient, dB |
| **GSD** | Ground sample distance, metres per pixel |
| **ECE** | Expected calibration error |
| **MFU** | Model FLOPs utilisation |

---

## 32. What changed in v2, and why

v1 was written from the master plan's §4.5 alone, and inherited its blind spots. v2 was written
after reading the frozen frontend contract, which had already answered several questions v1 left
open — and contradicted some of v1's invented vocabulary.

| # | v1 problem | v2 fix |
|---|---|---|
| 1 | `tile_scorer` in the tool catalogue but in **no** row of the task→tools matrix — unreachable | §15 corrected; §12 gives it a real job |
| 2 | Tiling handled in one line ("mosaic before recording") | §12 — full policy, two tool classes, budget, aggregation rules, Rule 14 |
| 3 | Prompt assembly unspecified — silent train/serve skew risk | §13 — one shared `assemble_prompt()`, imported by training, parity-tested |
| 4 | Bounding box coordinate contract undefined | §18 — scene pixel space always, GeoJSON when georeferenced, convert at the tool boundary |
| 5 | Multi-turn left open | §24 — **decided: single-shot.** The frozen API has no conversation id; chat history is client-side |
| 6 | Streaming unspecified | §19 — full frozen SSE event map, node by node; `emit` no-ops in headless |
| 7 | Overlay PNG generation unassigned | §18 — the API renders them from the GeoTIFF on demand; you set the URLs |
| 8 | Replan loop said "retry once" without saying what changes | §10 N4 step 8 — explicit correction ladder, plus a test that the plan differs |
| 9 | Invocation model unstated | §24 — Celery job in web, direct call in headless, cooperative cancellation at wave boundaries |
| 10 | **Invented refusal categories** that do not exist in the contract | §20 — the frozen five, plus the `remedy` object v1 omitted entirely |
| 11 | `QueryResult` used `list[Path]` for masks/evidence | §18 — `AssetRef` objects, matching the frozen schema |
| 12 | `supported_tasks` / `blocked_tasks` never mentioned | §21 — new `probe.py`, reuses the validator |
| 13 | `allow_llm_router` API option ignored | §10 N2 — honoured |
| 14 | Health endpoint fields unowned | §22 — serving client exposes them |

---

*Written against master plan v3.8, `frontend/contracts/` v1.0.0-frozen, and the repository state at
the time of writing. Where this document and the master plan disagree, the master plan wins — and
tell the team, because one of the two needs correcting.*
