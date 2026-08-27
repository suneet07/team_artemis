# SatQuery AI — Frontend Work Package (P10 + P8-UI)

**Derived from:** `../docs/01-plan/satquery-master-plan-v3.8.md` §3, §4.5.5, §4.8, §4.10, §4.11, Part 8, Part 9
**Owner:** Frontend engineer · **Effort:** ~3 weeks of build across Phases 1–4
**Status of this document:** the API contract below is **frozen for the frontend**. Backend builds to it; frontend mocks it. Changes require a version bump in `frontend/contracts/openapi.yaml` and a message to both sides on the same day.

---

## 0. Read this first (5 minutes)

You are building the only part of the system a judge actually touches. Three things matter more than polish:

1. **The trace viewer is the graded artifact.** The problem statement grades four fields: selected task, tool/model names, permitted parameters, outputs. They arrive in a top-level `graded` block. Render that block prominently and losslessly — never summarise it away.
2. **A refusal is a feature, not an error.** "Compatibility checking" is a named deliverable. When the backend refuses (change query with one image, spectral question on SAR-only input), it must look like a considered answer with a fix-it action, not a red toast.
3. **Two SLAs, shown separately.** Scene preparation (P1–P4) targets ≤ 5 min per scene and has a progress bar. Query latency targets < 20 s against an already-prepared bundle. The UI must display both numbers, measured, side by side. A measured "4 min prep + 12 s query" is a stronger claim than a hidden 20 s.

**You are not blocked on the backend.** Section 9 gives you MSW mocks and fixtures that satisfy the exact contract in Section 4. Build the whole app against them; swapping to the live API is one env var.

---

## 1. Scope

### In scope (yours)
| Area | From plan |
|---|---|
| Upload — drag/drop, multi-file, pair designation (optical/SAR, date1/date2) | §4.10 |
| Preparation progress — per-scene async P1–P4 job status | §4.10 [v3] |
| Compatibility panel — the `CompatibilityReport`, warnings visible not buried | §4.10 |
| Map viewer — MapLibre GL, geo-referenced overlays, opacity slider, layer toggle | §4.10 |
| Chat — query input, streamed response over SSE | §4.10 |
| Evidence panel — masks, overlays, area statistics, agreement verdict | §4.8, §4.10 |
| **Trace viewer** — step-by-step, per-tool params and confidence | §4.5.5, §4.10 |
| Export — PDF report + GeoTIFF mask download | §4.8 |
| Disagreement panel — optical vs SAR verdict, cause, winning modality | §4.7.2 |
| Offline demo mode — everything works with no internet, pre-cached scenes | §4.11, Phase 4 item 45 |

### Out of scope (not yours)
Model serving, tool execution, routing logic, calibration, PDF *rendering* (backend produces the file; you trigger and download it), tiling algorithm. If a number needs computing, the backend computes it — the frontend never derives area, IoU, or confidence itself.

---

## 2. Stack

Fixed, so mocks and CI agree:

| Concern | Choice | Notes |
|---|---|---|
| Build | Vite + React 18 + TypeScript (strict) | |
| Server state | TanStack Query v5 | retries, cache, polling fallback for SSE |
| UI state | Zustand | layer visibility, opacity, selected trace step |
| Map | MapLibre GL JS v4 (+ `react-map-gl` maplibre bindings) | never Mapbox GL ≥ v2 (licence) |
| Styling | Tailwind + shadcn/ui | |
| Streaming | native `EventSource` | all streams are GET SSE for this reason |
| Mocking | MSW v2 + fixtures in `frontend/mocks/fixtures/` | |
| Tests | Vitest (unit) + Playwright (one demo-flow E2E) | E2E runs against MSW, so it's green from Week 2 |
| Charts | Recharts (only for the reliability diagram / area bars) | |

Basemap: **no external tile provider on the demo path.** Default to a plain dark canvas + the scene's own raster tiles. An OSM raster basemap may be enabled behind `VITE_ENABLE_BASEMAP=1` for development only — the venue demo is assumed offline (Phase 4 item 45).

### Layout
```
frontend/
├── FRONTEND_PLAN.md            ← this file
├── contracts/
│   ├── openapi.yaml            ← frozen API contract (source of truth)
│   └── types.ts                ← generated/maintained TS types, mirrors trace_schema.json v2
├── mocks/
│   ├── handlers.ts             ← MSW handlers
│   └── fixtures/*.json         ← sample bundles, traces, refusals
└── src/
    ├── api/                    client + SSE hooks (one file per resource)
    ├── features/
    │   ├── upload/  prepare/  workspace/  chat/  evidence/  trace/  report/
    ├── components/             shared primitives
    ├── map/                    MapLibre wrappers, layer manager
    └── store/                  zustand slices
```

---

## 3. API conventions

- **Base URL:** `VITE_API_BASE`, default `http://localhost:8000/api/v1`. Backend must allow CORS from `http://localhost:5173`.
- **Auth:** none in Phase 1. Reserve an `X-Api-Key` header slot in the client so adding it later is one line.
- **IDs:** all opaque strings (UUIDv4). Never parse them.
- **Times:** ISO-8601 UTC strings. Durations always `*_ms` integers.
- **Errors:** every non-2xx returns
  ```json
  { "error": { "code": "BUNDLE_NOT_READY", "message": "Bundle is still preparing.", "hint": "Wait for job state=succeeded.", "details": {} } }
  ```
  Render `message` to the user, log `code`. Error codes are listed in §11.
- **Long operations** return `202 Accepted` with a `job_id` and a `stream_url`. Every stream has a polling twin (`GET /jobs/{job_id}`) — if `EventSource` fails twice, fall back to polling at 2 s.
- **SSE:** `text/event-stream`, named events, each with an `id:` so `Last-Event-ID` resumption works. Server sends a `:ping` comment every 15 s. Streams are **GET** so `EventSource` can be used directly.
- **Geo:** all bounds are WGS84 `[west, south, east, north]`. Masks are GeoTIFF in the *source* CRS (that's the graded output); the API also exposes web-friendly reprojected tiles for display. Never reproject in the browser.

---

## 4. Endpoint reference

Everything the frontend needs. Grouped by resource. `{}` = path param.

### 4.1 Scenes (uploaded rasters)

| # | Method | Path | Purpose |
|---|---|---|---|
| 1 | `POST` | `/scenes` | Upload one raster (multipart) |
| 2 | `GET` | `/scenes` | List scenes (`?is_demo=true` for pre-warmed) |
| 3 | `GET` | `/scenes/{scene_id}` | Scene + `CompatibilityReport` |
| 4 | `PATCH` | `/scenes/{scene_id}` | Set `role`, `acquired_at`, `declared_modality` |
| 5 | `DELETE` | `/scenes/{scene_id}` | Remove |
| 6 | `GET` | `/scenes/{scene_id}/preview.png` | Thumbnail (≤1024 px, stretched RGB or SAR dB) |
| 7 | `GET` | `/scenes/{scene_id}/tiles/{z}/{x}/{y}.png` | XYZ display tiles (EPSG:3857) |
| 8 | `GET` | `/scenes/{scene_id}/footprint.geojson` | Polygon for map fit-bounds |

**1. `POST /scenes`** — `multipart/form-data`
```
file: <binary>                 required, .tif/.tiff/.png/.jpg
declared_modality: optical|sar optional — sets modality_source="declared"
acquired_at: ISO8601           optional
role: optical|sar|t1|t2        optional, can also be set later via PATCH
```
→ `201`
```json
{ "scene_id":"sc_9f2…", "filename":"cartosat_mx.tif", "bytes":412334592,
  "status":"ingesting", "role":"optical", "acquired_at":"2024-03-11T05:20:00Z",
  "job_id":"job_a11…", "stream_url":"/api/v1/jobs/job_a11…/events" }
```
Files are large (a Cartosat scene is GBs). Use `XMLHttpRequest`/`fetch` with an upload-progress readout — show bytes and a percentage. Backend accepts up to `VITE_MAX_UPLOAD_BYTES` (ask for the real cap; assume 4 GB). If the cap forces it later, we add a chunked `POST /scenes/init` + `PUT /scenes/{id}/parts/{n}` pair — **design the upload component so the transport is swappable.**

**3. `GET /scenes/{scene_id}`** → `200 Scene` (see `contracts/types.ts`). Key part is `compatibility`:
```json
{ "scene_id":"sc_9f2…","filename":"cartosat_mx.tif","status":"ready","role":"optical",
  "is_demo":false,"acquired_at":"2024-03-11T05:20:00Z",
  "preview_url":"/api/v1/scenes/sc_9f2…/preview.png",
  "tile_url_template":"/api/v1/scenes/sc_9f2…/tiles/{z}/{x}/{y}.png",
  "footprint_url":"/api/v1/scenes/sc_9f2…/footprint.geojson",
  "bounds_wgs84":[77.05,28.44,77.31,28.68],
  "compatibility":{
    "format_ok":true,"crs_valid":true,"is_georeferenced":true,
    "modality":"optical","modality_source":"sensor_tag",
    "bands_present":["blue","green","nir","red"],
    "computable_indices":["NDVI","NDWI"],
    "nodata_frac":0.02,"bit_depth":12,"bit_depth_source":"tag:NBITS",
    "pixel_size_m":2.0,"native_gsd_m":2.0,
    "band_inventory":{"bands":{"blue":1,"green":2,"red":3,"nir":4},
      "has_swir":false,"has_nir":true,"is_pan_only":false,
      "polarisations":[],"sar_band":null,"sensor_hint":"CARTOSAT-3",
      "computable_indices":["NDVI","NDWI"]},
    "warnings":["NDBI unavailable: source lacks SWIR band"] } }
```
UI rules: `crs_valid:false` → **disable the map**, fall back to the plain image viewer (§8.4). `warnings[]` renders as an always-visible amber list on the compatibility panel, never collapsed by default. `computable_indices` drives the "what can I ask?" hints.

### 4.2 Bundles (a prepared analysis unit — P1→P4)

| # | Method | Path | Purpose |
|---|---|---|---|
| 9 | `POST` | `/bundles` | Create bundle from scenes, starts prep job |
| 10 | `GET` | `/bundles` | List (session history) |
| 11 | `GET` | `/bundles/{bundle_id}` | Bundle state, compatibility, provenance |
| 12 | `POST` | `/bundles/{bundle_id}/reprepare` | Re-run prep (after changing roles) |
| 13 | `DELETE` | `/bundles/{bundle_id}` | Remove |

**9. `POST /bundles`**
```json
{ "scenes":[{"scene_id":"sc_9f2…","role":"optical"},
             {"scene_id":"sc_7b1…","role":"sar"}],
  "pair_type":"crossmodal",
  "label":"Delhi flood — Mar 2024" }
```
`pair_type`: `single` | `crossmodal` | `bitemporal`. For `bitemporal`, roles are `t1`/`t2` and both scenes need `acquired_at`.
→ `202`
```json
{ "bundle_id":"bn_c3…","job_id":"job_d4…",
  "stream_url":"/api/v1/jobs/job_d4…/events","status":"preparing" }
```
`409 PAIR_INVALID` if roles don't match `pair_type` — validate client-side first to avoid the round trip.

**11. `GET /bundles/{bundle_id}`** → `200`
```json
{ "bundle_id":"bn_c3…","label":"Delhi flood — Mar 2024","created_at":"…",
  "pair_type":"crossmodal","status":"ready","prep_ms":238410,
  "scenes":[ /* Scene[] */ ],
  "pair_compatibility":{
    "coregistered":true,"rmse_px":0.8,"correction_applied":true,
    "method":"phase_correlation+AROSICS","common_crs":"EPSG:32644",
    "checks_passed":["crs_match","extent_overlap","gsd_ratio_ok"] },
  "tiles":{"tile_count":36,"tile_size_px":512,"overlap_frac":0.1},
  "provenance":[{"stage":"p2_sar_normalise","op":"refined_lee","params":{"window":5},"at":"…"}],
  "supported_tasks":["crossmodal_extraction","crossmodal_vqa","single_vqa","single_caption"],
  "blocked_tasks":[{"task":"change_vqa","reason":"needs two same-modality scenes"}],
  "bounds_wgs84":[77.05,28.44,77.31,28.68],
  "warnings":[] }
```
`supported_tasks` / `blocked_tasks` drive the chat's suggestion chips and let you grey out impossible questions **before** the user asks. Use them; don't reimplement routing rules in the client.

### 4.3 Jobs (prep, query, report — one progress model for all three)

| # | Method | Path | Purpose |
|---|---|---|---|
| 14 | `GET` | `/jobs/{job_id}` | Poll status (SSE fallback) |
| 15 | `GET` | `/jobs/{job_id}/events` | **SSE** progress stream |
| 16 | `POST` | `/jobs/{job_id}/cancel` | Cancel |

**15. SSE events** — prep stages are `p1_ingest`, `p2_sar_normalise`, `p3_coregister`, `p4_tiling`; a stage that doesn't apply arrives with `"skipped":true` and should render greyed, not hidden (it proves the pipeline considered it).
```
event: progress
id: 7
data: {"job_id":"job_d4…","kind":"prepare","state":"running","stage":"p2_sar_normalise",
       "stage_index":2,"stage_count":4,"percent":41,"message":"Speckle filter (Refined Lee)",
       "eta_s":92,"scene_id":"sc_7b1…"}

event: stage_complete
data: {"stage":"p1_ingest","scene_id":"sc_7b1…","skipped":false,"elapsed_ms":41220}

event: done
data: {"job_id":"job_d4…","bundle_id":"bn_c3…","state":"succeeded","elapsed_ms":238410}

event: error
data: {"job_id":"job_d4…","state":"failed",
       "error":{"code":"INGEST_UNREADABLE","message":"rasterio could not open the file.",
                "hint":"Is it a valid GeoTIFF?","details":{"scene_id":"sc_7b1…"}}}
```

### 4.4 Queries (the < 20 s path)

| # | Method | Path | Purpose |
|---|---|---|---|
| 17 | `POST` | `/queries` | Ask a question against a bundle |
| 18 | `GET` | `/queries/{query_id}/events` | **SSE** live execution stream |
| 19 | `GET` | `/queries/{query_id}` | Full result + trace |
| 20 | `GET` | `/queries/{query_id}/trace` | Raw trace JSON (schema v2) — download |
| 21 | `GET` | `/queries/{query_id}/evidence` | `AssetRef[]` for the map/evidence panel |
| 22 | `GET` | `/queries?bundle_id=…` | Chat history for a bundle |
| 23 | `POST` | `/queries/{query_id}/cancel` | Cancel a running query |

**17. `POST /queries`**
```json
{ "bundle_id":"bn_c3…", "question":"How much of the built-up area was flooded?",
  "options":{"allow_llm_router":true,"stream":true} }
```
→ `202 { "query_id":"qr_51…", "stream_url":"/api/v1/queries/qr_51…/events" }`
`409 BUNDLE_NOT_READY` if prep hasn't finished — the composer must be disabled until `bundle.status === "ready"`.

**18. SSE event sequence** (this *is* the trace being built live — render it as it arrives; do not wait for `done`):

| event | payload | UI |
|---|---|---|
| `accepted` | `{query_id, queued_ms}` | spinner in the message bubble |
| `router` | `{router_path:"rules"\|"llm", task_selected, notes:[…]}` | task badge + "rules" / "LLM tie-break" chip |
| `validator` | `{passed:bool, refusal?:Refusal, warnings:[…]}` | on `passed:false` → refusal card, stream ends |
| `plan` | `{steps:[{tool,params,within_manifest}], parameter_check:{passed,rejected:[…]}}` | trace skeleton appears, all steps pending |
| `step_started` | `{index, tool}` | step → running |
| `step_completed` | `{index, tool, outputs, confidence, latency_ms}` | step → done, params + latency shown |
| `evidence` | `{assets:[AssetRef]}` | layers appear on the map as they're produced |
| `agreement` | `{iou, verdict, disagreement_cause, winning_modality, explanation}` | **disagreement panel** (§7.3) |
| `token` | `{text}` | append to the streaming answer |
| `fusion` | `{model, answer, confidence, confidence_basis}` | final answer + confidence badge |
| `done` | `{query_id, total_latency_ms, trace_url}` | stop timer, show measured latency |
| `error` | `{error:{code,message,hint}}` | inline error in the message, retry action |

A refused query ends at `validator` (or at `plan` if `parameter_check.passed` is false) and then emits `done` with `state:"refused"`. **That is a success path.**

**19. `GET /queries/{query_id}`** → `200`
```json
{ "query_id":"qr_51…","bundle_id":"bn_c3…","question":"…","created_at":"…",
  "state":"succeeded","answer":"About 3.4 km² of built-up land is under water…",
  "confidence":0.81,"confidence_basis":"heuristic",
  "latency_ms":11840,"refusal":null,
  "evidence":[ /* AssetRef[] */ ],
  "warnings":["NDBI unavailable: source lacks SWIR band"],
  "trace":{ /* trace_schema.json v2, verbatim */ } }
```
`confidence_basis` is `"heuristic"` in Phases 1–2 and `"calibrated"` from Phase 3. **Render them differently** (§7.2) — an uncalibrated number shown as if calibrated is the kind of thing a judge catches.

**20. `GET /queries/{query_id}/trace`** → the raw trace object, `Content-Disposition: attachment` when `?download=1`. This file is the graded artifact; the download button is not optional.

### 4.5 Assets (masks, overlays, files)

| # | Method | Path | Purpose |
|---|---|---|---|
| 24 | `GET` | `/assets/{asset_id}` | Bytes (`?download=1` for attachment) |
| 25 | `GET` | `/assets/{asset_id}/meta` | `AssetRef` |
| 26 | `GET` | `/assets/{asset_id}/tiles/{z}/{x}/{y}.png` | Display tiles for a GeoTIFF mask |
| 27 | `GET` | `/assets/{asset_id}/overlay.png` | Single reprojected PNG + bounds (small scenes) |

`AssetRef`:
```json
{ "asset_id":"as_2a…","kind":"mask_geotiff","label":"Water mask (NDWI, Otsu 0.14)",
  "produced_by":"spectral_index","media_type":"image/tiff","bytes":8412300,
  "crs":"EPSG:32644","bounds_wgs84":[77.05,28.44,77.31,28.68],
  "tile_url_template":"/api/v1/assets/as_2a…/tiles/{z}/{x}/{y}.png",
  "overlay_url":"/api/v1/assets/as_2a…/overlay.png",
  "download_url":"/api/v1/assets/as_2a…?download=1",
  "colour":"#3BA3F2",
  "stats":{"area_km2":3.4,"pixel_count":851200} }
```
Masks download as GeoTIFF in the source CRS — label the download "GeoTIFF (opens in QGIS)". That sentence is worth a mark.

### 4.6 Reports

| # | Method | Path | Purpose |
|---|---|---|---|
| 28 | `POST` | `/queries/{query_id}/report` | Start PDF generation → `202 {report_id, job_id, stream_url}` |
| 29 | `GET` | `/reports/{report_id}` | Status |
| 30 | `GET` | `/reports/{report_id}/file` | The PDF |

Body: `{"include":["evidence","trace","warnings"],"format":"pdf"}`.
**Cut-list fallback (Part 8, cut #7):** if PDF generation is cut, the frontend must have a styled **print stylesheet** on the query view so `Ctrl+P` produces the same content. Build the print CSS in Phase 2 — it costs an afternoon and it de-risks the cut.

### 4.7 Metadata (drives labels — never hardcode these)

| # | Method | Path | Purpose |
|---|---|---|---|
| 31 | `GET` | `/meta/tasks` | Task enum + human labels + descriptions |
| 32 | `GET` | `/meta/tools` | All tool manifests (name, description, permitted params with ranges/enums, outputs, expected latency, confidence source) |
| 33 | `GET` | `/meta/disagreement-causes` | cause code → plain-English explanation |
| 34 | `GET` | `/meta/health` | System state |

**32** is what makes the trace viewer smart: for each step you can show the tool's description, whether each parameter sits inside its declared range, and the manifest default. Fetch once, cache for the session.
```json
{ "tools":[{"name":"spectral_index","description":"Computes a normalised spectral index…",
  "version":1,"required_modalities":["optical"],"confidence_source":"threshold_statistics",
  "expected_latency_ms":200,
  "permitted_parameters":{
    "index":{"type":"enum","values":["NDVI","NDWI","MNDWI","NDBI"],
      "requires_bands":{"NDVI":["nir","red"],"NDWI":["green","nir"]}},
    "threshold_method":{"type":"enum","values":["otsu","fixed"],"default":"otsu"},
    "threshold_value":{"type":"float","range":[-1.0,1.0],"optional":true}},
  "outputs":{"mask_uri":{"type":"geotiff","crs":"source","resolution":"full_scene"},
             "area_km2":{"type":"float"}}}] }
```

**34. `GET /meta/health`**
```json
{ "status":"ok","version":"0.4.1","trace_schema_version":2,
  "serving":"vllm","gpu":true,"offline_mode":false,
  "adapters_loaded":["rs_vqa@v2","change_vqa@v1"],
  "demo_bundles_warm":3 }
```
`serving:"local_quantised"` or `gpu:false` → show a small "degraded / offline mode" badge. At the venue this will be the real state; make it look deliberate.

### 4.8 Demo mode

| # | Method | Path | Purpose |
|---|---|---|---|
| 35 | `GET` | `/demo/bundles` | Pre-warmed bundles, ready instantly |

Landing page shows these as one-click cards. **This is the demo path** — if uploads are slow at the venue, the judge still sees the system in 3 seconds.

---

## 5. Types

`frontend/contracts/types.ts` holds the full TypeScript surface, mirroring `configs/trace_schema.json` (schema_version 2) field-for-field. Rules:

- **Never widen the trace type.** If the backend adds a field, it lands in `trace_schema.json` first with a version bump.
- The trace's `refusal` has only `{reason, category}`. The API envelope may add a `remedy` (a UI action hint) at the `QueryResult.refusal` level — that enrichment lives outside the frozen trace. Use `QueryResult.refusal.remedy` for the button.
- `graded.outputs` is deliberately open (`additionalProperties: true`) — render unknown keys generically as `key: value` rather than dropping them.

---

## 6. Screens

### S1 — Landing / New session
Demo bundle cards (`/demo/bundles`), drag-drop zone, recent bundles. Empty state explains the two-SLA model in one line: *"Scenes are prepared once (~minutes); questions then answer in seconds."*

### S2 — Upload & pair designation
Multi-file drop → a row per file with: thumbnail (once `status:"ready"`), detected modality + `modality_source` (as a tooltip: "detected from sensor tag"), a role selector (`optical`/`SAR`/`date 1`/`date 2`), acquisition date input, upload progress, and a per-file compatibility chip. A `pair_type` selector at the top drives which roles are offered. "Prepare" button → `POST /bundles`.

### S3 — Preparation
Four stage rows (P1–P4), each queued/running/skipped/done with elapsed ms, driven by the job SSE. A live compatibility panel fills in as `stage_complete` events land. Total prep timer visible. On failure: the error's `message` + `hint`, and a "retry / change roles" action.

### S4 — Workspace (the main screen)
Three panes, resizable:
- **Left — Layers & evidence.** Scene layers (optical/SAR base), then one row per `AssetRef` produced: colour swatch, label, visibility toggle, opacity slider, `area_km2`, download button. A "produced by `spectral_index`" caption links to that trace step.
- **Centre — Map.** MapLibre, fit to `bounds_wgs84`, scale bar, coordinate readout, basemap toggle. A/B swipe for bitemporal bundles (t1 vs t2) — worth the day it costs, it demos beautifully.
- **Right — Chat + Trace tabs.** Chat streams the answer; each assistant message carries a task badge, router-path chip, confidence badge, latency, and a "View trace" affordance that switches the tab with that query selected.

### S5 — Trace viewer
Top: the **`graded` block**, rendered as four labelled sections in the PS's own order — *selected task · tools invoked · permitted parameters · outputs* — with `parameter_check` as a green/red banner. Below: `router_path`, `inputs[]`, `compatibility`, `routing_notes[]`, then the `steps[]` accordion (tool, params vs manifest range, `param_source`, outputs, confidence, latency), then `agreement`, `fusion`, `evidence`, `warnings`. Buttons: **Copy JSON**, **Download trace.json**.
Every parameter renders next to its manifest constraint, e.g. `threshold_value: 0.14` `range [-1.0, 1.0]` ✓. A rejected parameter renders red with the rejection reason from `parameter_check.rejected[]`.

### S6 — Report / export
Preview of what goes in the PDF, generate button, job progress, download. Print stylesheet as the fallback.

### S7 — Settings / system
Health badge, offline-mode indicator, API base, trace schema version, adapters loaded. Small, but it makes the system look operated rather than assembled.

---

## 7. The three behaviours that carry marks

### 7.1 Refusals are answers
When `validator.passed === false` or `parameter_check.passed === false`, render a **refusal card** inside the chat: an info-blue panel with the category label ("Missing input", "Modality limitation", "Parameter rejected"), the backend's `reason` verbatim, and — where `remedy` is present — a button that does the thing (`add_second_image` → opens the upload drawer with the right role pre-selected; `ask_different_question` → drops suggested valid questions from `bundle.supported_tasks`). Never a red error toast. Never a retry that just repeats the same query.

Categories to handle: `missing_input`, `validator`, `modality_limitation`, `parameter_gate`, `unsupported_class`.

### 7.2 Confidence, honestly
`confidence_basis:"heuristic"` → grey badge, tooltip "interim heuristic — calibrator not yet fitted".
`confidence_basis:"calibrated"` → solid badge, tooltip "isotonic calibration, ECE reported on held-out set".
Never round a confidence to a percentage without showing the raw value on hover.

### 7.3 The disagreement panel (the centrepiece)
On an `agreement` event with `verdict !== "consistent"`, show a two-column card: **Optical says** vs **SAR says**, each with its mask thumbnail and statistic; below, the `disagreement_cause` translated via `/meta/disagreement-causes` (e.g. `cloud_over_water` → "Cloud cover hid the water in the optical image; SAR sees through cloud"), the `winning_modality`, the IoU, and the lowered confidence. This is the thing no competing system has. Give it real estate — it should be the screenshot on the last slide.

---

## 8. Map viewer specifics

1. **Scene base layer** — raster source from `tile_url_template`, `tileSize: 256`, `maxzoom` from the scene meta.
2. **Mask overlays** — same pattern from `AssetRef.tile_url_template`; colourise with `raster-*` paint properties, opacity from the layer slider. For small scenes, `overlay_url` + `bounds_wgs84` as an `image` source is fine and simpler — support both, prefer tiles when present.
3. **Layer order** — base scene → masks (in production order) → vector footprints → bbox annotations (`object_box_fallback` / grounding outputs arrive as `bbox_list`; draw them as a GeoJSON layer with labels).
4. **Non-georeferenced fallback** — when `compatibility.crs_valid === false` (benchmark PNG/JPEG chips do this), swap the map for a pan/zoom canvas viewer showing `preview.png` with masks composited in pixel space. Do not silently show an empty map. This path is exercised by the benchmark chips, so it will be hit often in testing.
5. **Grounding boxes** on a non-georeferenced scene arrive in pixel coordinates — the same viewer draws them; the `AssetRef` will carry `crs:null` and `bbox_px` instead of `bounds_wgs84`.

---

## 9. Working before the backend exists

This is the whole point of freezing the contract on day one.

1. `frontend/mocks/handlers.ts` implements every endpoint in §4 against the fixtures in `frontend/mocks/fixtures/`.
2. SSE is mocked with a small generator that replays a scripted event list with realistic delays (prep: ~4 min compressed to ~20 s in mock mode; query: ~12 s compressed to ~4 s). Set `VITE_MOCK_SPEED` to scale.
3. Fixtures shipped:
   - `bundle_crossmodal_ready.json` — optical + SAR, prepared, `supported_tasks` populated
   - `trace_crossmodal_success.json` — full schema-v2 trace with two tools, an agreement block and a disagreement cause
   - `trace_refusal_missing_input.json` — change query with one image
   - `trace_param_rejected.json` — `parameter_check.passed:false`, one rejected parameter
   - `scene_pan_only.json` — pan-only optical, `computable_indices: []`, spectral question must be refused
   - `scene_no_crs.json` — benchmark PNG, `crs_valid:false` → non-geo viewer path
   - `tools.json`, `tasks.json`, `disagreement_causes.json`, `health.json`
4. **Validate the fixtures in CI.** Add a Vitest test that checks every trace fixture against `configs/trace_schema.json` with Ajv. If backend changes the schema, your fixtures fail first — which is exactly the alarm you want.
5. Switch to live with `VITE_USE_MOCKS=0`.

**Integration rule:** the first day the backend has `/health` and `/meta/tools` live, point the app at it and keep both paths working for the rest of the project. Don't save integration for one big day.

---

## 10. Milestones

Aligned to the master plan's phases (Part 8). Frontend starts Phase 1.

| When | Deliverable | Definition of done |
|---|---|---|
| **Week 2, days 1–2** | Shell + contract + mocks | App boots, MSW serves every §4 endpoint, Ajv fixture test green in CI |
| **Week 2** | S2 upload + S3 preparation | Multi-file upload with progress, roles, prep SSE, compatibility panel with warnings visible |
| **Week 3** | **Frontend v1 (plan item 25)** — upload → prep → ask → answer → trace panel | Full loop works against mocks *and* whatever backend exists; trace viewer renders the `graded` block; refusal card works for `missing_input` |
| **Week 4** | Map viewer + evidence panel | MapLibre with scene tiles, mask overlays, opacity/toggle, GeoTIFF download, non-geo fallback viewer |
| **Week 4–5** | Disagreement panel (§7.3) + agreement rendering | Demoable on the crossmodal fixture — this is the Week-4 centrepiece deadline in the plan |
| **Week 6** | Print stylesheet + all five refusal categories + parameter-rejection rendering | Ctrl+P produces a report-shaped page; every refusal category has copy and an action |
| **Weeks 7–8** | PDF report flow, tiling viewer (tile grid + relevance highlight), calibration display (`calibrated` badge, reliability diagram in S7) | |
| **Weeks 9–10** | Offline demo hardening | Zero external requests with `VITE_ENABLE_BASEMAP=0`; app works against local Docker Compose with pre-cached demo bundles; rehearsed twice; Playwright demo-flow E2E green |

**Scheduling rule inherited from the plan:** the frontend never blocks on training. Every model-dependent surface must render sensibly against the zero-shot base and improve silently as adapters land.

---

## 11. Error codes & edge cases

| Code | HTTP | Meaning | UI |
|---|---|---|---|
| `UPLOAD_TOO_LARGE` | 413 | Over the cap | Pre-check client-side against `VITE_MAX_UPLOAD_BYTES` |
| `UNSUPPORTED_FORMAT` | 415 | Not a readable raster | Name the accepted formats |
| `INGEST_UNREADABLE` | 422 | rasterio failed | Show `hint`, offer re-upload |
| `NO_CRS` | 200 (warning) | Not georeferenced | Non-geo viewer, banner |
| `PAIR_INVALID` | 409 | Roles ≠ pair_type | Block the Prepare button instead |
| `COREG_FAILED` | 422 | RMSE above threshold | Show RMSE, offer "proceed anyway" only if backend allows |
| `BUNDLE_NOT_READY` | 409 | Query before prep done | Composer disabled |
| `QUERY_REFUSED` | 200 | Validator refusal | Refusal card (not an error) |
| `PARAM_REJECTED` | 200 | Parameter gate | Refusal card + red trace step |
| `MODEL_UNAVAILABLE` | 503 | vLLM down / adapter missing | Degraded badge; deterministic tools may still answer |
| `JOB_CANCELLED` | 200 | User cancelled | Neutral state |
| `INTERNAL` | 500 | Anything else | Generic message + `code` in the console |

Other cases to design for: zero evidence assets (answer is text-only — the evidence panel needs an empty state); a step that returns `confidence: null`; `nodata_frac > 0.4` (mask stats shown with a "masked statistics" caveat); a query cancelled mid-stream; an SSE reconnect that replays from `Last-Event-ID`; a 10k×10k scene where the preview takes 30 s (skeleton, not a spinner-forever).

---

## 12. Performance & offline

- The venue is assumed to have **no internet and no cloud GPU**. Everything ships in the Docker Compose bundle: fonts self-hosted, no CDN, no external basemap, no telemetry.
- Bundle budget: < 500 KB gzipped JS excluding MapLibre; lazy-load the report and reliability-diagram routes.
- Never load a full-resolution GeoTIFF into the browser. Tiles or preview PNGs only.
- Keep a visible timer for both SLAs. Judges ask "how fast is it" — the answer should already be on screen.

---

## 13. Open questions for backend (answer in week 1)

1. Max upload size and whether chunked upload is needed → decides the upload transport.
2. Are scene display tiles served by the API (endpoint 7) or should the frontend rely on `preview.png` + bounds for Phase 1? *(Recommendation: `preview.png` for Phase 1, tiles by Week 4 when full scenes arrive.)*
3. Does `bundle.supported_tasks` come from the router's rule table, or should the frontend derive hints from `computable_indices`? *(Recommendation: backend — the rules live there.)*
4. Grounding output shape on non-georeferenced input: `bbox_px` confirmed?
5. Report generation: synchronous or job-based? Spec assumes job-based; synchronous is fine for Phase 1 if the file is small.
6. CORS origin list and whether the API is served behind the same origin in the Docker Compose demo.
