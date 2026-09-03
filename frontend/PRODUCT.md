# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vite + React 18 + TypeScript (strict), TanStack Query v5 (server state), Zustand (UI state),
MapLibre GL JS v4 + react-map-gl (map), Tailwind + shadcn/ui (styling), native `EventSource`
(streaming), MSW v2 + `frontend/mocks/fixtures/` (mocking), Vitest + Playwright (tests),
Recharts (reliability diagram / area bars only).

Fixed by `frontend/FRONTEND_PLAN.md` §2 so mocks and CI agree. The user briefly proposed
Next.js and then withdrew it in favour of the frozen stack. Next.js is ruled out: the venue
demo ships as an offline Docker Compose bundle with self-hosted fonts and no CDN, so a
server runtime buys nothing here.

Never Mapbox GL ≥ v2 (licence). No external tile provider on the demo path.

## Users

**Primary: SIH hackathon judges evaluating SIH26167 (ISRO / SAC, Space Technology).** They
sit down for a short, high-stakes session, click through a system they have never seen, and
grade four named fields: selected task, tool/model names, permitted parameters, outputs.
They are technically literate and specifically hunting for systems that overclaim.

**Secondary: the remote-sensing analyst the product is built for.** Holds optical, SAR,
bi-temporal or cross-modal scenes; wants an answer, a mask, and a defensible record of how
the answer was produced.

**Operating scene:** a venue with **no internet and no cloud GPU**. Everything runs from a
local Docker Compose bundle. Degraded serving (`serving: "local_quantised"`, `gpu: false`)
is the expected state at the venue, not an edge case.

## Product Purpose

SatQuery AI answers natural-language questions about satellite imagery — optical, SAR,
bi-temporal, cross-modal — returning answers, captions, grounded boxes and change maps,
each accompanied by a **trace** showing which tool ran, on what parameters, with what
confidence.

Success on the frontend is narrow and gradeable: a judge touches the app and comes away
believing the system is *operated*, not assembled — because the evidence for every claim
was on screen without being asked for.

## Positioning

Four LoRA adapters over Qwen3-VL-4B (`rs_vqa`, `rs_ground_caption`, `change_vqa`,
`optsar_fusion`) sit behind a **deterministic geospatial pipeline** that validates inputs,
routes queries, enforces tool parameters, and fuses optical with SAR decisions.

Three things a neighbouring project cannot truthfully copy:

1. **The trace is the product surface, not a debug log.** The graded block is rendered
   losslessly, in the problem statement's own field order.
2. **Compatibility checking is a named deliverable.** The system refuses questions it
   cannot honestly answer, and the refusal carries a remedy that does the fix.
3. **Optical/SAR disagreement is explained, not averaged.** When the two modalities
   disagree, the UI names the cause, the winning modality, and the lowered confidence.

## Operating Context

**Two SLAs, always shown separately and measured, never hidden:**

- Scene preparation (P1 ingest → P2 SAR normalise → P3 coregister → P4 tiling): target
  ≤ 5 min per scene, has a progress bar.
- Query against an already-prepared bundle: target < 20 s.

A displayed "4 min prep + 12 s query" is a stronger claim than a concealed 20 s.

**The core loop:** upload scenes → designate roles (optical/SAR, or t1/t2) → prepare a
bundle → ask questions in chat → read evidence on the map → open the trace → export.

**API contract is frozen** at `frontend/contracts/openapi.yaml`; types mirror
`configs/trace_schema.json` (schema_version 2) field-for-field. The frontend never derives
area, IoU, or confidence itself — if a number needs computing, the backend computes it.

Backend is largely unbuilt (`sar/`, `coreg/`, `tiling/`, `fusion/`, `confidence/`,
`report/`, `serving/`, `api/` are empty). The frontend is explicitly not blocked on it:
MSW handlers plus fixtures satisfy the frozen contract, and going live is one env var.

## Capabilities and Constraints

**Seven screens:** S1 landing / new session · S2 upload & pair designation · S3 preparation
· S4 workspace (layers + map + chat/trace) · S5 trace viewer · S6 report / export ·
S7 settings & system health.

**Five refusal categories**, each needing copy and an action: `missing_input`, `validator`,
`modality_limitation`, `parameter_gate`, `unsupported_class`.

**Two confidence bases, rendered differently:** `heuristic` (Phases 1–2, interim, calibrator
not yet fitted) and `calibrated` (Phase 3+, isotonic, ECE reported). Showing an
uncalibrated number as if calibrated is the kind of thing a judge catches.

**Constraints:**
- Bundle budget < 500 KB gzipped JS excluding MapLibre; lazy-load report and reliability
  routes.
- Zero external requests on the demo path. Fonts self-hosted. No CDN, no telemetry.
- Never load a full-resolution GeoTIFF into the browser — tiles or preview PNGs only.
- `compatibility.crs_valid === false` (benchmark PNG/JPEG chips) must fall back to a
  pan/zoom canvas viewer, never a silently empty map. This path is hit often.
- A stage that does not apply arrives `skipped: true` and renders greyed, never hidden —
  it proves the pipeline considered it.
- Masks download as GeoTIFF in the *source* CRS, labelled "opens in QGIS".
- Print stylesheet on the query view is a required de-risking fallback for the PDF report.

**Undecided (open questions to backend, §13):** max upload size and whether chunked upload
is needed; whether scene display tiles exist in Phase 1 or only `preview.png`; grounding
output shape on non-georeferenced input; report generation sync vs job-based.

## Brand Commitments

Name: **SatQuery AI**. Problem statement: **SIH26167**, ISRO / SAC, Space Technology.
No logo, wordmark, palette, or typographic asset exists yet. No brand constraint has been
made binding by the user.

## Evidence on Hand

Real, in-repo, usable as design material:

- `frontend/mocks/fixtures/` — `bundle_crossmodal_ready.json`,
  `trace_crossmodal_success.json` (full schema-v2 trace, two tools, agreement block,
  disagreement cause), `trace_refusal_missing_input.json`, `trace_param_rejected.json`,
  `scene_pan_only.json`, `scene_no_crs.json`, `tools.json`, `tasks.json`,
  `disagreement_causes.json`, `health.json`.
- `frontend/contracts/openapi.yaml` and `contracts/types.ts` — the frozen surface.
- `configs/trace_schema.json` — schema v2.
- `CREDITS.md` — every source, licence, and status; judge-facing.

**Absent, and not to be fabricated:** real satellite rasters, real preview PNGs, real map
tiles, benchmark scores, user counts, deployment claims, pricing, partners, testimonials.
Any raster or thumbnail the UI needs in this phase is synthetic and must be labelled as
such, with a replacement list handed to the user.

## Product Principles

1. **Evidence beats assertion.** Every number on screen carries its provenance — which
   tool, which parameters, which manifest range, how confident, how measured.
2. **A refusal is a considered answer.** Never a red toast. Info-toned, categorised, the
   backend's reason verbatim, and a button that performs the remedy.
3. **Show the measurement, not the promise.** Both SLAs visible and measured. Degraded
   serving looks deliberate, not broken.
4. **The frontend computes nothing.** It renders what the backend asserts, losslessly,
   including keys it does not recognise.
5. **Nothing on the demo path leaves the machine.** Offline is the design constraint, not
   a hardening task deferred to the end.

## Accessibility & Inclusion

No user-specific requirement has been established. Standard obligations apply: keyboard
operability across the composer, layer controls and trace accordion; visible focus; text
contrast that survives the instrument ground; colour never the sole carrier of verdict,
agreement, or parameter-check state — evidence-layer colours come from the backend's
`AssetRef.colour` and must be paired with a label and a shape or pattern.
