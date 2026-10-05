import { http, HttpResponse, delay } from "msw";
import type {
  ApiErrorEnvelope,
  AssetRef,
  Bundle,
  CreateBundleRequest,
  CreateQueryRequest,
  DisagreementCause,
  Health,
  JobStatus,
  QueryResult,
  Report,
  Scene,
  SceneRole,
  TaskMeta,
  ToolManifest,
  Trace,
} from "@contracts/types";
import bundleCrossmodalReady from "@fixtures/bundle_crossmodal_ready.json";
import traceCrossmodalSuccess from "@fixtures/trace_crossmodal_success.json";
import traceRefusalMissingInput from "@fixtures/trace_refusal_missing_input.json";
import traceParamRejected from "@fixtures/trace_param_rejected.json";
import sceneNoCrs from "@fixtures/scene_no_crs.json";
import scenePanOnly from "@fixtures/scene_pan_only.json";
import toolsFixture from "@fixtures/tools.json";
import tasksFixture from "@fixtures/tasks.json";
import causesFixture from "@fixtures/disagreement_causes.json";
import healthFixture from "@fixtures/health.json";
import {
  renderMaskDownload,
  renderOverlay,
  renderPreview,
  renderTile,
  type RasterKind,
} from "./synthetic";
import {
  parseLastEventId,
  scriptDuration,
  sseResponse,
  type ScriptedEvent,
} from "./sse";

const BASE = "http://localhost:8000/api/v1";
const R = (path: string) => `${BASE}${path}`;

/* ── in-memory session state ─────────────────────────────────────────── */

const demoBundle = bundleCrossmodalReady as unknown as Bundle;

const scenes = new Map<string, Scene>();
const bundles = new Map<string, Bundle>();
const queries = new Map<string, QueryResult>();
const reports = new Map<string, Report>();
const jobs = new Map<string, JobStatus>();
/** job_id → the script an SSE consumer will replay. */
const jobScripts = new Map<string, ScriptedEvent[]>();
/** query_id → the script, so a reconnect replays the same execution. */
const queryScripts = new Map<string, ScriptedEvent[]>();

for (const scene of demoBundle.scenes) scenes.set(scene.scene_id, scene);
bundles.set(demoBundle.bundle_id, demoBundle);

/** A second pre-warmed bundle: the benchmark chip with no CRS (§8.4 path). */
const noCrsScene = sceneNoCrs as unknown as Scene;
const panOnlyScene = scenePanOnly as unknown as Scene;
scenes.set(noCrsScene.scene_id, noCrsScene);
scenes.set(panOnlyScene.scene_id, panOnlyScene);

const noCrsBundle: Bundle = {
  bundle_id: "bn_benchmark01",
  label: "Benchmark chip — RSVQA, no CRS",
  created_at: "2026-08-27T08:40:00Z",
  pair_type: "single",
  status: "ready",
  prep_ms: 4120,
  scenes: [noCrsScene],
  pair_compatibility: null,
  tiles: null,
  provenance: [
    {
      stage: "p1_ingest",
      op: "read+band_inventory",
      params: {},
      at: "2026-08-27T08:40:03Z",
      scene_id: noCrsScene.scene_id,
    },
  ],
  supported_tasks: ["single_vqa", "single_caption", "single_grounding"],
  blocked_tasks: [
    { task: "change_vqa", reason: "needs two scenes of the same modality" },
    { task: "change_map", reason: "needs two scenes of the same modality" },
    {
      task: "crossmodal_extraction",
      reason: "needs one optical and one SAR scene",
    },
  ],
  bounds_wgs84: null,
  warnings: [
    "Scene is not georeferenced: masks are produced in pixel space and the map is replaced by the image viewer.",
  ],
};
bundles.set(noCrsBundle.bundle_id, noCrsBundle);

const panOnlyBundle: Bundle = {
  bundle_id: "bn_panonly01",
  label: "Cartosat panchromatic — single band",
  created_at: "2026-08-27T08:52:00Z",
  pair_type: "single",
  status: "ready",
  prep_ms: 61200,
  scenes: [panOnlyScene],
  pair_compatibility: null,
  tiles: { tile_count: 16, tile_size_px: 512, overlap_frac: 0.1 },
  provenance: [
    {
      stage: "p1_ingest",
      op: "read+band_inventory",
      params: {},
      at: "2026-08-27T08:52:41Z",
      scene_id: panOnlyScene.scene_id,
    },
    {
      stage: "p4_tiling",
      op: "adaptive_partition",
      params: { tile_size_px: 512, overlap_frac: 0.1 },
      at: "2026-08-27T08:53:02Z",
    },
  ],
  supported_tasks: ["single_vqa", "single_caption", "single_grounding"],
  blocked_tasks: [
    {
      task: "crossmodal_extraction",
      reason: "needs one optical and one SAR scene",
    },
    { task: "change_map", reason: "needs two scenes of the same modality" },
  ],
  bounds_wgs84: panOnlyScene.bounds_wgs84,
  warnings: [
    "Panchromatic source: no spectral index can be computed, so index-based questions will be refused.",
  ],
};
bundles.set(panOnlyBundle.bundle_id, panOnlyBundle);

/* ── evidence assets ─────────────────────────────────────────────────── */

const RASTER_BY_ASSET: Record<string, RasterKind> = {
  as_2a11: "mask_water_optical",
  as_2a12: "mask_water_sar",
  as_2a13: "mask_builtup",
};

const assets = new Map<string, AssetRef>([
  [
    "as_2a11",
    {
      asset_id: "as_2a11",
      kind: "mask_geotiff",
      label: "Water mask — NDWI, Otsu 0.14",
      produced_by: "spectral_index",
      media_type: "image/tiff",
      bytes: 8_412_300,
      crs: "EPSG:32644",
      bounds_wgs84: [77.05, 28.44, 77.31, 28.68],
      tile_url_template: "/api/v1/assets/as_2a11/tiles/{z}/{x}/{y}.png",
      overlay_url: "/api/v1/assets/as_2a11/overlay.png",
      download_url: "/api/v1/assets/as_2a11?download=1",
      colour: "#3BA3F2",
      stats: { area_km2: 3.4, pixel_count: 851_200 },
    },
  ],
  [
    "as_2a12",
    {
      asset_id: "as_2a12",
      kind: "mask_geotiff",
      label: "Water mask — SAR VV, −18.0 dB",
      produced_by: "sar_backscatter",
      media_type: "image/tiff",
      bytes: 6_118_940,
      crs: "EPSG:32644",
      bounds_wgs84: [77.05, 28.44, 77.31, 28.68],
      tile_url_template: "/api/v1/assets/as_2a12/tiles/{z}/{x}/{y}.png",
      overlay_url: "/api/v1/assets/as_2a12/overlay.png",
      download_url: "/api/v1/assets/as_2a12?download=1",
      colour: "#EBB22A",
      stats: { area_km2: 3.9, pixel_count: 975_000 },
    },
  ],
  [
    "as_2a13",
    {
      asset_id: "as_2a13",
      kind: "mask_geotiff",
      label: "Built-up extent — SAR texture",
      produced_by: "sar_backscatter",
      media_type: "image/tiff",
      bytes: 5_204_112,
      crs: "EPSG:32644",
      bounds_wgs84: [77.05, 28.44, 77.31, 28.68],
      tile_url_template: "/api/v1/assets/as_2a13/tiles/{z}/{x}/{y}.png",
      overlay_url: "/api/v1/assets/as_2a13/overlay.png",
      download_url: "/api/v1/assets/as_2a13?download=1",
      colour: "#C55CD6",
      stats: { area_km2: 28.3, pixel_count: 7_075_000 },
    },
  ],
]);

/* ── helpers ─────────────────────────────────────────────────────────── */

let sequence = 0;
const nextId = (prefix: string) =>
  `${prefix}_${(Date.now() % 1e6).toString(36)}${(sequence += 1).toString(36)}`;

function apiError(
  status: number,
  code: ApiErrorEnvelope["error"]["code"],
  message: string,
  hint?: string,
  details?: Record<string, unknown>,
) {
  return HttpResponse.json<ApiErrorEnvelope>(
    { error: { code, message, hint, details: details ?? {} } },
    { status },
  );
}

async function png(buffer: ArrayBuffer, filename?: string) {
  return new HttpResponse(buffer, {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "public, max-age=3600",
      ...(filename
        ? { "Content-Disposition": `attachment; filename="${filename}"` }
        : {}),
    },
  });
}

function sceneRaster(scene: Scene): RasterKind {
  const compatibility = scene.compatibility;
  if (compatibility?.modality === "sar") return "sar";
  // A panchromatic source has one broadband channel; showing it in colour
  // would contradict the refusal this scene produces for index questions.
  if (compatibility?.band_inventory?.is_pan_only) return "pan";
  return "optical";
}

/* ── preparation script (P1–P4) ──────────────────────────────────────── */

function buildPrepScript(
  jobId: string,
  bundle: Bundle,
): ScriptedEvent[] {
  const hasSar = bundle.scenes.some(
    (s) => s.compatibility?.modality === "sar" || s.role === "sar",
  );
  const isPair = bundle.scenes.length > 1;
  const opticalId = bundle.scenes[0]?.scene_id;
  const sarId = bundle.scenes.find((s) => s.role === "sar")?.scene_id;

  const script: ScriptedEvent[] = [];
  const push = (after: number, event: string, data: unknown) =>
    script.push({ after, event, data });

  const progress = (
    stage: string,
    stageIndex: number,
    percent: number,
    message: string,
    etaS: number,
    sceneId?: string,
  ) =>
    ({
      job_id: jobId,
      kind: "prepare",
      state: "running",
      stage,
      stage_index: stageIndex,
      stage_count: 4,
      percent,
      message,
      eta_s: etaS,
      scene_id: sceneId,
      started_at: new Date().toISOString(),
      finished_at: null,
      elapsed_ms: null,
      error: null,
    }) satisfies Partial<JobStatus> as unknown;

  // P1 — ingest, per scene.
  push(400, "progress", progress("p1_ingest", 1, 4, "Reading raster and band inventory", 210, opticalId));
  push(2600, "progress", progress("p1_ingest", 1, 18, "Radiometric stretch bounds from percentiles", 190, opticalId));
  push(2200, "stage_complete", {
    stage: "p1_ingest",
    scene_id: opticalId,
    skipped: false,
    elapsed_ms: 41_220,
    compatibility: bundle.scenes[0]?.compatibility ?? undefined,
  });

  // P2 — SAR normalisation. Skipped, visibly, when there is no SAR scene.
  if (hasSar) {
    push(500, "progress", progress("p2_sar_normalise", 2, 26, "Calibrating to sigma-nought", 165, sarId));
    push(2400, "progress", progress("p2_sar_normalise", 2, 41, "Speckle filter (Refined Lee, window 5)", 132, sarId));
    push(2600, "stage_complete", {
      stage: "p2_sar_normalise",
      scene_id: sarId,
      skipped: false,
      elapsed_ms: 63_400,
      compatibility: bundle.scenes[1]?.compatibility ?? undefined,
    });
  } else {
    push(700, "stage_complete", {
      stage: "p2_sar_normalise",
      skipped: true,
      elapsed_ms: 0,
    });
  }

  // P3 — coregistration. Only meaningful for a pair.
  if (isPair) {
    push(500, "progress", progress("p3_coregister", 3, 55, "Phase correlation, coarse shift estimate", 96, undefined));
    push(2600, "progress", progress("p3_coregister", 3, 72, "AROSICS local shift correction", 58, undefined));
    push(2400, "stage_complete", {
      stage: "p3_coregister",
      skipped: false,
      elapsed_ms: 74_900,
    });
  } else {
    push(700, "stage_complete", {
      stage: "p3_coregister",
      skipped: true,
      elapsed_ms: 0,
    });
  }

  // P4 — tiling.
  push(500, "progress", progress("p4_tiling", 4, 86, "Adaptive partition, 512 px, 10% overlap", 24, undefined));
  push(2400, "progress", progress("p4_tiling", 4, 97, "Writing tile index", 6, undefined));
  push(1600, "stage_complete", {
    stage: "p4_tiling",
    skipped: false,
    elapsed_ms: 58_890,
  });

  push(600, "done", {
    job_id: jobId,
    bundle_id: bundle.bundle_id,
    state: "succeeded",
    elapsed_ms: bundle.prep_ms ?? 238_410,
  });

  return script;
}

/* ── query scripts ───────────────────────────────────────────────────── */

type Scenario = "crossmodal" | "missing_input" | "param_gate" | "modality";

function chooseScenario(question: string, bundle: Bundle): Scenario {
  const q = question.toLowerCase();
  const wantsChange = /\bchang|between the two|since|before and after|t1|t2\b/.test(q);
  const indices = bundle.scenes.flatMap(
    (s) => s.compatibility?.computable_indices ?? [],
  );
  const mentionsNdbi = /ndbi|built[- ]?up.*(index|ndbi)/.test(q);
  const mentionsSpectral = /ndvi|ndwi|mndwi|ndbi|spectral|vegetation index/.test(q);

  if (wantsChange && bundle.pair_type !== "bitemporal") return "missing_input";
  if (mentionsNdbi) return "param_gate";
  if (mentionsSpectral && indices.length === 0) return "modality";
  if (bundle.pair_type === "crossmodal") return "crossmodal";
  return "crossmodal";
}

const CROSSMODAL_ANSWER =
  "About 3.4 km² of built-up land is under water — roughly 12% of the built-up extent in the scene. The optical and SAR readings disagree over the western reach, where cloud hides the flood from the optical sensor; the SAR extent is trusted there and the confidence is lowered accordingly.";

/**
 * Chunk the answer the way a served model actually streams it — a few words
 * per frame rather than one. Word-at-a-time framing also puts the replay at
 * the mercy of the browser's background-timer clamp, which turns a four
 * second answer into a minute whenever the tab is not foregrounded.
 */
function tokenise(text: string, wordsPerChunk = 6): string[] {
  const words = text.match(/\S+\s*/g) ?? [text];
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += wordsPerChunk) {
    chunks.push(words.slice(i, i + wordsPerChunk).join(""));
  }
  return chunks;
}

function buildCrossmodalScript(queryId: string): ScriptedEvent[] {
  const trace = traceCrossmodalSuccess as unknown as Trace;
  const script: ScriptedEvent[] = [
    { after: 120, event: "accepted", data: { query_id: queryId, queued_ms: 38 } },
    {
      after: 260,
      event: "router",
      data: {
        router_path: "rules",
        task_selected: "crossmodal_extraction",
        notes: trace.routing_notes ?? [],
      },
    },
    {
      after: 220,
      event: "validator",
      data: { passed: true, warnings: trace.warnings ?? [] },
    },
    {
      after: 300,
      event: "plan",
      data: {
        steps: trace.graded.permitted_parameters,
        parameter_check: trace.graded.parameter_check,
      },
    },
  ];

  (trace.steps ?? []).forEach((step, index) => {
    script.push({
      after: index === 0 ? 220 : 180,
      event: "step_started",
      data: { index, tool: step.tool },
    });
    script.push({
      after: Math.min(step.latency_ms ?? 200, 1400),
      event: "step_completed",
      data: { index, ...step },
    });
    if (index === 0) {
      script.push({
        after: 120,
        event: "evidence",
        data: { assets: [assets.get("as_2a11")] },
      });
    }
    if (index === 1) {
      script.push({
        after: 120,
        event: "evidence",
        data: { assets: [assets.get("as_2a12"), assets.get("as_2a13")] },
      });
      script.push({
        after: 260,
        event: "agreement",
        data: {
          iou: 0.62,
          verdict: "disagreement",
          disagreement_cause: "cloud_over_water",
          winning_modality: "sar",
          explanation:
            "The optical NDWI mask stops at the cloud edge while the SAR mask continues beneath it. Radar is unaffected by cloud, so the SAR extent is taken as correct over the western reach.",
        },
      });
    }
  });

  for (const token of tokenise(CROSSMODAL_ANSWER)) {
    script.push({ after: 90, event: "token", data: { text: token } });
  }

  script.push({
    after: 180,
    event: "fusion",
    data: {
      model: "qwen3vl-4b-instruct+optsar_fusion@v3",
      answer: CROSSMODAL_ANSWER,
      confidence: 0.81,
      confidence_basis: "heuristic",
    },
  });
  script.push({
    after: 140,
    event: "done",
    data: {
      query_id: queryId,
      state: "succeeded",
      total_latency_ms: 11_840,
      trace_url: `/api/v1/queries/${queryId}/trace`,
    },
  });
  return script;
}

function buildRefusalScript(
  queryId: string,
  kind: "missing_input" | "param_gate" | "modality",
): ScriptedEvent[] {
  if (kind === "param_gate") {
    const trace = traceParamRejected as unknown as Trace;
    return [
      { after: 110, event: "accepted", data: { query_id: queryId, queued_ms: 31 } },
      {
        after: 240,
        event: "router",
        data: {
          router_path: "rules",
          task_selected: trace.graded.task_selected,
          notes: trace.routing_notes ?? [],
        },
      },
      {
        after: 200,
        event: "validator",
        data: { passed: true, warnings: trace.warnings ?? [] },
      },
      {
        after: 300,
        event: "plan",
        data: {
          steps: trace.graded.permitted_parameters,
          parameter_check: trace.graded.parameter_check,
        },
      },
      {
        after: 220,
        event: "validator",
        data: {
          passed: false,
          warnings: trace.warnings ?? [],
          refusal: {
            ...(trace.graded.outputs.refusal ?? {
              reason: "Parameters rejected.",
              category: "parameter_gate",
            }),
            remedy: {
              action: "add_sar",
              label: "Add a SAR scene and re-ask",
              suggested_questions: [
                "How much built-up area is visible from SAR backscatter?",
                "Where is open water in this scene?",
              ],
            },
          },
        },
      },
      {
        after: 120,
        event: "done",
        data: {
          query_id: queryId,
          state: "refused",
          total_latency_ms: 1_190,
          trace_url: `/api/v1/queries/${queryId}/trace`,
        },
      },
    ];
  }

  if (kind === "modality") {
    return [
      { after: 110, event: "accepted", data: { query_id: queryId, queued_ms: 29 } },
      {
        after: 230,
        event: "router",
        data: {
          router_path: "rules",
          task_selected: "single_vqa",
          notes: [
            "spectral_index requires computable_indices; source reports none",
          ],
        },
      },
      {
        after: 260,
        event: "validator",
        data: {
          passed: false,
          warnings: [],
          refusal: {
            reason:
              "This scene is panchromatic — it carries one broadband channel and no separable colour or infrared bands, so no spectral index can be computed from it. Vegetation and water questions that depend on an index cannot be answered honestly here.",
            category: "modality_limitation",
            remedy: {
              action: "ask_different_question",
              label: "Ask something this scene can answer",
              suggested_questions: [
                "Describe what is visible in this scene.",
                "Locate the bridges over the river.",
                "How many distinct settlements are visible?",
              ],
            },
          },
        },
      },
      {
        after: 110,
        event: "done",
        data: {
          query_id: queryId,
          state: "refused",
          total_latency_ms: 740,
          trace_url: `/api/v1/queries/${queryId}/trace`,
        },
      },
    ];
  }

  const trace = traceRefusalMissingInput as unknown as Trace;
  return [
    { after: 110, event: "accepted", data: { query_id: queryId, queued_ms: 27 } },
    {
      after: 240,
      event: "router",
      data: {
        router_path: "rules",
        task_selected: trace.graded.task_selected,
        notes: trace.routing_notes ?? [],
      },
    },
    {
      after: 280,
      event: "validator",
      data: {
        passed: false,
        warnings: [],
        refusal: {
          ...(trace.graded.outputs.refusal ?? {
            reason: "Missing input.",
            category: "missing_input",
          }),
          remedy: {
            action: "add_second_image",
            label: "Add a second date",
            suggested_questions: [
              "Describe what is visible in this scene.",
              "How much of the scene is open water?",
            ],
          },
        },
      },
    },
    {
      after: 110,
      event: "done",
      data: {
        query_id: queryId,
        state: "refused",
        total_latency_ms: 690,
        trace_url: `/api/v1/queries/${queryId}/trace`,
      },
    },
  ];
}

/** Rebuild the persisted QueryResult from the script we just replayed. */
function finaliseQuery(
  queryId: string,
  bundleId: string,
  question: string,
  scenario: Scenario,
): QueryResult {
  const base = {
    query_id: queryId,
    bundle_id: bundleId,
    question,
    created_at: new Date().toISOString(),
  };

  if (scenario === "crossmodal") {
    const trace = structuredClone(
      traceCrossmodalSuccess,
    ) as unknown as Trace;
    trace.query_id = queryId;
    trace.query_text = question;
    return {
      ...base,
      state: "succeeded",
      answer: CROSSMODAL_ANSWER,
      confidence: 0.81,
      confidence_basis: "heuristic",
      latency_ms: 11_840,
      refusal: null,
      evidence: ["as_2a11", "as_2a12", "as_2a13"].map(
        (id) => assets.get(id)!,
      ),
      warnings: trace.warnings ?? [],
      trace,
    };
  }

  const source =
    scenario === "param_gate"
      ? traceParamRejected
      : traceRefusalMissingInput;
  const trace = structuredClone(source) as unknown as Trace;
  trace.query_id = queryId;
  trace.query_text = question;

  if (scenario === "modality") {
    trace.graded.task_selected = "single_vqa";
    trace.routing_notes = [
      "spectral_index requires computable_indices; source reports none",
    ];
    trace.graded.permitted_parameters = [];
    trace.graded.outputs = {
      answer: null,
      masks: [],
      refusal: {
        reason:
          "This scene is panchromatic — it carries one broadband channel and no separable colour or infrared bands, so no spectral index can be computed from it. Vegetation and water questions that depend on an index cannot be answered honestly here.",
        category: "modality_limitation",
      },
    };
  }

  const refusal = trace.graded.outputs.refusal;
  const remedy =
    scenario === "param_gate"
      ? { action: "add_sar" as const, label: "Add a SAR scene and re-ask" }
      : scenario === "modality"
        ? {
            action: "ask_different_question" as const,
            label: "Ask something this scene can answer",
            suggested_questions: [
              "Describe what is visible in this scene.",
              "Locate the bridges over the river.",
            ],
          }
        : {
            action: "add_second_image" as const,
            label: "Add a second date",
          };

  return {
    ...base,
    state: "refused",
    answer: null,
    confidence: null,
    confidence_basis: null,
    latency_ms: scenario === "param_gate" ? 1_190 : 740,
    refusal: refusal ? { ...refusal, remedy } : null,
    evidence: [],
    warnings: trace.warnings ?? [],
    trace,
  };
}

/* ── handlers ────────────────────────────────────────────────────────── */

export const handlers = [
  /* 4.7 metadata ───────────────────────────────────────────────────── */
  http.get(R("/meta/tools"), () =>
    HttpResponse.json(toolsFixture as unknown as { tools: ToolManifest[] }),
  ),
  http.get(R("/meta/tasks"), () =>
    HttpResponse.json(tasksFixture as unknown as { tasks: TaskMeta[] }),
  ),
  http.get(R("/meta/disagreement-causes"), () =>
    HttpResponse.json(causesFixture as unknown as { causes: DisagreementCause[] }),
  ),
  http.get(R("/meta/health"), () =>
    HttpResponse.json(healthFixture as unknown as Health),
  ),

  /* 4.8 demo ───────────────────────────────────────────────────────── */
  http.get(R("/demo/bundles"), () =>
    HttpResponse.json({
      bundles: [demoBundle, panOnlyBundle, noCrsBundle],
    }),
  ),

  /* 4.1 scenes ─────────────────────────────────────────────────────── */
  http.get(R("/scenes"), ({ request }) => {
    const isDemo = new URL(request.url).searchParams.get("is_demo");
    const all = [...scenes.values()];
    return HttpResponse.json({
      scenes:
        isDemo === null
          ? all
          : all.filter((s) => s.is_demo === (isDemo === "true")),
    });
  }),

  http.post(R("/scenes"), async ({ request }) => {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return apiError(
        415,
        "UNSUPPORTED_FORMAT",
        "No raster file was attached to the upload.",
        "Accepted formats: GeoTIFF (.tif/.tiff), PNG, JPEG.",
      );
    }
    if (
      !/\.(tiff?|png|jpe?g|jfif|jp2|j2k|webp|bmp|gif|avif|hei[cf])$/i.test(
        file.name,
      ) &&
      !file.type.startsWith("image/")
    ) {
      return apiError(
        415,
        "UNSUPPORTED_FORMAT",
        `${file.name} is not a raster this system can read.`,
        "Accepted formats: any raster image — GeoTIFF, JPEG 2000, PNG, JPEG, WebP.",
      );
    }

    const declared = form.get("declared_modality");
    const role = form.get("role");
    const acquiredAt = form.get("acquired_at");
    const sceneId = nextId("sc");
    const jobId = nextId("job");

    // Modality: an explicit declaration wins; otherwise infer from the name
    // the way the backend infers from the sensor tag.
    const modality: "optical" | "sar" =
      declared === "sar" || declared === "optical"
        ? declared
        : /sar|risat|sentinel-?1|s1|vv|vh/i.test(file.name)
          ? "sar"
          : "optical";

    const template = (
      modality === "sar" ? demoBundle.scenes[1] : demoBundle.scenes[0]
    ) as Scene;

    const scene: Scene = {
      ...structuredClone(template),
      scene_id: sceneId,
      filename: file.name,
      bytes: file.size,
      uploaded_at: new Date().toISOString(),
      status: "ingesting",
      role: (role as SceneRole) ?? null,
      acquired_at: typeof acquiredAt === "string" ? acquiredAt : null,
      is_demo: false,
      preview_url: `/api/v1/scenes/${sceneId}/preview.png`,
      tile_url_template: `/api/v1/scenes/${sceneId}/tiles/{z}/{x}/{y}.png`,
      footprint_url: `/api/v1/scenes/${sceneId}/footprint.geojson`,
      compatibility: null,
      job_id: jobId,
      stream_url: `/api/v1/jobs/${jobId}/events`,
    };
    scenes.set(sceneId, scene);

    // Ingest resolves shortly after the bytes land, filling in compatibility.
    setTimeout(() => {
      const current = scenes.get(sceneId);
      if (!current) return;
      scenes.set(sceneId, {
        ...current,
        status: "ready",
        compatibility: structuredClone(template.compatibility),
      });
    }, 1400);

    await delay(120);
    return HttpResponse.json(
      {
        scene_id: sceneId,
        filename: scene.filename,
        bytes: scene.bytes,
        status: scene.status,
        role: scene.role,
        acquired_at: scene.acquired_at,
        job_id: jobId,
        stream_url: scene.stream_url,
      },
      { status: 201 },
    );
  }),

  http.get(R("/scenes/:sceneId"), ({ params }) => {
    const scene = scenes.get(String(params.sceneId));
    if (!scene) {
      return apiError(404, "INTERNAL", "No scene with that id.");
    }
    return HttpResponse.json(scene);
  }),

  http.patch(R("/scenes/:sceneId"), async ({ params, request }) => {
    const scene = scenes.get(String(params.sceneId));
    if (!scene) return apiError(404, "INTERNAL", "No scene with that id.");
    const patch = (await request.json()) as Record<string, unknown>;
    const updated: Scene = {
      ...scene,
      role: (patch.role as SceneRole) ?? scene.role,
      acquired_at:
        patch.acquired_at !== undefined
          ? (patch.acquired_at as string | null)
          : scene.acquired_at,
    };
    if (patch.declared_modality && updated.compatibility) {
      updated.compatibility = {
        ...updated.compatibility,
        modality: patch.declared_modality as "optical" | "sar",
        modality_source: "declared",
      };
    }
    scenes.set(updated.scene_id, updated);
    return HttpResponse.json(updated);
  }),

  http.delete(R("/scenes/:sceneId"), ({ params }) => {
    scenes.delete(String(params.sceneId));
    return new HttpResponse(null, { status: 204 });
  }),

  http.get(R("/scenes/:sceneId/preview.png"), async ({ params }) => {
    const scene = scenes.get(String(params.sceneId));
    if (!scene) return apiError(404, "INTERNAL", "No scene with that id.");
    return png(await renderPreview(sceneRaster(scene), 640));
  }),

  http.get(
    R("/scenes/:sceneId/tiles/:z/:x/:y.png"),
    async ({ params }) => {
      const scene = scenes.get(String(params.sceneId));
      if (!scene) return apiError(404, "INTERNAL", "No scene with that id.");
      const buffer = await renderTile(
        sceneRaster(scene),
        Number(params.z),
        Number(params.x),
        Number(params.y),
      );
      return png(buffer);
    },
  ),

  http.get(R("/scenes/:sceneId/footprint.geojson"), ({ params }) => {
    const scene = scenes.get(String(params.sceneId));
    if (!scene?.bounds_wgs84) {
      return apiError(
        404,
        "NO_CRS",
        "This scene is not georeferenced, so it has no footprint.",
      );
    }
    const [w, s, e, n] = scene.bounds_wgs84;
    return HttpResponse.json({
      type: "Feature",
      properties: { scene_id: scene.scene_id, filename: scene.filename },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [w, s],
            [e, s],
            [e, n],
            [w, n],
            [w, s],
          ],
        ],
      },
    });
  }),

  /* 4.2 bundles ────────────────────────────────────────────────────── */
  http.get(R("/bundles"), () =>
    HttpResponse.json({ bundles: [...bundles.values()] }),
  ),

  http.post(R("/bundles"), async ({ request }) => {
    const body = (await request.json()) as CreateBundleRequest;
    const chosen = body.scenes
      .map((s) => scenes.get(s.scene_id))
      .filter((s): s is Scene => Boolean(s));

    if (chosen.length !== body.scenes.length) {
      return apiError(404, "INTERNAL", "One of the scenes no longer exists.");
    }

    const roles = body.scenes.map((s) => s.role);
    const required =
      body.pair_type === "crossmodal"
        ? ["optical", "sar"]
        : body.pair_type === "bitemporal"
          ? ["t1", "t2"]
          : ["optical"];
    if (
      body.pair_type !== "single" &&
      !required.every((role) => roles.includes(role as SceneRole))
    ) {
      return apiError(
        409,
        "PAIR_INVALID",
        `A ${body.pair_type} bundle needs scenes in the ${required.join(" and ")} roles.`,
        "Change the role selectors and prepare again.",
        { required_roles: required, received_roles: roles },
      );
    }

    const bundleId = nextId("bn");
    const jobId = nextId("job");
    const bundle: Bundle = {
      bundle_id: bundleId,
      label: body.label ?? null,
      created_at: new Date().toISOString(),
      pair_type: body.pair_type,
      status: "preparing",
      prep_ms: null,
      scenes: chosen.map((scene, index) => ({
        ...scene,
        role: body.scenes[index].role,
      })),
      pair_compatibility: null,
      tiles: null,
      provenance: [],
      supported_tasks: [],
      blocked_tasks: [],
      bounds_wgs84: chosen[0]?.bounds_wgs84 ?? null,
      warnings: [],
      job_id: jobId,
      stream_url: `/api/v1/jobs/${jobId}/events`,
    };
    bundles.set(bundleId, bundle);

    const script = buildPrepScript(jobId, {
      ...bundle,
      prep_ms: 238_410,
    });
    jobScripts.set(jobId, script);
    jobs.set(jobId, {
      job_id: jobId,
      kind: "prepare",
      state: "running",
      stage: "p1_ingest",
      stage_index: 1,
      stage_count: 4,
      percent: 0,
      message: "Queued",
      eta_s: 240,
      started_at: new Date().toISOString(),
      finished_at: null,
      elapsed_ms: null,
      error: null,
    });

    // When the script finishes, the bundle becomes queryable, with the
    // routing surface (supported/blocked tasks) the router actually derives.
    const settleAfter =
      scriptDuration(script) / (Number(import.meta.env?.VITE_MOCK_SPEED ?? 1) || 1);
    setTimeout(() => {
      const isCross = body.pair_type === "crossmodal";
      const isBitemporal = body.pair_type === "bitemporal";
      bundles.set(bundleId, {
        ...bundle,
        status: "ready",
        prep_ms: 238_410,
        pair_compatibility:
          body.pair_type === "single"
            ? null
            : structuredClone(demoBundle.pair_compatibility),
        tiles: { tile_count: 36, tile_size_px: 512, overlap_frac: 0.1 },
        provenance: structuredClone(demoBundle.provenance),
        supported_tasks: isCross
          ? ["crossmodal_extraction", "crossmodal_vqa", "single_vqa", "single_caption", "single_grounding"]
          : isBitemporal
            ? ["change_vqa", "change_map", "change_description", "single_vqa", "single_caption"]
            : ["single_vqa", "single_caption", "single_grounding"],
        blocked_tasks: isCross
          ? [
              { task: "change_vqa", reason: "needs two scenes of the same modality" },
              { task: "change_map", reason: "needs two scenes of the same modality" },
            ]
          : isBitemporal
            ? [
                {
                  task: "crossmodal_extraction",
                  reason: "needs one optical and one SAR scene",
                },
              ]
            : [
                { task: "change_vqa", reason: "needs two scenes of the same modality" },
                {
                  task: "crossmodal_extraction",
                  reason: "needs one optical and one SAR scene",
                },
              ],
        warnings: chosen.flatMap((s) => s.compatibility?.warnings ?? []),
        job_id: jobId,
      });
      jobs.set(jobId, {
        ...jobs.get(jobId)!,
        state: "succeeded",
        percent: 100,
        message: "Bundle ready",
        finished_at: new Date().toISOString(),
        elapsed_ms: 238_410,
      });
    }, settleAfter + 200);

    return HttpResponse.json(
      {
        bundle_id: bundleId,
        job_id: jobId,
        stream_url: bundle.stream_url,
        status: "preparing",
      },
      { status: 202 },
    );
  }),

  http.get(R("/bundles/:bundleId"), ({ params }) => {
    const bundle = bundles.get(String(params.bundleId));
    if (!bundle) return apiError(404, "INTERNAL", "No bundle with that id.");
    return HttpResponse.json(bundle);
  }),

  http.post(R("/bundles/:bundleId/reprepare"), ({ params }) => {
    const bundle = bundles.get(String(params.bundleId));
    if (!bundle) return apiError(404, "INTERNAL", "No bundle with that id.");
    const jobId = nextId("job");
    const script = buildPrepScript(jobId, bundle);
    jobScripts.set(jobId, script);
    bundles.set(bundle.bundle_id, { ...bundle, status: "preparing", job_id: jobId });
    const settleAfter =
      scriptDuration(script) / (Number(import.meta.env?.VITE_MOCK_SPEED ?? 1) || 1);
    setTimeout(() => {
      bundles.set(bundle.bundle_id, { ...bundle, status: "ready" });
    }, settleAfter + 200);
    return HttpResponse.json(
      {
        bundle_id: bundle.bundle_id,
        job_id: jobId,
        stream_url: `/api/v1/jobs/${jobId}/events`,
        status: "preparing",
      },
      { status: 202 },
    );
  }),

  http.delete(R("/bundles/:bundleId"), ({ params }) => {
    bundles.delete(String(params.bundleId));
    return new HttpResponse(null, { status: 204 });
  }),

  /* 4.3 jobs ───────────────────────────────────────────────────────── */
  http.get(R("/jobs/:jobId"), ({ params }) => {
    const job = jobs.get(String(params.jobId));
    if (!job) return apiError(404, "INTERNAL", "No job with that id.");
    return HttpResponse.json(job);
  }),

  http.get(R("/jobs/:jobId/events"), ({ params, request }) => {
    const script = jobScripts.get(String(params.jobId));
    if (!script) {
      return apiError(404, "INTERNAL", "No job with that id.");
    }
    return sseResponse(script, { resumeFrom: parseLastEventId(request) });
  }),

  http.post(R("/jobs/:jobId/cancel"), ({ params }) => {
    const job = jobs.get(String(params.jobId));
    if (!job) return apiError(404, "INTERNAL", "No job with that id.");
    const cancelled: JobStatus = {
      ...job,
      state: "cancelled",
      message: "Cancelled",
      finished_at: new Date().toISOString(),
    };
    jobs.set(job.job_id, cancelled);
    jobScripts.delete(job.job_id);
    return HttpResponse.json(cancelled);
  }),

  /* 4.4 queries ────────────────────────────────────────────────────── */
  http.post(R("/queries"), async ({ request }) => {
    const body = (await request.json()) as CreateQueryRequest;
    const bundle = bundles.get(body.bundle_id);
    if (!bundle) return apiError(404, "INTERNAL", "No bundle with that id.");
    if (bundle.status !== "ready") {
      return apiError(
        409,
        "BUNDLE_NOT_READY",
        "This bundle is still preparing.",
        "Wait for the preparation job to reach state=succeeded.",
        { bundle_id: bundle.bundle_id, status: bundle.status },
      );
    }

    const queryId = nextId("qr");
    const scenario = chooseScenario(body.question, bundle);
    const script =
      scenario === "crossmodal"
        ? buildCrossmodalScript(queryId)
        : buildRefusalScript(queryId, scenario);
    queryScripts.set(queryId, script);
    queries.set(
      queryId,
      finaliseQuery(queryId, bundle.bundle_id, body.question, scenario),
    );

    await delay(80);
    return HttpResponse.json(
      { query_id: queryId, stream_url: `/api/v1/queries/${queryId}/events` },
      { status: 202 },
    );
  }),

  http.get(R("/queries/:queryId/events"), ({ params, request }) => {
    const script = queryScripts.get(String(params.queryId));
    if (!script) return apiError(404, "INTERNAL", "No query with that id.");
    return sseResponse(script, { resumeFrom: parseLastEventId(request) });
  }),

  http.get(R("/queries/:queryId/trace"), ({ params, request }) => {
    const query = queries.get(String(params.queryId));
    if (!query) return apiError(404, "INTERNAL", "No query with that id.");
    const download = new URL(request.url).searchParams.get("download");
    return HttpResponse.json(query.trace, {
      headers: download
        ? {
            "Content-Disposition": `attachment; filename="trace_${query.query_id}.json"`,
          }
        : {},
    });
  }),

  http.get(R("/queries/:queryId/evidence"), ({ params }) => {
    const query = queries.get(String(params.queryId));
    if (!query) return apiError(404, "INTERNAL", "No query with that id.");
    return HttpResponse.json({ assets: query.evidence });
  }),

  http.post(R("/queries/:queryId/cancel"), ({ params }) => {
    const query = queries.get(String(params.queryId));
    if (!query) return apiError(404, "INTERNAL", "No query with that id.");
    queries.set(query.query_id, { ...query, state: "cancelled" });
    queryScripts.delete(query.query_id);
    return HttpResponse.json({
      query_id: query.query_id,
      state: "cancelled",
    });
  }),

  http.get(R("/queries"), ({ request }) => {
    const bundleId = new URL(request.url).searchParams.get("bundle_id");
    return HttpResponse.json({
      queries: [...queries.values()].filter(
        (q) => !bundleId || q.bundle_id === bundleId,
      ),
    });
  }),

  http.get(R("/queries/:queryId"), ({ params }) => {
    const query = queries.get(String(params.queryId));
    if (!query) return apiError(404, "INTERNAL", "No query with that id.");
    return HttpResponse.json(query);
  }),

  /* 4.5 assets ─────────────────────────────────────────────────────── */
  http.get(R("/assets/:assetId/meta"), ({ params }) => {
    const asset = assets.get(String(params.assetId));
    if (!asset) return apiError(404, "INTERNAL", "No asset with that id.");
    return HttpResponse.json(asset);
  }),

  http.get(R("/assets/:assetId/tiles/:z/:x/:y.png"), async ({ params }) => {
    const kind = RASTER_BY_ASSET[String(params.assetId)];
    if (!kind) return apiError(404, "INTERNAL", "No asset with that id.");
    return png(
      await renderTile(kind, Number(params.z), Number(params.x), Number(params.y)),
    );
  }),

  http.get(R("/assets/:assetId/overlay.png"), async ({ params }) => {
    const kind = RASTER_BY_ASSET[String(params.assetId)];
    if (!kind) return apiError(404, "INTERNAL", "No asset with that id.");
    return png(await renderOverlay(kind));
  }),

  http.get(R("/assets/:assetId"), async ({ params, request }) => {
    const assetId = String(params.assetId);
    const kind = RASTER_BY_ASSET[assetId];
    if (!kind) return apiError(404, "INTERNAL", "No asset with that id.");
    const download = new URL(request.url).searchParams.get("download");
    return png(
      await renderMaskDownload(kind),
      download ? `${assetId}_mask_MOCK.png` : undefined,
    );
  }),

  /* 4.6 reports ────────────────────────────────────────────────────── */
  http.post(R("/queries/:queryId/report"), ({ params }) => {
    const query = queries.get(String(params.queryId));
    if (!query) return apiError(404, "INTERNAL", "No query with that id.");
    const reportId = nextId("rp");
    const jobId = nextId("job");
    const report: Report = {
      report_id: reportId,
      query_id: query.query_id,
      state: "running",
      file_url: null,
      bytes: null,
      job_id: jobId,
      stream_url: `/api/v1/jobs/${jobId}/events`,
    };
    reports.set(reportId, report);
    jobScripts.set(jobId, [
      {
        after: 500,
        event: "progress",
        data: {
          job_id: jobId,
          kind: "report",
          state: "running",
          stage: "compose",
          stage_index: 1,
          stage_count: 3,
          percent: 20,
          message: "Composing evidence figures",
          eta_s: 4,
        },
      },
      {
        after: 1200,
        event: "progress",
        data: {
          job_id: jobId,
          kind: "report",
          state: "running",
          stage: "trace",
          stage_index: 2,
          stage_count: 3,
          percent: 62,
          message: "Appending trace and warnings",
          eta_s: 2,
        },
      },
      {
        after: 1100,
        event: "progress",
        data: {
          job_id: jobId,
          kind: "report",
          state: "running",
          stage: "render",
          stage_index: 3,
          stage_count: 3,
          percent: 94,
          message: "Rendering PDF",
          eta_s: 1,
        },
      },
      {
        after: 700,
        event: "done",
        data: { job_id: jobId, state: "succeeded", elapsed_ms: 3_500 },
      },
    ]);
    setTimeout(() => {
      reports.set(reportId, {
        ...report,
        state: "succeeded",
        file_url: `/api/v1/reports/${reportId}/file`,
        bytes: 1_284_100,
      });
    }, 3_700);
    return HttpResponse.json(report, { status: 202 });
  }),

  http.get(R("/reports/:reportId"), ({ params }) => {
    const report = reports.get(String(params.reportId));
    if (!report) return apiError(404, "INTERNAL", "No report with that id.");
    return HttpResponse.json(report);
  }),

  http.get(R("/reports/:reportId/file"), ({ params }) => {
    const report = reports.get(String(params.reportId));
    if (!report || report.state !== "succeeded") {
      return apiError(
        409,
        "INTERNAL",
        "The report is not finished yet.",
        "Wait for state=succeeded.",
      );
    }
    // Mock mode has no PDF renderer; the print stylesheet is the real
    // fallback and it is wired on the query view.
    return apiError(
      503,
      "MODEL_UNAVAILABLE",
      "PDF rendering runs in the backend and is not available in mock mode.",
      "Use Print (Ctrl+P) on the query view — the print stylesheet produces the same content.",
    );
  }),
];
