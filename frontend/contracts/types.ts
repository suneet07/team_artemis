/**
 * SatQuery AI — frontend API contract types.
 * Mirrors configs/trace_schema.json (schema_version 2) and configs/tool_manifest_schema.json.
 * Source of truth for the frontend. Backend changes require a schema version bump.
 */

/* ── enums ─────────────────────────────────────────────────────────── */

export type Task =
  | "single_vqa"
  | "single_caption"
  | "single_grounding"
  | "change_description"
  | "change_vqa"
  | "change_map"
  | "crossmodal_extraction"
  | "crossmodal_vqa";

export type RouterPath = "rules" | "llm";
export type Modality = "optical" | "sar" | "unknown";
export type ModalitySource = "declared" | "sensor_tag" | "intensity_signature" | "band_count";
export type PairType = "single" | "crossmodal" | "bitemporal";
export type SceneRole = "optical" | "sar" | "t1" | "t2";
export type SarBand = "C" | "X" | "L" | "S" | "P" | null;
export type ConfidenceBasis = "heuristic" | "calibrated";
export type PrepStage = "p1_ingest" | "p2_sar_normalise" | "p3_coregister" | "p4_tiling";
export type RefusalCategory =
  | "parameter_gate"
  | "validator"
  | "modality_limitation"
  | "missing_input"
  | "unsupported_class";

/* ── errors ────────────────────────────────────────────────────────── */

export type ErrorCode =
  | "UPLOAD_TOO_LARGE" | "UNSUPPORTED_FORMAT" | "INGEST_UNREADABLE" | "NO_CRS"
  | "PAIR_INVALID" | "COREG_FAILED" | "BUNDLE_NOT_READY" | "QUERY_REFUSED"
  | "PARAM_REJECTED" | "MODEL_UNAVAILABLE" | "JOB_CANCELLED" | "INTERNAL";

export interface ApiError {
  code: ErrorCode;
  message: string;
  hint?: string;
  details?: Record<string, unknown>;
}
export interface ApiErrorEnvelope { error: ApiError }

/* ── scenes ────────────────────────────────────────────────────────── */

export interface BandInventory {
  bands: Record<string, number>;      // band name -> 1-based raster index
  has_swir: boolean;
  has_nir: boolean;
  is_pan_only: boolean;
  polarisations: string[];            // e.g. ["VV","VH"]
  sar_band: SarBand;
  sensor_hint: string | null;
  computable_indices: string[];       // e.g. ["NDVI","NDWI"]
}

export interface CompatibilityReport {
  format_ok: boolean;
  crs_valid: boolean;
  is_georeferenced: boolean;
  modality: Modality;
  modality_source: ModalitySource | null;
  bands_present: string[];
  computable_indices: string[];
  nodata_frac: number;
  bit_depth: 8 | 12 | 16 | 32;
  bit_depth_source: string | null;
  pixel_size_m: number | null;
  native_gsd_m: number | null;
  band_inventory: BandInventory;
  warnings: string[];
}

export type SceneStatus = "uploading" | "ingesting" | "ready" | "failed";

export interface Scene {
  scene_id: string;
  filename: string;
  bytes: number;
  uploaded_at: string;
  status: SceneStatus;
  role: SceneRole | null;
  acquired_at: string | null;
  is_demo: boolean;
  preview_url: string;
  tile_url_template: string | null;   // null until display tiles exist
  footprint_url: string | null;
  bounds_wgs84: [number, number, number, number] | null;
  width_px?: number;
  height_px?: number;
  compatibility: CompatibilityReport | null;
  job_id?: string;
  stream_url?: string;
  error?: ApiError;
}

/* ── bundles ───────────────────────────────────────────────────────── */

export interface PairCompatibility {
  coregistered: boolean;
  rmse_px: number | null;
  correction_applied: boolean;
  method: string | null;
  common_crs: string | null;
  checks_passed: string[];
}

export interface ProvenanceStep {
  stage: PrepStage;
  op: string;
  params: Record<string, unknown>;
  at: string;
  scene_id?: string;
}

export interface TileIndex {
  tile_count: number;
  tile_size_px: number;
  overlap_frac: number;
}

export type BundleStatus = "queued" | "preparing" | "ready" | "failed";

export interface Bundle {
  bundle_id: string;
  label: string | null;
  created_at: string;
  pair_type: PairType;
  status: BundleStatus;
  prep_ms: number | null;
  scenes: Scene[];
  pair_compatibility: PairCompatibility | null;
  tiles: TileIndex | null;
  provenance: ProvenanceStep[];
  supported_tasks: Task[];
  blocked_tasks: { task: Task; reason: string }[];
  bounds_wgs84: [number, number, number, number] | null;
  warnings: string[];
  job_id?: string;
  stream_url?: string;
}

export interface CreateBundleRequest {
  scenes: { scene_id: string; role: SceneRole }[];
  pair_type: PairType;
  label?: string;
}

/* ── jobs ──────────────────────────────────────────────────────────── */

export type JobKind = "prepare" | "query" | "report";
export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface JobStatus {
  job_id: string;
  kind: JobKind;
  state: JobState;
  stage: PrepStage | string | null;
  stage_index: number | null;
  stage_count: number | null;
  percent: number;              // 0..100
  message: string | null;
  eta_s: number | null;
  scene_id?: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_ms: number | null;
  error: ApiError | null;
}

export type PrepEvent =
  | { event: "progress"; data: JobStatus }
  | { event: "stage_complete"; data: { stage: PrepStage; scene_id?: string; skipped: boolean; elapsed_ms: number; compatibility?: CompatibilityReport } }
  | { event: "done"; data: { job_id: string; bundle_id?: string; state: "succeeded"; elapsed_ms: number } }
  | { event: "error"; data: { job_id: string; state: "failed"; error: ApiError } };

/* ── assets ────────────────────────────────────────────────────────── */

export type AssetKind =
  | "mask_geotiff" | "overlay_png" | "chart_png" | "report_pdf"
  | "scene_preview" | "bbox_geojson";

export interface AssetStats {
  area_km2?: number;
  pixel_count?: number;
  class_counts?: Record<string, number>;
  [k: string]: unknown;
}

export interface AssetRef {
  asset_id: string;
  kind: AssetKind;
  label: string;
  produced_by: string;              // tool name
  media_type: string;
  bytes: number;
  crs: string | null;               // null => not georeferenced, use bbox_px
  bounds_wgs84: [number, number, number, number] | null;
  bbox_px?: [number, number, number, number][];   // grounding boxes on non-geo input
  tile_url_template: string | null;
  overlay_url: string | null;
  download_url: string;
  colour?: string;                  // suggested render colour
  stats?: AssetStats;
}

/* ── trace (configs/trace_schema.json, schema_version 2) ───────────── */

export interface Refusal {
  reason: string;
  category: RefusalCategory;
}

/** API-level refusal = frozen trace refusal + a UI action hint. */
export interface RefusalWithRemedy extends Refusal {
  remedy?: {
    action: "add_second_image" | "add_optical" | "add_sar" | "ask_different_question" | "reupload" | "none";
    label: string;
    suggested_questions?: string[];
  };
}

export interface PermittedParameterRecord {
  tool: string;
  params: Record<string, unknown>;
  within_manifest: boolean;
  defaults_applied?: string[];
}

export interface ParameterCheck {
  passed: boolean;
  rejected: string[];
}

export interface GradedOutputs {
  answer?: string | null;
  masks?: string[];
  area_km2?: number;
  confidence?: number;
  refusal?: Refusal;
  [k: string]: unknown;             // open by schema — render unknown keys generically
}

export interface GradedBlock {
  task_selected: Task;
  tools_invoked: string[];
  permitted_parameters: PermittedParameterRecord[];
  parameter_check: ParameterCheck;
  outputs: GradedOutputs;
}

export interface InputRecord {
  file: string;
  modality: Modality;
  modality_source?: ModalitySource;
  computable_indices?: string[];
  native_gsd_m?: number;
  pixel_size_m?: number;
  crs?: string;
  bands?: string[];
  swir_available?: boolean;
  bit_depth?: 8 | 12 | 16 | 32;
  bit_depth_source?: string;
  stretch_bounds?: [number, number];
  nodata_frac?: number;
  polarisations?: string[];
  sar_band?: SarBand;
}

export interface IngestChecks {
  format_ok?: boolean;
  crs_valid?: boolean;
  georeferenced_required_for_masks?: boolean;
  modality?: Modality;
  modality_source?: ModalitySource;
  bands_present?: string[];
  computable_indices?: string[];
  nodata_frac?: number;
  bit_depth?: 8 | 12 | 16 | 32;
  bit_depth_source?: string;
  pixel_size_m?: number | null;
  native_gsd_m?: number | null;
  warnings?: string[];
}

export interface TraceCompatibility {
  coregistered?: boolean;
  rmse_px?: number | null;
  correction_applied?: boolean;
  method?: string | null;
  common_crs?: string | null;
  checks_passed?: string[];
  ingest?: IngestChecks;
}

export interface StepRecord {
  tool: string;
  params: Record<string, unknown>;
  param_source?: string;
  outputs?: Record<string, unknown>;
  confidence?: number | null;
  latency_ms?: number | null;
}

export interface Agreement {
  iou?: number | null;
  verdict?: string | null;              // "consistent" | "disagreement" | ...
  disagreement_cause?: string | null;   // key into /meta/disagreement-causes
}

export interface Fusion {
  model?: string;
  answer?: string | null;
  confidence?: number | null;
}

export interface Trace {
  schema_version: 2;
  query_id: string;
  timestamp: string;
  query_text: string;
  graded: GradedBlock;
  router_path?: RouterPath;
  inputs?: InputRecord[];
  compatibility?: TraceCompatibility;
  routing_notes?: string[];
  steps?: StepRecord[];
  agreement?: Agreement;
  fusion?: Fusion;
  evidence?: string[];
  warnings?: string[];
}

/* ── queries ───────────────────────────────────────────────────────── */

export type QueryState = "queued" | "running" | "succeeded" | "refused" | "failed" | "cancelled";

export interface CreateQueryRequest {
  bundle_id: string;
  question: string;
  options?: { allow_llm_router?: boolean; stream?: boolean };
}

export interface QueryResult {
  query_id: string;
  bundle_id: string;
  question: string;
  created_at: string;
  state: QueryState;
  answer: string | null;
  confidence: number | null;
  confidence_basis: ConfidenceBasis | null;
  latency_ms: number | null;
  refusal: RefusalWithRemedy | null;
  evidence: AssetRef[];
  warnings: string[];
  trace: Trace;
}

/* ── query SSE ─────────────────────────────────────────────────────── */

export type QueryEvent =
  | { event: "accepted";       data: { query_id: string; queued_ms: number } }
  | { event: "router";         data: { router_path: RouterPath; task_selected: Task; notes: string[] } }
  | { event: "validator";      data: { passed: boolean; refusal?: RefusalWithRemedy; warnings: string[] } }
  | { event: "plan";           data: { steps: PermittedParameterRecord[]; parameter_check: ParameterCheck } }
  | { event: "step_started";   data: { index: number; tool: string } }
  | { event: "step_completed"; data: StepRecord & { index: number } }
  | { event: "evidence";       data: { assets: AssetRef[] } }
  | { event: "agreement";      data: Agreement & { winning_modality: Modality | null; explanation: string | null } }
  | { event: "token";          data: { text: string } }
  | { event: "fusion";         data: { model: string; answer: string; confidence: number; confidence_basis: ConfidenceBasis } }
  | { event: "done";           data: { query_id: string; state: QueryState; total_latency_ms: number; trace_url: string } }
  | { event: "error";          data: ApiErrorEnvelope };

/* ── reports ───────────────────────────────────────────────────────── */

export interface CreateReportRequest {
  include?: ("evidence" | "trace" | "warnings")[];
  format?: "pdf";
}
export interface Report {
  report_id: string;
  query_id: string;
  state: JobState;
  file_url: string | null;
  bytes: number | null;
  job_id?: string;
  stream_url?: string;
}

/* ── meta ──────────────────────────────────────────────────────────── */

export interface ParamSpec {
  type: "enum" | "float" | "int" | "bool" | "string";
  values?: (string | number | boolean)[];
  range?: [number, number];
  optional?: boolean;
  default?: unknown;
  requires_bands?: Record<string, string[]>;
}

export interface OutputSpec {
  type: "geotiff" | "png_overlay" | "float" | "int" | "text" | "bbox_list" | "stats";
  crs?: "source" | "none";
  resolution?: "full_scene" | "tile";
}

export interface ToolManifest {
  name: string;
  description: string;
  version: number;
  required_modalities: ("optical" | "sar" | "any" | "pair")[];
  confidence_source?: "learned_logprob" | "threshold_statistics" | "heuristic" | "deterministic_fallback";
  low_confidence_proposer?: boolean;
  expected_latency_ms?: number;
  permitted_parameters: Record<string, ParamSpec>;
  outputs: Record<string, OutputSpec>;
}

export interface TaskMeta {
  task: Task;
  label: string;
  description: string;
  requires: PairType[];
}

export interface DisagreementCause {
  code: string;                 // "cloud_over_water"
  label: string;
  explanation: string;
  trusted_modality: Modality;
}

export interface Health {
  status: "ok" | "degraded" | "down";
  version: string;
  trace_schema_version: number;
  serving: "vllm" | "local_quantised" | "down";
  gpu: boolean;
  offline_mode: boolean;
  adapters_loaded: string[];
  demo_bundles_warm: number;
}
