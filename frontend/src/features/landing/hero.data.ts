import type { ConfidenceBasis } from "@contracts/types";

/**
 * Front matter for S1 — the masthead, the capability record, and the four
 * exemplars the console mock replays.
 *
 * Every claim here is transcribed from a source in this repository, never
 * paraphrased upward:
 *
 * - capability figures and their comparisons  → `docs/ppt/slide-2-solution.md`
 *   and `frontend/BENCHMARKS.md` — the same numbers the evaluation plots
 *   carry, so the masthead and the record can never drift apart
 * - task names, tools, routing terminals and refusal wording
 *                                             → `docs/ppt/slide-3-technical.md`
 * - the cross-modal exemplar is the mock API's own fixture,
 *   `mocks/fixtures/query_crossmodal_success.json` — question, answer,
 *   confidence, evidence colours, thresholds and warning, unaltered
 *
 * The exemplars are an ILLUSTRATION of the console and are labelled as one
 * wherever they are drawn. They are not a recorded session. Nothing here is
 * presented as a measurement except the benchmark figures, which are.
 */

/* ── capability record ───────────────────────────────────────────────── */

export interface Capability {
  /** The task id the router emits, verbatim. */
  task: string;
  name: string;
  /** What it is handed. */
  scope: string;
  /** Why a general model fails at it. */
  problem: string;
  /** What was built instead. */
  fix: string;
  /** The headline figure, as the source table carries it. */
  score: string;
  scoreOn: string;
  /** The strongest thing it is measured against. */
  against: string;
  /** The one line that turns the number into a claim. */
  mechanism: string;
  /** What it costs in parameters. */
  weight: string;
}

export const CAPABILITIES: Capability[] = [
  {
    task: "rs_vqa",
    name: "Single-image VQA",
    scope: "One optical or radar scene · a question in plain language",
    problem:
      "General models have no sense of physical scale — a 10 m pixel and a 0.5 m pixel read alike — and answer from dataset priors rather than from the image.",
    fix: "A 40.3M-parameter LoRA adapter trained on BigEarthNet, RSVQA-HR and RSVQA-LR, with ground sample distance injected into every prompt.",
    score: "85.06",
    scoreOn: "AA · RSVQA-HR",
    against: "vs 83.12 — the model the dataset's own authors published",
    mechanism: "GSD in every prompt",
    weight: "40.3M LoRA · 0.899%",
  },
  {
    task: "rs_ground_caption",
    name: "Grounding and captioning",
    scope: "One scene · a referring expression, or a request to describe it",
    problem:
      "Coordinate extraction degrades sharply out of domain. Our own detector pipeline had a 63.7% ceiling and scored 34.7% once it was built end to end.",
    fix: "Measurement showed training was unnecessary. The planned adapter was cancelled and a strictly constrained coordinate prompt applied to the base model, which serves captioning from the same weights.",
    score: "62.7%",
    scoreOn: "acc@0.5 · VRSBench",
    against: "vs 60.6% — GeoChat, fine-tuned on this benchmark",
    mechanism: "Zero training, zero added parameters",
    weight: "base model only",
  },
  {
    task: "change_vqa",
    name: "Bi-temporal change",
    scope: "Two registered acquisitions of the same ground · what changed",
    problem:
      "The strongest change-detection models are trained on 0.5 m academic-licence data, which cannot ship in an operational system.",
    fix: "A second dedicated LoRA adapter trained on CDVQA, keeping the whole path Apache-2.0. A licence-clean model beats an unusable one.",
    score: "68.0",
    scoreOn: "AA · CDVQA Val",
    against: "vs 65.9 — VisTA, the prior purpose-built SOTA",
    mechanism: "Licence-clean, inside one epoch",
    weight: "40.3M LoRA · 0.899%",
  },
  {
    task: "crossmodal_extraction",
    name: "SAR and cross-modal fusion",
    scope: "Optical and radar over the same ground · one answer from both",
    problem:
      "Optical sensors are blind under monsoon cloud, and averaging the two sensors destroys signal — the dataset authors' own fused score falls below their optical-only result.",
    fix: "The MIT-licensed BIFOLD radar classifier, reconciled by five physical rules rather than by neural averaging. Cloud over water goes to radar, because cloud is opaque to light and transparent to C-band.",
    score: "74.95%",
    scoreOn: "accuracy · reBEN SAR",
    against: "vs 50.1 — the verified majority-answer floor",
    mechanism: "No rule applies → neither sensor wins",
    weight: "resnet50-s1 · MIT",
  },
];

/** The fifth section: not a benchmark, a different kind of claim. */
export const ORCHESTRATION = {
  name: "Deterministic orchestration",
  claim: "Where a measurement can be computed, nothing is inferred.",
  body: "A rules-first router. Where a mask or a quantifiable metric exists, the agent bypasses the neural network entirely and runs dependency waves of geospatial tools — NDVI, Otsu thresholding, co-registration, set arithmetic — emitting a full audit trace of every step, every threshold, and the reason it was chosen.",
  figures: [
    {
      value: "100%",
      label: "Set arithmetic · 2,012 rows",
      note: "the same questions, answered by tools",
      muted: false,
    },
    {
      value: "43.5%",
      label: "The fine-tuned adapter · same rows",
      note: "which is why the tool answers and the model does not",
      muted: true,
    },
    {
      value: "285 / 285",
      label: "Unambiguous routing cases",
      note: "resolved by rule, each with an auditable trace",
      muted: false,
    },
  ],
} as const;

/** The line that survives every cut. */
export const BLIND_BASELINE = {
  headline: "Every score sits beside the score a blind model gets.",
  body: "RSVQA presence questions are 76.3% “yes” — a system that never opens the image scores that. We report the gap, not the number: vision contribution is 9.9–16 points, established by re-running every question against the wrong image.",
} as const;

/* ── the masthead ────────────────────────────────────────────────────── */

/** The specification strip, in the order a spec plate reads. */
export const SPEC_STRIP: { label: string; value: string }[] = [
  { label: "Backbone", value: "Qwen3-VL-4B · Apache-2.0" },
  { label: "Adapters", value: "2 × LoRA · 40.3M each" },
  { label: "Radar", value: "BIFOLD resnet50-s1 · MIT" },
  { label: "Serving", value: "one L4 GPU · PEFT hot-swap" },
  { label: "Verification", value: "440 unit · 50 route · 200 replay" },
];

/** The four numerals, and what each one beat. */
export const PROOF_RAIL: {
  value: string;
  benchmark: string;
  against: string;
}[] = [
  { value: "85.06", benchmark: "RSVQA-HR", against: "vs 83.12 authors' model" },
  {
    value: "62.7%",
    benchmark: "VRSBench grounding",
    against: "vs 60.6% fine-tuned",
  },
  { value: "68.0", benchmark: "CDVQA change", against: "vs 65.9 prior SOTA" },
  { value: "74.95%", benchmark: "reBEN SAR", against: "vs 50.1 blind floor" },
];

/* ── exemplars — the console, illustrated ────────────────────────────── */

export type MockScene = "optical" | "grounding" | "crossmodal" | "sar";

export interface Exemplar {
  id: string;
  /** Tab label — the task, as an operator would name it. */
  tab: string;
  /** The task id the router selects. */
  task: string;
  /** Which branch of the routing tree answered. */
  router: "rules" | "llm tie-break";
  question: string;
  scene: MockScene;
  /** What is in the window, stated so nobody mistakes it for observed data. */
  sceneNote: string;
  answer: string;
  /** Present only on a refusal, which is an answer and reads as one. */
  refusal: { reason: string; remedy: string } | null;
  confidence: number | null;
  confidenceBasis: ConfidenceBasis | null;
  latencyMs: number;
  evidence: { label: string; by: string; colour: string | null }[];
  trace: { tool: string; detail: string; ms: number }[];
  warning: string | null;
}

export const EXEMPLARS: Exemplar[] = [
  {
    id: "e-crossmodal",
    tab: "Cross-modal",
    task: "crossmodal_extraction",
    router: "rules",
    question: "How much of the built-up area is under water?",
    scene: "crossmodal",
    sceneNote: "Optical + SAR over one ground",
    answer:
      "About 3.4 km² of built-up land is under water — roughly 12% of the built-up extent in the scene.",
    refusal: null,
    confidence: 0.81,
    confidenceBasis: "heuristic",
    latencyMs: 11840,
    evidence: [
      {
        label: "Water mask · NDWI, Otsu 0.14",
        by: "spectral_index",
        colour: "#3BA3F2",
      },
      {
        label: "Water from SAR · VV < −18 dB",
        by: "sar_backscatter",
        colour: "#F2A03B",
      },
      { label: "Fused overlay", by: "optsar_fusion", colour: null },
    ],
    trace: [
      {
        tool: "coreg_check",
        detail: "phase_correlation + AROSICS · RMSE 0.8 px",
        ms: 240,
      },
      {
        tool: "spectral_index",
        detail: "NDWI · otsu 0.14 · bimodality passed",
        ms: 180,
      },
      { tool: "sar_backscatter", detail: "VV · −18.0 dB · 3.9 km²", ms: 210 },
      {
        tool: "optsar_fusion",
        detail: "IoU 0.71 ≥ 0.60 → union · confidence raised",
        ms: 460,
      },
    ],
    warning: "NDBI unavailable: source lacks SWIR band",
  },
  {
    id: "e-vqa",
    tab: "Single-image",
    task: "single_vqa",
    router: "llm tie-break",
    question: "Is there a residential area next to the river?",
    scene: "optical",
    sceneNote: "One optical scene · GSD 2.0 m",
    answer:
      "Yes. A dense residential block sits on the north bank, directly adjacent to the channel.",
    refusal: null,
    confidence: 0.86,
    confidenceBasis: "heuristic",
    latencyMs: 3120,
    evidence: [
      {
        label: "No mask — the answer is the adapter's",
        by: "rs_vqa",
        colour: null,
      },
    ],
    trace: [
      {
        tool: "router",
        detail: "one image · not a locate request · ends in ? → single_vqa",
        ms: 4,
      },
      {
        tool: "ingest",
        detail: "optical · 4 bands · GSD 2.0 m · EPSG:32644",
        ms: 90,
      },
      {
        tool: "rs_vqa",
        detail: "LoRA adapter · GSD injected into the prompt",
        ms: 2980,
      },
    ],
    warning: null,
  },
  {
    id: "e-grounding",
    tab: "Grounding",
    task: "single_grounding",
    router: "rules",
    question: "Where is the storage tank farm?",
    scene: "grounding",
    sceneNote: "One optical scene · box in pixel and map space",
    answer:
      "The tank farm is on the western edge of the industrial block, centred at 28.5514°N 77.1183°E.",
    refusal: null,
    confidence: 0.74,
    confidenceBasis: "heuristic",
    latencyMs: 2650,
    evidence: [
      {
        label: "Bounding box · GeoJSON",
        by: "rs_ground_caption",
        colour: "#3BA3F2",
      },
      { label: "Centroid · EPSG:32644", by: "centroid_prior", colour: null },
    ],
    trace: [
      {
        tool: "router",
        detail: "referring expression · names a known object → grounding",
        ms: 3,
      },
      {
        tool: "rs_ground_caption",
        detail: "base model + PRECISE_PROMPT · no adapter loaded",
        ms: 2510,
      },
      { tool: "centroid_prior", detail: "box centre → map coordinate", ms: 12 },
    ],
    warning: null,
  },
  {
    id: "e-refusal",
    tab: "Refusal",
    task: "refused",
    router: "rules",
    question: "What colour are the rooftops in this scene?",
    scene: "sar",
    sceneNote: "One radar scene · Sentinel-1 VV",
    answer: "",
    refusal: {
      reason:
        "Radar measures backscatter, not light. There is no colour in this acquisition to report.",
      remedy: "Add an optical acquisition of the same ground and ask again.",
    },
    confidence: null,
    confidenceBasis: null,
    latencyMs: 210,
    evidence: [],
    trace: [
      {
        tool: "ingest",
        detail: "sar · VV · C-band · no optical bands present",
        ms: 86,
      },
      {
        tool: "router",
        detail: "colour question + radar-only input → refusal terminal",
        ms: 3,
      },
    ],
    warning: null,
  },
];
