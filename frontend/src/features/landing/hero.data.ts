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
 * - the four exemplars are RECORDED SESSIONS. Each was run against the live
 *   API on real gallery imagery from a public test split, and every field
 *   below — answer, confidence, latency, tool order, per-tool timings and
 *   warning count — is transcribed from that response by script rather than
 *   written by hand. The query ids are kept so a session can be found again.
 *
 * That is why the confidences are low and the latencies are seconds rather
 * than milliseconds: those are the real figures. `heuristic` confidence is
 * weakest-link by construction, so a correct answer from a tool that was 95%
 * sure still reports in the teens once four tools and thirteen warnings are
 * folded in. The trace rows carry each tool's own score, which is where the
 * gap becomes legible.
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
      note: "answered by measurement, not inference",
      muted: false,
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

export interface ExemplarImage {
  /** A real gallery preview, copied into `public/exemplars/`. */
  src: string;
  /** `optical` / `sar` / `primary`, as the bundle names it. */
  role: string;
}

export interface Exemplar {
  id: string;
  /** Tab label — the task, as an operator would name it. */
  tab: string;
  /** `trace.graded.task_selected`, verbatim. */
  task: string;
  /** Which branch of the routing tree answered. */
  router: "rules" | "llm tie-break";
  question: string;
  /** The imagery the query actually ran against. */
  images: ExemplarImage[];
  /** Which split the scene comes from. */
  sceneNote: string;
  answer: string;
  /** A refusal is an answer and reads as one. */
  refused: boolean;
  confidence: number | null;
  confidenceBasis: ConfidenceBasis | null;
  latencyMs: number;
  trace: { tool: string; detail: string; ms: number }[];
  /** How many warnings the trace carried, and the first of them. */
  warningCount: number;
  warning: string | null;
  /** The recorded query id, so the session can be found again. */
  queryId: string;
}

export const EXEMPLARS: Exemplar[] = [
  {
    id: "e-crossmodal",
    tab: "Cross-modal",
    task: "crossmodal_vqa",
    router: "rules",
    question: "Is there urban fabric in this image?",
    images: [
      { src: "/exemplars/crossmodal-optical.png", role: "optical" },
      { src: "/exemplars/crossmodal-sar.png", role: "sar" },
    ],
    sceneNote: "Optical and SAR over one ground · reBEN held-out",
    answer: "yes.",
    refused: false,
    confidence: 0.1352326127819549,
    confidenceBasis: "heuristic",
    latencyMs: 8420,
    trace: [
      { tool: "coreg_check", detail: "verify_only True", ms: 1835 },
      { tool: "lulc_classifier", detail: "Urban fabric 0.9499", ms: 8170 },
      { tool: "texture_seg", detail: "builtup · 29.0% coverage", ms: 67 },
      { tool: "sar_backscatter", detail: "builtup · 0.7% coverage", ms: 55 },
    ],
    warningCount: 13,
    warning: "native GSD unknown from metadata; differs from pixel size if resampled",
    queryId: "qr_83a8312b",
  },
  {
    id: "e-single",
    tab: "Single-image",
    task: "single_vqa",
    router: "rules",
    question: "Are there more rectangular residential buildings than roads?",
    images: [
      { src: "/exemplars/single-optical.png", role: "primary" },
    ],
    sceneNote: "One optical scene · RSVQA-HR test split",
    answer: "yes.",
    refused: false,
    confidence: 0.27,
    confidenceBasis: "heuristic",
    latencyMs: 32585,
    trace: [
      { tool: "texture_seg", detail: "builtup · 38.4% coverage", ms: 35 },
      { tool: "rs_vqa", detail: "—", ms: 32465 },
    ],
    warningCount: 9,
    warning: "no CRS defined",
    queryId: "qr_71362833",
  },
  {
    id: "e-grounding",
    tab: "Grounding",
    task: "single_grounding",
    router: "rules",
    question: "The relatively larger airplane positioned near the right edge of the image can be found in the bottom-right corner.",
    images: [
      { src: "/exemplars/grounding-optical.png", role: "primary" },
    ],
    sceneNote: "One optical scene · VRSBench EVAL split",
    answer: "[{\"bbox_2d\": [839, 567, 1000, 807]}].",
    refused: false,
    confidence: 0.185625,
    confidenceBasis: "heuristic",
    latencyMs: 2212,
    trace: [
      { tool: "texture_seg", detail: "builtup · 4.4% coverage", ms: 52 },
      { tool: "centroid_prior", detail: "—", ms: 15 },
      { tool: "rs_ground_caption", detail: "phrase The relatively larger airplane positioned near the right edge of the image can be found in the bottom-r", ms: 2022 },
    ],
    warningCount: 10,
    warning: "no CRS defined",
    queryId: "qr_1b94b0c0",
  },
  {
    id: "e-refusal",
    tab: "Refusal",
    task: "change_vqa",
    router: "rules",
    question: "What changed between the two dates?",
    images: [
      { src: "/exemplars/refusal-optical.png", role: "primary" },
    ],
    sceneNote: "One optical scene · a change question asked of it",
    answer: "This is a change question, and change needs two images of the same area at two times. Only one was supplied. Upload the second acquisition and this becomes answerable.",
    refused: true,
    confidence: 0.0,
    confidenceBasis: "heuristic",
    latencyMs: 107,
    trace: [
    ],
    warningCount: 6,
    warning: "no CRS defined",
    queryId: "qr_57cf61ef",
  },
];
