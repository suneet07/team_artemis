import { create } from "zustand";
import type {
  Agreement,
  AssetRef,
  ConfidenceBasis,
  Modality,
  ParameterCheck,
  PermittedParameterRecord,
  QueryState,
  RefusalWithRemedy,
  RouterPath,
  StepRecord,
  Task,
} from "@contracts/types";

/**
 * A live query run.
 *
 * The SSE sequence *is* the trace being assembled, so this store holds it in
 * exactly the shape the trace viewer renders — the panel is populated as the
 * events land rather than after `done`. Nothing here is derived: every field
 * arrives from the backend verbatim.
 */

export type StepStatus = "pending" | "running" | "done";

export interface LiveStep extends Partial<StepRecord> {
  index: number;
  tool: string;
  status: StepStatus;
  plannedParams?: Record<string, unknown>;
  withinManifest?: boolean;
  defaultsApplied?: string[];
}

export interface LiveAgreement extends Agreement {
  winning_modality?: Modality | null;
  explanation?: string | null;
}

export interface Run {
  queryId: string;
  bundleId: string;
  question: string;
  state: QueryState;
  /**
   * True from the moment the user submits until the API acknowledges the
   * query. The backend answers inside that one request, so this is where
   * nearly all of the waiting happens — and until it returns there is no
   * query id, no stream, and nothing from the server to show. The chat pane
   * renders this state itself rather than leaving the turn blank.
   */
  awaitingServer: boolean;
  /** When the user pressed Ask, so the clock covers the whole wait. */
  startedAt: number;
  /** Wall-clock the UI measured, alongside the backend's own total. */
  finishedAt: number | null;
  queuedMs: number | null;

  routerPath: RouterPath | null;
  taskSelected: Task | null;
  routingNotes: string[];

  validatorPassed: boolean | null;
  refusal: RefusalWithRemedy | null;
  warnings: string[];

  plan: PermittedParameterRecord[];
  parameterCheck: ParameterCheck | null;
  steps: LiveStep[];

  evidence: AssetRef[];
  agreement: LiveAgreement | null;

  streamedAnswer: string;
  answer: string | null;
  confidence: number | null;
  confidenceBasis: ConfidenceBasis | null;
  fusionModel: string | null;

  totalLatencyMs: number | null;
  traceUrl: string | null;
  error: { code: string; message: string; hint?: string } | null;
}

function emptyRun(
  queryId: string,
  bundleId: string,
  question: string,
): Run {
  return {
    queryId,
    bundleId,
    question,
    state: "queued",
    awaitingServer: true,
    startedAt: Date.now(),
    finishedAt: null,
    queuedMs: null,
    routerPath: null,
    taskSelected: null,
    routingNotes: [],
    validatorPassed: null,
    refusal: null,
    warnings: [],
    plan: [],
    parameterCheck: null,
    steps: [],
    evidence: [],
    agreement: null,
    streamedAnswer: "",
    answer: null,
    confidence: null,
    confidenceBasis: null,
    fusionModel: null,
    totalLatencyMs: null,
    traceUrl: null,
    error: null,
  };
}

interface RunStore {
  runs: Record<string, Run>;
  order: string[];
  /** Open a turn the instant the user submits, under a provisional id. */
  begin: (provisionalId: string, bundleId: string, question: string) => void;
  /** The API acknowledged it: move the turn onto its real query id. */
  confirm: (provisionalId: string, queryId: string) => void;
  /** The submission never produced a query: take the turn back out. */
  discard: (provisionalId: string) => void;
  apply: (queryId: string, event: string, data: unknown) => void;
  reset: () => void;
}

export const useRuns = create<RunStore>((set) => ({
  runs: {},
  order: [],

  begin: (provisionalId, bundleId, question) =>
    set((state) => ({
      runs: {
        ...state.runs,
        [provisionalId]: emptyRun(provisionalId, bundleId, question),
      },
      order: [...state.order, provisionalId],
    })),

  confirm: (provisionalId, queryId) =>
    set((state) => {
      const run = state.runs[provisionalId];
      if (!run) return state;
      const { [provisionalId]: _provisional, ...rest } = state.runs;
      return {
        // `startedAt` is kept, so the clock does not restart on acknowledgement.
        runs: { ...rest, [queryId]: { ...run, queryId, awaitingServer: false } },
        order: state.order.map((id) => (id === provisionalId ? queryId : id)),
      };
    }),

  discard: (provisionalId) =>
    set((state) => {
      if (!state.runs[provisionalId]) return state;
      const { [provisionalId]: _provisional, ...rest } = state.runs;
      return {
        runs: rest,
        order: state.order.filter((id) => id !== provisionalId),
      };
    }),

  apply: (queryId, event, data) =>
    set((state) => {
      const run = state.runs[queryId];
      if (!run) return state;
      const next = reduce(run, event, data as Record<string, unknown>);
      return { runs: { ...state.runs, [queryId]: next } };
    }),

  reset: () => set({ runs: {}, order: [] }),
}));

function reduce(run: Run, event: string, data: Record<string, any>): Run {
  switch (event) {
    case "accepted":
      return { ...run, state: "running", queuedMs: data.queued_ms ?? null };

    case "router":
      return {
        ...run,
        routerPath: data.router_path ?? null,
        taskSelected: data.task_selected ?? null,
        routingNotes: data.notes ?? [],
      };

    case "validator":
      return {
        ...run,
        validatorPassed: data.passed,
        refusal: data.refusal ?? run.refusal,
        warnings: data.warnings?.length ? data.warnings : run.warnings,
      };

    case "plan": {
      const plan: PermittedParameterRecord[] = data.steps ?? [];
      return {
        ...run,
        plan,
        parameterCheck: data.parameter_check ?? null,
        // The skeleton appears immediately, every step pending.
        steps: plan.map((record, index) => ({
          index,
          tool: record.tool,
          status: "pending" as const,
          plannedParams: record.params,
          withinManifest: record.within_manifest,
          defaultsApplied: record.defaults_applied,
        })),
      };
    }

    case "step_started":
      return {
        ...run,
        steps: upsertStep(run.steps, data.index, data.tool, (step) => ({
          ...step,
          status: "running",
        })),
      };

    case "step_completed":
      return {
        ...run,
        steps: upsertStep(run.steps, data.index, data.tool, (step) => ({
          ...step,
          status: "done",
          params: data.params ?? step.params,
          param_source: data.param_source ?? step.param_source,
          outputs: data.outputs ?? step.outputs,
          confidence: data.confidence ?? null,
          latency_ms: data.latency_ms ?? null,
        })),
      };

    case "evidence": {
      const incoming: AssetRef[] = (data.assets ?? []).filter(Boolean);
      const known = new Set(run.evidence.map((a) => a.asset_id));
      return {
        ...run,
        evidence: [
          ...run.evidence,
          ...incoming.filter((a) => !known.has(a.asset_id)),
        ],
      };
    }

    case "agreement":
      return { ...run, agreement: data as LiveAgreement };

    case "token":
      return { ...run, streamedAnswer: run.streamedAnswer + (data.text ?? "") };

    case "fusion":
      return {
        ...run,
        answer: data.answer ?? run.streamedAnswer,
        confidence: data.confidence ?? null,
        confidenceBasis: data.confidence_basis ?? null,
        fusionModel: data.model ?? null,
      };

    case "done":
      return {
        ...run,
        state: (data.state as QueryState) ?? "succeeded",
        totalLatencyMs: data.total_latency_ms ?? null,
        traceUrl: data.trace_url ?? null,
        finishedAt: Date.now(),
      };

    case "error":
      return {
        ...run,
        state: "failed",
        finishedAt: Date.now(),
        error: data.error ?? {
          code: "INTERNAL",
          message: "The stream ended unexpectedly.",
        },
      };

    default:
      return run;
  }
}

function upsertStep(
  steps: LiveStep[],
  index: number,
  tool: string,
  update: (step: LiveStep) => LiveStep,
): LiveStep[] {
  const existing = steps.find((step) => step.index === index);
  if (existing) return steps.map((step) => (step.index === index ? update(step) : step));
  return [
    ...steps,
    update({ index, tool, status: "pending" }),
  ].sort((a, b) => a.index - b.index);
}
