import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import type {
  ApiError,
  CompatibilityReport,
  PrepStage,
} from "@contracts/types";
import { useBundle, useRepreparebundle } from "@/api/bundles";
import { useCancelJob, useJobPolling } from "@/api/jobs";
import { PREP_EVENTS, useEventStream } from "@/api/stream";
import {
  Button,
  Field,
  StatusLamp,
  Tag,
  Tip,
  WarningList,
  ZoneHeader,
  type Lamp,
} from "@/components/primitives";
import { IconArrowRight, IconRefresh, IconStop } from "@/components/icons";
import { formatClock, formatDuration, formatEta } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * S3 — preparation.
 *
 * Four stage rows driven by the job stream. A stage that does not apply
 * arrives `skipped: true` and renders greyed rather than hidden: that the
 * pipeline considered SAR normalisation and found nothing to normalise is
 * information, and hiding it would make the pipeline look shorter than it is.
 */

const STAGES: { stage: PrepStage; label: string; detail: string }[] = [
  {
    stage: "p1_ingest",
    label: "P1 · Ingest",
    detail: "Read the raster, inventory its bands, derive radiometric bounds.",
  },
  {
    stage: "p2_sar_normalise",
    label: "P2 · SAR normalise",
    detail: "Calibrate to sigma-nought and filter speckle.",
  },
  {
    stage: "p3_coregister",
    label: "P3 · Coregister",
    detail: "Align the pair onto a common grid and report the residual.",
  },
  {
    stage: "p4_tiling",
    label: "P4 · Tiling",
    detail: "Partition into overlapping tiles the tools can reason over.",
  },
];

interface StageState {
  status: "queued" | "running" | "done" | "skipped" | "failed";
  percent: number;
  message: string | null;
  etaS: number | null;
  elapsedMs: number | null;
}

const INITIAL: Record<string, StageState> = Object.fromEntries(
  STAGES.map((s) => [
    s.stage,
    {
      status: "queued" as const,
      percent: 0,
      message: null,
      etaS: null,
      elapsedMs: null,
    },
  ]),
);

const LAMP: Record<StageState["status"], Lamp> = {
  queued: "idle",
  running: "active",
  done: "pass",
  skipped: "skipped",
  failed: "fail",
};

export function PrepareScreen() {
  const { bundleId } = useParams<{ bundleId: string }>();
  const navigate = useNavigate();
  const { data: bundle } = useBundle(bundleId ?? null, true);
  const reprepare = useRepreparebundle();
  const cancelJob = useCancelJob();

  const [stages, setStages] = useState(INITIAL);
  const [compatibility, setCompatibility] = useState<CompatibilityReport[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [finished, setFinished] = useState<number | null>(null);
  const [pollingFallback, setPollingFallback] = useState(false);
  const startedAt = useRef(Date.now());
  const [now, setNow] = useState(Date.now());

  const jobId = bundle?.job_id ?? null;
  const streamUrl = bundle?.stream_url ?? null;

  useEffect(() => {
    if (finished !== null) return;
    const timer = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, [finished]);

  const onEvent = useCallback((name: string, data: unknown) => {
    const payload = data as Record<string, any>;
    if (name === "progress" && payload.stage) {
      setStages((current) => ({
        ...current,
        [payload.stage]: {
          ...current[payload.stage],
          status: "running",
          percent: payload.percent ?? current[payload.stage]?.percent ?? 0,
          message: payload.message ?? null,
          etaS: payload.eta_s ?? null,
          elapsedMs: current[payload.stage]?.elapsedMs ?? null,
        },
      }));
    }
    if (name === "stage_complete") {
      setStages((current) => ({
        ...current,
        [payload.stage]: {
          status: payload.skipped ? "skipped" : "done",
          percent: 100,
          message: current[payload.stage]?.message ?? null,
          etaS: null,
          elapsedMs: payload.elapsed_ms ?? null,
        },
      }));
      if (payload.compatibility) {
        setCompatibility((list) => [...list, payload.compatibility]);
      }
    }
    if (name === "done") {
      setFinished(payload.elapsed_ms ?? null);
    }
    if (name === "error") {
      setError(payload.error ?? null);
      setStages((current) => {
        const next = { ...current };
        for (const key of Object.keys(next)) {
          if (next[key].status === "running") next[key] = { ...next[key], status: "failed" };
        }
        return next;
      });
    }
  }, []);

  const transport = useEventStream(streamUrl, {
    events: PREP_EVENTS,
    enabled: Boolean(streamUrl) && finished === null && !error,
    onEvent,
    onFallback: () => setPollingFallback(true),
  });

  const polled = useJobPolling(jobId, pollingFallback && finished === null);
  useEffect(() => {
    if (polled.data?.stage) onEvent("progress", polled.data);
    if (polled.data?.state === "succeeded") setFinished(polled.data.elapsed_ms);
  }, [polled.data, onEvent]);

  const overall = useMemo(() => {
    const values = STAGES.map((s) => {
      const state = stages[s.stage];
      if (state.status === "done" || state.status === "skipped") return 100;
      if (state.status === "running") return state.percent;
      return 0;
    });
    return Math.round(values.reduce((a, b) => a + b, 0) / STAGES.length);
  }, [stages]);

  const elapsed = finished ?? now - startedAt.current;
  const overBudget = elapsed > 300_000;
  const ready = bundle?.status === "ready" || finished !== null;

  return (
    <div className="mx-auto w-full max-w-[900px] px-6 pb-16 pt-8">
      <div className="mb-5 flex flex-wrap items-baseline gap-x-3 gap-y-2">
        <h1 className="t-plate text-[22px] text-ink-0">Preparing</h1>
        <span className="t-data text-[13px] text-ink-2">
          {bundle?.label ?? bundleId}
        </span>
        <Tag className="ml-auto">{bundle?.pair_type ?? "—"}</Tag>
      </div>

      {/* the slow clock, against its budget */}
      <section className="m-sheet armature-field mb-5 p-4">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
          <span className="t-code text-ink-1">
            SLA-1 · Scene preparation
          </span>
          <span className="t-code-sm text-ink-3">BUDGET ≤ 5 MIN PER SCENE</span>
        </div>
        <div className="flex items-center gap-4">
          <span
            className={cn(
              "t-data-strong text-[28px] leading-none",
              overBudget ? "text-signal-ink" : "text-ink-0",
            )}
          >
            {finished !== null ? formatDuration(finished) : formatClock(elapsed)}
          </span>
          <div className="flex-1">
            <div className="h-[10px] w-full border border-rule bg-panel-sunk">
              <div
                className={cn(
                  "h-full transition-[width] duration-300",
                  overBudget ? "bg-signal" : "bg-pass",
                )}
                style={{ width: `${overall}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between">
              <span className="t-code-sm text-ink-3">{overall}% COMPLETE</span>
              {transport === "polling" || pollingFallback ? (
                <span className="t-code-sm text-caution">POLLING FALLBACK</span>
              ) : null}
            </div>
          </div>
        </div>
      </section>

      {/* the four stages */}
      <section className="m-sheet mb-5">
        <ZoneHeader code="S3" title="Pipeline stages" />
        <ol>
          {STAGES.map((definition) => {
            const state = stages[definition.stage];
            return (
              <li
                key={definition.stage}
                className={cn(
                  "flex flex-wrap items-start gap-x-4 gap-y-2 border-b border-rule-hair px-4 py-3 last:border-b-0",
                  state.status === "skipped" && "bg-panel-1",
                )}
              >
                <span className="mt-[5px]">
                  <StatusLamp state={LAMP[state.status]} size={11} />
                </span>
                <span className="min-w-0 flex-1">
                  <span
                    className={cn(
                      "t-code block",
                      state.status === "skipped" ? "text-ink-3" : "text-ink-0",
                    )}
                  >
                    {definition.label}
                  </span>
                  <span
                    className={cn(
                      "mt-0.5 block text-[12px] leading-[1.45]",
                      state.status === "skipped" ? "text-ink-3" : "text-ink-2",
                    )}
                  >
                    {state.status === "skipped" ? (
                      <Tip content="This stage does not apply to these inputs. It is shown rather than hidden so the pipeline's full shape stays visible.">
                        <span>
                          Not applicable to these inputs — considered and
                          skipped.
                        </span>
                      </Tip>
                    ) : (
                      (state.message ?? definition.detail)
                    )}
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-3">
                  {state.status === "running" && state.etaS !== null ? (
                    <span className="t-code-sm text-ink-2">
                      ETA {formatEta(state.etaS)}
                    </span>
                  ) : null}
                  {state.elapsedMs !== null ? (
                    <span
                      className="t-data text-[11.5px] text-ink-1"
                      title={`${state.elapsedMs} ms`}
                    >
                      {formatDuration(state.elapsedMs)}
                    </span>
                  ) : null}
                  <span className="t-code-sm w-[62px] text-right text-ink-3">
                    {state.status.toUpperCase()}
                  </span>
                </span>
                {state.status === "running" ? (
                  <span className="relative h-[3px] w-full overflow-hidden bg-panel-sunk carriage" />
                ) : null}
              </li>
            );
          })}
        </ol>
      </section>

      {/* compatibility fills in as the stages land */}
      {compatibility.length > 0 ? (
        <section className="m-sheet mb-5">
          <ZoneHeader code="S3·C" title="Compatibility, as determined" />
          <div className="flex flex-col divide-y divide-rule-hair">
            {compatibility.map((report, index) => (
              <div key={index} className="px-4 py-3">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-4">
                  <Field label="MODALITY" value={report.modality} />
                  <Field
                    label="DETECTED VIA"
                    value={report.modality_source ?? "—"}
                  />
                  <Field
                    label="GEOREFERENCED"
                    value={report.crs_valid ? "yes" : "no"}
                    tone={report.crs_valid ? "pass" : "caution"}
                  />
                  <Field
                    label="NATIVE GSD"
                    value={
                      report.native_gsd_m ? `${report.native_gsd_m} m` : "—"
                    }
                  />
                  <Field
                    label="BANDS"
                    value={report.bands_present.join(", ")}
                    className="col-span-2"
                  />
                  <Field
                    label="COMPUTABLE INDICES"
                    value={
                      report.computable_indices.length
                        ? report.computable_indices.join(", ")
                        : "none"
                    }
                    tone={
                      report.computable_indices.length ? "default" : "caution"
                    }
                    className="col-span-2"
                  />
                </div>
                {report.warnings.length ? (
                  <WarningList warnings={report.warnings} className="mt-3" />
                ) : null}
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {/* failure carries the backend's own message and hint */}
      {error ? (
        <section className="mb-5 border border-signal/45">
          <div className="m-hazard h-[6px]" />
          <div className="bg-signal-wash px-4 py-3.5">
            <p className="t-code mb-1.5 text-signal-ink">
              Preparation failed · {error.code}
            </p>
            <p className="t-doc text-[13px] text-ink-0">{error.message}</p>
            {error.hint ? (
              <p className="mt-1.5 text-[12.5px] text-ink-2">{error.hint}</p>
            ) : null}
            <div className="mt-3 flex flex-wrap gap-2">
              <Button
                variant="panel"
                icon={<IconRefresh size={12} />}
                onClick={() => {
                  if (bundleId) void reprepare.mutateAsync(bundleId);
                  setStages(INITIAL);
                  setError(null);
                  startedAt.current = Date.now();
                }}
              >
                Retry preparation
              </Button>
              <Button variant="ghost" onClick={() => navigate("/upload")}>
                Change roles
              </Button>
            </div>
          </div>
        </section>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          disabled={!ready}
          icon={<IconArrowRight size={13} />}
          onClick={() => navigate(`/workspace/${bundleId}`)}
        >
          {ready ? "Open workspace" : "Preparing…"}
        </Button>
        {!ready && !error && jobId ? (
          <Button
            variant="ghost"
            icon={<IconStop size={12} />}
            onClick={() => void cancelJob.mutateAsync(jobId)}
          >
            Cancel
          </Button>
        ) : null}
        <p className="t-code-sm ml-auto text-ink-3">
          PREPARATION RUNS ONCE · QUESTIONS THEN ANSWER IN SECONDS
        </p>
      </div>
    </div>
  );
}
