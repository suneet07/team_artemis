import { useSearchParams } from "react-router-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Bundle } from "@contracts/types";
import { useTasks } from "@/api/meta";
import { useCancelQuery } from "@/api/queries";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import {
  IconArrowRight,
  IconClock,
  IconFile,
  IconStop,
  IconWarning,
} from "@/components/icons";
import {
  Button,
  EmptyState,
  StatusLamp,
  Tag,
  Tip,
  WarningList,
} from "@/components/primitives";
import { DisagreementPanel } from "@/features/evidence/DisagreementPanel";
import { useRuns, type Run } from "@/store/runs";
import { useWorkspace } from "@/store/workspace";
import { formatClock, formatDuration } from "@/lib/format";
import { cn } from "@/lib/cn";
import { RefusalCard } from "./RefusalCard";
import { useQueryStream } from "./useQueryStream";

/**
 * Right pane — chat.
 *
 * Each assistant turn carries the four things a judge checks without being
 * asked: which task the router selected, whether rules or the LLM tie-break
 * chose it, how confident the system is and on what basis, and how long it
 * actually took.
 */

function LiveClock({ run }: { run: Run }) {
  const [now, setNow] = useState(Date.now());
  const finished = run.finishedAt !== null;

  useEffect(() => {
    if (finished) return;
    const timer = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, [finished]);

  // The backend's own total is authoritative once it arrives; until then the
  // wall clock shows the user that something is genuinely running.
  const elapsed = finished
    ? (run.totalLatencyMs ?? (run.finishedAt ?? now) - run.startedAt)
    : now - run.startedAt;
  const overBudget = elapsed > 20_000;

  return (
    <Tip
      content={
        finished
          ? `Backend-reported total: ${run.totalLatencyMs ?? "—"} ms. Budget is 20 s per question against a prepared bundle.`
          : "Measuring against the 20 s query budget."
      }
    >
      <span
        className={cn(
          "t-data-strong inline-flex items-center gap-1 text-[11.5px]",
          overBudget ? "text-signal-ink" : "text-ink-1",
        )}
      >
        <IconClock size={11} />
        {finished ? formatDuration(elapsed) : formatClock(elapsed)}
      </span>
    </Tip>
  );
}

function RunTurn({
  run,
  bundle,
  onAskSuggested,
  onCancel,
}: {
  run: Run;
  bundle: Bundle | undefined;
  onAskSuggested: (question: string) => void;
  onCancel: () => void;
}) {
  const { data: tasks } = useTasks();
  const setTab = useWorkspace((s) => s.setTab);
  const selectQuery = useWorkspace((s) => s.selectQuery);
  const selectedQueryId = useWorkspace((s) => s.selectedQueryId);

  const taskMeta = run.taskSelected
    ? tasks?.byTask.get(run.taskSelected)
    : undefined;
  const answer = run.answer ?? run.streamedAnswer;
  const running = run.finishedAt === null;

  return (
    <article
      className={cn(
        "print-break flex flex-col gap-2.5 border-b border-rule px-3 py-3.5",
        selectedQueryId === run.queryId && "bg-panel-2",
      )}
    >
      {/* the question */}
      <div className="flex items-start gap-2">
        <span className="t-code-sm mt-[3px] shrink-0 border border-ink-0 bg-ink-0 px-1 py-[2px] text-panel-2">
          ASK
        </span>
        <p className="min-w-0 flex-1 text-[13.5px] font-medium leading-[1.45] text-ink-0">
          {run.question}
        </p>
      </div>

      {/* execution header */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
        {run.taskSelected ? (
          <Tip content={taskMeta?.description ?? "Task selected by the router."}>
            <span>
              <Tag tone="solid">{taskMeta?.label ?? run.taskSelected}</Tag>
            </span>
          </Tip>
        ) : null}

        {run.routerPath ? (
          <Tip
            content={
              run.routerPath === "rules"
                ? "Chosen by the deterministic rule table — no model was asked to route this."
                : "The rule table was ambiguous, so the LLM broke the tie. The notes are in the trace."
            }
          >
            <span>
              <Tag tone={run.routerPath === "rules" ? "default" : "caution"}>
                {run.routerPath === "rules" ? "RULES" : "LLM TIE-BREAK"}
              </Tag>
            </span>
          </Tip>
        ) : null}

        {run.confidence !== null ? (
          <ConfidenceBadge
            value={run.confidence}
            basis={run.confidenceBasis}
          />
        ) : null}

        <span className="ml-auto flex items-center gap-2">
          <LiveClock run={run} />
          {running ? (
            <button
              type="button"
              onClick={onCancel}
              className="t-code-sm inline-flex items-center gap-1 border border-rule px-1.5 py-[3px] text-ink-2 transition-colors hover:border-signal hover:text-signal-ink"
            >
              <IconStop size={10} />
              CANCEL
            </button>
          ) : null}
        </span>
      </div>

      {/* live execution readout, while the trace is still assembling */}
      {running ? (
        <ol className="flex flex-col gap-1 border-l-2 border-rule pl-2.5">
          {run.steps.map((step) => (
            <li
              key={step.index}
              className="print-row flex items-center gap-2 text-[11.5px]"
            >
              <StatusLamp
                state={
                  step.status === "done"
                    ? "pass"
                    : step.status === "running"
                      ? "active"
                      : "idle"
                }
              />
              <span className="t-data text-ink-1">{step.tool}</span>
              {step.latency_ms ? (
                <span className="t-data ml-auto text-ink-3">
                  {step.latency_ms} ms
                </span>
              ) : null}
            </li>
          ))}
          {run.steps.length === 0 ? (
            <li className="t-code-sm text-ink-3">
              {run.validatorPassed === null
                ? "VALIDATING INPUTS…"
                : "PLANNING…"}
            </li>
          ) : null}
        </ol>
      ) : null}

      {/* the refusal path — a success, not an error */}
      {run.refusal ? (
        <RefusalCard
          refusal={run.refusal}
          bundle={bundle}
          onAskSuggested={onAskSuggested}
        />
      ) : null}

      {/* the answer */}
      {answer && !run.refusal ? (
        <p className="t-doc border-l-2 border-ink-0 pl-2.5 text-[13.5px] text-ink-0">
          {answer}
          {running ? (
            <span
              aria-hidden="true"
              className="ml-[1px] inline-block h-[14px] w-[7px] translate-y-[2px] bg-signal pulse-mark"
            />
          ) : null}
        </p>
      ) : null}

      {/* the disagreement panel gets real estate, inline with the answer */}
      {run.agreement ? (
        <DisagreementPanel agreement={run.agreement} evidence={run.evidence} />
      ) : null}

      {run.warnings.length ? <WarningList warnings={run.warnings} /> : null}

      {run.error ? (
        <div className="flex items-start gap-2 border border-signal/40 bg-signal-wash px-2.5 py-2">
          <IconWarning size={14} className="mt-[2px] shrink-0 text-signal-ink" />
          <span className="min-w-0">
            <span className="block text-[12.5px] leading-[1.4] text-ink-0">
              {run.error.message}
            </span>
            {run.error.hint ? (
              <span className="mt-1 block text-[11.5px] text-ink-2">
                {run.error.hint}
              </span>
            ) : null}
            <span className="t-data mt-1 block text-[10.5px] text-ink-3">
              {run.error.code}
            </span>
          </span>
        </div>
      ) : null}

      {/* the graded artifact is always one click away */}
      {!running ? (
        <div className="flex flex-wrap items-center gap-2 pt-0.5">
          <Button
            variant="panel"
            icon={<IconFile size={12} />}
            onClick={() => {
              selectQuery(run.queryId);
              setTab("trace");
            }}
          >
            View trace
          </Button>
          {run.fusionModel ? (
            <span className="t-data text-[10.5px] text-ink-3">
              {run.fusionModel}
            </span>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

export function ChatPane({ bundle }: { bundle: Bundle | undefined }) {
  const bundleId = bundle?.bundle_id ?? "";
  const { ask, activeQueryId, streaming, transport, submitError } =
    useQueryStream(bundleId);
  const cancelQuery = useCancelQuery();
  const runs = useRuns((s) => s.runs);
  const order = useRuns((s) => s.order);
  const { data: tasks } = useTasks();
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement | null>(null);

  /* A question handed over in the URL, which is how the testing corpus opens an
     item: it loads the bundle, then lands here with the benchmark's own wording
     already in the box. Placed in the composer rather than asked automatically
     -- the visitor should see the exact question that is about to be sent, and
     be able to change it, which is most of the point of showing a held-out row
     with its published answer beside it.

     Consumed once, keyed on the parameter itself, so a later edit to the draft
     is not overwritten on the next render. */
  const [searchParams, setSearchParams] = useSearchParams();
  const handedOver = searchParams.get("q");
  useEffect(() => {
    if (!handedOver) return;
    setDraft(handedOver);
    const next = new URLSearchParams(searchParams);
    next.delete("q");
    setSearchParams(next, { replace: true });
  }, [handedOver]);

  const sessionRuns = useMemo(
    () =>
      order
        .map((id) => runs[id])
        .filter((run): run is Run => Boolean(run) && run.bundleId === bundleId),
    [order, runs, bundleId],
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [sessionRuns.length, runs[activeQueryId ?? ""]?.streamedAnswer]);

  const ready = bundle?.status === "ready";

  /* Suggestion chips come from the bundle's own supported task list, so an
     impossible question is never offered in the first place. */
  const suggestions = useMemo(() => {
    if (!bundle) return [];
    const byTask: Partial<Record<string, string>> = {
      crossmodal_extraction:
        "How much of the built-up area is under water?",
      crossmodal_vqa:
        "Do the optical and SAR views agree about the flood extent?",
      change_vqa: "What changed between the two dates?",
      change_map: "Map the change between the two dates.",
      single_caption: "Describe what is visible in this scene.",
      single_grounding: "Locate the bridges over the river.",
      single_vqa: "How much of the scene is open water?",
    };
    return bundle.supported_tasks
      .map((task) => byTask[task])
      .filter((question): question is string => Boolean(question))
      .slice(0, 4);
  }, [bundle]);

  const submit = () => {
    if (!ready || streaming) return;
    void ask(draft);
    setDraft("");
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-panel-1">
      <div className="min-h-0 flex-1 overflow-y-auto">
        {sessionRuns.length === 0 ? (
          <div className="p-3">
            <EmptyState
              code="NO QUERIES"
              title="Ask this bundle a question"
            >
              The bundle is prepared, so answers come back in seconds rather
              than minutes. Every answer arrives with the tools, parameters and
              confidence that produced it.
            </EmptyState>
          </div>
        ) : (
          sessionRuns.map((run) => (
            <RunTurn
              key={run.queryId}
              run={run}
              bundle={bundle}
              onAskSuggested={(question) => void ask(question)}
              onCancel={() => {
                void cancelQuery.mutateAsync(run.queryId);
              }}
            />
          ))
        )}
        <div ref={endRef} />
      </div>

      {/* composer */}
      <div className="border-t border-rule bg-panel-2 p-3 no-print">
        {suggestions.length > 0 && sessionRuns.length === 0 ? (
          <div className="mb-2.5 flex flex-col gap-1.5">
            <span className="t-code-sm text-ink-3">
              SUPPORTED BY THIS BUNDLE
            </span>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  disabled={!ready}
                  onClick={() => void ask(suggestion)}
                  className="border border-rule bg-panel-1 px-2 py-1.5 text-left text-[12px] leading-[1.35] text-ink-1 transition-colors hover:border-ink-0 hover:bg-panel-2 hover:text-ink-0 disabled:cursor-not-allowed disabled:text-ink-3"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {bundle && bundle.blocked_tasks.length > 0 ? (
          <details className="mb-2.5">
            <summary className="t-code-sm cursor-pointer text-ink-3 transition-colors hover:text-ink-1">
              {bundle.blocked_tasks.length} TASKS THIS BUNDLE CANNOT SUPPORT
            </summary>
            <ul className="mt-1.5 flex flex-col gap-1">
              {bundle.blocked_tasks.map((blocked) => (
                <li
                  key={blocked.task}
                  className="flex gap-2 text-[11.5px] leading-[1.4] text-ink-2"
                >
                  <span className="t-data shrink-0 text-ink-3">
                    {tasks?.byTask.get(blocked.task)?.label ?? blocked.task}
                  </span>
                  <span>— {blocked.reason}</span>
                </li>
              ))}
            </ul>
          </details>
        ) : null}

        {submitError ? (
          <p className="mb-2 border border-signal/40 bg-signal-wash px-2 py-1.5 text-[12px] leading-[1.4] text-ink-0">
            {submitError.error.message}
            {submitError.error.hint ? (
              <span className="mt-0.5 block text-ink-2">
                {submitError.error.hint}
              </span>
            ) : null}
          </p>
        ) : null}

        <div className="flex items-end gap-2">
          <label className="sr-only" htmlFor="composer">
            Ask a question about this bundle
          </label>
          <textarea
            id="composer"
            rows={2}
            value={draft}
            disabled={!ready || streaming}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={
              ready
                ? "Ask about this imagery…"
                : "The bundle is still preparing."
            }
            className="m-sunk min-h-[52px] flex-1 resize-none px-2 py-1.5 text-[13px] leading-[1.45] text-ink-0 placeholder:text-ink-3 disabled:cursor-not-allowed disabled:text-ink-3"
          />
          <Button
            variant="primary"
            disabled={!ready || streaming || draft.trim().length === 0}
            onClick={submit}
            icon={<IconArrowRight size={13} />}
            className="h-[52px]"
          >
            Ask
          </Button>
        </div>

        <div className="mt-1.5 flex items-center justify-between gap-2">
          <span className="t-code-sm text-ink-3">
            {ready
              ? "ENTER TO SEND · SHIFT+ENTER FOR A NEW LINE"
              : "COMPOSER DISABLED UNTIL PREPARATION COMPLETES"}
          </span>
          {transport === "polling" ? (
            <Tip content="The event stream dropped twice, so progress is being polled at 2 s instead. Nothing is lost.">
              <span>
                <Tag tone="caution">POLLING FALLBACK</Tag>
              </span>
            </Tip>
          ) : null}
        </div>
      </div>
    </div>
  );
}
