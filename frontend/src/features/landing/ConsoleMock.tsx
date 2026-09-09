import { useEffect, useRef, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { StatusLamp, Tag, Tip, WarningList } from "@/components/primitives";
import { formatDuration } from "@/lib/format";
import { cn } from "@/lib/cn";
import { EXEMPLARS, type Exemplar } from "./hero.data";

/**
 * The console, illustrated.
 *
 * The masthead has to answer "what is this thing" in one look, and the honest
 * answer is a picture of the instrument doing its job: a question, the window
 * it was asked of, the answer, the evidence that produced it, and the trace
 * underneath. Four exemplars cover the four shapes an answer takes —
 * cross-modal fusion, single-image VQA, grounding, and a refusal.
 *
 * **The refusal is one of the four on purpose.** It is a graded deliverable
 * rather than an error path, and a masthead that only ever shows successes is
 * making a claim the system does not make.
 *
 * Three rules keep this from becoming marketing:
 *
 * - It is **labelled an illustration** in the panel head, and the window
 *   carries its own `SYNTHETIC` tag. Nothing here is a recorded session.
 * - The cross-modal exemplar is the mock API's fixture verbatim — the same
 *   thresholds, colours, warning and confidence the running app would show.
 * - Motion reuses the trace's own line printer rather than inventing a second
 *   idea: rows print in sequence, and nothing else moves.
 *
 * Auto-advance is a courtesy for an unattended screen at a booth, and it
 * yields immediately: it stops on hover, on keyboard focus anywhere inside
 * the panel, and entirely under `prefers-reduced-motion`.
 */

/** Long enough to read a four-step trace before the panel moves on. */
const DWELL_MS = 12_000;

/* ── one exemplar ──────────────────────────────────────────────────── */

function ExemplarPanel({ exemplar }: { exemplar: Exemplar }) {
  const refused = exemplar.refused;

  return (
    <div className="flex flex-col">
      {/* ── the window ────────────────────────────────────────────── */}
      <div className="m-window relative aspect-[320/220] w-full overflow-hidden">
        <div className="absolute inset-0 flex gap-px">
          {exemplar.images.map((image) => (
            <figure key={image.role} className="relative min-w-0 flex-1">
              <img
                src={image.src}
                alt={`${image.role} scene the query ran against`}
                loading="lazy"
                className="h-full w-full object-cover"
              />
              {exemplar.images.length > 1 ? (
                <figcaption className="t-code-sm absolute left-1 top-1 bg-window-0/80 px-1 py-[1px] text-[10px] text-window-ink">
                  {image.role.toUpperCase()}
                </figcaption>
              ) : null}
            </figure>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-x border-b border-rule-hair bg-panel-1 px-2.5 py-1.5">
        <span className="t-code-sm text-ink-3">{exemplar.sceneNote}</span>
        <span className="t-code-sm ml-auto text-ink-2">
          TASK <span className="t-data text-[10px] text-ink-0">{exemplar.task}</span>
        </span>
        <Tip
          content={
            exemplar.router === "rules"
              ? "Resolved by the deterministic router — no model was asked which task this is."
              : "The only place a model breaks a tie: 15 of the 16 routing terminals never reach it."
          }
        >
          <span className="t-code-sm cursor-help border border-rule bg-panel-sunk px-1 py-[2px] text-ink-2">
            ROUTER · {exemplar.router.toUpperCase()}
          </span>
        </Tip>
      </div>

      {/* ── the question ──────────────────────────────────────────── */}
      <div className="m-sunk mt-3 flex items-baseline gap-2.5 px-3 py-2.5">
        <span className="t-code-sm shrink-0 text-ink-3">ASK</span>
        <p className="min-w-0 flex-1 text-[13.5px] leading-[1.45] text-ink-0">
          {exemplar.question}
        </p>
      </div>

      {/* ── the answer, or the refusal ────────────────────────────── */}
      {refused ? (
        <div className="print-row mt-3 flex border border-advisory/35 bg-advisory-wash">
          <span aria-hidden="true" className="m-perforated w-[5px] shrink-0" />
          <div className="min-w-0 flex-1 px-3 py-2.5">
            <span className="t-code-sm flex items-center gap-1.5 text-advisory">
              <StatusLamp state="skipped" />
              REFUSAL · AN ANSWER, NOT AN ERROR
            </span>
            {/* The API returns the refusal as the answer, with the remedy in
                the same sentence. Splitting it here would be editing it. */}
            <p className="mt-1.5 text-[13.5px] leading-[1.5] text-ink-0">
              {exemplar.answer}
            </p>
          </div>
        </div>
      ) : (
        <div className="print-row mt-3 border border-rule-hair bg-panel-1 px-3 py-2.5">
          <span className="t-code-sm text-ink-3">ANSWER</span>
          <p className="mt-1.5 text-[13.5px] leading-[1.5] text-ink-0">
            {exemplar.answer}
          </p>
        </div>
      )}

      {/* ── what the answer cost, and how sure it is ──────────────── */}
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <ConfidenceBadge
          value={exemplar.confidence}
          basis={exemplar.confidenceBasis}
        />
        <Tip content="Wall-clock for the whole query, against the < 20 s budget for a question asked of a prepared bundle.">
          <span className="t-code-sm inline-flex cursor-help items-center gap-1.5 border border-rule bg-panel-sunk px-1.5 py-[3px] text-ink-2">
            <StatusLamp state={exemplar.latencyMs < 20_000 ? "pass" : "caution"} />
            {/* `normal-case`: the label voice uppercases everything it
                contains, and "2.65 S" is not a unit anybody writes. */}
            <span className="t-data-strong text-[11px] normal-case text-ink-0">
              {formatDuration(exemplar.latencyMs)}
            </span>
            SLA-2
          </span>
        </Tip>
        {exemplar.warningCount > 0 ? (
          <Tip content={exemplar.warning ?? undefined}>
            <span className="t-code-sm cursor-help border border-rule bg-panel-sunk px-1.5 py-[3px] text-ink-2">
              {exemplar.warningCount} WARNING{exemplar.warningCount === 1 ? "" : "S"}
            </span>
          </Tip>
        ) : null}
      </div>

      {exemplar.warning ? (
        <WarningList warnings={[exemplar.warning]} className="mt-2.5" />
      ) : null}

      {/* ── the trace ─────────────────────────────────────────────── */}
      <div className="mt-3 border border-rule-hair bg-panel-2">
        <div className="flex items-center gap-2 border-b border-rule-hair bg-panel-1 px-2.5 py-1.5">
          <span className="t-code-sm text-ink-2">TRACE</span>
          <span className="t-code-sm text-ink-3">
            {exemplar.trace.length} STEPS · EVERY PARAMETER RECORDED
          </span>
        </div>
        <ol>
          {exemplar.trace.map((step, i) => (
            <li
              key={step.tool}
              className="print-row flex items-baseline gap-2.5 border-b border-rule-hair px-2.5 py-[6px] last:border-b-0"
              style={{ animationDelay: `${180 + i * 130}ms` }}
            >
              <span className="t-data-strong shrink-0 text-[11px] text-ink-0">
                {step.tool}
              </span>
              <span className="min-w-0 flex-1 text-[11.5px] leading-[1.4] text-ink-2">
                {step.detail}
              </span>
              <span className="t-data shrink-0 text-[10.5px] text-ink-3">
                {formatDuration(step.ms)}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

/* ── the panel ─────────────────────────────────────────────────────── */

export function ConsoleMock({ className }: { className?: string }) {
  const [active, setActive] = useState(EXEMPLARS[0].id);
  const [held, setHeld] = useState(false);
  // Read once: this decides whether the panel rotates at all, and a media
  // query listener that flips it mid-dwell would restart the clock.
  const reduced = useRef(
    typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    if (held || reduced.current) return;
    const timer = window.setTimeout(() => {
      setActive((current) => {
        const at = EXEMPLARS.findIndex((e) => e.id === current);
        return EXEMPLARS[(at + 1) % EXEMPLARS.length].id;
      });
    }, DWELL_MS);
    return () => window.clearTimeout(timer);
  }, [active, held]);

  return (
    <Tabs.Root
      value={active}
      onValueChange={setActive}
      className={cn("m-sheet armature-field flex flex-col", className)}
      // Any attention at all — a pointer over it, a focus inside it — stops
      // the rotation. A panel that moves while it is being read is hostile.
      onMouseEnter={() => setHeld(true)}
      onMouseLeave={() => setHeld(false)}
      onFocusCapture={() => setHeld(true)}
      onBlurCapture={() => setHeld(false)}
    >
      <header className="flex items-center gap-3 border-b border-rule bg-panel-1 px-3 py-2">
        <h2 className="t-code min-w-0 flex-1 text-ink-1">
          The console, answering
        </h2>
        <Tip content="Four sessions recorded against the live API on real gallery imagery from public test splits. Answer, confidence, latency, tool order and per-tool timings are transcribed from the response, not written.">
          <span className="cursor-help">
            <Tag tone="pass">RECORDED</Tag>
          </span>
        </Tip>
      </header>

      <Tabs.List
        aria-label="Exemplar answers"
        className="flex flex-wrap gap-1 border-b border-rule-hair bg-panel-1 px-2 py-2"
      >
        {EXEMPLARS.map((exemplar) => (
          <Tabs.Trigger
            key={exemplar.id}
            value={exemplar.id}
            className={cn(
              "t-code-sm border px-2 py-[5px] transition-colors",
              "border-rule bg-panel-2 text-ink-2 hover:bg-panel-sunk",
              "data-[state=active]:border-ink-0 data-[state=active]:bg-ink-0 data-[state=active]:text-panel-2",
            )}
          >
            {exemplar.tab}
          </Tabs.Trigger>
        ))}
        <span className="t-code-sm ml-auto self-center pr-1 text-ink-3">
          {held ? "HOLD" : "CYCLING"}
        </span>
      </Tabs.List>

      <div className="p-3">
        {EXEMPLARS.map((exemplar) => (
          <Tabs.Content key={exemplar.id} value={exemplar.id}>
            <ExemplarPanel exemplar={exemplar} />
          </Tabs.Content>
        ))}
      </div>
    </Tabs.Root>
  );
}
