import { StatusLamp, Tip } from "@/components/primitives";
import { cn } from "@/lib/cn";
import {
  BLIND_BASELINE,
  CAPABILITIES,
  ORCHESTRATION,
  type Capability,
} from "./hero.data";

/**
 * What the system does, as a record rather than as a feature grid.
 *
 * Four capabilities, each one row: what it is handed, why a general model
 * fails at it, what was built instead, and the figure that settles it with
 * the strongest thing it was measured against printed directly underneath.
 * The numeral is the row's right-hand column at every width, because a judge
 * reads four numbers before reading a single sentence.
 *
 * The fifth block is deliberately a different material — a sunk band rather
 * than a sheet. It carries a different *kind* of claim: not a benchmark, but
 * the argument that where a measurement can be computed, arithmetic beats
 * inference. `100%` against `43.5%` on the same 2,012 rows is the whole case,
 * so those are the band's own numerals.
 *
 * The strip at the foot is the last thing read and the one that separates
 * this from a leaderboard screenshot. It is drawn on the same hatched floor
 * the benchmark plots use for their blind baselines, so the two are visibly
 * the same idea before anything is read.
 */

function CapabilityRow({ capability }: { capability: Capability }) {
  return (
    <article className="m-sheet armature-field grid gap-x-5 gap-y-3 px-4 py-3.5 sm:grid-cols-[minmax(0,1fr)_200px]">
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
          <h3 className="t-plate text-[14px] text-ink-0">{capability.name}</h3>
          <Tip content="The task id the router emits, and the name the trace records.">
            <span className="t-data cursor-help text-[11px] text-ink-3">
              {capability.task}
            </span>
          </Tip>
        </div>
        <p className="t-code-sm mt-1.5 normal-case tracking-normal text-ink-3">
          {capability.scope}
        </p>

        <dl className="mt-2.5 flex flex-col gap-1.5">
          <div className="flex gap-2.5">
            <dt className="t-code-sm w-[72px] shrink-0 pt-[3px] text-ink-3">
              Problem
            </dt>
            <dd className="min-w-0 flex-1 text-[12.5px] leading-[1.45] text-ink-2">
              {capability.problem}
            </dd>
          </div>
          <div className="flex gap-2.5">
            <dt className="t-code-sm w-[72px] shrink-0 pt-[3px] text-ink-3">
              Built
            </dt>
            <dd className="min-w-0 flex-1 text-[12.5px] leading-[1.45] text-ink-1">
              {capability.fix}
            </dd>
          </div>
        </dl>
      </div>

      {/* the numeral column — the thing read first, kept out of the prose */}
      <div className="flex flex-col justify-start border-t border-rule-hair pt-3 sm:border-l sm:border-t-0 sm:pl-5 sm:pt-0">
        <span className="t-data-strong text-[30px] leading-none text-ink-0">
          {capability.score}
        </span>
        <span className="t-code-sm mt-1.5 text-ink-2">{capability.scoreOn}</span>
        <span className="mt-2 flex items-start gap-1.5 text-[11.5px] leading-[1.4] text-pass-ink">
          <span className="mt-[3px]">
            <StatusLamp state="pass" />
          </span>
          {capability.against}
        </span>
        <span className="t-code-sm mt-2.5 border-t border-rule-hair pt-2 text-ink-1">
          {capability.mechanism}
        </span>
        <span className="t-data mt-1 text-[10.5px] text-ink-3">
          {capability.weight}
        </span>
      </div>
    </article>
  );
}

export function CapabilityRecord() {
  return (
    <div className="flex flex-col gap-3">
      {CAPABILITIES.map((capability) => (
        <CapabilityRow key={capability.task} capability={capability} />
      ))}

      {/* ── the orchestrator: a different claim, on a different ground ── */}
      <section className="m-sunk armature-field px-4 py-4">
        <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
          <h3 className="t-plate text-[14px] text-ink-0">
            {ORCHESTRATION.name}
          </h3>
          <span className="t-code-sm text-ink-2">{ORCHESTRATION.claim}</span>
        </div>

        <div className="mt-3 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,420px)] lg:items-start">
          <p className="t-doc text-[12.5px] leading-[1.5] text-ink-1">
            {ORCHESTRATION.body}
          </p>
          <dl className="grid gap-x-4 gap-y-3 sm:grid-cols-3">
            {ORCHESTRATION.figures.map((figure) => (
              <div key={figure.label} className="min-w-0">
                <dt
                  className={cn(
                    "t-data-strong text-[22px] leading-none",
                    figure.muted ? "text-ink-3" : "text-ink-0",
                  )}
                >
                  {figure.value}
                </dt>
                <dd className="mt-1.5">
                  <span
                    className={cn(
                      "t-code-sm block",
                      figure.muted ? "text-ink-3" : "text-ink-1",
                    )}
                  >
                    {figure.label}
                  </span>
                  <span className="mt-1 block text-[11.5px] leading-[1.4] text-ink-2">
                    {figure.note}
                  </span>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* ── the blind baseline, on the plots' own floor ──────────────── */}
      <section className="m-floor armature-field border border-rule-heavy px-4 py-3.5">
        <h3 className="t-plate text-[13px] text-ink-0">
          {BLIND_BASELINE.headline}
        </h3>
        <p className="t-doc mt-1.5 text-[12.5px] leading-[1.5] text-ink-1">
          {BLIND_BASELINE.body}
        </p>
      </section>
    </div>
  );
}
