import { StatusLamp, Tip } from "@/components/primitives";
import { cn } from "@/lib/cn";
import {
  BENCHMARKS,
  type Benchmark,
  type BenchmarkEntry,
} from "./benchmarks.data";

/**
 * The evaluation record, plotted.
 *
 * Seven bar plots, all on screen at once. No tabs, no accordion, nothing to
 * open: a judge with eight minutes takes the whole result in by scrolling,
 * then reads whichever panel they want.
 *
 * The drawing rules are the argument:
 *
 * - **One 0–100 domain across all seven plots.** A bar's height means the
 *   same thing everywhere, and no axis is cropped to flatter a lead.
 * - **The blind baseline is a hatched floor under the bars**, not a bar
 *   beside them. It is the score reachable without looking at the image, so
 *   a bar that fails to clear it is visibly standing in the floor. On the
 *   BigEarthNet binary panel that is three published remote-sensing VLMs at
 *   once, and it is the single most useful frame in the record.
 * - **Every bar carries its own number**, so the plot is read, not trusted.
 *
 * Panels size themselves from their bar count and wrap, so the row packs
 * without hand-tuned column spans at any width.
 *
 * The bars' natural state is their final state, and the reveal only takes
 * that away and gives it back (`backwards`, never `both`). Nothing is gated
 * on a scroll observer or on a paint that may not come: in print, in a
 * background tab, or anywhere the frame loop is stalled, the plot is still
 * a plot rather than an empty axis.
 */

/** Reproduce the source table's own precision — never invent a decimal. */
function scoreText(value: number): string {
  return Number.isInteger(value * 10) ? value.toFixed(1) : value.toFixed(2);
}

/** Gridlines, and the only numerals on the y axis. */
const GRID = [25, 50, 75, 100];

// The record is seven panels deep and it sits under the masthead, so the
// plot is drawn at the shortest height that still separates 85.06 from 83.12
// legibly. Height was traded away; the 0-100 domain was not.
const PLOT_HEIGHT = 108;

/* ── one bar ───────────────────────────────────────────────────────── */

function Bar({
  entry,
  benchmark,
  delay,
}: {
  entry: BenchmarkEntry;
  benchmark: Benchmark;
  delay: number;
}) {
  const ours = entry.kind === "ours";
  const inFloor = benchmark.blind !== null && entry.score <= benchmark.blind;

  return (
    <Tip
      content={
        <>
          <span className="t-code-sm block">{entry.system}</span>
          <span className="mt-1 block">
            {scoreText(entry.score)} {benchmark.metricAbbr}
            {entry.note ? ` · ${entry.note}` : ""}
            {inFloor ? " · inside the blind floor" : ""}
          </span>
        </>
      }
    >
      <div className="relative h-full min-w-0 flex-1 cursor-help">
        {/* The value is a sibling of the bar, not a child: the reveal
            clips to the bar's own box, which would crop anything sitting
            above it. It settles once its bar has finished being drawn. */}
        <span
          title={`${entry.score} ${benchmark.metricAbbr}`}
          className={cn(
            "absolute inset-x-0 text-center text-[9.5px] leading-none",
            ours ? "t-data-strong text-ink-0" : "t-data text-ink-2",
            "plot-label",
          )}
          style={{
            bottom: `calc(${entry.score}% + 4px)`,
            animationDelay: `${delay + 280}ms`,
          }}
        >
          {scoreText(entry.score)}
        </span>
        <span
          className={cn(
            "absolute inset-x-0 bottom-0 block border",
            ours
              ? "border-ink-0 bg-ink-0"
              : inFloor
                ? "border-rule-heavy bg-panel-2"
                : "border-plate-edge bg-plate-1",
            "plot-rise",
          )}
          style={{
            height: `${entry.score}%`,
            animationDelay: `${delay}ms`,
          }}
        />
      </div>
    </Tip>
  );
}

/* ── one plot ──────────────────────────────────────────────────────── */

function BenchmarkPlot({
  benchmark,
  order,
}: {
  benchmark: Benchmark;
  order: number;
}) {
  const bars = benchmark.entries;
  const win = benchmark.verdict === "win";

  return (
    <section
      aria-labelledby={`bm-${benchmark.id}`}
      className="m-sheet armature-field flex min-w-0 flex-col"
      style={{
        // A panel earns its width from the number of bars it carries, so a
        // two-bar panel never sprawls and a seven-bar panel never crushes.
        // Whatever is left over is shared out along the row.
        flexGrow: bars.length,
        flexBasis: `${bars.length * 50 + 84}px`,
        minWidth: `min(${bars.length * 34 + 50}px, 100%)`,
      }}
    >
      {/* ── panel head ────────────────────────────────────────────── */}
      <header className="flex items-start gap-2 border-b border-rule px-2.5 py-[7px]">
        <span className="min-w-0 flex-1">
          <h3
            id={`bm-${benchmark.id}`}
            className="t-code block text-[10.5px] text-ink-0"
          >
            {benchmark.name}
          </h3>
          <span className="mt-[2px] block text-[10.5px] leading-[1.3] text-ink-2">
            {benchmark.subtitle}
          </span>
        </span>
        <Tip content={<>Metric: {benchmark.metric}. Plotted 0–100.</>}>
          <span className="t-code-sm mt-[2px] shrink-0 cursor-help text-ink-3">
            {benchmark.metricAbbr}
          </span>
        </Tip>
      </header>

      {/* ── the plot ──────────────────────────────────────────────── */}
      <div className="overflow-x-auto overscroll-x-contain px-2.5 pb-0.5 pt-5">
        <div style={{ minWidth: `${bars.length * 36 + 18}px` }}>
          <div className="flex">
            {/* y axis — labelled, because an unlabelled scale is an assertion */}
            <div
              className="relative w-[18px] shrink-0"
              style={{ height: PLOT_HEIGHT }}
              aria-hidden="true"
            >
              {[0, ...GRID].map((value) => (
                <span
                  key={value}
                  className="t-code-sm absolute right-[5px] translate-y-1/2 text-ink-3"
                  style={{ bottom: `${value}%` }}
                >
                  {value}
                </span>
              ))}
            </div>

            <div
              className="relative min-w-0 flex-1 border-b border-l border-rule-heavy"
              style={{ height: PLOT_HEIGHT }}
            >
              {/* the blind floor, drawn behind everything else */}
              {benchmark.blind !== null ? (
                <span
                  aria-hidden="true"
                  className="m-floor absolute inset-x-0 bottom-0"
                  style={{ height: `${benchmark.blind}%` }}
                />
              ) : null}
              {GRID.map((value) => (
                <span
                  key={value}
                  aria-hidden="true"
                  className="absolute inset-x-0 h-px bg-rule-hair"
                  style={{ bottom: `${value}%` }}
                />
              ))}
              {/* the floor's own edge reads heavier than a gridline */}
              {benchmark.blind !== null ? (
                <span
                  aria-hidden="true"
                  className="absolute inset-x-0 h-px bg-rule-heavy"
                  style={{ bottom: `${benchmark.blind}%` }}
                />
              ) : null}

              <div
                className="absolute inset-0 flex items-end gap-[5px] px-[5px]"
                aria-hidden="true"
              >
                {bars.map((entry, index) => (
                  <Bar
                    key={`${entry.short}-${entry.score}`}
                    entry={entry}
                    benchmark={benchmark}
                    delay={order * 90 + index * 55}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* x axis labels, on the same flex geometry as the bars */}
          <div className="flex">
            <span className="w-[18px] shrink-0" aria-hidden="true" />
            <div className="flex min-w-0 flex-1 gap-[5px] px-[5px] pt-[5px]">
              {bars.map((entry) => (
                <span
                  key={`${entry.short}-label`}
                  className={cn(
                    "t-code-sm min-w-0 flex-1 break-words text-center leading-[1.3]",
                    entry.kind === "ours" ? "text-ink-0" : "text-ink-3",
                  )}
                >
                  {entry.short}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* The same record as a table, for anyone not reading the picture.
          The bars themselves are decorative to assistive tech, so nobody
          has to tab through twenty-seven of them to hear the numbers. */}
      {/* `sr-only` must sit on a block wrapper: a table sizes to its own
          content and will not honour the 1px clamp, which is a silent
          horizontal overflow on narrow screens. */}
      <div className="sr-only">
        <table>
          <caption>
            {benchmark.name} — {benchmark.metric}
          </caption>
          <thead>
            <tr>
              <th scope="col">System</th>
              <th scope="col">{benchmark.metricAbbr}</th>
              <th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            {bars.map((entry) => (
              <tr key={`${entry.short}-row`}>
                <th scope="row">
                  {entry.kind === "ours"
                    ? `SatQuery AI — ${entry.system}`
                    : entry.system}
                </th>
                <td>{scoreText(entry.score)}</td>
                <td>
                  {entry.note ?? ""}
                  {benchmark.blind !== null && entry.score <= benchmark.blind
                    ? " (inside the blind floor)"
                    : ""}
                </td>
              </tr>
            ))}
            {benchmark.blind !== null ? (
              <tr>
                <th scope="row">Blind baseline — {benchmark.blindLabel}</th>
                <td>{scoreText(benchmark.blind)}</td>
                <td>reachable without seeing the image</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {/* ── verdict and finding ───────────────────────────────────── */}
      <div className="mt-auto flex flex-wrap items-baseline gap-x-2 gap-y-1 border-t border-rule-hair px-2.5 pb-0.5 pt-2">
        <span className="flex items-center gap-1.5">
          <StatusLamp state={win ? "pass" : "idle"} />
          <span
            className={cn("t-code-sm", win ? "text-pass-ink" : "text-ink-2")}
          >
            {win ? "WIN" : "LEVEL"}
          </span>
        </span>
        <span
          className={cn(
            "t-data-strong text-[12px]",
            win ? "text-pass-ink" : "text-ink-1",
          )}
        >
          {benchmark.delta}
        </span>
        <span className="t-code-sm text-ink-3">{benchmark.verdictNote}</span>
      </div>
      <p className="px-2.5 pb-2.5 pt-1 text-[11px] leading-[1.4] text-ink-1">
        {benchmark.finding}
      </p>
    </section>
  );
}

/* ── the key ───────────────────────────────────────────────────────── */

function PlotKey() {
  return (
    <div className="m-sheet armature-field flex flex-wrap items-center gap-x-5 gap-y-2 px-3 py-2">
      <span className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className="h-[11px] w-[11px] border border-ink-0 bg-ink-0"
        />
        <span className="t-code-sm text-ink-0">SatQuery AI</span>
      </span>
      <span className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className="h-[11px] w-[11px] border border-plate-edge bg-plate-1"
        />
        <span className="t-code-sm text-ink-2">Comparison system</span>
      </span>
      <span className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className="m-floor h-[11px] w-[22px] border border-rule"
        />
        <span className="t-code-sm text-ink-2">
          Blind baseline — reachable without seeing the image
        </span>
      </span>
      <span className="t-code-sm ml-auto text-ink-3">
        ALL PLOTS 0–100 · UNTRUNCATED
      </span>
    </div>
  );
}

/* ── the record ────────────────────────────────────────────────────── */

export function BenchmarkRecord() {
  return (
    <div className="flex flex-col gap-3">
      <PlotKey />
      {/* One wrapping row of panels. Each panel's basis and grow come from
          its bar count, so the packing is a property of the data rather
          than a set of breakpoints somebody has to maintain. */}
      <div className="flex flex-wrap gap-3">
        {BENCHMARKS.map((benchmark, order) => (
          <BenchmarkPlot
            key={benchmark.id}
            benchmark={benchmark}
            order={order}
          />
        ))}
      </div>
    </div>
  );
}
