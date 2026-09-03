import { useId } from "react";
import { cn } from "@/lib/cn";
import { formatDuration } from "@/lib/format";

/**
 * The two-SLA gauge.
 *
 * Scene preparation and query latency are different budgets against
 * different clocks, so they get two scales rather than one blended number.
 * A measured "4 min prep + 12 s query" is a stronger claim than a hidden
 * 20 s, and this is where the app makes that claim — ruled, marked, and
 * carrying the actual measurement rather than the target.
 *
 * The scale is logarithmic because the two budgets are an order of
 * magnitude apart; the decade ticks are labelled so nobody has to take the
 * spacing on trust.
 */

const MIN_MS = 100;
const MAX_MS = 600_000;

function position(ms: number): number {
  const clamped = Math.min(Math.max(ms, MIN_MS), MAX_MS);
  return (
    (Math.log(clamped) - Math.log(MIN_MS)) /
    (Math.log(MAX_MS) - Math.log(MIN_MS))
  );
}

/**
 * Decade ticks only. The 5-minute mark is deliberately absent: it is the
 * preparation budget, and it already gets a heavier rule of its own — a
 * label there would sit on top of the 10-minute one and say the same thing
 * twice.
 */
const TICKS: { ms: number; label: string }[] = [
  { ms: 100, label: "0.1s" },
  { ms: 1_000, label: "1s" },
  { ms: 10_000, label: "10s" },
  { ms: 60_000, label: "1min" },
  { ms: 600_000, label: "10min" },
];

export interface GaugeTrack {
  code: string;
  label: string;
  /** The published budget, in ms. */
  budgetMs: number;
  budgetLabel: string;
  /** The value actually measured, or null before anything has run. */
  measuredMs: number | null;
  measuredNote: string;
}

function Track({ track }: { track: GaugeTrack }) {
  const clipId = useId();
  const budgetX = position(track.budgetMs) * 100;
  const measuredX =
    track.measuredMs === null ? null : position(track.measuredMs) * 100;
  const overBudget =
    track.measuredMs !== null && track.measuredMs > track.budgetMs;

  return (
    <div className="grid grid-cols-1 items-center gap-x-5 gap-y-2 md:grid-cols-[152px_1fr]">
      <div className="flex items-baseline gap-2 md:flex-col md:items-start md:gap-[3px]">
        <span className="t-code-sm text-ink-3">{track.code}</span>
        <span className="t-code text-ink-0">{track.label}</span>
      </div>

      <div className="relative">
        {/* the ruled scale */}
        <svg
          viewBox="0 0 100 13"
          preserveAspectRatio="none"
          className="h-[13px] w-full"
          role="img"
          aria-label={`${track.label}: budget ${track.budgetLabel}, measured ${
            track.measuredMs === null
              ? "not yet measured"
              : formatDuration(track.measuredMs)
          }`}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x="0" y="0" width="100" height="13" />
            </clipPath>
          </defs>
          <g clipPath={`url(#${clipId})`}>
            {/* the region beyond budget, hatched: this is the failure zone */}
            <rect
              x={budgetX}
              y="0"
              width={100 - budgetX}
              height="13"
              fill="var(--color-panel-sunk)"
            />
            {/* the region within budget */}
            <rect
              x="0"
              y="0"
              width={budgetX}
              height="13"
              fill="var(--color-panel-2)"
            />
            <rect
              x="0"
              y="0"
              width="100"
              height="13"
              fill="none"
              stroke="var(--color-rule)"
              strokeWidth="0.6"
              vectorEffect="non-scaling-stroke"
            />
            {/* decade ticks */}
            {TICKS.map((tick) => (
              <line
                key={tick.ms}
                x1={position(tick.ms) * 100}
                y1="0"
                x2={position(tick.ms) * 100}
                y2="13"
                stroke="var(--color-rule-hair)"
                strokeWidth="0.6"
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {/* the budget limit */}
            <line
              x1={budgetX}
              y1="0"
              x2={budgetX}
              y2="13"
              stroke="var(--color-ink-0)"
              strokeWidth="1.4"
              vectorEffect="non-scaling-stroke"
            />
            {/* the measured value, as a filled bar up to its mark */}
            {measuredX !== null ? (
              <rect
                x="0"
                y="4"
                width={measuredX}
                height="5"
                fill={
                  overBudget ? "var(--color-signal)" : "var(--color-pass)"
                }
              />
            ) : null}
            {measuredX !== null ? (
              <line
                x1={measuredX}
                y1="0"
                x2={measuredX}
                y2="13"
                stroke={
                  overBudget ? "var(--color-signal)" : "var(--color-pass-ink)"
                }
                strokeWidth="1.6"
                vectorEffect="non-scaling-stroke"
              />
            ) : null}
          </g>
        </svg>

        {/* tick legend — the end labels anchor inside the scale rather than
            straddling it, so nothing collides or overhangs the column */}
        <div className="relative mt-[3px] h-[11px]">
          {TICKS.map((tick, index) => {
            const first = index === 0;
            const last = index === TICKS.length - 1;
            return (
              <span
                key={tick.ms}
                className={cn(
                  "t-code-sm absolute top-0 text-ink-3",
                  first && "translate-x-0",
                  last && "-translate-x-full",
                  !first && !last && "-translate-x-1/2",
                )}
                style={{ left: `${position(tick.ms) * 100}%` }}
              >
                {tick.label}
              </span>
            );
          })}
        </div>
      </div>

      <div className="md:col-start-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="t-code-sm text-ink-3">
            BUDGET {track.budgetLabel}
          </span>
          {track.measuredMs === null ? (
            <span className="t-code-sm text-ink-3">
              NOT YET MEASURED THIS SESSION
            </span>
          ) : (
            <>
              <span
                className={cn(
                  "t-data-strong text-[13px]",
                  overBudget ? "text-signal-ink" : "text-pass-ink",
                )}
                title={`${track.measuredMs} ms`}
              >
                {formatDuration(track.measuredMs)}
              </span>
              <span className="t-code-sm text-ink-2">
                {track.measuredNote}
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function SlaGauge({
  tracks,
  className,
}: {
  tracks: GaugeTrack[];
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-5", className)}>
      {tracks.map((track) => (
        <Track key={track.code} track={track} />
      ))}
    </div>
  );
}
