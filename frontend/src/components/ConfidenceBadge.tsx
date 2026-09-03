import type { ConfidenceBasis } from "@contracts/types";
import { cn } from "@/lib/cn";
import { formatConfidence } from "@/lib/format";
import { Tip } from "./primitives";

/**
 * Confidence, honestly (§7.2).
 *
 * A heuristic number and a calibrated one are different claims, so they are
 * different objects: the heuristic badge is unfilled and hatched — the panel
 * marking for "reading not yet certified" — and the calibrated badge is
 * solid. The raw value is always one hover away; the percentage never
 * travels alone.
 */
export function ConfidenceBadge({
  value,
  basis,
  className,
}: {
  value: number | null | undefined;
  basis: ConfidenceBasis | null | undefined;
  className?: string;
}) {
  if (value === null || value === undefined) {
    return (
      <Tip content="This step reported no confidence. Nothing is inferred in its place.">
        <span
          className={cn(
            "t-code-sm inline-flex items-center gap-1 border border-dashed border-rule px-1.5 py-[3px] text-ink-3",
            className,
          )}
        >
          CONF n/a
        </span>
      </Tip>
    );
  }

  const calibrated = basis === "calibrated";

  return (
    <Tip
      content={
        <span className="flex flex-col gap-1">
          <span className="t-data">raw {value}</span>
          <span>
            {calibrated
              ? "Isotonic calibration; ECE reported on a held-out set."
              : "Interim heuristic — the calibrator is not fitted yet, so treat this as an ordering, not a probability."}
          </span>
        </span>
      }
    >
      <span
        className={cn(
          "t-code-sm inline-flex items-center gap-1.5 border px-1.5 py-[3px]",
          calibrated
            ? "border-ink-0 bg-ink-0 text-panel-2"
            : "m-hazard-soft border-rule-heavy bg-panel-1 text-ink-1",
          className,
        )}
      >
        <span className={cn("t-data-strong text-[11px]", !calibrated && "text-ink-0")}>
          {formatConfidence(value)}
        </span>
        <span className={calibrated ? "text-panel-2/75" : "text-ink-2"}>
          {calibrated ? "CALIBRATED" : "HEURISTIC"}
        </span>
      </span>
    </Tip>
  );
}
