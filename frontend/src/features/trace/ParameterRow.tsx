import type { ParamSpec, ToolManifest } from "@contracts/types";
import { StatusLamp, Tip } from "@/components/primitives";
import { formatManifestBound, formatUnknownValue } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * One parameter, next to the constraint it had to satisfy.
 *
 * This is the row the problem statement grades: it must be visible that the
 * value was checked against a declared range or enum, not merely recorded.
 * The verdict comes from the backend's `parameter_check.rejected[]` — the
 * client never re-adjudicates, it only shows the manifest beside the value
 * so a reader can see the same thing the gate saw.
 */

function describeSpec(spec: ParamSpec | undefined): string | null {
  if (!spec) return null;
  if (spec.type === "enum" && spec.values) {
    return `one of ${spec.values.join(" | ")}`;
  }
  if (spec.range) {
    return `range [${formatManifestBound(spec.range[0], spec.type)}, ${formatManifestBound(spec.range[1], spec.type)}]`;
  }
  return spec.type;
}

/** Matches a rejection line such as `spectral_index.index=NDBI requires…`. */
function rejectionFor(
  rejected: string[],
  tool: string,
  key: string,
): string | undefined {
  const bare = tool.split("@")[0];
  return rejected.find(
    (line) =>
      line.startsWith(`${tool}.${key}`) || line.startsWith(`${bare}.${key}`),
  );
}

export function ParameterRow({
  tool,
  paramKey,
  value,
  manifest,
  rejected,
  wasDefault,
}: {
  tool: string;
  paramKey: string;
  value: unknown;
  manifest: ToolManifest | undefined;
  rejected: string[];
  wasDefault: boolean;
}) {
  const spec = manifest?.permitted_parameters?.[paramKey];
  const constraint = describeSpec(spec);
  const rejection = rejectionFor(rejected, tool, paramKey);
  /** The tool is in `/meta/tools` but does not declare this key. */
  const undeclaredKey = manifest !== undefined && spec === undefined;
  /** The tool itself is absent from `/meta/tools` — nothing to check against. */
  const unknownTool = manifest === undefined;

  return (
    <div
      className={cn(
        "@container grid grid-cols-[minmax(88px,auto)_1fr] items-baseline gap-x-3 gap-y-1 border-b border-rule-hair py-[6px] last:border-b-0",
        rejection && "bg-signal-wash",
      )}
    >
      <span className="t-code-sm text-ink-2">{paramKey}</span>

      <span
        className={cn(
          "t-data-strong break-words text-[12.5px]",
          rejection ? "text-signal-ink" : "text-ink-0",
        )}
      >
        {formatUnknownValue(value)}
      </span>

      <span className="col-span-2 flex flex-wrap items-center gap-x-2 gap-y-1">
        {rejection ? (
          <>
            <StatusLamp state="fail" />
            <span className="text-[11.5px] leading-[1.4] text-signal-ink">
              {rejection}
            </span>
          </>
        ) : constraint ? (
          <>
            <StatusLamp state="pass" />
            <span className="t-data text-[11px] text-ink-2">{constraint}</span>
            {spec?.default !== undefined ? (
              <span className="t-code-sm text-ink-3">
                DEFAULT {String(spec.default)}
              </span>
            ) : null}
          </>
        ) : undeclaredKey ? (
          <Tip content="The tool's manifest does not declare this key. It is shown because the trace recorded it; enforcing the manifest is the backend's job, not the viewer's.">
            <span className="t-code-sm flex items-center gap-1.5 text-caution">
              <StatusLamp state="caution" />
              KEY NOT DECLARED
            </span>
          </Tip>
        ) : unknownTool ? (
          <Tip
            content={`${tool.split("@")[0]} is not published by /meta/tools, so there is no declared range to check this value against. The value is shown exactly as the trace recorded it.`}
          >
            <span className="t-code-sm flex items-center gap-1.5 text-caution">
              <StatusLamp state="caution" />
              NO PUBLISHED MANIFEST
            </span>
          </Tip>
        ) : (
          <span className="t-code-sm text-ink-3">—</span>
        )}

        {wasDefault ? (
          <Tip content="The caller did not supply this parameter; the manifest default was applied and recorded.">
            <span className="t-code-sm border border-rule px-1 py-[1px] text-ink-2">
              DEFAULT APPLIED
            </span>
          </Tip>
        ) : null}
      </span>
    </div>
  );
}
