import { useMemo, useState } from "react";
import * as Accordion from "@radix-ui/react-accordion";
import type { Trace } from "@contracts/types";
import { lookupTool, useTasks, useTools } from "@/api/meta";
import { traceDownloadUrl } from "@/api/queries";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import {
  IconCheck,
  IconChevron,
  IconCopy,
  IconCross,
  IconDownload,
} from "@/components/icons";
import {
  Button,
  EmptyState,
  Field,
  FieldRow,
  StatusLamp,
  Tag,
  Tip,
  WarningList,
} from "@/components/primitives";
import { useWorkspace } from "@/store/workspace";
import {
  formatDuration,
  formatTimestamp,
  formatUnknownValue,
  humaniseKey,
} from "@/lib/format";
import { cn } from "@/lib/cn";
import { ParameterRow } from "./ParameterRow";

/**
 * S5 — the trace viewer.
 *
 * This is the graded artifact. The problem statement grades four fields, so
 * they lead, in its own order, in a block that is never summarised away:
 * selected task · tools invoked · permitted parameters · outputs. Everything
 * the backend sent is rendered, including keys this build has never seen —
 * `graded.outputs` is open by schema and dropping an unknown key would be
 * dropping evidence.
 */

function GradedSection({
  index,
  title,
  children,
}: {
  index: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-b border-rule last:border-b-0">
      <h4 className="t-code-sm flex items-center gap-2 bg-panel-sunk px-3 py-1.5 text-ink-2">
        <span className="border border-rule bg-panel-2 px-1 py-[1px]">
          {index}
        </span>
        {title}
      </h4>
      <div className="px-3 py-2.5">{children}</div>
    </section>
  );
}

export function TracePanel({ trace }: { trace: Trace | undefined }) {
  const { data: tools } = useTools();
  const { data: tasks } = useTasks();
  const selectedStepIndex = useWorkspace((s) => s.selectedStepIndex);
  const selectStep = useWorkspace((s) => s.selectStep);
  const [copied, setCopied] = useState(false);

  const rejected = trace?.graded.parameter_check.rejected ?? [];
  const parameterCheckPassed = trace?.graded.parameter_check.passed ?? true;

  const openItems = useMemo(
    () =>
      selectedStepIndex !== null
        ? [`step-${selectedStepIndex}`]
        : (trace?.steps ?? []).map((_, index) => `step-${index}`),
    [selectedStepIndex, trace?.steps],
  );

  if (!trace) {
    return (
      <div className="p-3">
        <EmptyState code="NO TRACE" title="No query selected">
          Ask a question, or choose an answer in the chat, and its trace
          appears here as it is produced.
        </EmptyState>
      </div>
    );
  }

  const copy = async () => {
    await navigator.clipboard.writeText(JSON.stringify(trace, null, 2));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  const graded = trace.graded;
  const taskMeta = tasks?.byTask.get(graded.task_selected);

  return (
    <div className="flex h-full min-h-0 flex-col bg-panel-1">
      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* ── the graded block ─────────────────────────────────────── */}
        <section className="m-sheet m-2.5 print-break">
          <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-rule bg-plate-1 px-3 py-2">
            <span className="t-code-sm border border-ink-0 bg-ink-0 px-1.5 py-[3px] text-panel-2">
              GRADED
            </span>
            <h3 className="t-plate text-[13px] text-ink-0">
              The four assessed fields
            </h3>
            <span className="t-data ml-auto text-[10.5px] text-ink-3">
              schema v{trace.schema_version}
            </span>
          </header>

          {/* the parameter gate verdict, as a banner */}
          <div
            className={cn(
              "flex items-center gap-2 border-b px-3 py-2",
              parameterCheckPassed
                ? "border-pass/30 bg-pass-wash"
                : "border-signal/35 bg-signal-wash",
            )}
          >
            {parameterCheckPassed ? (
              <IconCheck size={14} className="shrink-0 text-pass-ink" />
            ) : (
              <IconCross size={14} className="shrink-0 text-signal-ink" />
            )}
            <span
              className={cn(
                "t-code",
                parameterCheckPassed ? "text-pass-ink" : "text-signal-ink",
              )}
            >
              {parameterCheckPassed
                ? "Parameter check passed"
                : `Parameter check failed · ${rejected.length} rejected`}
            </span>
            <span className="ml-auto text-[11.5px] text-ink-2">
              {parameterCheckPassed
                ? "Every parameter sat inside its declared manifest."
                : "No tool was invoked with a rejected parameter."}
            </span>
          </div>

          <GradedSection index={1} title="SELECTED TASK">
            <div className="flex flex-wrap items-center gap-2">
              <Tag tone="solid">{graded.task_selected}</Tag>
              {taskMeta ? (
                <span className="text-[12.5px] leading-[1.4] text-ink-1">
                  {taskMeta.label} — {taskMeta.description}
                </span>
              ) : null}
            </div>
            {trace.router_path ? (
              <p className="t-code-sm mt-2 text-ink-3">
                ROUTED BY {trace.router_path === "rules" ? "RULE TABLE" : "LLM TIE-BREAK"}
              </p>
            ) : null}
          </GradedSection>

          <GradedSection index={2} title="TOOLS AND MODELS INVOKED">
            {graded.tools_invoked.length === 0 ? (
              <p className="text-[12.5px] text-ink-2">
                None. The request was refused before any tool ran — which is
                itself the result.
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {graded.tools_invoked.map((invoked) => {
                  const manifest = lookupTool(tools?.byName, invoked);
                  return (
                    <li key={invoked} className="flex items-start gap-2">
                      <StatusLamp state="pass" />
                      <span className="min-w-0">
                        <span className="t-data-strong block text-[12.5px] text-ink-0">
                          {invoked}
                        </span>
                        {manifest ? (
                          <span className="block text-[11.5px] leading-[1.45] text-ink-2">
                            {manifest.description}
                          </span>
                        ) : null}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </GradedSection>

          <GradedSection index={3} title="PERMITTED PARAMETERS">
            {graded.permitted_parameters.length === 0 ? (
              <p className="text-[12.5px] text-ink-2">
                No parameters were admitted.
              </p>
            ) : (
              <div className="flex flex-col gap-3">
                {graded.permitted_parameters.map((record) => {
                  const manifest = lookupTool(tools?.byName, record.tool);
                  return (
                    <div key={record.tool}>
                      <div className="mb-1 flex flex-wrap items-center gap-2">
                        <span className="t-data-strong text-[12px] text-ink-0">
                          {record.tool}
                        </span>
                        <Tag
                          tone={record.within_manifest ? "pass" : "signal"}
                        >
                          {record.within_manifest
                            ? "WITHIN MANIFEST"
                            : "OUTSIDE MANIFEST"}
                        </Tag>
                      </div>
                      <div className="border border-rule-hair bg-panel-2 px-2.5">
                        {Object.entries(record.params).map(([key, value]) => (
                          <ParameterRow
                            key={key}
                            tool={record.tool}
                            paramKey={key}
                            value={value}
                            manifest={manifest}
                            rejected={rejected}
                            wasDefault={Boolean(
                              record.defaults_applied?.includes(key),
                            )}
                          />
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {rejected.length > 0 ? (
              <ul className="mt-3 flex flex-col gap-1.5 border border-signal/35 bg-signal-wash px-2.5 py-2">
                {rejected.map((line) => (
                  <li
                    key={line}
                    className="flex items-start gap-1.5 text-[12px] leading-[1.45] text-ink-0"
                  >
                    <IconCross
                      size={12}
                      className="mt-[3px] shrink-0 text-signal-ink"
                    />
                    <span className="t-data">{line}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </GradedSection>

          <GradedSection index={4} title="OUTPUTS">
            {/* Open by schema: unknown keys render generically rather than
                being dropped. */}
            <div className="flex flex-col">
              {Object.entries(graded.outputs).map(([key, value]) => {
                if (key === "refusal" && value && typeof value === "object") {
                  const refusal = value as { reason: string; category: string };
                  return (
                    <FieldRow key={key} label="REFUSAL">
                      <span className="flex flex-col gap-1">
                        <Tag tone="advisory">{refusal.category}</Tag>
                        <span className="text-ink-0">{refusal.reason}</span>
                      </span>
                    </FieldRow>
                  );
                }
                return (
                  <FieldRow key={key} label={humaniseKey(key)}>
                    <span
                      className={cn(
                        value === null ? "text-ink-3" : "text-ink-0",
                        typeof value === "number" ||
                          Array.isArray(value) ||
                          key.endsWith("_id")
                          ? "t-data"
                          : "",
                      )}
                    >
                      {formatUnknownValue(value)}
                    </span>
                  </FieldRow>
                );
              })}
            </div>
          </GradedSection>
        </section>

        {/* ── the record around it ─────────────────────────────────── */}
        <section className="m-sheet m-2.5 print-break">
          <h3 className="t-code border-b border-rule bg-panel-1 px-3 py-2 text-ink-1">
            Execution record
          </h3>
          <div className="px-3 py-2.5">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-3">
              <Field label="QUERY ID" value={trace.query_id} />
              <Field label="TIMESTAMP" value={formatTimestamp(trace.timestamp)} />
              <Field label="ROUTER PATH" value={trace.router_path ?? "—"} />
            </div>

            {trace.routing_notes?.length ? (
              <div className="mt-3">
                <span className="t-code-sm text-ink-3">ROUTING NOTES</span>
                <ul className="mt-1 flex flex-col gap-1">
                  {trace.routing_notes.map((note) => (
                    <li
                      key={note}
                      className="t-data flex gap-1.5 text-[11.5px] leading-[1.45] text-ink-1"
                    >
                      <span aria-hidden="true" className="text-ink-3">
                        ›
                      </span>
                      {note}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </section>

        {/* ── inputs ───────────────────────────────────────────────── */}
        {trace.inputs?.length ? (
          <section className="m-sheet m-2.5 print-break">
            <h3 className="t-code border-b border-rule bg-panel-1 px-3 py-2 text-ink-1">
              Inputs · {trace.inputs.length}
            </h3>
            <div className="flex flex-col divide-y divide-rule-hair">
              {trace.inputs.map((input) => (
                <div key={input.file} className="px-3 py-2.5">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <span className="t-data-strong text-[12.5px] text-ink-0">
                      {input.file}
                    </span>
                    <Tag>{input.modality}</Tag>
                    {input.modality_source ? (
                      <Tip content="How the modality was determined. A declared value is the caller's word; a sensor tag is the file's own metadata.">
                        <span>
                          <Tag>VIA {input.modality_source}</Tag>
                        </span>
                      </Tip>
                    ) : null}
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
                    <Field label="CRS" value={input.crs ?? "none"} />
                    <Field
                      label="NATIVE GSD"
                      value={
                        input.native_gsd_m ? `${input.native_gsd_m} m` : "—"
                      }
                    />
                    <Field
                      label="BIT DEPTH"
                      value={
                        input.bit_depth
                          ? `${input.bit_depth}${input.bit_depth_source ? ` · ${input.bit_depth_source}` : ""}`
                          : "—"
                      }
                    />
                    <Field
                      label="NODATA"
                      value={
                        input.nodata_frac !== undefined
                          ? `${(input.nodata_frac * 100).toFixed(1)}%`
                          : "—"
                      }
                      tone={
                        (input.nodata_frac ?? 0) > 0.4 ? "caution" : "default"
                      }
                      title={
                        (input.nodata_frac ?? 0) > 0.4
                          ? "Over 40% nodata — statistics from this scene are masked statistics."
                          : undefined
                      }
                    />
                    <Field
                      label="BANDS"
                      value={input.bands?.join(", ") ?? "—"}
                      className="col-span-2"
                    />
                    <Field
                      label="COMPUTABLE INDICES"
                      value={
                        input.computable_indices?.length
                          ? input.computable_indices.join(", ")
                          : "none"
                      }
                      tone={
                        input.computable_indices?.length ? "default" : "caution"
                      }
                      className="col-span-2"
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {/* ── compatibility ────────────────────────────────────────── */}
        {trace.compatibility ? (
          <section className="m-sheet m-2.5 print-break">
            <h3 className="t-code border-b border-rule bg-panel-1 px-3 py-2 text-ink-1">
              Compatibility
            </h3>
            <div className="px-3 py-2.5">
              <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-4">
                <Field
                  label="COREGISTERED"
                  value={trace.compatibility.coregistered ? "yes" : "no"}
                  tone={trace.compatibility.coregistered ? "pass" : "caution"}
                />
                <Field
                  label="RMSE"
                  value={
                    trace.compatibility.rmse_px !== null &&
                    trace.compatibility.rmse_px !== undefined
                      ? `${trace.compatibility.rmse_px} px`
                      : "—"
                  }
                />
                <Field
                  label="METHOD"
                  value={trace.compatibility.method ?? "—"}
                  className="col-span-2"
                />
              </div>
              {trace.compatibility.checks_passed?.length ? (
                <div className="mt-2.5 flex flex-wrap gap-1.5">
                  {trace.compatibility.checks_passed.map((check) => (
                    <Tag key={check} tone="pass">
                      <IconCheck size={9} />
                      {check}
                    </Tag>
                  ))}
                </div>
              ) : null}
              {trace.compatibility.ingest?.warnings?.length ? (
                <WarningList
                  warnings={trace.compatibility.ingest.warnings}
                  className="mt-3"
                  title="Ingest findings"
                />
              ) : null}
            </div>
          </section>
        ) : null}

        {/* ── steps ────────────────────────────────────────────────── */}
        {trace.steps?.length ? (
          <section className="m-sheet m-2.5 print-break">
            <h3 className="t-code border-b border-rule bg-panel-1 px-3 py-2 text-ink-1">
              Steps · {trace.steps.length}
            </h3>
            <Accordion.Root
              type="multiple"
              value={openItems}
              onValueChange={(values) => {
                selectStep(
                  values.length === 1
                    ? Number(values[0].replace("step-", ""))
                    : null,
                );
              }}
            >
              {trace.steps.map((step, index) => {
                const manifest = lookupTool(tools?.byName, step.tool);
                return (
                  <Accordion.Item
                    key={`${step.tool}-${index}`}
                    value={`step-${index}`}
                    className="border-b border-rule-hair last:border-b-0"
                  >
                    <Accordion.Header>
                      <Accordion.Trigger className="group flex w-full flex-wrap items-center gap-x-2 gap-y-1 px-3 py-2 text-left transition-colors hover:bg-panel-1">
                        <IconChevron
                          size={11}
                          className="shrink-0 text-ink-2 transition-transform group-data-[state=open]:rotate-90"
                        />
                        <span className="t-code-sm text-ink-3">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="t-data-strong min-w-0 flex-1 break-words text-left text-[12.5px] text-ink-0">
                          {step.tool}
                        </span>
                        {step.latency_ms !== null &&
                        step.latency_ms !== undefined ? (
                          <Tip
                            content={
                              manifest?.expected_latency_ms
                                ? `Manifest expects about ${manifest.expected_latency_ms} ms.`
                                : "Measured latency for this step."
                            }
                          >
                            <span className="t-data shrink-0 text-[11px] text-ink-2">
                              {formatDuration(step.latency_ms)}
                            </span>
                          </Tip>
                        ) : null}
                        <ConfidenceBadge
                          value={step.confidence}
                          basis={
                            manifest?.confidence_source === "learned_logprob"
                              ? "calibrated"
                              : "heuristic"
                          }
                        />
                      </Accordion.Trigger>
                    </Accordion.Header>

                    <Accordion.Content className="overflow-hidden data-[state=open]:animate-[print-row_320ms_var(--ease-print)]">
                      <div className="border-t border-rule-hair bg-panel-1 px-3 py-2.5">
                        {manifest ? (
                          <p className="mb-2.5 text-[11.5px] leading-[1.5] text-ink-2">
                            {manifest.description}
                          </p>
                        ) : null}

                        <div className="mb-2.5">
                          <span className="t-code-sm text-ink-3">
                            PARAMETERS AGAINST MANIFEST
                          </span>
                          <div className="mt-1 border border-rule-hair bg-panel-2 px-2.5">
                            {Object.entries(step.params).map(([key, value]) => (
                              <ParameterRow
                                key={key}
                                tool={step.tool}
                                paramKey={key}
                                value={value}
                                manifest={manifest}
                                rejected={rejected}
                                wasDefault={false}
                              />
                            ))}
                          </div>
                          {step.param_source ? (
                            <p className="t-code-sm mt-1.5 text-ink-3">
                              SOURCE {step.param_source}
                            </p>
                          ) : null}
                        </div>

                        {step.outputs &&
                        Object.keys(step.outputs).length > 0 ? (
                          <div>
                            <span className="t-code-sm text-ink-3">
                              OUTPUTS
                            </span>
                            <div className="mt-1 border border-rule-hair bg-panel-2 px-2.5">
                              {Object.entries(step.outputs).map(
                                ([key, value]) => (
                                  <FieldRow key={key} label={humaniseKey(key)}>
                                    <span className="t-data text-ink-0">
                                      {formatUnknownValue(value)}
                                    </span>
                                  </FieldRow>
                                ),
                              )}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </Accordion.Content>
                  </Accordion.Item>
                );
              })}
            </Accordion.Root>
          </section>
        ) : null}

        {/* ── fusion ───────────────────────────────────────────────── */}
        {trace.fusion ? (
          <section className="m-sheet m-2.5 print-break">
            <h3 className="t-code border-b border-rule bg-panel-1 px-3 py-2 text-ink-1">
              Fusion
            </h3>
            <div className="px-3 py-2.5">
              <Field label="MODEL" value={trace.fusion.model ?? "—"} />
              {trace.fusion.answer ? (
                <p className="t-doc mt-2 text-[12.5px] text-ink-0">
                  {trace.fusion.answer}
                </p>
              ) : null}
            </div>
          </section>
        ) : null}

        {trace.warnings?.length ? (
          <div className="m-2.5">
            <WarningList warnings={trace.warnings} />
          </div>
        ) : null}
      </div>

      {/* ── the artifact itself ──────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2 border-t border-rule bg-panel-2 p-3 no-print">
        <Button
          variant="panel"
          onClick={() => void copy()}
          icon={copied ? <IconCheck size={12} /> : <IconCopy size={12} />}
        >
          {copied ? "Copied" : "Copy JSON"}
        </Button>
        <a
          href={traceDownloadUrl(trace.query_id)}
          download={`trace_${trace.query_id}.json`}
          className="t-code inline-flex items-center gap-1.5 border border-ink-0 bg-ink-0 px-2.5 py-[7px] text-panel-2 transition-colors hover:bg-[#20282a]"
        >
          <IconDownload size={12} />
          Download trace.json
        </a>
        <span className="t-code-sm ml-auto text-ink-3">
          SCHEMA v{trace.schema_version} · VERBATIM
        </span>
      </div>
    </div>
  );
}
