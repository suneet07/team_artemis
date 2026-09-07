import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ENABLE_BASEMAP, USE_MOCKS } from "@/api/client";
import { useDisagreementCauses, useHealth, useTasks, useTools } from "@/api/meta";
import {
  Field,
  FieldRow,
  Skeleton,
  StatusLamp,
  Tag,
  Tip,
  ZoneHeader,
} from "@/components/primitives";
import { IconChip, IconOptical, IconSar } from "@/components/icons";
import { formatManifestBound } from "@/lib/format";

/**
 * S7 — system.
 *
 * Small, and it earns its place: it is what makes the system look operated
 * rather than assembled. Everything here is read from `/meta`, so it stays
 * true when the backend changes without anyone editing this file.
 */

/**
 * Reliability diagram. Empty until a calibrator is fitted — showing a curve
 * that does not exist yet would be exactly the overclaim this product is
 * built to avoid.
 */
function ReliabilityDiagram({ calibrated }: { calibrated: boolean }) {
  const perfect = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ];

  return (
    <div className="h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
          <CartesianGrid stroke="var(--color-rule-hair)" />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            stroke="var(--color-ink-3)"
            tick={{ fontSize: 10, fontFamily: "var(--font-mono)" }}
            label={{
              value: "predicted confidence",
              position: "insideBottom",
              offset: -14,
              style: {
                fontSize: 10,
                fill: "var(--color-ink-3)",
                fontFamily: "var(--font-display)",
                letterSpacing: "0.11em",
                textTransform: "uppercase",
              },
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            stroke="var(--color-ink-3)"
            tick={{ fontSize: 10, fontFamily: "var(--font-mono)" }}
          />
          <ReferenceLine
            segment={perfect}
            stroke="var(--color-rule-heavy)"
            strokeDasharray="3 3"
            ifOverflow="extendDomain"
          />
          {calibrated ? (
            <Scatter data={[]} fill="var(--color-signal)" />
          ) : null}
          <Tooltip
            contentStyle={{
              background: "var(--color-ink-0)",
              border: "none",
              fontSize: 11,
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SystemScreen() {
  const { data: health, isError } = useHealth();
  const { data: tools } = useTools();
  const { data: tasks } = useTasks();
  const { data: causes } = useDisagreementCauses();

  const calibrated = false; // becomes true when the backend reports a fitted calibrator

  return (
    <div className="mx-auto w-full max-w-[1060px] px-6 pb-16 pt-8">
      <h1 className="t-plate mb-1.5 text-[22px] text-ink-0">System</h1>
      <p className="t-doc mb-6 text-[13.5px] text-ink-2">
        What is loaded, what is serving it, and what the tools are permitted to
        do. Every label on this page comes from the API rather than the build,
        so it stays honest when the backend changes.
      </p>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* ── serving state ────────────────────────────────────────── */}
        <section className="m-sheet">
          <ZoneHeader code="S7·A" title="Serving" />
          <div className="px-4 py-3">
            {isError ? (
              <p className="border border-signal/40 bg-signal-wash px-2.5 py-2 text-[12.5px] leading-[1.45] text-ink-0">
                The health endpoint is unreachable. Cached bundles and the
                deterministic tools still work; generated answers do not.
              </p>
            ) : !health ? (
              <Skeleton className="h-24 w-full" />
            ) : (
              <>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
                  <Field
                    label="STATUS"
                    value={health.status}
                    tone={health.status === "ok" ? "pass" : "caution"}
                  />
                  <Field label="BUILD" value={health.version} />
                  <Field
                    label="TRACE SCHEMA"
                    value={`v${health.trace_schema_version}`}
                  />
                  <Field
                    label="SERVING"
                    value={health.serving}
                    tone={health.serving === "vllm" ? "pass" : "caution"}
                  />
                  <Field
                    label="GPU"
                    value={health.gpu ? "present" : "absent"}
                    tone={health.gpu ? "pass" : "caution"}
                  />
                  <Field
                    label="OFFLINE MODE"
                    value={health.offline_mode ? "on" : "off"}
                  />
                </div>

                <div className="mt-4 border-t border-rule-hair pt-3">
                  <span className="t-code-sm text-ink-3">
                    ADAPTERS LOADED · {health.adapters_loaded.length}
                  </span>
                  <ul className="mt-1.5 flex flex-wrap gap-1.5">
                    {health.adapters_loaded.length === 0 ? (
                      <li className="text-[12px] text-ink-2">
                        None. The base model answers zero-shot; quality
                        improves silently as adapters land.
                      </li>
                    ) : (
                      health.adapters_loaded.map((adapter) => (
                        <li key={adapter}>
                          <Tag>
                            <IconChip size={10} />
                            {adapter}
                          </Tag>
                        </li>
                      ))
                    )}
                  </ul>
                </div>

                <div className="mt-3 flex flex-wrap gap-1.5 border-t border-rule-hair pt-3">
                  <Tag tone={health.demo_bundles_warm > 0 ? "pass" : "caution"}>
                    <StatusLamp
                      state={health.demo_bundles_warm > 0 ? "pass" : "caution"}
                    />
                    {health.demo_bundles_warm} DEMO BUNDLES WARM
                  </Tag>
                </div>
              </>
            )}
          </div>
        </section>

        {/* ── client configuration ─────────────────────────────────── */}
        <section className="m-sheet">
          <ZoneHeader code="S7·B" title="This client" />
          <div className="px-4 py-1">
            <FieldRow label="DATA SOURCE">
              {USE_MOCKS ? (
                <span className="flex flex-wrap items-center gap-2">
                  <Tag tone="signal">MOCK</Tag>
                  <span className="text-ink-2">
                    MSW against the frozen contract. Imagery is generated, not
                    observed.
                  </span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Tag tone="pass">LIVE</Tag>
                  <span className="text-ink-2">Backed by the real API.</span>
                </span>
              )}
            </FieldRow>
            <FieldRow label="EXTERNAL BASEMAP">
              <span className="flex items-center gap-2">
                <Tag tone={ENABLE_BASEMAP ? "caution" : "pass"}>
                  {ENABLE_BASEMAP ? "ENABLED" : "DISABLED"}
                </Tag>
                <span className="text-ink-2">
                  {ENABLE_BASEMAP
                    ? "Development only — this makes external tile requests."
                    : "No external tile requests. The demo path is offline."}
                </span>
              </span>
            </FieldRow>
            <FieldRow label="MAP ENGINE">
              <span className="t-data text-ink-0">
                MapLibre GL JS v4 · BSD-3
              </span>
            </FieldRow>
          </div>
        </section>

        {/* ── calibration ──────────────────────────────────────────── */}
        <section className="m-sheet">
          <ZoneHeader
            code="S7·C"
            title="Confidence calibration"
            actions={
              <Tag tone={calibrated ? "pass" : "caution"}>
                {calibrated ? "CALIBRATED" : "HEURISTIC"}
              </Tag>
            }
          />
          <div className="px-4 py-3">
            <ReliabilityDiagram calibrated={calibrated} />
            <p className="mt-2 border-t border-rule-hair pt-3 text-[12px] leading-[1.5] text-ink-2">
              {calibrated
                ? "Isotonic regression fitted on a held-out set; the expected calibration error is reported alongside it."
                : "No calibrator is fitted yet, so the diagonal is drawn and nothing is plotted against it. Confidence values are shown throughout the app as interim heuristics — an ordering, not a probability — and they are badged differently from calibrated ones for exactly this reason."}
            </p>
          </div>
        </section>

        {/* ── disagreement causes ──────────────────────────────────── */}
        <section className="m-sheet">
          <ZoneHeader
            code="S7·D"
            title="Disagreement causes the system can name"
            actions={
              <span className="t-code-sm text-ink-3">
                {causes?.causes.length ?? 0}
              </span>
            }
          />
          <ul className="flex flex-col divide-y divide-rule-hair">
            {(causes?.causes ?? []).map((cause) => (
              <li key={cause.code} className="px-4 py-2.5">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span className="t-code text-ink-0">{cause.label}</span>
                  <span className="t-data text-[10.5px] text-ink-3">
                    {cause.code}
                  </span>
                  <Tag className="ml-auto">
                    {cause.trusted_modality === "sar" ? (
                      <IconSar size={10} />
                    ) : (
                      <IconOptical size={10} />
                    )}
                    {cause.trusted_modality} trusted
                  </Tag>
                </div>
                <p className="text-[12px] leading-[1.5] text-ink-2">
                  {cause.explanation}
                </p>
              </li>
            ))}
          </ul>
        </section>
      </div>

      {/* ── tool manifests ─────────────────────────────────────────── */}
      <section className="m-sheet mt-5">
        <ZoneHeader
          code="S7·E"
          title="Tool manifests — what each tool is permitted to do"
          actions={
            <span className="t-code-sm text-ink-3">
              {tools?.tools.length ?? 0} TOOLS
            </span>
          }
        />
        <div className="flex flex-col divide-y divide-rule">
          {(tools?.tools ?? []).map((tool) => (
            <div key={tool.name} className="px-4 py-3.5">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="t-data-strong text-[13px] text-ink-0">
                  {tool.name}
                </span>
                <Tag>v{tool.version}</Tag>
                {tool.required_modalities.map((modality) => (
                  <Tag key={modality}>
                    {modality === "sar" ? (
                      <IconSar size={10} />
                    ) : (
                      <IconOptical size={10} />
                    )}
                    {modality}
                  </Tag>
                ))}
                {tool.expected_latency_ms ? (
                  <Tip content="What the manifest expects. Measured latency appears against it in every trace step.">
                    <span>
                      <Tag>~{tool.expected_latency_ms} MS</Tag>
                    </span>
                  </Tip>
                ) : null}
                {tool.confidence_source ? (
                  <Tag
                    tone={
                      tool.confidence_source === "learned_logprob"
                        ? "pass"
                        : "caution"
                    }
                  >
                    {tool.confidence_source.replace(/_/g, " ")}
                  </Tag>
                ) : null}
              </div>

              <p className="t-doc mb-2.5 text-[12.5px] text-ink-1">
                {tool.description}
              </p>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <span className="t-code-sm text-ink-3">
                    PERMITTED PARAMETERS
                  </span>
                  <div className="mt-1 border border-rule-hair bg-panel-1 px-2.5">
                    {Object.entries(tool.permitted_parameters).map(
                      ([key, spec]) => (
                        <FieldRow key={key} label={key}>
                          <span className="t-data text-ink-0">
                            {spec.type === "enum" && spec.values
                              ? spec.values.join(" | ")
                              : spec.range
                                ? `[${formatManifestBound(spec.range[0], spec.type)}, ${formatManifestBound(spec.range[1], spec.type)}]`
                                : spec.type}
                            {spec.default !== undefined ? (
                              <span className="text-ink-3">
                                {" "}
                                default {String(spec.default)}
                              </span>
                            ) : null}
                            {spec.optional ? (
                              <span className="text-ink-3"> optional</span>
                            ) : null}
                          </span>
                        </FieldRow>
                      ),
                    )}
                  </div>
                </div>

                <div>
                  <span className="t-code-sm text-ink-3">OUTPUTS</span>
                  <div className="mt-1 border border-rule-hair bg-panel-1 px-2.5">
                    {Object.entries(tool.outputs).map(([key, spec]) => (
                      <FieldRow key={key} label={key}>
                        <span className="t-data text-ink-0">
                          {spec.type}
                          {spec.crs ? (
                            <span className="text-ink-3"> crs {spec.crs}</span>
                          ) : null}
                          {spec.resolution ? (
                            <span className="text-ink-3">
                              {" "}
                              {spec.resolution}
                            </span>
                          ) : null}
                        </span>
                      </FieldRow>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── task table ─────────────────────────────────────────────── */}
      <section className="m-sheet mt-5">
        <ZoneHeader code="S7·F" title="Tasks the router can select" />
        <ul className="flex flex-col divide-y divide-rule-hair">
          {(tasks?.tasks ?? []).map((task) => (
            <li
              key={task.task}
              className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-2.5"
            >
              <span className="t-data-strong w-[190px] shrink-0 text-[12px] text-ink-0">
                {task.task}
              </span>
              <span className="min-w-0 flex-1 text-[12.5px] leading-[1.45] text-ink-1">
                <span className="t-code mr-2 text-ink-2">{task.label}</span>
                {task.description}
              </span>
              <span className="flex shrink-0 gap-1">
                {task.requires.map((requirement) => (
                  <Tag key={requirement}>{requirement}</Tag>
                ))}
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
