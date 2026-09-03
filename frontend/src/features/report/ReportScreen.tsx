import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { CreateReportRequest } from "@contracts/types";
import { useBundle } from "@/api/bundles";
import { useQueryResult } from "@/api/queries";
import { reportFileUrl, useCreateReport, useReport } from "@/api/reports";
import { resolveUrl, ApiFailure } from "@/api/client";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import {
  Button,
  Field,
  Skeleton,
  StatusLamp,
  Tag,
  WarningList,
  ZoneHeader,
} from "@/components/primitives";
import { IconDownload, IconFile, IconPrint } from "@/components/icons";
import { TracePanel } from "@/features/trace/TracePanel";
import { formatArea, formatDuration, formatTimestamp } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * S6 — report.
 *
 * The page previews exactly what the PDF contains, and the print stylesheet
 * makes Ctrl+P produce the same document. That is deliberate: PDF generation
 * is on the plan's cut list, and if it is cut this page still delivers the
 * artifact rather than a dead button.
 */

type Include = NonNullable<CreateReportRequest["include"]>[number];

const SECTIONS: { value: Include; label: string; detail: string }[] = [
  {
    value: "evidence",
    label: "Evidence",
    detail: "Masks, overlays and their area statistics.",
  },
  {
    value: "trace",
    label: "Trace",
    detail: "The graded block and every step, verbatim.",
  },
  {
    value: "warnings",
    label: "Warnings",
    detail: "Compatibility findings raised during ingest and execution.",
  },
];

export function ReportScreen() {
  const { bundleId, queryId } = useParams<{
    bundleId: string;
    queryId: string;
  }>();
  const { data: bundle } = useBundle(bundleId ?? null);
  const { data: query, isLoading } = useQueryResult(queryId ?? null);
  const createReport = useCreateReport();
  const [reportId, setReportId] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [include, setInclude] = useState<Include[]>([
    "evidence",
    "trace",
    "warnings",
  ]);
  const { data: report } = useReport(
    reportId,
    Boolean(reportId) && !failure,
  );

  const generate = async () => {
    if (!queryId) return;
    setFailure(null);
    try {
      const created = await createReport.mutateAsync({
        queryId,
        body: { include, format: "pdf" },
      });
      setReportId(created.report_id);
    } catch (error) {
      setFailure(
        error instanceof ApiFailure
          ? `${error.error.message}${error.error.hint ? ` ${error.error.hint}` : ""}`
          : "The report job could not be started.",
      );
    }
  };

  if (isLoading || !query) {
    return (
      <div className="mx-auto w-full max-w-[900px] px-6 py-8">
        <Skeleton className="mb-3 h-8 w-72" />
        <Skeleton className="h-72 w-full" />
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-[900px] px-6 pb-16 pt-8">
      <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-2 no-print">
        <Link
          to={`/workspace/${bundleId}`}
          className="t-code-sm text-ink-3 transition-colors hover:text-ink-0"
        >
          ← WORKSPACE
        </Link>
        <h1 className="t-plate text-[20px] text-ink-0">Report</h1>
        <Tag className="ml-auto">S6</Tag>
      </div>

      {/* ── the document itself ─────────────────────────────────────── */}
      <article className="m-sheet mb-5">
        <header className="border-b border-rule px-5 py-4">
          <p className="t-code-sm mb-2 text-ink-3">
            SATQUERY AI · SIH26167 · ISRO / SAC
          </p>
          <h2 className="t-plate mb-3 text-[17px] text-ink-0">
            {query.question}
          </h2>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-4">
            <Field label="BUNDLE" value={bundle?.label ?? query.bundle_id} mono={false} />
            <Field label="QUERY ID" value={query.query_id} />
            <Field label="PRODUCED" value={formatTimestamp(query.created_at)} />
            <Field
              label="QUERY LATENCY"
              value={formatDuration(query.latency_ms)}
              tone={
                (query.latency_ms ?? 0) > 20_000 ? "signal" : "pass"
              }
              title={query.latency_ms ? `${query.latency_ms} ms` : undefined}
            />
          </div>
        </header>

        <section className="border-b border-rule px-5 py-4">
          <h3 className="t-code mb-2 text-ink-2">Answer</h3>
          {query.answer ? (
            <p className="t-doc text-[14px] text-ink-0">{query.answer}</p>
          ) : query.refusal ? (
            <div className="border border-advisory/35 bg-advisory-wash px-3 py-2.5">
              <Tag tone="advisory" className="mb-2">
                REFUSED · {query.refusal.category}
              </Tag>
              <p className="t-doc text-[13px] text-ink-0">
                {query.refusal.reason}
              </p>
            </div>
          ) : null}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <ConfidenceBadge
              value={query.confidence}
              basis={query.confidence_basis}
            />
            <Tag>{query.trace.graded.task_selected}</Tag>
            {query.trace.router_path ? (
              <Tag>ROUTED BY {query.trace.router_path}</Tag>
            ) : null}
          </div>
        </section>

        {include.includes("evidence") && query.evidence.length > 0 ? (
          <section className="border-b border-rule px-5 py-4 print-break">
            <h3 className="t-code mb-3 text-ink-2">
              Evidence · {query.evidence.length}
            </h3>
            <ul className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {query.evidence.map((asset) => (
                <li key={asset.asset_id} className="border border-rule-hair">
                  <figure className="m-window aspect-[4/3] overflow-hidden">
                    {asset.overlay_url ? (
                      <img
                        src={resolveUrl(asset.overlay_url)}
                        alt={`${asset.label} — synthetic imagery, not observed data`}
                        loading="lazy"
                        className="h-full w-full object-cover"
                        style={{ imageRendering: "pixelated" }}
                      />
                    ) : null}
                  </figure>
                  <div className="flex flex-col gap-1.5 p-2.5">
                    <span className="flex items-start gap-1.5">
                      <span
                        aria-hidden="true"
                        className="mt-[3px] h-[11px] w-[11px] shrink-0 border border-ink-0"
                        style={{ background: asset.colour ?? "#3BA3F2" }}
                      />
                      <span className="text-[12px] leading-[1.35] text-ink-0">
                        {asset.label}
                      </span>
                    </span>
                    <Field
                      label="AREA"
                      value={formatArea(asset.stats?.area_km2)}
                    />
                    <a
                      href={resolveUrl(asset.download_url)}
                      download
                      className="t-code-sm inline-flex items-center gap-1 text-ink-2 underline decoration-rule underline-offset-[3px] hover:text-ink-0 no-print"
                    >
                      <IconDownload size={10} />
                      GeoTIFF (opens in QGIS)
                    </a>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {include.includes("warnings") && query.warnings.length > 0 ? (
          <section className="border-b border-rule px-5 py-4 print-break">
            <WarningList warnings={query.warnings} />
          </section>
        ) : null}

        {include.includes("trace") ? (
          <section className="print-break">
            <h3 className="t-code border-b border-rule bg-panel-1 px-5 py-2.5 text-ink-2">
              Trace · the graded artifact
            </h3>
            <div className="max-h-[560px] overflow-y-auto print:max-h-none print:overflow-visible">
              <TracePanel trace={query.trace} />
            </div>
          </section>
        ) : null}
      </article>

      {/* ── generation controls ─────────────────────────────────────── */}
      <section className="m-sheet p-4 no-print">
        <ZoneHeader
          code="S6·G"
          title="Generate"
          className="-mx-4 -mt-4 mb-4 border-t-0"
        />
        <fieldset className="mb-4">
          <legend className="t-code-sm mb-2 text-ink-3">
            SECTIONS TO INCLUDE
          </legend>
          <div className="flex flex-col gap-1.5">
            {SECTIONS.map((section) => (
              <label
                key={section.value}
                className="flex cursor-pointer items-start gap-2.5 border border-rule-hair bg-panel-1 px-2.5 py-2 transition-colors hover:border-rule"
              >
                <input
                  type="checkbox"
                  checked={include.includes(section.value)}
                  onChange={(event) =>
                    setInclude((current) =>
                      event.target.checked
                        ? [...current, section.value]
                        : current.filter((item) => item !== section.value),
                    )
                  }
                  className="mt-[3px] h-[13px] w-[13px] shrink-0 accent-[#dc4a1e]"
                />
                <span className="min-w-0">
                  <span className="t-code block text-ink-0">
                    {section.label}
                  </span>
                  <span className="mt-0.5 block text-[12px] leading-[1.4] text-ink-2">
                    {section.detail}
                  </span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            onClick={() => void generate()}
            disabled={createReport.isPending || include.length === 0}
            icon={<IconFile size={13} />}
          >
            {createReport.isPending ? "Starting…" : "Generate PDF"}
          </Button>

          <Button
            variant="panel"
            onClick={() => window.print()}
            icon={<IconPrint size={12} />}
          >
            Print this page
          </Button>

          {report?.state === "succeeded" && report.file_url ? (
            <a
              href={reportFileUrl(report.report_id)}
              download
              className="t-code inline-flex items-center gap-1.5 border border-ink-0 bg-ink-0 px-2.5 py-[7px] text-panel-2 transition-colors hover:bg-[#20282a]"
            >
              <IconDownload size={12} />
              Download PDF
            </a>
          ) : null}

          {report && report.state !== "succeeded" ? (
            <span className="t-code-sm flex items-center gap-1.5 text-ink-2">
              <StatusLamp
                state={report.state === "failed" ? "fail" : "active"}
              />
              {report.state}
            </span>
          ) : null}
        </div>

        <p
          className={cn(
            "mt-3 border-t border-rule-hair pt-3 text-[12px] leading-[1.5]",
            failure ? "text-signal-ink" : "text-ink-2",
          )}
        >
          {failure ??
            "PDF rendering runs in the backend. If it is unavailable, Print produces the same document from this page — the print stylesheet is built for it, not an afterthought."}
        </p>
      </section>
    </div>
  );
}
