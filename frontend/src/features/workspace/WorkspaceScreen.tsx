import { useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import * as Tabs from "@radix-ui/react-tabs";
import { useBundle } from "@/api/bundles";
import { useQueryResult } from "@/api/queries";
import { ChatPane } from "@/features/chat/ChatPane";
import { EvidencePanel } from "@/features/evidence/EvidencePanel";
import { TracePanel } from "@/features/trace/TracePanel";
import { SceneMap } from "@/map/SceneMap";
import { ImageViewer } from "@/map/ImageViewer";
import {
  Skeleton,
  StatusLamp,
  Tag,
  Tip,
  ZoneHeader,
} from "@/components/primitives";
import { IconFile, IconPrint } from "@/components/icons";
import { useRuns } from "@/store/runs";
import { useWorkspace } from "@/store/workspace";
import { formatDuration } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * S4 — the workspace.
 *
 * Three panes: evidence on the left, the imagery window in the centre, chat
 * and trace on the right. The map is the dominant field and the panels are
 * subordinate instruments around it — a judge should be looking at imagery,
 * not at chrome.
 */
export function WorkspaceScreen() {
  const { bundleId } = useParams<{ bundleId: string }>();
  const { data: bundle, isLoading } = useBundle(bundleId ?? null);

  const tab = useWorkspace((s) => s.tab);
  const setTab = useWorkspace((s) => s.setTab);
  const selectedQueryId = useWorkspace((s) => s.selectedQueryId);
  const clearLayers = useWorkspace((s) => s.clearLayers);
  const runs = useRuns((s) => s.runs);

  // Layers belong to a bundle; switching bundles must not carry evidence
  // from the previous one onto a different piece of ground.
  useEffect(() => {
    clearLayers();
  }, [bundleId, clearLayers]);

  const liveRun = selectedQueryId ? runs[selectedQueryId] : undefined;
  const { data: persisted } = useQueryResult(
    liveRun && liveRun.finishedAt !== null ? selectedQueryId : null,
  );
  const trace = persisted?.trace;

  const georeferenced = useMemo(
    () => bundle?.scenes.some((scene) => scene.compatibility?.crs_valid) ?? false,
    [bundle],
  );

  const lastLatency = useMemo(() => {
    const finished = Object.values(runs)
      .filter((run) => run.bundleId === bundleId && run.totalLatencyMs !== null)
      .sort((a, b) => (b.finishedAt ?? 0) - (a.finishedAt ?? 0));
    return finished[0]?.totalLatencyMs ?? null;
  }, [runs, bundleId]);

  if (isLoading || !bundle) {
    return (
      <div className="grid flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-[290px_1fr_400px]">
        <Skeleton className="h-full min-h-[300px]" />
        <Skeleton className="h-full min-h-[300px]" />
        <Skeleton className="h-full min-h-[300px]" />
      </div>
    );
  }

  const nonGeoScene = bundle.scenes.find(
    (scene) => scene.compatibility?.crs_valid === false,
  );

  return (
    /* Viewport-bounded on large screens. The shell is `min-h-dvh`, so a tall
       child lengthens the whole page -- and the image viewer sizes to the
       image's own aspect ratio, so a portrait screenshot pushed the chat box
       hundreds of pixels below the fold. Panes scroll internally instead.
       Small screens keep the stacked, scrolling layout, where that is right. */
    <div className="flex min-h-0 flex-1 flex-col lg:h-[calc(100dvh-var(--identity-plate-h,44px))] lg:overflow-hidden">
      {/* ── bundle bar: the two SLAs stay on screen ─────────────────── */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-rule bg-panel-1 px-4 py-2 no-print">
        <Link
          to="/"
          className="t-code-sm text-ink-3 transition-colors hover:text-ink-0"
        >
          ← SESSIONS
        </Link>
        <h1 className="t-plate text-[13px] text-ink-0">
          {bundle.label ?? "Untitled bundle"}
        </h1>
        <Tag tone={bundle.status === "ready" ? "pass" : "caution"}>
          <StatusLamp state={bundle.status === "ready" ? "pass" : "caution"} />
          {bundle.status}
        </Tag>
        <Tag>{bundle.pair_type}</Tag>

        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <Tip content="Measured preparation time for this bundle — the slow clock, paid once. Budget is 5 minutes per scene.">
            <span className="flex items-baseline gap-1.5">
              <span className="t-code-sm text-ink-3">PREP</span>
              <span
                className="t-data-strong text-[12px] text-pass-ink"
                title={bundle.prep_ms ? `${bundle.prep_ms} ms` : undefined}
              >
                {formatDuration(bundle.prep_ms)}
              </span>
            </span>
          </Tip>
          <Tip content="Measured latency of the most recent question against this prepared bundle. Budget is 20 seconds.">
            <span className="flex items-baseline gap-1.5">
              <span className="t-code-sm text-ink-3">LAST QUERY</span>
              <span
                className={cn(
                  "t-data-strong text-[12px]",
                  lastLatency === null
                    ? "text-ink-3"
                    : lastLatency > 20_000
                      ? "text-signal-ink"
                      : "text-pass-ink",
                )}
                title={lastLatency ? `${lastLatency} ms` : undefined}
              >
                {lastLatency === null ? "—" : formatDuration(lastLatency)}
              </span>
            </span>
          </Tip>
          {selectedQueryId ? (
            <Link
              to={`/workspace/${bundle.bundle_id}/report/${selectedQueryId}`}
              className="t-code inline-flex items-center gap-1.5 border border-rule-heavy bg-plate-1 px-2.5 py-[6px] text-ink-0 transition-colors hover:bg-plate-0"
            >
              <IconFile size={12} />
              Report
            </Link>
          ) : null}
        </div>
      </div>

      {/* ── three panes ─────────────────────────────────────────────── */}
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[288px_minmax(0,1fr)_408px]">
        <div className="order-2 min-h-[320px] lg:order-1 lg:min-h-0">
          <EvidencePanel bundle={bundle} />
        </div>

        <div className="order-1 flex min-h-[380px] flex-col border-b border-rule lg:order-2 lg:min-h-0 lg:border-b-0">
          <ZoneHeader
            code="S4·C"
            title={
              georeferenced
                ? `Imagery window · ${bundle.pair_compatibility?.common_crs ?? "WGS84"}`
                : "Image viewer · pixel space"
            }
            className="no-print"
          />
          <div className="armature-field relative min-h-0 flex-1">
            {georeferenced ? (
              <SceneMap bundle={bundle} />
            ) : nonGeoScene ? (
              <ImageViewer scene={nonGeoScene} />
            ) : null}
          </div>
        </div>

        <div className="order-3 flex min-h-[420px] flex-col border-l border-rule lg:min-h-0">
          <Tabs.Root
            value={tab}
            onValueChange={(value) => setTab(value as "chat" | "trace")}
            className="flex min-h-0 flex-1 flex-col"
          >
            <Tabs.List className="flex shrink-0 border-b border-rule bg-panel-1 no-print">
              {(
                [
                  { value: "chat", code: "S4·R", label: "Chat" },
                  { value: "trace", code: "S5", label: "Trace" },
                ] as const
              ).map((item) => (
                <Tabs.Trigger
                  key={item.value}
                  value={item.value}
                  className={cn(
                    "t-code flex items-center gap-1.5 border-r border-rule px-3 py-2.5 transition-colors",
                    "data-[state=active]:bg-panel-2 data-[state=active]:text-ink-0",
                    "data-[state=inactive]:text-ink-2 data-[state=inactive]:hover:bg-panel-2",
                  )}
                >
                  <span className="opacity-55">{item.code}</span>
                  {item.label}
                </Tabs.Trigger>
              ))}
              <span className="flex flex-1 items-center justify-end px-2">
                <button
                  type="button"
                  onClick={() => window.print()}
                  title="Print this query — the print stylesheet produces the same content as the PDF report"
                  className="t-code-sm inline-flex items-center gap-1.5 border border-rule px-1.5 py-[4px] text-ink-2 transition-colors hover:border-rule-heavy hover:text-ink-0"
                >
                  <IconPrint size={11} />
                  PRINT
                </button>
              </span>
            </Tabs.List>

            <Tabs.Content
              value="chat"
              className="min-h-0 flex-1 data-[state=inactive]:hidden"
              forceMount
            >
              <ChatPane bundle={bundle} />
            </Tabs.Content>

            <Tabs.Content value="trace" className="min-h-0 flex-1">
              <TracePanel trace={trace} />
            </Tabs.Content>
          </Tabs.Root>
        </div>
      </div>
    </div>
  );
}
