import * as Slider from "@radix-ui/react-slider";
import type { Bundle } from "@contracts/types";
import { resolveUrl } from "@/api/client";
import {
  IconDownload,
  IconGrid,
  IconHidden,
  IconOptical,
  IconSar,
  IconVisible,
} from "@/components/icons";
import {
  EmptyState,
  Field,
  IconButton,
  Tag,
  Tip,
  WarningList,
  ZoneHeader,
} from "@/components/primitives";
import { bundlePanes } from "@/map/panes";
import { useWorkspace } from "@/store/workspace";
import { formatArea, formatBytes } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * Left pane — layers and evidence.
 *
 * Every produced asset gets a row carrying the four things that make it
 * checkable: what it is, which tool made it, what it measures, and how to
 * take it away as a file. The "produced by" caption is a control: it selects
 * that tool's step in the trace, so a number on the map is never more than
 * one click from the parameters that produced it.
 */

function OpacityControl({
  value,
  onChange,
  label,
}: {
  value: number;
  onChange: (next: number) => void;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <Slider.Root
        className="relative flex h-4 flex-1 touch-none select-none items-center"
        value={[Math.round(value * 100)]}
        onValueChange={([next]) => onChange(next / 100)}
        max={100}
        step={1}
        aria-label={label}
      >
        <Slider.Track className="relative h-[5px] flex-1 border border-rule bg-panel-sunk">
          <Slider.Range className="absolute h-full bg-ink-1" />
        </Slider.Track>
        <Slider.Thumb className="block h-[15px] w-[9px] border border-ink-0 bg-plate-1 shadow-[inset_0_1px_0_rgb(255_255_255/0.7)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal" />
      </Slider.Root>
      <span className="t-data w-[34px] shrink-0 text-right text-[11px] text-ink-2">
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

export function EvidencePanel({ bundle }: { bundle: Bundle }) {
  const layers = useWorkspace((s) => s.layers);
  const setLayerVisible = useWorkspace((s) => s.setLayerVisible);
  const setLayerOpacity = useWorkspace((s) => s.setLayerOpacity);
  const baseLayer = useWorkspace((s) => s.baseLayer);
  const setBaseLayer = useWorkspace((s) => s.setBaseLayer);
  const baseOpacity = useWorkspace((s) => s.baseOpacity);
  const setBaseOpacity = useWorkspace((s) => s.setBaseOpacity);
  const compareMode = useWorkspace((s) => s.compareMode);
  const setCompareMode = useWorkspace((s) => s.setCompareMode);
  const showTileGrid = useWorkspace((s) => s.showTileGrid);
  const setShowTileGrid = useWorkspace((s) => s.setShowTileGrid);
  const setTab = useWorkspace((s) => s.setTab);
  const selectQuery = useWorkspace((s) => s.selectQuery);
  const selectStep = useWorkspace((s) => s.selectStep);

  const georeferenced = bundle.scenes.some((s) => s.compatibility?.crs_valid);
  const canCompare = bundle.scenes.length > 1;
  const { paneB } = bundlePanes(bundle);

  return (
    <aside className="flex h-full min-h-0 flex-col border-r border-rule bg-panel-1">
      <ZoneHeader title="Layers & evidence" />

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* ── base imagery ─────────────────────────────────────────── */}
        <section className="border-b border-rule px-3 py-3">
          <h3 className="t-code-sm mb-2 text-ink-3">BASE IMAGERY</h3>
          <div className="mb-2.5 flex flex-col gap-1">
            {bundle.scenes.map((scene) => {
              const sar = scene.compatibility?.modality === "sar";
              // The switch selects a pane, not a modality: the second date of
              // a bi-temporal pair is optical too, and still has to be pane B.
              const second = scene.scene_id === paneB?.scene_id;
              const active =
                baseLayer !== "none" &&
                (!canCompare || compareMode !== "single"
                  ? true
                  : second
                    ? baseLayer === "sar"
                    : baseLayer === "optical");
              return (
                <button
                  key={scene.scene_id}
                  type="button"
                  onClick={() => {
                    // Picking one scene is asking for the single view.
                    setBaseLayer(second ? "sar" : "optical");
                    setCompareMode("single");
                  }}
                  aria-pressed={active}
                  className={cn(
                    "flex w-full items-center gap-2 border px-2 py-1.5 text-left transition-colors",
                    // A signal-coloured edge and a SHOWN tag, so which scene is
                    // on screen reads at a glance and not from a border weight.
                    active
                      ? "border-ink-0 bg-signal-wash shadow-[inset_3px_0_0_var(--color-signal)]"
                      : "border-rule-hair bg-panel-1 hover:border-rule hover:bg-panel-2",
                  )}
                >
                  {sar ? (
                    <IconSar size={13} className="shrink-0 text-ink-1" />
                  ) : (
                    <IconOptical size={13} className="shrink-0 text-ink-1" />
                  )}
                  <span className="min-w-0 flex-1">
                    <span className="t-code block truncate text-ink-0">
                      {scene.role ?? scene.compatibility?.modality ?? "scene"}
                    </span>
                    <span className="t-data block truncate text-[11px] text-ink-2">
                      {scene.filename}
                    </span>
                  </span>
                  {scene.compatibility?.native_gsd_m ? (
                    <span className="t-code-sm shrink-0 text-ink-3">
                      {scene.compatibility.native_gsd_m} M
                    </span>
                  ) : null}
                  {active && canCompare ? (
                    <span className="t-code-sm shrink-0 border border-signal bg-signal px-1 py-[1px] text-white">
                      SHOWN
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
          <OpacityControl
            value={baseOpacity}
            onChange={setBaseOpacity}
            label="Base imagery opacity"
          />

          <div className="mt-3 flex flex-wrap gap-1.5">
            {bundle.tiles && georeferenced ? (
              <Tip
                content={`${bundle.tiles.tile_count} tiles of ${bundle.tiles.tile_size_px} px with ${Math.round(bundle.tiles.overlap_frac * 100)}% overlap — the partition the pipeline actually reasons over.`}
              >
                <button
                  type="button"
                  onClick={() => setShowTileGrid(!showTileGrid)}
                  aria-pressed={showTileGrid}
                  className={cn(
                    "t-code-sm flex items-center gap-1.5 border px-1.5 py-[5px] transition-colors",
                    showTileGrid
                      ? "border-signal bg-signal text-white"
                      : "border-rule bg-panel-2 text-ink-1 hover:border-rule-heavy",
                  )}
                >
                  <IconGrid size={12} />
                  TILE GRID
                </button>
              </Tip>
            ) : null}
          </div>
        </section>

        {/* ── produced evidence ────────────────────────────────────── */}
        <section className="px-3 py-3">
          <h3 className="t-code-sm mb-2 flex items-baseline justify-between text-ink-3">
            <span>PRODUCED EVIDENCE</span>
            <span>{layers.length}</span>
          </h3>

          {layers.length === 0 ? (
            <EmptyState code="NO ASSETS" title="Nothing produced yet">
              Masks, overlays and grounded boxes appear here as the tools
              produce them — each with its area, its source tool, and a
              GeoTIFF download.
            </EmptyState>
          ) : (
            <ul className="flex flex-col gap-2">
              {layers.map((layer) => {
                const { asset } = layer;
                return (
                  <li
                    key={asset.asset_id}
                    className="m-sheet flex flex-col gap-2 p-2.5"
                  >
                    <div className="flex items-start gap-2">
                      <span
                        aria-hidden="true"
                        className="mt-[3px] h-[13px] w-[13px] shrink-0 border border-ink-0"
                        style={{ background: asset.colour ?? "#3BA3F2" }}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12.5px] font-medium leading-[1.35] text-ink-0">
                          {asset.label}
                        </span>
                        <button
                          type="button"
                          onClick={() => {
                            selectQuery(layer.queryId);
                            selectStep(null);
                            setTab("trace");
                          }}
                          className="t-code-sm mt-[3px] inline-flex items-center gap-1 text-ink-2 underline decoration-rule underline-offset-[3px] transition-colors hover:text-signal-ink hover:decoration-signal"
                        >
                          PRODUCED BY {asset.produced_by}
                        </button>
                      </span>
                      <IconButton
                        label={
                          layer.visible ? "Hide this layer" : "Show this layer"
                        }
                        onClick={() =>
                          setLayerVisible(asset.asset_id, !layer.visible)
                        }
                        className="shrink-0"
                      >
                        {layer.visible ? (
                          <IconVisible size={15} />
                        ) : (
                          <IconHidden size={15} />
                        )}
                      </IconButton>
                    </div>

                    <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                      <Field
                        label="AREA"
                        value={formatArea(asset.stats?.area_km2)}
                        title={
                          asset.stats?.area_km2 !== undefined
                            ? `${asset.stats.area_km2} km² · reported by ${asset.produced_by}`
                            : undefined
                        }
                      />
                      <Field
                        label="CRS"
                        value={asset.crs ?? "pixel space"}
                        tone={asset.crs ? "default" : "caution"}
                      />
                    </div>

                    <OpacityControl
                      value={layer.opacity}
                      onChange={(next) => setLayerOpacity(asset.asset_id, next)}
                      label={`${asset.label} opacity`}
                    />

                    <div className="flex items-center justify-between gap-2 border-t border-rule-hair pt-2">
                      <span className="t-code-sm text-ink-3">
                        {formatBytes(asset.bytes)}
                      </span>
                      <a
                        href={resolveUrl(asset.download_url)}
                        download
                        className="t-code-sm inline-flex items-center gap-1.5 border border-rule-heavy bg-plate-1 px-2 py-[5px] text-ink-0 transition-colors hover:bg-plate-0"
                      >
                        <IconDownload size={12} />
                        {asset.media_type === "image/tiff"
                          ? "GeoTIFF (opens in QGIS)"
                          : "Download"}
                      </a>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        {/* ── compatibility findings ───────────────────────────────── */}
        {bundle.warnings.length ? (
          <section className="border-t border-rule px-3 py-3">
            <WarningList warnings={bundle.warnings} />
          </section>
        ) : null}

        {/* ── provenance ───────────────────────────────────────────── */}
        {bundle.provenance.length ? (
          <section className="border-t border-rule px-3 py-3">
            <h3 className="t-code-sm mb-2 text-ink-3">PREPARATION RECORD</h3>
            <ol className="flex flex-col">
              {bundle.provenance.map((step, index) => (
                <li
                  key={`${step.stage}-${index}`}
                  className="flex items-baseline gap-2 border-b border-rule-hair py-1.5 last:border-b-0"
                >
                  <span className="t-code-sm w-[104px] shrink-0 text-ink-3">
                    {step.stage.replace("_", " ")}
                  </span>
                  <span className="t-data min-w-0 flex-1 text-[11.5px] text-ink-1">
                    {step.op}
                    {Object.keys(step.params).length ? (
                      <span className="text-ink-3">
                        {" "}
                        {JSON.stringify(step.params)}
                      </span>
                    ) : null}
                  </span>
                </li>
              ))}
            </ol>
            {bundle.pair_compatibility?.coregistered ? (
              <Tag tone="pass" className="mt-2">
                COREG {bundle.pair_compatibility.method} · RMSE{" "}
                {bundle.pair_compatibility.rmse_px} PX
              </Tag>
            ) : null}
          </section>
        ) : null}
      </div>
    </aside>
  );
}
