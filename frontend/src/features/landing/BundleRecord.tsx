import { useNavigate } from "react-router-dom";
import type { Bundle, Scene } from "@contracts/types";
import { resolveUrl } from "@/api/client";
import {
  IconArrowRight,
  IconOptical,
  IconSar,
} from "@/components/icons";
import {
  Button,
  Field,
  StatusLamp,
  Tag,
  Tip,
} from "@/components/primitives";
import { formatDate, formatDuration } from "@/lib/format";
import { cn } from "@/lib/cn";

/**
 * A specimen record.
 *
 * Not a card. The imagery sits in a dark window on the left, the way it does
 * on the instrument, and everything printed about the specimen sits in the
 * marginalia on the right — the sheet's own convention. A judge should be
 * able to read the whole provenance of a bundle without opening it.
 */

function ScenePane({ scene, span }: { scene: Scene; span: boolean }) {
  const modality = scene.compatibility?.modality ?? "unknown";
  const geo = scene.compatibility?.crs_valid ?? false;

  return (
    <figure
      className={cn(
        "relative min-w-0 overflow-hidden bg-window-0",
        span ? "col-span-2" : "",
      )}
    >
      {/* Positioned, so a square preview cannot drive the record's height
          from its own intrinsic size and leave the marginalia short. */}
      <img
        src={resolveUrl(scene.preview_url)}
        alt={`Preview of ${scene.filename} — synthetic imagery, not observed data`}
        loading="lazy"
        decoding="async"
        className="absolute inset-0 h-full w-full object-cover"
      />
      <figcaption className="absolute inset-x-0 bottom-0 flex items-center gap-1.5 bg-gradient-to-t from-window-0 via-window-0/85 to-transparent px-2 pb-1.5 pt-5">
        {modality === "sar" ? (
          <IconSar size={12} className="shrink-0 text-window-ink-2" />
        ) : (
          <IconOptical size={12} className="shrink-0 text-window-ink-2" />
        )}
        <span className="t-code-sm truncate text-window-ink">
          {scene.role ?? modality}
        </span>
        {!geo ? (
          <span className="t-code-sm ml-auto shrink-0 border border-caution/60 px-1 text-caution">
            NO CRS
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}

export function BundleRecord({
  bundle,
  index,
}: {
  bundle: Bundle;
  index: number;
}) {
  const navigate = useNavigate();
  const scenes = bundle.scenes;
  const crossmodal = bundle.pair_type === "crossmodal";
  const geoValid = scenes.some((s) => s.compatibility?.crs_valid);
  const gsd = scenes
    .map((s) => s.compatibility?.native_gsd_m)
    .filter((v): v is number => typeof v === "number");

  const open = () => navigate(`/workspace/${bundle.bundle_id}`);

  return (
    <article className="m-sheet armature-field group relative grid grid-cols-1 md:grid-cols-[minmax(200px,264px)_1fr]">
      {/* specimen number, in the sheet's margin */}
      <span className="t-code-sm absolute -top-[1px] left-0 z-10 border border-ink-0 bg-ink-0 px-1.5 py-[3px] text-panel-2">
        {String(index + 1).padStart(2, "0")}
      </span>

      {/* the imagery window */}
      <div className="m-window grid aspect-[4/3] grid-cols-2 gap-px md:aspect-auto md:min-h-[178px]">
        {scenes.map((scene) => (
          <ScenePane
            key={scene.scene_id}
            scene={scene}
            span={scenes.length === 1}
          />
        ))}
      </div>

      {/* the marginalia */}
      <div className="flex min-w-0 flex-col gap-3 p-4">
        <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
          <h3 className="t-plate min-w-0 flex-1 text-[15px] text-ink-0">
            {bundle.label ?? "Untitled bundle"}
          </h3>
          <Tag tone={bundle.status === "ready" ? "pass" : "default"}>
            <StatusLamp
              state={bundle.status === "ready" ? "pass" : "idle"}
            />
            {bundle.status}
          </Tag>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-4">
          <Field label="PAIR TYPE" value={bundle.pair_type} mono={false} />
          <Field
            label="PREP TIME"
            value={formatDuration(bundle.prep_ms)}
            title={bundle.prep_ms ? `${bundle.prep_ms} ms` : undefined}
          />
          <Field
            label="COMMON CRS"
            value={
              bundle.pair_compatibility?.common_crs ??
              (geoValid ? "single scene" : "not georeferenced")
            }
            tone={geoValid ? "default" : "caution"}
          />
          <Field
            label="NATIVE GSD"
            value={gsd.length ? `${gsd.join(" / ")} m` : "—"}
          />
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {crossmodal && bundle.pair_compatibility ? (
            <Tip
              content={`Coregistered by ${bundle.pair_compatibility.method}. Checks passed: ${bundle.pair_compatibility.checks_passed.join(", ")}.`}
            >
              <span>
                <Tag tone="pass">
                  COREG RMSE {bundle.pair_compatibility.rmse_px} PX
                </Tag>
              </span>
            </Tip>
          ) : null}
          {bundle.tiles ? (
            <Tag>
              {bundle.tiles.tile_count} TILES · {bundle.tiles.tile_size_px} PX
            </Tag>
          ) : null}
          <Tag>{bundle.supported_tasks.length} TASKS AVAILABLE</Tag>
          {bundle.blocked_tasks.length ? (
            <Tip
              content={
                <span className="flex flex-col gap-1">
                  {bundle.blocked_tasks.map((blocked) => (
                    <span key={blocked.task}>
                      <span className="t-data">{blocked.task}</span> —{" "}
                      {blocked.reason}
                    </span>
                  ))}
                </span>
              }
            >
              <span>
                <Tag tone="caution">
                  {bundle.blocked_tasks.length} BLOCKED
                </Tag>
              </span>
            </Tip>
          ) : null}
        </div>

        {bundle.warnings.length ? (
          <p className="flex gap-1.5 border-t border-rule-hair pt-2.5 text-[12px] leading-[1.45] text-ink-2">
            <span aria-hidden="true" className="text-caution">
              △
            </span>
            {bundle.warnings[0]}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Button variant="primary" onClick={open} icon={<IconArrowRight size={13} />}>
            Open workspace
          </Button>
          <span className="t-code-sm text-ink-3">
            {scenes.length} scene{scenes.length === 1 ? "" : "s"} ·{" "}
            {formatDate(bundle.created_at)}
          </span>
        </div>
      </div>
    </article>
  );
}
