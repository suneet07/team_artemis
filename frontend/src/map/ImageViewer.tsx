import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import type { Scene } from "@contracts/types";
import { resolveUrl } from "@/api/client";
import { useWorkspace, type LayerState } from "@/store/workspace";
import { cn } from "@/lib/cn";
import { SwipeDivider } from "./SwipeDivider";

interface Transform {
  x: number;
  y: number;
  scale: number;
}

type PaneKey = "a" | "b";

/**
 * One scene with its evidence, under the viewer's transform.
 *
 * Every frame is laid out from the same top-left origin and given the same
 * transform, which is what holds two scenes at the same position: pan or zoom
 * one and the other is looking at the same pixels.
 */
function SceneFrame({
  scene,
  pane,
  fill,
  transform,
  layers,
  baseOpacity,
}: {
  scene: Scene;
  pane: PaneKey;
  /**
   * Draw the scene at the full width of its viewport, whatever its own pixel
   * size. Two scenes of one pair are only at "the same position" if they are
   * also at the same scale, and their previews need not be the same size.
   */
  fill: boolean;
  transform: Transform;
  layers: LayerState[];
  baseOpacity: number;
}) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(
    null,
  );

  const boxes = layers.flatMap((layer) =>
    (layer.asset.bbox_px ?? []).map((box, index) => ({
      box,
      colour: layer.asset.colour ?? "#dc4a1e",
      label: layer.asset.label,
      key: `${layer.asset.asset_id}-${index}`,
      visible: layer.visible,
    })),
  );

  return (
    <div
      /* `w-fit` is load-bearing, not cosmetic. Masks and grounding boxes are
         absolutely positioned as a percentage of THIS element, so it has to
         be exactly the size of the image. While the image was `w-full` the
         two matched by accident; bounding the image to the pane left this
         wrapper full-width, and every box landed to the right of the
         picture it was describing. */
      className={cn(
        "relative origin-top-left will-change-transform",
        // Either way this element is exactly the size of the image.
        fill ? "w-full" : "w-fit",
      )}
      style={{
        transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
      }}
    >
      <img
        src={resolveUrl(scene.preview_url)}
        alt={`${scene.filename} — synthetic imagery, not observed data`}
        draggable={false}
        data-pane={pane}
        onLoad={(event) =>
          setNaturalSize({
            w: event.currentTarget.naturalWidth,
            h: event.currentTarget.naturalHeight,
          })
        }
        /* `w-full` alone lets a portrait image set the pane's height from
           its aspect ratio. Bounded by the pane and centred instead, so the
           viewer fills the space it is given rather than the space the
           image wants. Pan and zoom still work -- the transform wrapper is
           unchanged. */
        className={cn(
          "block select-none",
          fill ? "h-auto w-full" : "max-h-full w-auto max-w-full",
        )}
        style={{ opacity: baseOpacity, imageRendering: "pixelated" }}
      />

      {/* masks composited in pixel space */}
      {layers
        .filter((layer) => layer.visible && layer.asset.overlay_url)
        .map((layer) => (
          <img
            key={layer.asset.asset_id}
            src={resolveUrl(layer.asset.overlay_url!)}
            alt={layer.asset.label}
            draggable={false}
            className="pointer-events-none absolute inset-0 block h-full w-full select-none"
            style={{
              opacity: layer.opacity,
              imageRendering: "pixelated",
            }}
          />
        ))}

      {/* grounding boxes, in the source's own pixel coordinates */}
      {naturalSize
        ? boxes
            .filter((entry) => entry.visible)
            .map((entry) => {
              const [x0, y0, x1, y1] = entry.box;
              /* Boxes are in the SOURCE raster's pixels; the element on
                 screen is the preview, which `_write_preview` thumbnails to
                 at most 512 px. Dividing source pixels by the preview's
                 natural size overshoots by exactly that ratio. Scene
                 dimensions when the API reports them, preview size only as
                 a fallback. */
              const boxW = scene.width_px || naturalSize.w;
              const boxH = scene.height_px || naturalSize.h;
              return (
                <span
                  key={entry.key}
                  className="pointer-events-none absolute border-2"
                  style={{
                    left: `${(x0 / boxW) * 100}%`,
                    top: `${(y0 / boxH) * 100}%`,
                    width: `${((x1 - x0) / boxW) * 100}%`,
                    height: `${((y1 - y0) / boxH) * 100}%`,
                    borderColor: entry.colour,
                  }}
                >
                  <span
                    className="t-code-sm absolute -top-[15px] left-0 whitespace-nowrap px-1 text-white"
                    style={{ background: entry.colour }}
                  >
                    {entry.label}
                  </span>
                </span>
              );
            })
        : null}
    </div>
  );
}

/**
 * The non-georeferenced viewer (§8.4).
 *
 * Benchmark chips arrive as PNG or JPEG with no CRS. Showing an empty map
 * for those would be a lie about what the system knows, so the map is
 * replaced by a pan/zoom canvas that composites the masks in pixel space and
 * says plainly why it is here. Grounding boxes on this path arrive as
 * `bbox_px` and are drawn in the same coordinate system.
 *
 * A two-scene bundle passes `pair`, and the workspace's compare mode then
 * chooses between one scene, an A/B swipe, and the two side by side — all
 * three under one transform, so the scenes never drift apart.
 */
export function ImageViewer({
  scene,
  pair,
  labels = ["A", "B"],
}: {
  /** The scene shown in the single view. */
  scene: Scene;
  pair?: { a: Scene; b: Scene } | null;
  labels?: readonly [string, string];
}) {
  const layers = useWorkspace((s) => s.layers);
  const baseOpacity = useWorkspace((s) => s.baseOpacity);
  const compareMode = useWorkspace((s) => s.compareMode);
  const swipePosition = useWorkspace((s) => s.swipePosition);
  const setSwipePosition = useWorkspace((s) => s.setSwipePosition);

  const mode = pair ? compareMode : "single";

  const rootRef = useRef<HTMLDivElement | null>(null);
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 1 });
  const dragging = useRef<{ x: number; y: number } | null>(null);
  const [readout, setReadout] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
  } | null>(null);

  const reset = useCallback(() => setTransform({ x: 0, y: 0, scale: 1 }), []);

  // A new bundle starts from the origin. Switching between the two scenes of
  // one pair does not: holding the position is the point of comparing them.
  const viewKey = pair
    ? `${pair.a.scene_id}|${pair.b.scene_id}`
    : scene.scene_id;
  useEffect(() => {
    reset();
  }, [viewKey, reset]);

  const onWheel = (event: ReactWheelEvent) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const originX = event.clientX - rect.left;
    const originY = event.clientY - rect.top;
    setTransform((current) => {
      const next = Math.min(
        12,
        Math.max(0.5, current.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12)),
      );
      const ratio = next / current.scale;
      return {
        scale: next,
        x: originX - (originX - current.x) * ratio,
        y: originY - (originY - current.y) * ratio,
      };
    });
  };

  const onPointerDown = (event: ReactPointerEvent) => {
    dragging.current = {
      x: event.clientX - transform.x,
      y: event.clientY - transform.y,
    };
    // currentTarget, not target: the child under the cursor can unmount mid-drag
    // (a box overlay re-rendering), and capture on a removed node is lost --
    // which stranded `dragging.current` set with no pointerup to clear it.
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Capture is an optimisation; dragging still works without it.
    }
  };

  const onPointerMove = (event: ReactPointerEvent, pane: PaneKey) => {
    const rect = event.currentTarget.getBoundingClientRect();
    // Under a swipe both scenes share the viewport; the divider decides which
    // one the cursor is actually over.
    const over: PaneKey =
      mode === "swipe"
        ? (event.clientX - rect.left) / rect.width > swipePosition
          ? "b"
          : "a"
        : pane;
    const image = event.currentTarget.querySelector<HTMLImageElement>(
      `img[data-pane="${over}"]`,
    );
    if (image && image.naturalWidth && image.clientWidth) {
      const localX = (event.clientX - rect.left - transform.x) / transform.scale;
      const localY = (event.clientY - rect.top - transform.y) / transform.scale;
      const ratio = image.naturalWidth / image.clientWidth;
      setReadout({
        x: Math.round(localX * ratio),
        y: Math.round(localY * ratio),
        w: image.naturalWidth,
        h: image.naturalHeight,
      });
    }
    // Read the origin once, here, rather than inside the updater. React runs
    // the updater during the render phase, not at call time -- so a pointerup
    // arriving in between ran `endDrag`, `dragging.current` was null, and the
    // non-null assertion threw *during render*. With no error boundary above
    // it, React unmounts the whole tree: the screen went blank the moment you
    // dragged an image and released at the wrong instant.
    const origin = dragging.current;
    if (!origin) return;
    const { clientX, clientY } = event;
    setTransform((current) => ({
      ...current,
      x: clientX - origin.x,
      y: clientY - origin.y,
    }));
  };

  const endDrag = () => {
    dragging.current = null;
  };

  const frame = (target: Scene, pane: PaneKey) => (
    <SceneFrame
      scene={target}
      pane={pane}
      fill={Boolean(pair)}
      transform={transform}
      layers={layers}
      baseOpacity={baseOpacity}
    />
  );

  const viewport = (
    pane: PaneKey,
    content: React.ReactNode,
    className?: string,
  ) => (
    <div
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={(event) => onPointerMove(event, pane)}
      onPointerUp={endDrag}
      onPointerLeave={() => {
        endDrag();
        setReadout(null);
      }}
      className={cn(
        "relative h-full min-w-0 touch-none overflow-hidden",
        dragging.current ? "cursor-grabbing" : "cursor-grab",
        className,
      )}
    >
      {content}
    </div>
  );

  const tag = (text: string) => (
    <span className="t-code-sm pointer-events-none absolute bottom-2 left-2 z-10 border border-signal bg-window-0/85 px-1.5 py-[4px] text-signal">
      {text}
    </span>
  );

  return (
    <div
      ref={rootRef}
      className="on-window relative h-full w-full overflow-hidden bg-window-0"
    >
      {mode === "side" && pair ? (
        <div className="grid h-full w-full grid-cols-2">
          {viewport(
            "a",
            <>
              {frame(pair.a, "a")}
              {tag(labels[0])}
            </>,
          )}
          {viewport(
            "b",
            <>
              {frame(pair.b, "b")}
              {tag(labels[1])}
            </>,
            "border-l border-signal",
          )}
        </div>
      ) : mode === "swipe" && pair ? (
        <>
          {viewport(
            "a",
            <>
              {frame(pair.a, "a")}
              <div
                className="pointer-events-none absolute inset-0"
                style={{ clipPath: `inset(0 0 0 ${swipePosition * 100}%)` }}
              >
                {/* An opaque backing, so pane A does not show through where
                    the second scene is smaller or partly transparent. */}
                <div className="absolute inset-0 bg-window-0" />
                <div className="absolute inset-0">{frame(pair.b, "b")}</div>
              </div>
            </>,
            "w-full",
          )}
          <SwipeDivider
            containerRef={rootRef}
            position={swipePosition}
            onChange={setSwipePosition}
            labels={labels}
          />
        </>
      ) : (
        viewport(
          "a",
          <>
            {frame(scene, "a")}
            {pair
              ? tag(scene.scene_id === pair.b.scene_id ? labels[1] : labels[0])
              : null}
          </>,
          "w-full",
        )
      )}

      <div className="pointer-events-none absolute inset-x-2 top-2 z-10 flex flex-wrap items-center gap-2">
        <span className="t-code-sm border border-caution/60 bg-window-0/90 px-1.5 py-[4px] text-caution">
          PIXEL SPACE · NO CRS
        </span>
        <span className="border border-window-2 bg-window-0/85 px-1.5 py-[4px] text-[11px] leading-tight text-window-ink-2">
          This scene is not georeferenced, so the map is replaced by the image
          viewer and masks are composited in pixel coordinates.
        </span>
      </div>

      <div className="pointer-events-none absolute bottom-2 right-2 z-10 flex flex-col items-end gap-1">
        <span className="t-data border border-window-2 bg-window-0/85 px-1.5 py-[3px] text-[10.5px] text-window-ink">
          {readout
            ? `px ${Math.max(0, Math.min(readout.w, readout.x))}, ${Math.max(0, Math.min(readout.h, readout.y))}`
            : "hover the image for pixel coordinates"}
        </span>
        <span className="t-code-sm border border-window-2 bg-window-0/85 px-1.5 py-[3px] text-window-ink-2">
          {readout ? `${readout.w} × ${readout.h} PX` : "—"} ·{" "}
          {transform.scale.toFixed(1)}×
        </span>
      </div>

      <button
        type="button"
        onClick={reset}
        className="t-code-sm absolute right-2 top-11 z-10 border border-window-2 bg-window-0/85 px-2 py-[5px] text-window-ink transition-colors hover:border-window-ink-2 hover:text-white"
      >
        Reset view
      </button>
    </div>
  );
}
