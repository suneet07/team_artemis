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
import { useWorkspace } from "@/store/workspace";
import { cn } from "@/lib/cn";

/**
 * The non-georeferenced viewer (§8.4).
 *
 * Benchmark chips arrive as PNG or JPEG with no CRS. Showing an empty map
 * for those would be a lie about what the system knows, so the map is
 * replaced by a pan/zoom canvas that composites the masks in pixel space and
 * says plainly why it is here. Grounding boxes on this path arrive as
 * `bbox_px` and are drawn in the same coordinate system.
 */
export function ImageViewer({ scene }: { scene: Scene }) {
  const layers = useWorkspace((s) => s.layers);
  const baseOpacity = useWorkspace((s) => s.baseOpacity);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(
    null,
  );
  const dragging = useRef<{ x: number; y: number } | null>(null);
  const [pixel, setPixel] = useState<{ x: number; y: number } | null>(null);

  const reset = useCallback(() => setTransform({ x: 0, y: 0, scale: 1 }), []);

  useEffect(() => {
    reset();
  }, [scene.scene_id, reset]);

  const onWheel = (event: ReactWheelEvent) => {
    event.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
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
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: ReactPointerEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect && naturalSize) {
      const localX = (event.clientX - rect.left - transform.x) / transform.scale;
      const localY = (event.clientY - rect.top - transform.y) / transform.scale;
      const displayed = rect.width;
      const ratio = naturalSize.w / displayed;
      setPixel({
        x: Math.round(localX * ratio),
        y: Math.round(localY * ratio),
      });
    }
    if (!dragging.current) return;
    setTransform((current) => ({
      ...current,
      x: event.clientX - dragging.current!.x,
      y: event.clientY - dragging.current!.y,
    }));
  };

  const endDrag = () => {
    dragging.current = null;
  };

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
    <div className="on-window relative h-full w-full overflow-hidden bg-window-0">
      <div
        ref={containerRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={() => {
          endDrag();
          setPixel(null);
        }}
        className={cn(
          "h-full w-full touch-none",
          dragging.current ? "cursor-grabbing" : "cursor-grab",
        )}
      >
        <div
          className="relative origin-top-left will-change-transform"
          style={{
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
          }}
        >
          <img
            src={resolveUrl(scene.preview_url)}
            alt={`${scene.filename} — synthetic imagery, not observed data`}
            draggable={false}
            onLoad={(event) =>
              setNaturalSize({
                w: event.currentTarget.naturalWidth,
                h: event.currentTarget.naturalHeight,
              })
            }
            className="block w-full select-none"
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
                  return (
                    <span
                      key={entry.key}
                      className="pointer-events-none absolute border-2"
                      style={{
                        left: `${(x0 / naturalSize.w) * 100}%`,
                        top: `${(y0 / naturalSize.h) * 100}%`,
                        width: `${((x1 - x0) / naturalSize.w) * 100}%`,
                        height: `${((y1 - y0) / naturalSize.h) * 100}%`,
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
      </div>

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
          {pixel && naturalSize
            ? `px ${Math.max(0, Math.min(naturalSize.w, pixel.x))}, ${Math.max(0, Math.min(naturalSize.h, pixel.y))}`
            : "hover the image for pixel coordinates"}
        </span>
        <span className="t-code-sm border border-window-2 bg-window-0/85 px-1.5 py-[3px] text-window-ink-2">
          {naturalSize ? `${naturalSize.w} × ${naturalSize.h} PX` : "—"} ·{" "}
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
