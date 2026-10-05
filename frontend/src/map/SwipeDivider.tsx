import {
  useCallback,
  useEffect,
  useRef,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";

/**
 * The draggable divider of an A/B swipe: pane A to its left, pane B to its
 * right. Shared by the map and the pixel-space image viewer so the two
 * comparisons handle identically.
 */
export function SwipeDivider({
  containerRef,
  position,
  onChange,
  labels,
}: {
  /** The element the position is a fraction of. */
  containerRef: RefObject<HTMLElement | null>;
  position: number;
  onChange: (position: number) => void;
  labels: readonly [string, string];
}) {
  const dragging = useRef(false);

  const move = useCallback(
    (clientX: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      onChange(
        Math.min(0.98, Math.max(0.02, (clientX - rect.left) / rect.width)),
      );
    },
    [containerRef, onChange],
  );

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (dragging.current) move(event.clientX);
    };
    const onUp = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [move]);

  return (
    <div
      role="separator"
      aria-label="Comparison divider"
      aria-orientation="vertical"
      aria-valuenow={Math.round(position * 100)}
      aria-valuemin={2}
      aria-valuemax={98}
      tabIndex={0}
      onPointerDown={(event: ReactPointerEvent) => {
        // Not a pan of the imagery underneath.
        event.stopPropagation();
        dragging.current = true;
        move(event.clientX);
      }}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") onChange(Math.max(0.02, position - 0.02));
        if (event.key === "ArrowRight")
          onChange(Math.min(0.98, position + 0.02));
      }}
      className="absolute inset-y-0 z-10 -ml-3 w-6 cursor-ew-resize touch-none"
      style={{ left: `${position * 100}%` }}
    >
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-signal"
      />
      <span className="pointer-events-none absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-1 border border-signal bg-window-0 px-1.5 py-1">
        <span className="t-code-sm text-signal">{labels[0]}</span>
        <span aria-hidden="true" className="text-signal">
          ◂▸
        </span>
        <span className="t-code-sm text-signal">{labels[1]}</span>
      </span>
    </div>
  );
}
