import { useEffect, useId, useRef } from "react";
import { cn } from "@/lib/cn";
import type { MockScene } from "./hero.data";
import {
  BOX,
  FLOOD_LOBE,
  SCALE_BAR,
  SCENE_H,
  SCENE_W,
  floodPath,
  paintScene,
  type SceneKind,
} from "./sceneRaster";

/**
 * The imagery window.
 *
 * Two layers, and the split is deliberate:
 *
 * 1. **The scene is pixels** — a procedural raster painted by `sceneRaster`,
 *    sampled per pixel through an optical or a radar sensor model. Imagery has
 *    no crisp edges and no flat fills, so it is not drawn with polygons.
 * 2. **Everything the product produced is vector** — the masks, the returned
 *    box, the scale bar and the chrome. That is the same division the real
 *    console has: raster underneath, evidence and instrument on top.
 *
 * Nothing here is observed data. The frame carries `SYNTHETIC` in its own
 * corner, and the mask geometry is sampled from the same functions that
 * painted the water, so an overlay always lands on the water in the picture
 * rather than near it.
 */

const KIND: Record<MockScene, SceneKind> = {
  optical: "optical",
  grounding: "optical",
  crossmodal: "optical-flood",
  sar: "sar",
};

export function SceneMock({
  scene,
  className,
}: {
  scene: MockScene;
  className?: string;
}) {
  // Ids must be unique per instance: two windows sharing a pattern id is how
  // the second one silently renders untextured.
  const uid = useId().replace(/:/g, "");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const kind = KIND[scene];

  useEffect(() => {
    if (canvasRef.current) paintScene(canvasRef.current, kind);
  }, [kind]);

  const radar = scene === "sar";
  const flooded = scene === "crossmodal";

  const chip = radar
    ? "SAR · VV · C-BAND · 6.0 M"
    : flooded
      ? "OPTICAL + SAR · COREGISTERED"
      : "OPTICAL · 4 BAND · GSD 2.0 M";

  const label = `Illustration of the imagery window: a procedurally rendered ${
    radar ? "radar" : "optical"
  } scene of a channel, a built-up block and field parcels${
    flooded
      ? ", with the optical and radar water masks disagreeing over the flood extent"
      : ""
  }${
    scene === "grounding"
      ? ", with the returned bounding box drawn around a storage tank farm"
      : ""
  }. Not observed data.`;

  return (
    <div className={cn("relative overflow-hidden bg-window-0", className)}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={label}
        className="absolute inset-0 h-full w-full"
      />

      <svg
        viewBox={`0 0 ${SCENE_W} ${SCENE_H}`}
        preserveAspectRatio="none"
        aria-hidden="true"
        className="absolute inset-0 h-full w-full"
      >
        <defs>
          {/* The SAR mask is hatched as well as coloured: where the two water
              masks overlap, hue alone would just make a third colour. */}
          <pattern
            id={`${uid}-sarhatch`}
            width="5"
            height="5"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width="5" height="5" fill="#F2A03B" opacity="0.04" />
            <rect width="1.4" height="5" fill="#F2A03B" opacity="0.3" />
          </pattern>
          <linearGradient id={`${uid}-vignette`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#000" stopOpacity="0.34" />
            <stop offset="30%" stopColor="#000" stopOpacity="0" />
            <stop offset="72%" stopColor="#000" stopOpacity="0" />
            <stop offset="100%" stopColor="#000" stopOpacity="0.46" />
          </linearGradient>
        </defs>

        {/* ── evidence ───────────────────────────────────────────────── */}

        {flooded ? (
          <>
            {/* Opacity sits on the group, not on each shape: the lobe is a
                union of overlapping bulges, and per-shape opacity would draw
                their seams as darker bands through the water. */}
            <g opacity="0.22">
              <path d={floodPath()} fill="#3BA3F2" />
              {FLOOD_LOBE.map((l) => (
                <ellipse key={`l-${l.cx}-${l.cy}`} {...l} fill="#3BA3F2" />
              ))}
            </g>
            {/* radar: the same water read through C-band, 0.5 km² larger.
                Offset and hatched, so the disagreement is the thing seen. */}
            <g transform="translate(2 -4)">
              <path d={floodPath()} fill={`url(#${uid}-sarhatch)`} />
              {FLOOD_LOBE.map((l) => (
                <ellipse
                  key={`s-${l.cx}-${l.cy}`}
                  {...l}
                  fill={`url(#${uid}-sarhatch)`}
                />
              ))}
            </g>
            <path
              d={floodPath()}
              fill="none"
              stroke="#3BA3F2"
              strokeWidth="1.1"
              opacity="0.95"
            />
          </>
        ) : null}

        {scene === "grounding" ? (
          <g>
            <rect {...BOX} fill="#3BA3F2" opacity="0.1" />
            <rect
              {...BOX}
              fill="none"
              stroke="#3BA3F2"
              strokeWidth="1.3"
            />
            {/* corner ticks — a returned box, not a drawn rectangle */}
            {[
              [BOX.x, BOX.y, 1, 1],
              [BOX.x + BOX.w, BOX.y, -1, 1],
              [BOX.x, BOX.y + BOX.h, 1, -1],
              [BOX.x + BOX.w, BOX.y + BOX.h, -1, -1],
            ].map(([cx, cy, sx, sy]) => (
              <path
                key={`t-${cx}-${cy}`}
                d={`M${cx} ${cy + sy * 9} L${cx} ${cy} L${cx + sx * 9} ${cy}`}
                stroke="#3BA3F2"
                strokeWidth="2"
                fill="none"
              />
            ))}
            {/* The label hangs below the box: above it would collide with the
                modality chip, and a box whose caption is unreadable is not
                evidence of anything. */}
            <rect
              x={BOX.x}
              y={BOX.y + BOX.h + 2}
              width="118"
              height="12"
              fill="#0a0f11"
              opacity="0.82"
            />
            <text
              x={BOX.x + 4}
              y={BOX.y + BOX.h + 10.5}
              className="t-code-sm"
              fill="#3BA3F2"
              fontSize="7"
            >
              STORAGE TANK FARM · 0.74
            </text>
          </g>
        ) : null}

        {/* ── window chrome ─────────────────────────────────────────── */}

        <rect width={SCENE_W} height={SCENE_H} fill={`url(#${uid}-vignette)`} />

        <g stroke="#8f9b98" strokeWidth="1" opacity="0.55" fill="none">
          <path d="M6 6 L6 15 M6 6 L15 6" />
          <path d="M314 6 L314 15 M314 6 L305 6" />
          <path d="M6 214 L6 205 M6 214 L15 214" />
          <path d="M314 214 L314 205 M314 214 L305 214" />
        </g>

        <rect x="16" y="12" width="134" height="13" fill="#0a0f11" opacity="0.72" />
        <text x="21" y="21.5" className="t-code-sm" fill="#cfd8d5" fontSize="7">
          {chip}
        </text>

        <g
          transform="translate(292 34)"
          stroke="#cfd8d5"
          fill="#cfd8d5"
          opacity="0.8"
        >
          <path d="M0 -10 L4 4 L0 1 L-4 4 Z" strokeWidth="0.5" />
          <text x="-2.6" y="14" className="t-code-sm" fontSize="7" stroke="none">
            N
          </text>
        </g>

        {/* a scale bar — a scene without one is a picture, not a measurement */}
        <g transform="translate(16 198)">
          <rect
            x="0"
            y="0"
            width={SCALE_BAR.units}
            height="4"
            fill="#cfd8d5"
            opacity="0.85"
          />
          <rect
            x="0"
            y="0"
            width={SCALE_BAR.units / 2}
            height="4"
            fill="#0a0f11"
            opacity="0.75"
          />
          <rect
            x="0"
            y="0"
            width={SCALE_BAR.units}
            height="4"
            fill="none"
            stroke="#cfd8d5"
            strokeWidth="0.7"
            opacity="0.9"
          />
          <text
            x={SCALE_BAR.units + 6}
            y="4.5"
            className="t-code-sm"
            fill="#cfd8d5"
            fontSize="7"
            opacity="0.8"
          >
            {SCALE_BAR.label}
          </text>
        </g>

        {/* the honesty tag: this is rendered, and it never leaves the frame */}
        <text
          x="304"
          y="207"
          textAnchor="end"
          className="t-code-sm"
          fill="#cfd8d5"
          fontSize="7"
          opacity="0.75"
        >
          SYNTHETIC · NOT OBSERVED DATA
        </text>
      </svg>
    </div>
  );
}
