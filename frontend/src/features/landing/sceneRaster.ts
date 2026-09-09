/**
 * The imagery in the masthead window, rendered pixel by pixel.
 *
 * **This is not observed data, and the window says so on its own face.** It is
 * a procedural scene: one continuous world — a meandering channel, a built-up
 * block on the north bank, an industrial apron to the west, field parcels to
 * the south — sampled per pixel and coloured through a sensor model. Nothing
 * is downloaded, nothing is licensed, and the venue is offline.
 *
 * It is drawn this way rather than as vector shapes because a satellite scene
 * has no crisp edges and no flat fills. What makes imagery read as imagery is
 * continuous tone, texture at every scale, and sensor noise — none of which a
 * polygon can produce. So the world is a set of scalar fields sampled by
 * fractal noise, and the two sensors are two different functions over the
 * *same* fields:
 *
 * - **Optical** is reflectance: water dark and slightly blue, vegetation dark
 *   green with canopy texture, bare soil warm, roofs bright and hard-edged,
 *   plus atmospheric haze that lifts the blacks the way a real L2A product
 *   does.
 * - **SAR** is backscatter, and the inversion is the point: open water is
 *   nearly black because it reflects away from the sensor, built-up is nearly
 *   white from double-bounce off walls, and everything carries multiplicative
 *   speckle rather than additive grain. It is the same ground, and it looks
 *   nothing alike — which is the argument the cross-modal capability makes.
 *
 * The scene is deterministic (one seeded integer hash, no `Math.random`) so it
 * is identical on every machine and between renders, and each variant is
 * rasterised once and cached. Geometry used by the overlays — the channel, the
 * flood extent — is exported from here, so a mask drawn on top lands on the
 * water that is actually in the picture.
 *
 * Ground sample distance is 2.0 m, matching the optical scene in the API
 * fixture, so 320 world units span roughly 1.3 km.
 */

/* ── deterministic noise ─────────────────────────────────────────────── */

/**
 * Integer hash. Every multiply goes through `Math.imul` and stays in int32:
 * in floating point a large seed term swallows the coordinates entirely and
 * the field collapses to a constant.
 */
function hash2(x: number, y: number, seed: number): number {
  let h =
    Math.imul(x | 0, 374761393) ^
    Math.imul(y | 0, 668265263) ^
    Math.imul(seed | 0, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

function valueNoise(x: number, y: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = smooth(x - xi);
  const yf = smooth(y - yi);
  const a = hash2(xi, yi, seed);
  const b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed);
  const d = hash2(xi + 1, yi + 1, seed);
  return (
    a * (1 - xf) * (1 - yf) + b * xf * (1 - yf) + c * (1 - xf) * yf + d * xf * yf
  );
}

function fbm(x: number, y: number, seed: number, octaves: number): number {
  let value = 0;
  let amplitude = 0.5;
  let frequency = 1;
  let norm = 0;
  for (let i = 0; i < octaves; i += 1) {
    value += amplitude * valueNoise(x * frequency, y * frequency, seed + i * 71);
    norm += amplitude;
    amplitude *= 0.5;
    frequency *= 2.07;
  }
  return value / norm;
}

/** Ridged noise — the fine linear structure that reads as canopy and furrow. */
function ridge(x: number, y: number, seed: number): number {
  return 1 - Math.abs(fbm(x, y, seed, 3) * 2 - 1);
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep(edge0: number, edge1: number, v: number): number {
  return smooth(clamp01((v - edge0) / (edge1 - edge0)));
}

/* ── the world ───────────────────────────────────────────────────────── */

/** World size in the same units the overlays use. */
export const SCENE_W = 320;
export const SCENE_H = 220;

/** Rasterised at 3× so the window stays sharp on a retina panel. */
const SCALE = 3;

/** Metres per world unit, and the scale bar that follows from it. */
export const GSD_M = 4;
export const SCALE_BAR = { units: 50, label: "200 M" };

/** The channel's centre line, as a function of easting. */
export function channelCentre(x: number): number {
  return (
    132 +
    17 * Math.sin(x * 0.0182 + 0.6) +
    7 * Math.sin(x * 0.041 + 2.1) +
    3.2 * Math.sin(x * 0.093 + 0.3)
  );
}

/** Half-width of open water, and of the inundated extent over its banks. */
function channelHalf(x: number): number {
  return 9 + 3.2 * Math.sin(x * 0.031 + 1.4) + 1.4 * Math.sin(x * 0.077);
}

function floodHalf(x: number): number {
  return channelHalf(x) + 8 + 4.5 * Math.sin(x * 0.024 + 0.2);
}

/**
 * The flood lobe: where water backs up into the low ground on the north bank
 * and stands over the settlement. Returns 1 inside, falling to 0 at the edge.
 */
function lobe(x: number, y: number): number {
  const dx = (x - 205) / 52;
  const dy = (y - 112) / 26;
  const d = Math.sqrt(dx * dx + dy * dy);
  // The boundary is perturbed by noise, because standing water follows ground,
  // not an ellipse.
  const edge = 1 + 0.22 * (fbm(x * 0.05, y * 0.05, 991, 3) - 0.5) * 2;
  return smoothstep(edge, edge - 0.28, d);
}

/**
 * The settlement, in its own rotated frame.
 *
 * A grid squared to the pixel raster is the tell that gives a synthetic scene
 * away — no town is aligned to a satellite's scan lines. So the whole
 * settlement sits at 9° to the frame, and every block-scale function works in
 * these rotated coordinates.
 */
const TOWN_SIN = Math.sin(0.157);
const TOWN_COS = Math.cos(0.157);

function townFrame(x: number, y: number): [number, number] {
  const dx = x - 148;
  const dy = y - 14;
  return [dx * TOWN_COS - dy * TOWN_SIN, dx * TOWN_SIN + dy * TOWN_COS];
}

/** 1 on a carriageway, 0 inside a block. */
function street(x: number, y: number): number {
  const [tx, ty] = townFrame(x, y);
  // Block pitch varies across the town: an old core of small blocks to the
  // west, larger plots east of it.
  const pitch = 15 + 6 * smoothstep(20, 130, tx);
  const bx = tx / pitch + 0.05 * fbm(x * 0.02, y * 0.02, 313, 2);
  const by = ty / 15.5;
  const fx = Math.abs(bx - Math.round(bx));
  const fy = Math.abs(by - Math.round(by));
  return fx < 0.075 || fy < 0.07 ? 1 : 0;
}

/**
 * Individual roofs inside a block — small, hard-edged, unevenly bright, and
 * unevenly present. Density falls off at the edge of town, so the settlement
 * thins into the fields instead of ending at a line.
 */
function roof(x: number, y: number): number {
  const [tx, ty] = townFrame(x, y);
  const cellX = 3.2;
  const cellY = 2.9;
  const cx = Math.floor(tx / cellX);
  const cy = Math.floor(ty / cellY);
  const density = 0.34 + 0.5 * fbm(x * 0.045, y * 0.045, 1723, 3);
  if (hash2(cx, cy, 7717) > density) return 0;
  const fx = tx / cellX - cx;
  const fy = ty / cellY - cy;
  const mx = 0.08 + 0.24 * hash2(cx, cy, 4231);
  const my = 0.08 + 0.22 * hash2(cx, cy, 6607);
  if (fx < mx || fx > 1 - mx || fy < my || fy > 1 - my) return 0;
  return 0.25 + 0.75 * hash2(cx, cy, 5519);
}

/**
 * Field parcels south of the channel — a jittered lattice, one crop each,
 * ploughed in a direction of its own. Parcels are ~40 x 28 m, and the furrows
 * inside them are what stops a field from reading as a flat swatch.
 */
function parcel(
  x: number,
  y: number,
): { crop: number; furrow: number; edge: number; green: number } {
  const gx = x / 17 + 0.34 * fbm(x * 0.014, y * 0.014, 271, 2);
  const gy = y / 12 + 0.3 * fbm(x * 0.017, y * 0.017, 611, 2);
  const cx = Math.floor(gx);
  const cy = Math.floor(gy);
  const fx = gx - cx;
  const fy = gy - cy;
  const crop = hash2(cx, cy, 1361);
  const angle = hash2(cx, cy, 2213) * Math.PI;
  const u = x * Math.cos(angle) + y * Math.sin(angle);
  const furrow = 0.5 + 0.5 * Math.sin(u * (1.9 + 1.6 * hash2(cx, cy, 3907)));
  // A field boundary in imagery is a hedgerow or a farm track — a dark line,
  // never the abrupt tone change that a flat lattice produces.
  const edge =
    Math.min(fx, 1 - fx) < 0.045 || Math.min(fy, 1 - fy) < 0.055 ? 1 : 0;
  // Roughly a third of parcels are standing crop rather than worked soil.
  const green = smoothstep(0.58, 0.78, crop);
  return { crop, furrow, edge, green };
}

export type SceneKind = "optical" | "optical-flood" | "sar";

interface Sample {
  /** 0–1 openness of water at this pixel. */
  water: number;
  /** 0–1 how built-up. */
  urban: number;
  /** Carriageway. */
  road: number;
  /** Roof brightness where a roof stands, else 0. */
  roof: number;
  /** 0–1 vegetation vigour. */
  veg: number;
  /** Tank-farm apron and its vessels. */
  industrial: number;
  tank: number;
}

function sampleWorld(x: number, y: number, flooded: boolean): Sample {
  // The bank is not a line: the distance to the channel is perturbed by the
  // same noise field that textures the ground, so the shoreline is ragged.
  const jitter = (fbm(x * 0.09, y * 0.09, 143, 4) - 0.5) * 5;
  const d = Math.abs(y - channelCentre(x)) + jitter;

  const open = smoothstep(channelHalf(x) + 0.7, channelHalf(x) - 0.7, d);
  let water = open;
  if (flooded) {
    const sheet = smoothstep(floodHalf(x) + 1, floodHalf(x) - 1, d);
    water = Math.max(water, sheet * 0.92, lobe(x, y) * 0.88);
  }

  const inTown =
    smoothstep(140, 156, x) * smoothstep(112, 100, y) * smoothstep(4, 12, y);
  const road = inTown * street(x, y);
  const roofs = inTown > 0.5 && road < 0.5 ? roof(x, y) : 0;
  const urban = inTown * (0.55 + 0.45 * fbm(x * 0.02, y * 0.02, 55, 3));

  const apron =
    smoothstep(43, 46, x) *
    smoothstep(107, 104, x) *
    smoothstep(27, 30, y) *
    smoothstep(79, 76, y);
  let tank = 0;
  for (const t of TANKS) {
    const dt = Math.hypot(x - t[0], y - t[1]);
    tank = Math.max(tank, smoothstep(t[2] + 0.8, t[2] - 0.8, dt));
  }

  const south = smoothstep(0, 26, y - channelCentre(x) - channelHalf(x));
  const veg = clamp01(
    south * (0.35 + 0.65 * fbm(x * 0.016, y * 0.016, 907, 4)) +
      0.18 * smoothstep(60, 20, x) * smoothstep(0, 40, y),
  );

  return { water, urban, road, roof: roofs, veg, industrial: apron, tank };
}

const TANKS: [number, number, number][] = [
  [58, 46, 6],
  [74, 44, 5.4],
  [90, 48, 5.8],
  [60, 62, 5.2],
  [77, 62, 6],
  [92, 64, 5],
];

/* ── the two sensors ─────────────────────────────────────────────────── */

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Optical reflectance, in the muted register of an atmospherically corrected
 * product.
 *
 * The frequencies matter more than the hues. A satellite scene carries
 * structure at every scale, so three noise bands are combined: metre-scale
 * grain, ten-metre texture, and a slow field that varies the whole scene.
 * Flat fills are what made the first attempt read as a diagram.
 */
function optical(
  s: Sample,
  x: number,
  y: number,
  out: [number, number, number],
): void {
  const micro = fbm(x * 1.35, y * 1.35, 401, 3);
  const meso = fbm(x * 0.3, y * 0.3, 233, 4);
  const slow = fbm(x * 0.06, y * 0.06, 617, 3);
  // Pushed away from the mean: real land cover is high-contrast at this
  // scale, and an average of noise bands is not.
  const tone = clamp01(0.5 + 1.5 * (0.34 * micro + 0.44 * meso + 0.22 * slow - 0.5));
  const canopy = ridge(x * 0.9, y * 0.9, 733);
  const { crop, furrow, edge, green } = parcel(x, y);

  // bare ground and cropland: warm, mid-dark, ploughed, and hedged
  const cropLift = (crop * 44 + furrow * 16 - 16) * (1 - green * 0.75);
  let r = 52 + 46 * tone + cropLift;
  let g = 47 + 42 * tone + cropLift * 0.92;
  let b = 35 + 30 * tone + cropLift * 0.6;
  // standing crop: greener and darker than the worked ground beside it
  r = mix(r, 30 + 24 * tone, green * 0.8);
  g = mix(g, 52 + 40 * tone + 8 * furrow, green * 0.8);
  b = mix(b, 28 + 20 * tone, green * 0.8);
  // the hedgerow or track along the boundary
  r = mix(r, 26 + 16 * tone, edge * 0.7);
  g = mix(g, 30 + 20 * tone, edge * 0.7);
  b = mix(b, 22 + 14 * tone, edge * 0.7);

  // vegetation: darker, greener, with canopy structure
  const vg = s.veg * (0.72 + 0.28 * canopy);
  r = mix(r, 22 + 26 * tone, vg);
  g = mix(g, 38 + 40 * tone + 10 * canopy, vg);
  b = mix(b, 22 + 22 * tone, vg);

  // the industrial apron is bare hardstanding: brighter, flatter, dusty
  // The apron is concrete: bright, but stained and patched, never a flat fill.
  const ap = s.industrial * 0.8;
  const stain = 22 * (fbm(x * 0.5, y * 0.5, 3121, 3) - 0.5);
  r = mix(r, 98 + 24 * micro + stain, ap);
  g = mix(g, 95 + 22 * micro + stain, ap);
  b = mix(b, 84 + 20 * micro + stain, ap);

  // settlement: grey ground between roofs, roads cut into it, roofs on top
  const u = s.urban * (1 - s.water * 0.9);
  r = mix(r, 66 + 30 * micro, u * 0.85);
  g = mix(g, 64 + 28 * micro, u * 0.85);
  b = mix(b, 60 + 26 * micro, u * 0.85);
  r = mix(r, 42 + 10 * micro, s.road * 0.85);
  g = mix(g, 43 + 10 * micro, s.road * 0.85);
  b = mix(b, 44 + 10 * micro, s.road * 0.85);
  if (s.roof > 0) {
    const rf = 1 - s.water * 0.85;
    const bright = 92 + 118 * s.roof + 18 * micro;
    r = mix(r, bright, rf);
    g = mix(g, bright * 0.97, rf);
    b = mix(b, bright * 0.9, rf);
  }
  const tk = s.tank * 0.9;
  r = mix(r, 138 + 30 * micro, tk);
  g = mix(g, 140 + 28 * micro, tk);
  b = mix(b, 138 + 26 * micro, tk);

  // water: dark, blue-shifted, with ripple glint rather than a flat fill
  const ripple = fbm(x * 1.6, y * 0.55, 88, 3);
  const sediment = clamp01(fbm(x * 0.22, y * 0.5, 1229, 3) * 1.5 - 0.35);
  const glint = 30 * Math.max(0, ripple - 0.7);
  r = mix(r, 10 + 8 * ripple + 26 * sediment + glint, s.water);
  g = mix(g, 20 + 10 * ripple + 24 * sediment + glint, s.water);
  b = mix(b, 31 + 12 * ripple + 14 * sediment + glint * 1.2, s.water);

  // sensor noise, then a thin atmospheric veil that lifts the blacks
  const shot = (hash2(Math.round(x * SCALE), Math.round(y * SCALE), 6151) - 0.5) * 14;
  const haze = 0.055;
  out[0] = (r + shot) * (1 - haze) + 34 * haze;
  out[1] = (g + shot) * (1 - haze) + 40 * haze;
  out[2] = (b + shot) * (1 - haze) + 46 * haze;
}

/**
 * Radar backscatter, single band, rendered as amplitude.
 *
 * The inversion against the optical render is the whole point: water is
 * specular and returns nothing, walls double-bounce and return almost
 * everything, and the noise is multiplicative speckle rather than grain.
 */
function sar(s: Sample, x: number, y: number): number {
  const rough = 0.5 * fbm(x * 0.55, y * 0.55, 251, 3) + 0.5 * fbm(x * 0.12, y * 0.12, 251, 3);
  let sigma = 0.28 + 0.16 * rough; // bare ground
  sigma = mix(sigma, 0.34 + 0.2 * ridge(x * 0.8, y * 0.8, 617), s.veg);
  sigma = mix(sigma, 0.22 + 0.1 * rough, s.industrial * 0.6);
  sigma = mix(sigma, 0.52 + 0.3 * rough, s.urban * 0.7);
  sigma = mix(sigma, 0.12, s.road * 0.8);
  sigma = mix(sigma, 0.95, s.roof * 0.9); // double bounce off walls
  sigma = mix(sigma, 0.86, s.tank * 0.8); // and off cylinders
  sigma = mix(sigma, 0.03, s.water); // specular: it reflects away

  // Speckle is multiplicative and heavy-tailed — the texture that makes a SAR
  // image unmistakable, and the reason a single pixel is never trusted.
  const u = hash2(Math.round(x * SCALE), Math.round(y * SCALE), 33331);
  const v = hash2(Math.round(x * SCALE), Math.round(y * SCALE), 90211);
  const speckle = 0.45 + 1.25 * Math.pow(u * 0.65 + v * 0.35, 1.8);
  return clamp01(sigma * speckle + 0.02);
}

/* ── rasterisation ───────────────────────────────────────────────────── */

const cache = new Map<SceneKind, ImageData>();

/**
 * Paint one scene into a canvas. Each variant is rasterised once for the life
 * of the page: switching tabs re-blits cached pixels rather than re-sampling
 * a quarter of a million of them.
 */
export function paintScene(canvas: HTMLCanvasElement, kind: SceneKind): void {
  const w = SCENE_W * SCALE;
  const h = SCENE_H * SCALE;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let image = cache.get(kind);
  if (!image) {
    image = ctx.createImageData(w, h);
    const data = image.data;
    const flooded = kind === "optical-flood";
    const rgb: [number, number, number] = [0, 0, 0];

    for (let py = 0; py < h; py += 1) {
      const y = py / SCALE;
      for (let px = 0; px < w; px += 1) {
        const x = px / SCALE;
        const s = sampleWorld(x, y, flooded);
        const i = (py * w + px) * 4;

        if (kind === "sar") {
          const a = sar(s, x, y);
          // A radar amplitude image is single-channel; the faint warmth comes
          // from the display LUT, not from the data.
          const v = 255 * Math.pow(a, 0.82);
          data[i] = v * 0.98;
          data[i + 1] = v;
          data[i + 2] = v * 0.97;
        } else {
          optical(s, x, y, rgb);
          data[i] = rgb[0];
          data[i + 1] = rgb[1];
          data[i + 2] = rgb[2];
        }
        data[i + 3] = 255;
      }
    }
    cache.set(kind, image);
  }

  ctx.putImageData(image, 0, 0);
}

/* ── geometry the overlays share with the pixels ─────────────────────── */

/**
 * The inundated extent, as a closed path in world units.
 *
 * Sampled from the same functions the raster uses, so the mask an analyst is
 * shown lands on the water that is actually in the picture — which is the
 * whole claim the evidence panel makes.
 */
export function floodPath(inset = 0): string {
  const top: string[] = [];
  const bottom: string[] = [];
  for (let x = -6; x <= SCENE_W + 6; x += 8) {
    const c = channelCentre(x);
    const half = floodHalf(x) - inset;
    const wobble = (fbm(x * 0.05, 0, 143, 3) - 0.5) * 4;
    top.push(`${x.toFixed(1)} ${(c - half + wobble).toFixed(1)}`);
    bottom.push(`${x.toFixed(1)} ${(c + half - wobble).toFixed(1)}`);
  }
  bottom.reverse();
  return `M${top.join(" L")} L${bottom.join(" L")} Z`;
}

/** The lobe standing over the settlement, as four overlapping bulges. */
export const FLOOD_LOBE: { cx: number; cy: number; rx: number; ry: number }[] = [
  { cx: 175, cy: 116, rx: 30, ry: 19 },
  { cx: 206, cy: 108, rx: 32, ry: 23 },
  { cx: 238, cy: 116, rx: 26, ry: 17 },
  { cx: 208, cy: 124, rx: 46, ry: 18 },
];

/** The box a grounding answer returns, around the tank farm. */
export const BOX = { x: 42, y: 30, w: 66, h: 48 };
