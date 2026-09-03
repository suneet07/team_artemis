/**
 * Synthetic imagery for the mock API.
 *
 * NOT SATELLITE DATA. Every pixel here is generated from a seeded noise
 * field so the demo has something honest-looking to render before the
 * backend serves real tiles. It is deliberately continuous in world space,
 * so a tile at z/x/y stitches seamlessly with its neighbours and survives
 * zooming — the same property real tiles have, which is what makes the map
 * behave correctly rather than merely look busy.
 *
 * Replace with `GET /scenes/{id}/tiles/...` the day the backend serves them:
 * nothing outside this file knows these rasters are fabricated.
 */

/* ── seeded value noise ──────────────────────────────────────────────── */

/**
 * Integer hash. Every multiply goes through `Math.imul` and stays in int32:
 * doing this in floating point lets a large seed term swallow the x and y
 * contributions entirely, and the field collapses to a constant.
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
    a * (1 - xf) * (1 - yf) +
    b * xf * (1 - yf) +
    c * (1 - xf) * yf +
    d * xf * yf
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

/* ── the scene model ─────────────────────────────────────────────────── */

/**
 * One continuous "world" shared by every raster of a bundle, so the optical
 * scene, the SAR scene and the masks all describe the same ground — which
 * is the entire point of a cross-modal demo.
 */
export interface WorldSample {
  /** 0 = dry, 1 = open water. */
  water: number;
  /** 0 = rural, 1 = dense built-up. */
  builtup: number;
  /**
   * The same field before clamping. The clamped value saturates across far
   * more ground than the built-up extent the fixtures report, so the mask
   * thresholds on this instead — see scripts/calibrate-masks.mjs.
   */
  builtupRaw: number;
  /** 0 = bare, 1 = dense vegetation. */
  vegetation: number;
  /** 0 = clear, 1 = thick cloud. Optical only; SAR sees through it. */
  cloud: number;
}

const WORLD_SEED = 20260827;

/**
 * The scene spans this many noise cells. Every frequency below is expressed
 * against it, so the low-frequency fields (river course, settlement extent,
 * cloud bank) actually vary across one scene instead of resolving to a flat
 * value — which is what separates imagery from a gradient.
 */
const WORLD_SPAN = 9;

export function sampleWorld(u: number, v: number): WorldSample {
  const x = u * WORLD_SPAN;
  const y = v * WORLD_SPAN;

  // A river valley: a meandering corridor that floods outward. The meander
  // is a low-frequency wander plus noise, so the channel reads as a river
  // rather than a stripe.
  const wander =
    Math.sin(x * 0.72) * 0.55 +
    Math.sin(x * 1.63 + 1.7) * 0.26 +
    (fbm(x * 0.5, y * 0.18, WORLD_SEED, 3) - 0.5) * 1.5;
  const channelY = WORLD_SPAN * 0.52 + wander;
  const distance = Math.abs(y - channelY);
  const floodExtent =
    0.42 + fbm(x * 0.9, y * 0.9, WORLD_SEED + 900, 4) * 0.85;
  const water = Math.max(
    0,
    Math.min(1, 1 - distance / Math.max(floodExtent, 0.05)),
  );

  // The city: a coherent settlement extent, textured by a block grid at
  // high frequency and pushed back from the water.
  const settlement = fbm(x * 0.62 + 11, y * 0.62 - 4, WORLD_SEED + 310, 4);
  const blocks =
    fbm(x * 7.5, y * 7.5, WORLD_SEED + 55, 2) * 0.6 +
    fbm(x * 18, y * 18, WORLD_SEED + 56, 1) * 0.4;
  const builtupRaw =
    (settlement - 0.46) * 4.2 * (0.45 + blocks * 1.1) - water * 1.1;
  const builtup = Math.max(0, Math.min(1, builtupRaw));

  const vegetation = Math.max(
    0,
    Math.min(
      1,
      (fbm(x * 1.15 - 7, y * 1.15 + 3, WORLD_SEED + 620, 5) - 0.38) * 3.4 -
        builtup * 1.2 -
        water * 1.1,
    ),
  );

  // A cloud bank over the western reach — this is what makes the optical
  // and SAR verdicts disagree, and it is the story the demo tells.
  const cloudField = fbm(x * 0.55 + 40, y * 0.55 + 40, WORLD_SEED + 1500, 5);
  const cloudMask = Math.max(0, 0.58 - u) * 2.8;
  const cloud = Math.max(0, Math.min(1, (cloudField - 0.4) * 3.6 * cloudMask));

  return { water, builtup, builtupRaw, vegetation, cloud };
}

/* ── renderers ───────────────────────────────────────────────────────── */

type Painter = (u: number, v: number, out: Uint8ClampedArray, i: number) => void;

const paintOptical: Painter = (u, v, out, i) => {
  const s = sampleWorld(u, v);
  const grain = fbm(u * 260, v * 260, WORLD_SEED + 3, 2);

  // Base: bare soil.
  let r = 122 + grain * 26;
  let g = 108 + grain * 24;
  let b = 86 + grain * 20;

  if (s.vegetation > 0) {
    const t = s.vegetation;
    r += (58 - r) * t;
    g += (92 - g) * t;
    b += (48 - b) * t;
  }
  if (s.builtup > 0) {
    const t = s.builtup;
    r += (152 + grain * 40 - r) * t;
    g += (150 + grain * 38 - g) * t;
    b += (146 + grain * 36 - b) * t;
  }
  if (s.water > 0) {
    const t = s.water;
    r += (34 - r) * t;
    g += (58 - g) * t;
    b += (78 - b) * t;
  }
  if (s.cloud > 0) {
    const t = Math.min(1, s.cloud);
    const puff = 216 + grain * 34;
    r += (puff - r) * t;
    g += (puff - g) * t;
    b += (puff + 6 - b) * t;
  }

  out[i] = r;
  out[i + 1] = g;
  out[i + 2] = b;
  out[i + 3] = 255;
};

const paintSar: Painter = (u, v, out, i) => {
  const s = sampleWorld(u, v);
  // Backscatter, roughly: smooth water is dark, urban double-bounce is
  // bright, vegetation is mid. Cloud is absent — radar sees through it.
  let db = 0.34 + s.vegetation * 0.14 + s.builtup * 0.6 - s.water * 0.3;
  // Multiplicative speckle, the signature texture of an unfiltered SAR
  // scene. Kept below the structure it modulates, so the geography still
  // reads — an over-speckled render is just noise.
  const speckle =
    0.74 +
    hash2(Math.floor(u * 3200), Math.floor(v * 3200), WORLD_SEED + 77) * 0.52;
  db = Math.max(0, Math.min(1, db * speckle));
  const level = 18 + db * 224;
  out[i] = level;
  out[i + 1] = level;
  out[i + 2] = level * 0.99;
  out[i + 3] = 255;
};

/**
 * Panchromatic: one broadband channel, so no colour and no separable NIR —
 * which is exactly why a spectral index cannot be computed from it. Rendering
 * it in colour would contradict the refusal the system gives for this scene.
 * The finer grain stands in for its much sharper native GSD.
 */
const paintPan: Painter = (u, v, out, i) => {
  const s = sampleWorld(u, v);
  const grain = fbm(u * 620, v * 620, WORLD_SEED + 9, 2);
  let level = 150 + grain * 44;
  level += (96 - level) * s.vegetation;
  level += (188 + grain * 30 - level) * s.builtup;
  level += (52 - level) * s.water;
  level += (238 - level) * Math.min(1, s.cloud);
  out[i] = level;
  out[i + 1] = level;
  out[i + 2] = level;
  out[i + 3] = 255;
};

/**
 * Mask thresholds calibrated so each mask's painted coverage matches the area
 * the corresponding AssetRef reports. Produced by scripts/calibrate-masks.mjs
 * against the 674.6 km² scene footprint; re-run it if the world changes.
 */
const MASK_THRESHOLD = {
  /** 3.40 km² once the cloud bank is subtracted. */
  waterOptical: 0.9673,
  /** 3.90 km² — radar sees the reach the optical sensor cannot. */
  waterSar: 0.9676,
  /** 28.30 km², on the unclamped field. */
  builtup: 1.2066,
} as const;

/** Water extent as the optical NDWI tool would threshold it — cloud blinds it. */
const paintWaterMask: Painter = (u, v, out, i) => {
  const s = sampleWorld(u, v);
  const detected = s.water > MASK_THRESHOLD.waterOptical && s.cloud < 0.35;
  out[i] = 59;
  out[i + 1] = 163;
  out[i + 2] = 242;
  out[i + 3] = detected ? 255 : 0;
};

/** Water extent as SAR backscatter thresholding finds it — cloud is irrelevant. */
const paintSarWaterMask: Painter = (u, v, out, i) => {
  const s = sampleWorld(u, v);
  const detected = s.water > MASK_THRESHOLD.waterSar;
  out[i] = 235;
  out[i + 1] = 178;
  out[i + 2] = 42;
  out[i + 3] = detected ? 255 : 0;
};

const paintBuiltupMask: Painter = (u, v, out, i) => {
  const s = sampleWorld(u, v);
  out[i] = 197;
  out[i + 1] = 92;
  out[i + 2] = 214;
  out[i + 3] = s.builtupRaw > MASK_THRESHOLD.builtup ? 255 : 0;
};

export type RasterKind =
  | "optical"
  | "pan"
  | "sar"
  | "mask_water_optical"
  | "mask_water_sar"
  | "mask_builtup";

const PAINTERS: Record<RasterKind, Painter> = {
  optical: paintOptical,
  pan: paintPan,
  sar: paintSar,
  mask_water_optical: paintWaterMask,
  mask_water_sar: paintSarWaterMask,
  mask_builtup: paintBuiltupMask,
};

/* ── rasterisation ───────────────────────────────────────────────────── */

/**
 * The scene's ground footprint, in WGS84. Matches the bounds the fixtures
 * report, so a tile request and the bundle's `bounds_wgs84` describe the
 * same piece of Earth — derived from the bounds rather than a hardcoded
 * tile index, which is what keeps the two from silently drifting apart.
 */
const SCENE_BOUNDS = { west: 77.05, south: 28.44, east: 77.31, north: 28.68 };

/** Web-Mercator normalised coordinates, 0..1 over the whole world. */
function lonToMercatorX(lon: number): number {
  return (lon + 180) / 360;
}

function latToMercatorY(lat: number): number {
  const radians = (lat * Math.PI) / 180;
  return (
    (1 - Math.log(Math.tan(radians) + 1 / Math.cos(radians)) / Math.PI) / 2
  );
}

const SCENE_MERCATOR = {
  x0: lonToMercatorX(SCENE_BOUNDS.west),
  x1: lonToMercatorX(SCENE_BOUNDS.east),
  // Mercator Y grows southward, so the northern edge is the smaller value.
  y0: latToMercatorY(SCENE_BOUNDS.north),
  y1: latToMercatorY(SCENE_BOUNDS.south),
};

function tileToWorldUV(
  z: number,
  x: number,
  y: number,
  px: number,
  py: number,
  size: number,
): [number, number] {
  const n = Math.pow(2, z);
  const mercX = (x + px / size) / n;
  const mercY = (y + py / size) / n;
  return [
    (mercX - SCENE_MERCATOR.x0) / (SCENE_MERCATOR.x1 - SCENE_MERCATOR.x0),
    (mercY - SCENE_MERCATOR.y0) / (SCENE_MERCATOR.y1 - SCENE_MERCATOR.y0),
  ];
}

function renderToCanvas(
  width: number,
  height: number,
  toUV: (px: number, py: number) => [number, number],
  painter: Painter,
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas context unavailable");
  const image = context.createImageData(width, height);
  const data = image.data;
  for (let py = 0; py < height; py += 1) {
    for (let px = 0; px < width; px += 1) {
      const [u, v] = toUV(px, py);
      painter(u, v, data, (py * width + px) * 4);
    }
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

async function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("canvas.toBlob returned null"));
    }, "image/png");
  });
}

const tileCache = new Map<string, ArrayBuffer>();

export async function renderTile(
  kind: RasterKind,
  z: number,
  x: number,
  y: number,
): Promise<ArrayBuffer> {
  const key = `${kind}/${z}/${x}/${y}`;
  const cached = tileCache.get(key);
  if (cached) return cached;

  const size = 256;
  const painter = PAINTERS[kind];
  const canvas = renderToCanvas(
    size,
    size,
    (px, py) => tileToWorldUV(z, x, y, px, py, size),
    // Outside the scene footprint there is no data, so the pixel is
    // transparent and the dark canvas shows through. Extrapolating the noise
    // field would draw imagery where the sensor never looked.
    (u, v, out, i) => {
      if (u < 0 || u > 1 || v < 0 || v > 1) {
        out[i + 3] = 0;
        return;
      }
      painter(u, v, out, i);
    },
  );
  const buffer = await (await canvasToBlob(canvas)).arrayBuffer();
  if (tileCache.size > 400) tileCache.clear();
  tileCache.set(key, buffer);
  return buffer;
}

const previewCache = new Map<string, ArrayBuffer>();

/** Whole-scene preview (§4.1 endpoint 6) and the non-georeferenced viewer's image. */
export async function renderPreview(
  kind: RasterKind,
  size = 512,
): Promise<ArrayBuffer> {
  const key = `${kind}/${size}`;
  const cached = previewCache.get(key);
  if (cached) return cached;
  const canvas = renderToCanvas(
    size,
    size,
    (px, py) => [px / size, py / size],
    PAINTERS[kind],
  );
  const buffer = await (await canvasToBlob(canvas)).arrayBuffer();
  previewCache.set(key, buffer);
  return buffer;
}

/** The `overlay.png` form: one reprojected PNG for the whole bounds. */
export async function renderOverlay(kind: RasterKind): Promise<ArrayBuffer> {
  return renderPreview(kind, 768);
}

/**
 * Stand-in for a GeoTIFF download. Real masks are GeoTIFF in the source CRS;
 * this keeps the download button honest in mock mode by shipping a PNG with
 * a name that says exactly what it is.
 */
export async function renderMaskDownload(
  kind: RasterKind,
): Promise<ArrayBuffer> {
  return renderPreview(kind, 1024);
}
