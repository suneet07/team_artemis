/**
 * Measure the synthetic world's mask coverage against the areas the fixtures
 * assert, and report the thresholds that make them agree.
 *
 * The map must not contradict the numbers the evidence panel prints. A mask
 * that claims 3.4 km² and paints a fifth of the scene is exactly the kind of
 * thing a judge notices, so the thresholds are calibrated here rather than
 * guessed.
 *
 *   node scripts/calibrate-masks.mjs
 */
import process from "node:process";

/* — the world model, kept in step with mocks/synthetic.ts — */

function hash2(x, y, seed) {
  let h =
    Math.imul(x | 0, 374761393) ^
    Math.imul(y | 0, 668265263) ^
    Math.imul(seed | 0, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
const smooth = (t) => t * t * (3 - 2 * t);
function valueNoise(x, y, seed) {
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
function fbm(x, y, seed, octaves) {
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

const WORLD_SEED = 20260827;
const WORLD_SPAN = 9;

function sampleWorld(u, v) {
  const x = u * WORLD_SPAN;
  const y = v * WORLD_SPAN;
  const wander =
    Math.sin(x * 0.72) * 0.55 +
    Math.sin(x * 1.63 + 1.7) * 0.26 +
    (fbm(x * 0.5, y * 0.18, WORLD_SEED, 3) - 0.5) * 1.5;
  const channelY = WORLD_SPAN * 0.52 + wander;
  const distance = Math.abs(y - channelY);
  const floodExtent = 0.42 + fbm(x * 0.9, y * 0.9, WORLD_SEED + 900, 4) * 0.85;
  const water = Math.max(
    0,
    Math.min(1, 1 - distance / Math.max(floodExtent, 0.05)),
  );
  const settlement = fbm(x * 0.62 + 11, y * 0.62 - 4, WORLD_SEED + 310, 4);
  const blocks =
    fbm(x * 7.5, y * 7.5, WORLD_SEED + 55, 2) * 0.6 +
    fbm(x * 18, y * 18, WORLD_SEED + 56, 1) * 0.4;
  const builtupRaw =
    (settlement - 0.46) * 4.2 * (0.45 + blocks * 1.1) - water * 1.1;
  const builtup = Math.max(0, Math.min(1, builtupRaw));
  const cloudField = fbm(x * 0.55 + 40, y * 0.55 + 40, WORLD_SEED + 1500, 5);
  const cloudMask = Math.max(0, 0.58 - u) * 2.8;
  const cloud = Math.max(0, Math.min(1, (cloudField - 0.4) * 3.6 * cloudMask));
  return { water, builtup, builtupRaw, cloud };
}

/* — the scene's real ground area, from the fixture bounds — */

const BOUNDS = { west: 77.05, south: 28.44, east: 77.31, north: 28.68 };
const MEAN_LAT = ((BOUNDS.north + BOUNDS.south) / 2) * (Math.PI / 180);
const KM_PER_DEG_LAT = 110.574;
const KM_PER_DEG_LON = 111.32 * Math.cos(MEAN_LAT);
const SCENE_KM2 =
  (BOUNDS.east - BOUNDS.west) *
  KM_PER_DEG_LON *
  ((BOUNDS.north - BOUNDS.south) * KM_PER_DEG_LAT);

const N = 700;
const field = { water: [], builtup: [], builtupRaw: [], cloud: [] };
for (let j = 0; j < N; j += 1) {
  for (let i = 0; i < N; i += 1) {
    const s = sampleWorld((i + 0.5) / N, (j + 0.5) / N);
    field.water.push(s.water);
    field.builtup.push(s.builtup);
    field.builtupRaw.push(s.builtupRaw);
    field.cloud.push(s.cloud);
  }
}
const total = N * N;

/**
 * The threshold whose above-threshold fraction is closest to the target
 * coverage, found by bisection on the sorted field.
 */
function thresholdFor(values, targetKm2, gate) {
  const targetFraction = targetKm2 / SCENE_KM2;
  let lo = -2;
  let hi = 3;
  for (let step = 0; step < 40; step += 1) {
    const mid = (lo + hi) / 2;
    let count = 0;
    for (let i = 0; i < total; i += 1) {
      if (values[i] > mid && (!gate || gate(i))) count += 1;
    }
    if (count / total > targetFraction) lo = mid;
    else hi = mid;
  }
  const threshold = (lo + hi) / 2;
  let count = 0;
  for (let i = 0; i < total; i += 1) {
    if (values[i] > threshold && (!gate || gate(i))) count += 1;
  }
  return {
    threshold: Number(threshold.toFixed(4)),
    km2: Number(((count / total) * SCENE_KM2).toFixed(2)),
  };
}

const clearOfCloud = (i) => field.cloud[i] < 0.35;

process.stdout.write(
  `scene area           ${SCENE_KM2.toFixed(1)} km²\n` +
    `\ntarget → threshold that produces it:\n` +
    `  water, optical (cloud-blinded)  3.40 km²  ` +
    JSON.stringify(thresholdFor(field.water, 3.4, clearOfCloud)) +
    `\n  water, SAR (sees through cloud) 3.90 km²  ` +
    JSON.stringify(thresholdFor(field.water, 3.9)) +
    `\n  built-up extent                28.30 km²  ` +
    JSON.stringify(thresholdFor(field.builtupRaw, 28.3)) +
    `\n`,
);
