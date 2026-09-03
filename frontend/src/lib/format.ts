/**
 * Presentation-only formatting.
 *
 * The frontend never derives a quantity — no areas, no IoU, no confidence.
 * Everything here reshapes a number the backend already asserted, and every
 * rounded value keeps its raw form available for a title/tooltip.
 */

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/** Durations arrive as `*_ms` integers. Show the unit a human would use. */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) {
    const s = ms / 1000;
    return `${s < 10 ? s.toFixed(2) : s.toFixed(1)} s`;
  }
  const totalSeconds = Math.round(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m} min ${String(s).padStart(2, "0")} s`;
}

/** Compact form for a running clock that must not jitter in width. */
export function formatClock(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  const tenths = Math.floor((ms % 1000) / 100);
  if (m > 0) return `${m}:${String(s).padStart(2, "0")}.${tenths}`;
  return `${s}.${tenths} s`;
}

export function formatEta(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "—";
  }
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m} min ${String(s).padStart(2, "0")} s`;
}

/**
 * Confidence is displayed as a percentage for scanning, but the raw value
 * always travels with it — §7.2 forbids rounding without the original.
 */
export function formatConfidence(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${Math.round(value * 100)}%`;
}

export function formatArea(km2: number | null | undefined): string {
  if (km2 === null || km2 === undefined || !Number.isFinite(km2)) return "—";
  if (km2 < 0.01) return `${(km2 * 1_000_000).toFixed(0)} m²`;
  return `${km2 < 10 ? km2.toFixed(2) : km2.toFixed(1)} km²`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().replace("T", " ").replace(/\.\d+Z$/, "Z");
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 10);
}

export function formatLatLon(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(4)}°${ns}  ${Math.abs(lon).toFixed(4)}°${ew}`;
}

/**
 * `graded.outputs` is open by schema. Unknown keys must render, never drop.
 * Turns `area_km2` into `AREA KM2` for the equipment-code label voice.
 */
export function humaniseKey(key: string): string {
  return key.replace(/[_.]/g, " ").trim();
}

export function formatUnknownValue(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    if (value.every((v) => typeof v !== "object" || v === null)) {
      return value.map((v) => formatUnknownValue(v)).join(", ");
    }
  }
  return JSON.stringify(value);
}

/**
 * A manifest bound, in the manifest's own form. A float range declared as
 * [-1.0, 1.0] parses to [-1, 1]; showing it that way on one screen and with
 * decimals on another makes the same constraint look like two.
 */
export function formatManifestBound(
  value: number,
  type: string | undefined,
): string {
  if (type === "float" && Number.isInteger(value)) return value.toFixed(1);
  return String(value);
}
