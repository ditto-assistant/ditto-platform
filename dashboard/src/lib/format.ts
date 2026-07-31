// Pure formatting helpers. No DOM, no state — everything here is a direct
// port of the original formatters, exact output formats included.

import type { TimelineRelease } from "../types";

/** Fraction as a percent with one decimal: 0.123 → "12.3%". */
export function pct(x: number): string {
  return (x * 100).toFixed(1) + "%";
}

/** Fixed three decimals: 0.4917 → "0.492". */
export function fx(x: number): string {
  return x.toFixed(3);
}

// Trailing zeros on a consensus parameter read as false precision.
/** ≤4 decimals with trailing zeros stripped: 0.5000 → "0.5". */
export function num(x: number): string {
  return String(Number(x.toFixed(4)));
}

/** Human copy for the dethrone margin; a non-finite margin reads "incumbent". */
export function marginText(margin: number | null | undefined): string {
  if (!Number.isFinite(margin)) return "incumbent";
  return num(margin as number) + " composite points";
}

/** Clamp to [0, 1]. */
export function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/** "N ms" below 1s, seconds with 1 decimal below 10s, 0 decimals at/above 10s. */
export function fmtMs(ms: number): string {
  return ms >= 1000 ? (ms / 1000).toFixed(ms >= 10000 ? 0 : 1) + " s" : ms + " ms";
}

/** Median of a numeric array; 0 for empty input, never mutates the input. */
export function median(nums: number[]): number {
  if (!nums.length) return 0;
  const s = nums.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
}

/** Relative time from an ISO string: "5m ago". Invalid dates render "–" (en
 * dash); future timestamps clamp to "0s ago". */
export function relTime(iso: string | null | undefined): string {
  const t = iso == null ? NaN : new Date(iso).getTime();
  if (isNaN(t)) return "–";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return Math.floor(s) + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}

/** Abbreviate an ss58 hotkey for a compact row (full value stays in the
 * title): first 8 chars + "…" + last 6 when longer than 16. Null-safe. */
export function shortKey(k: string | null | undefined): string {
  return k && k.length > 16 ? k.slice(0, 8) + "…" + k.slice(-6) : k || "";
}

/** Strip the provider prefix from a model name (last "/" segment). */
export function shortModel(name: string): string {
  return name.indexOf("/") >= 0 ? (name.split("/").pop() ?? name) : name;
}

/** Coerce to a non-negative integer count; invalid/negative → 0. */
export function telemetryCount(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : 0;
}

/** Format an ms duration: invalid/negative → "—"; <1s → "N ms"; <60s →
 * seconds (1 decimal below 10s); else "<m>m <s>s". */
export function telemetryDuration(value: unknown): string {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "—";
  if (milliseconds < 1000) return Math.round(milliseconds) + " ms";
  if (milliseconds < 60000)
    return (milliseconds / 1000).toFixed(milliseconds < 10000 ? 1 : 0) + " s";
  return Math.floor(milliseconds / 60000) + "m " + Math.round((milliseconds % 60000) / 1000) + "s";
}

/** Elapsed time since an ISO timestamp as "<h>h <m>m <s>s" (leading units
 * omitted when zero); "" for an invalid/missing start. */
export function elapsedDuration(startedAt: string | null | undefined): string {
  const started = Date.parse(startedAt || "");
  if (!Number.isFinite(started)) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return (hours ? hours + "h " : "") + (hours || minutes ? minutes + "m " : "") + remainder + "s";
}

// Shared UTC date formatter for the memory timeline, so a release marker and
// a point placed on it never disagree across viewer timezones.
const TIMELINE_DATE = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

/** "Mon D" in UTC via the shared timeline formatter. */
export function timelineDate(value: string | number): string {
  return TIMELINE_DATE.format(new Date(value));
}

/** Epoch ms of a timeline release's released_at (NaN when unparsable). */
export function releaseTime(release: TimelineRelease): number {
  return Date.parse(release.released_at ?? "");
}

/** Locale date-time (medium/short); "Not recorded" for missing/unparsable. */
export function athDate(value: string | number | null | undefined): string {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not recorded";
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
