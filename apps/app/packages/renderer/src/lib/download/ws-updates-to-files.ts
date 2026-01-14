import type { UnifiedBucket, UnifiedDownloadWsUpdate } from "./api";

export type UnifiedDownloadProgressFile = {
  filename: string;
  downloadedBytes?: number;
  totalBytes?: number;
  status?: string;
  progress?: number | null; // normalized 0..1
  message?: string;
  bucket?: UnifiedBucket;
  label?: string;
  downloadSpeed?: number; // bytes/sec if provided by backend
};

function asNumber(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function asNonEmptyString(v: unknown): string | undefined {
  if (typeof v === "string") {
    const s = v.trim();
    return s ? s : undefined;
  }

  // Handles boxed strings: new String("foo")
  if (v instanceof String) {
    const s = String(v).trim();
    return s ? s : undefined;
  }

  // Some backends occasionally send filename/label as an array (e.g. split into characters).
  if (Array.isArray(v)) {
    const parts = v.filter((x) => typeof x === "string") as string[];
    if (!parts.length) return undefined;

    // If it looks like an array of characters, join without separators.
    if (parts.length > 1 && parts.every((p) => p.length === 1)) {
      const s = parts.join("").trim();
      return s ? s : undefined;
    }

    // Otherwise, treat it as multiple string parts.
    const s = parts.join(", ").trim();
    return s ? s : undefined;
  }

  return undefined;
}

function normalizeProgress(progress: unknown): number | null | undefined {
  const p = asNumber(progress);
  if (p == null) return undefined;
  const ratio = p > 1 ? p / 100 : p; // tolerate either 0..1 or 0..100
  return Math.max(0, Math.min(1, ratio));
}

/**
 * Convert a list of `UnifiedDownloadWsUpdate` events into per-file entries
 * consumable by download progress UIs (e.g. `ComponentCard2`).
 *
 * Notes:
 * - Groups by `metadata.filename` (fallbacks to `metadata.label`, then a stable
 *   synthetic name).
 * - Keeps the *latest* known values for each file.
 * - Normalizes `progress` to 0..1 so existing UIs can safely `* 100`.
 */
export function unifiedDownloadWsUpdatesToFiles(
  updates: UnifiedDownloadWsUpdate[] | undefined,
): UnifiedDownloadProgressFile[] {
  if (!updates?.length) return [];

  const byKey = new Map<string, UnifiedDownloadProgressFile>();
  const order: string[] = [];

  for (let i = 0; i < updates.length; i++) {
    const u = updates[i];
    const meta = (u?.metadata ?? {}) as Record<string, unknown>;

    const filename = asNonEmptyString(meta.filename);
    const label = asNonEmptyString(meta.label);

    // Prefer stable grouping keys. If the backend doesn't provide filename/label
    // (common for aggregated progress updates), avoid generating a new synthetic
    // "file-N" entry per event and instead collapse into a single stable row.
    const bucketKey =
      meta.bucket === "component" ||
      meta.bucket === "lora" ||
      meta.bucket === "preprocessor"
        ? (meta.bucket as UnifiedBucket)
        : undefined;

    const key = filename || label || bucketKey || "download";
    if (!byKey.has(key)) order.push(key);

    const prev = byKey.get(key);

    const downloadedBytes = asNumber(meta.downloaded) ?? prev?.downloadedBytes;
    const totalBytes = asNumber(meta.total) ?? prev?.totalBytes;

    // Backends sometimes include these in metadata instead of top-level fields.
    const downloadSpeed =
      asNumber((meta as any).downloadSpeed) ??
      asNumber((meta as any).speed) ??
      asNumber((meta as any).rate) ??
      prev?.downloadSpeed;

    const bucket =
      meta.bucket === "component" ||
      meta.bucket === "lora" ||
      meta.bucket === "preprocessor"
        ? (meta.bucket as UnifiedBucket)
        : prev?.bucket;

    byKey.set(key, {
      filename: filename || prev?.filename || key,
      downloadedBytes,
      totalBytes,
      status: (u as any)?.status ?? prev?.status,
      progress: normalizeProgress((u as any)?.progress) ?? prev?.progress,
      message: (u as any)?.message ?? prev?.message,
      bucket,
      label: label ?? prev?.label,
      downloadSpeed,
    });
  }

  return order.map((k) => byKey.get(k)!);
}


