import {
  deleteFile,
  getUserDataPath,
  readFileBuffer,
  sha256sum,
  writeFileBuffer,
} from "@app/preload";

type TimelineThumbnailKind = "video" | "audio";

type TimelineThumbnailManifestEntry = {
  file: string;
  kind: TimelineThumbnailKind;
  size: number;
  sourceHash: string;
  updatedAt: number;
  accessedAt: number;
  width: number;
  height: number;
};

type TimelineThumbnailManifest = {
  version: number;
  totalBytes: number;
  entries: Record<string, TimelineThumbnailManifestEntry>;
};

type ReadCacheParams = {
  kind: TimelineThumbnailKind;
  key: string;
  sourceSignature: string;
};

type WriteCacheParams = ReadCacheParams & {
  canvas: HTMLCanvasElement;
};

const CACHE_DIR_NAME = "apex-timeline-thumbnail-cache-v1";
const MANIFEST_FILE_NAME = "manifest.json";
const MANIFEST_VERSION = 1;
const CACHE_MAX_BYTES = 48 * 1024 * 1024;
const CACHE_MAX_ENTRIES = 384;
const CACHE_MAX_ENTRY_BYTES = 2 * 1024 * 1024;
const MANIFEST_FLUSH_DEBOUNCE_MS = 800;
const WRITE_DEDUPE_WINDOW_MS = 1200;

let cacheRootPromise: Promise<string | null> | null = null;
let manifestState: TimelineThumbnailManifest | null = null;
let manifestLoadPromise: Promise<TimelineThumbnailManifest> | null = null;
let manifestFlushTimer: ReturnType<typeof setTimeout> | null = null;
let mutationQueue: Promise<void> = Promise.resolve();
const recentWritesByKey = new Map<string, number>();

function toUint8Array(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  return new Uint8Array(0);
}

function joinPath(root: string, relative: string): string {
  const left = root.replace(/[\\/]+$/, "");
  const right = relative.replace(/^[\\/]+/, "");
  return `${left}/${right}`;
}

function buildEmptyManifest(): TimelineThumbnailManifest {
  return {
    version: MANIFEST_VERSION,
    totalBytes: 0,
    entries: {},
  };
}

function recomputeTotalBytes(manifest: TimelineThumbnailManifest): number {
  return Object.values(manifest.entries).reduce((sum, entry) => {
    const size = Number(entry.size);
    return sum + (Number.isFinite(size) && size > 0 ? size : 0);
  }, 0);
}

function enqueueMutation(task: () => Promise<void>): Promise<void> {
  mutationQueue = mutationQueue.then(task).catch(() => {});
  return mutationQueue;
}

async function getCacheRoot(): Promise<string | null> {
  if (cacheRootPromise) return cacheRootPromise;
  cacheRootPromise = (async () => {
    try {
      const res = await getUserDataPath();
      const userData = res?.success && res.data?.user_data
        ? String(res.data.user_data).trim()
        : "";
      if (!userData) return null;
      return joinPath(userData, CACHE_DIR_NAME);
    } catch {
      return null;
    }
  })();
  return cacheRootPromise;
}

async function flushManifestNow(): Promise<void> {
  const root = await getCacheRoot();
  const manifest = manifestState;
  if (!root || !manifest) return;
  try {
    const payload = new TextEncoder().encode(JSON.stringify(manifest));
    await writeFileBuffer(joinPath(root, MANIFEST_FILE_NAME), payload);
  } catch {
    // Best effort only.
  }
}

function scheduleManifestFlush(): void {
  if (manifestFlushTimer != null) return;
  manifestFlushTimer = setTimeout(() => {
    manifestFlushTimer = null;
    void enqueueMutation(async () => {
      await flushManifestNow();
    });
  }, MANIFEST_FLUSH_DEBOUNCE_MS);
}

async function loadManifest(): Promise<TimelineThumbnailManifest> {
  if (manifestState) return manifestState;
  if (manifestLoadPromise) return manifestLoadPromise;

  manifestLoadPromise = (async () => {
    try {
      const root = await getCacheRoot();
      if (!root) {
        manifestState = buildEmptyManifest();
        return manifestState;
      }

      try {
        const raw = await readFileBuffer(joinPath(root, MANIFEST_FILE_NAME));
        const bytes = toUint8Array(raw);
        const parsed = JSON.parse(new TextDecoder().decode(bytes)) as Partial<TimelineThumbnailManifest>;
        const manifest: TimelineThumbnailManifest = {
          version: Number(parsed?.version) || MANIFEST_VERSION,
          totalBytes: 0,
          entries: {},
        };
        const incomingEntries = parsed?.entries ?? {};
        for (const [hash, value] of Object.entries(incomingEntries)) {
          if (!value || typeof value !== "object") continue;
          const v = value as TimelineThumbnailManifestEntry;
          if (!v.file || !v.kind) continue;
          manifest.entries[hash] = {
            file: String(v.file),
            kind: v.kind === "audio" ? "audio" : "video",
            size: Number(v.size) || 0,
            sourceHash: String(v.sourceHash || ""),
            updatedAt: Number(v.updatedAt) || 0,
            accessedAt: Number(v.accessedAt) || 0,
            width: Math.max(1, Number(v.width) || 1),
            height: Math.max(1, Number(v.height) || 1),
          };
        }
        manifest.totalBytes = recomputeTotalBytes(manifest);
        manifestState = manifest;
        return manifest;
      } catch {
        manifestState = buildEmptyManifest();
        return manifestState;
      }
    } finally {
      manifestLoadPromise = null;
    }
  })();

  return manifestLoadPromise;
}

async function removeEntry(
  root: string,
  manifest: TimelineThumbnailManifest,
  keyHash: string,
): Promise<void> {
  const entry = manifest.entries[keyHash];
  if (!entry) return;
  delete manifest.entries[keyHash];
  manifest.totalBytes = Math.max(0, manifest.totalBytes - Math.max(0, entry.size || 0));
  try {
    await deleteFile(joinPath(root, entry.file));
  } catch {
    // Best effort.
  }
}

async function pruneManifest(
  root: string,
  manifest: TimelineThumbnailManifest,
): Promise<void> {
  if (
    manifest.totalBytes <= CACHE_MAX_BYTES &&
    Object.keys(manifest.entries).length <= CACHE_MAX_ENTRIES
  ) {
    return;
  }

  const entriesByAge = Object.entries(manifest.entries).sort((a, b) => {
    const aAge = Number(a[1]?.accessedAt || a[1]?.updatedAt || 0);
    const bAge = Number(b[1]?.accessedAt || b[1]?.updatedAt || 0);
    return aAge - bAge;
  });

  for (const [keyHash] of entriesByAge) {
    if (
      manifest.totalBytes <= CACHE_MAX_BYTES &&
      Object.keys(manifest.entries).length <= CACHE_MAX_ENTRIES
    ) {
      break;
    }
    await removeEntry(root, manifest, keyHash);
  }
}

function cloneCanvas(source: HTMLCanvasElement): HTMLCanvasElement | null {
  const width = Math.max(1, Math.floor(source.width || 0));
  const height = Math.max(1, Math.floor(source.height || 0));
  if (width <= 0 || height <= 0) return null;
  const clone = document.createElement("canvas");
  clone.width = width;
  clone.height = height;
  const ctx = clone.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, width, height);
  return clone;
}

function canvasToBlob(
  canvas: HTMLCanvasElement,
  type: string,
  quality?: number,
): Promise<Blob | null> {
  return new Promise((resolve) => {
    try {
      canvas.toBlob(
        (blob) => resolve(blob),
        type,
        quality,
      );
    } catch {
      resolve(null);
    }
  });
}

async function decodeCanvas(bytes: Uint8Array): Promise<HTMLCanvasElement | null> {
  if (!bytes.length) return null;
  const normalized = new Uint8Array(bytes.byteLength);
  normalized.set(bytes);
  const blob = new Blob([normalized]);

  try {
    if (typeof createImageBitmap === "function") {
      const bitmap = await createImageBitmap(blob);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, bitmap.width);
      canvas.height = Math.max(1, bitmap.height);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        bitmap.close();
        return null;
      }
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      return canvas;
    }
  } catch {
    // Fall through to HTMLImageElement decode.
  }

  return await new Promise((resolve) => {
    try {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, img.naturalWidth || img.width || 1);
        canvas.height = Math.max(1, img.naturalHeight || img.height || 1);
        const ctx = canvas.getContext("2d");
        URL.revokeObjectURL(url);
        if (!ctx) {
          resolve(null);
          return;
        }
        ctx.drawImage(img, 0, 0);
        resolve(canvas);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      img.src = url;
    } catch {
      resolve(null);
    }
  });
}

export function stableSerializeCacheKey(value: unknown): string {
  const seen = new WeakSet<object>();

  const normalize = (input: unknown): unknown => {
    if (input === null || typeof input !== "object") return input;

    const typed = input as object;
    if (seen.has(typed)) return undefined;
    seen.add(typed);

    if (Array.isArray(input)) {
      return input.map((item) => normalize(item));
    }

    if (input instanceof Map) {
      return Array.from(input.entries())
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([k, v]) => [k, normalize(v)]);
    }

    if (input instanceof Set) {
      return Array.from(input.values())
        .map((v) => normalize(v))
        .sort((a, b) => String(a).localeCompare(String(b)));
    }

    const out: Record<string, unknown> = {};
    const record = input as Record<string, unknown>;
    for (const key of Object.keys(record).sort()) {
      const normalized = normalize(record[key]);
      if (typeof normalized === "undefined") continue;
      out[key] = normalized;
    }
    return out;
  };

  return JSON.stringify(normalize(value));
}

export async function readTimelineThumbnailDiskCache(
  params: ReadCacheParams,
): Promise<HTMLCanvasElement | null> {
  const root = await getCacheRoot();
  if (!root) return null;

  const manifest = await loadManifest();
  const keyHash = sha256sum(params.key);
  const sourceHash = sha256sum(params.sourceSignature);
  const entry = manifest.entries[keyHash];
  if (!entry) return null;
  if (entry.kind !== params.kind || entry.sourceHash !== sourceHash) {
    void enqueueMutation(async () => {
      await removeEntry(root, manifest, keyHash);
      await flushManifestNow();
    });
    return null;
  }

  try {
    const raw = await readFileBuffer(joinPath(root, entry.file));
    const bytes = toUint8Array(raw);
    const canvas = await decodeCanvas(bytes);
    if (!canvas) {
      throw new Error("decode-failed");
    }
    entry.accessedAt = Date.now();
    scheduleManifestFlush();
    return canvas;
  } catch {
    void enqueueMutation(async () => {
      await removeEntry(root, manifest, keyHash);
      await flushManifestNow();
    });
    return null;
  }
}

export function writeTimelineThumbnailDiskCacheInBackground(
  params: WriteCacheParams,
): void {
  const dedupeKey = `${params.kind}:${params.key}`;
  const now = Date.now();
  const prev = recentWritesByKey.get(dedupeKey) ?? 0;
  if (now - prev < WRITE_DEDUPE_WINDOW_MS) return;
  recentWritesByKey.set(dedupeKey, now);
  if (recentWritesByKey.size > 4096) {
    recentWritesByKey.clear();
  }

  const snapshot = cloneCanvas(params.canvas);
  if (!snapshot) return;

  void enqueueMutation(async () => {
    const root = await getCacheRoot();
    if (!root) return;

    const manifest = await loadManifest();
    const keyHash = sha256sum(params.key);
    const sourceHash = sha256sum(params.sourceSignature);
    const prevEntry = manifest.entries[keyHash];

    const webp = await canvasToBlob(snapshot, "image/webp", 0.5);
    const blob = webp ?? (await canvasToBlob(snapshot, "image/png"));
    if (!blob) return;
    const bytes = new Uint8Array(await blob.arrayBuffer());
    if (!bytes.length || bytes.byteLength > CACHE_MAX_ENTRY_BYTES) return;

    const ext = blob.type === "image/png" ? "png" : "webp";
    const relativePath = `${params.kind}/${keyHash}.${ext}`;
    await writeFileBuffer(joinPath(root, relativePath), bytes);

    if (prevEntry?.file && prevEntry.file !== relativePath) {
      try {
        await deleteFile(joinPath(root, prevEntry.file));
      } catch {
        // Best effort.
      }
    }

    manifest.entries[keyHash] = {
      file: relativePath,
      kind: params.kind,
      size: bytes.byteLength,
      sourceHash,
      updatedAt: now,
      accessedAt: now,
      width: Math.max(1, snapshot.width || 1),
      height: Math.max(1, snapshot.height || 1),
    };

    manifest.totalBytes = recomputeTotalBytes(manifest);
    await pruneManifest(root, manifest);
    await flushManifestNow();
  });
}
