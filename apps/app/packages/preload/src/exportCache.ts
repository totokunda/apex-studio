import Store from "electron-store";
import fs from "node:fs";
import { promises as fsp } from "node:fs";
import { dirname, join } from "node:path";
import { getUserDataPath } from "./config.js";

type ExportCacheEntry = {
  path: string;
  createdAt: number;
  lastAccessAt: number;
  sizeBytes?: number;
};

type ExportCacheStoreShape = {
  exports: Record<string, ExportCacheEntry>;
};

const CACHE_VERSION = 1;
const MAX_ENTRIES = 256;

const store = new Store<ExportCacheStoreShape>({
  name: `apex-export-cache-v${CACHE_VERSION}`,
  defaults: { exports: {} },
});

async function getExportCacheDir(): Promise<string | null> {
  try {
    const res = await getUserDataPath();
    if (!res?.success || !res.data?.user_data) return null;
    const dir = join(res.data.user_data, "apex-export-cache");
    await fsp.mkdir(dir, { recursive: true });
    return dir;
  } catch {
    return null;
  }
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await fsp.access(p, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function ensureDirForFile(p: string): Promise<void> {
  try {
    await fsp.mkdir(dirname(p), { recursive: true });
  } catch {
    // ignore
  }
}

async function copyOrLinkFile(src: string, dest: string): Promise<void> {
  await ensureDirForFile(dest);
  // If destination already exists, treat it as materialized.
  try {
    if (await pathExists(dest)) return;
  } catch {}

  // Try hardlink first (fast, no extra space). Falls back to copy.
  try {
    await fsp.link(src, dest);
    return;
  } catch {
    // fall through
  }
  await fsp.copyFile(src, dest);
}

async function pruneExportCacheIfNeeded(): Promise<void> {
  try {
    const exportsMap = (store.get("exports") || {}) as Record<
      string,
      ExportCacheEntry
    >;
    const keys = Object.keys(exportsMap);
    if (keys.length <= MAX_ENTRIES) return;

    const sorted = keys
      .map((k) => ({ key: k, entry: exportsMap[k] }))
      .sort((a, b) => (a.entry.lastAccessAt || 0) - (b.entry.lastAccessAt || 0));

    const toRemove = sorted.slice(0, Math.max(0, keys.length - MAX_ENTRIES));
    for (const { key, entry } of toRemove) {
      try {
        await fsp.rm(entry.path, { force: true });
      } catch {}
      delete exportsMap[key];
    }
    store.set("exports", exportsMap);
  } catch {
    // ignore
  }
}

/**
 * Returns the cached ABSOLUTE path for a given export hash, if present and on disk.
 * If the store contains a stale entry, it is removed.
 */
async function exportCacheGet(hash: string): Promise<string | null> {
  try {
    const exportsMap = (store.get("exports") || {}) as Record<
      string,
      ExportCacheEntry
    >;
    const hit = exportsMap[hash];
    if (!hit?.path) return null;
    const ok = await pathExists(hit.path);
    if (!ok) {
      delete exportsMap[hash];
      store.set("exports", exportsMap);
      return null;
    }
    hit.lastAccessAt = Date.now();
    exportsMap[hash] = hit;
    store.set("exports", exportsMap);
    return hit.path;
  } catch {
    return null;
  }
}

/**
 * Copies (or hard-links) the cached export to a requested destination path.
 * Returns the destination path on success.
 */
async function exportCacheMaterialize(
  hash: string,
  targetPath: string,
): Promise<string | null> {
  try {
    const cached = await exportCacheGet(hash);
    if (!cached) return null;
    if (!targetPath) return cached;
    await copyOrLinkFile(cached, targetPath);
    return targetPath;
  } catch {
    return null;
  }
}

/**
 * Stores an exported file into the internal cache directory for a given hash.
 * This is done by copying (or hard-linking) into userData/apex-export-cache.
 * Returns the cached absolute path on success.
 */
async function exportCachePut(
  hash: string,
  srcPath: string,
  opts?: { ext?: string },
): Promise<string | null> {
  try {
    if (!hash || !srcPath) return null;
    const ok = await pathExists(srcPath);
    if (!ok) return null;

    const cacheDir = await getExportCacheDir();
    if (!cacheDir) return null;

    const extRaw = typeof opts?.ext === "string" ? opts!.ext : "";
    const ext = extRaw
      ? extRaw.startsWith(".")
        ? extRaw
        : `.${extRaw}`
      : "";
    const cachedPath = join(cacheDir, `${hash}${ext}`);

    await copyOrLinkFile(srcPath, cachedPath);

    let sizeBytes: number | undefined;
    try {
      const st = await fsp.stat(cachedPath);
      sizeBytes = st.size;
    } catch {}

    const exportsMap = (store.get("exports") || {}) as Record<
      string,
      ExportCacheEntry
    >;
    const now = Date.now();
    exportsMap[hash] = {
      path: cachedPath,
      createdAt: exportsMap[hash]?.createdAt || now,
      lastAccessAt: now,
      sizeBytes,
    };
    store.set("exports", exportsMap);
    await pruneExportCacheIfNeeded();
    return cachedPath;
  } catch {
    return null;
  }
}

async function exportCacheDelete(hash: string): Promise<void> {
  try {
    const exportsMap = (store.get("exports") || {}) as Record<
      string,
      ExportCacheEntry
    >;
    const hit = exportsMap[hash];
    if (hit?.path) {
      try {
        await fsp.rm(hit.path, { force: true });
      } catch {}
    }
    delete exportsMap[hash];
    store.set("exports", exportsMap);
  } catch {
    // ignore
  }
}

async function exportCacheClear(): Promise<void> {
  try {
    const exportsMap = (store.get("exports") || {}) as Record<
      string,
      ExportCacheEntry
    >;
    const keys = Object.keys(exportsMap);
    for (const k of keys) {
      const p = exportsMap[k]?.path;
      if (p) {
        try {
          await fsp.rm(p, { force: true });
        } catch {}
      }
    }
    store.set("exports", {});
  } catch {
    // ignore
  }
}

export {
  exportCacheGet,
  exportCachePut,
  exportCacheMaterialize,
  exportCacheDelete,
  exportCacheClear,
};


