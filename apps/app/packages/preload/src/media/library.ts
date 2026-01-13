import fs from "node:fs";
import { promises as fsp } from "node:fs";
import os from "node:os";
import { basename, extname, join, dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { ipcRenderer } from "electron";
import { spawn } from "node:child_process";
import { resolveFfmpegCommand } from "./ffmpegBin.js";
import { getMediaRootAbsolute } from "./root.js";
import { symlinksDir, proxyDir } from "./paths.js";
import {
  AUDIO_EXTS,
  IMAGE_EXTS,
  VIDEO_EXTS,
  getLowercaseExtension,
} from "./fileExts.js";
import { ensureUniqueNameSync } from "./links.js";
import type { ClipType } from "../../../renderer/src/lib/types.js";
import { getUserDataPath } from "../config.js";
import {
  renameMediaPairInRoot,
  deleteMediaPairInRoot,
} from "./renameDelete.js";

type ProxyIndex = {
  [symlinkName: string]: string;
};

export type ConvertedMediaItem = {
  name: string;
  absPath: string;
  assetUrl: string;
  originalAssetUrl?: string;
  dateAddedMs: number;
  type: ClipType;
  hasProxy?: boolean;
};

export type ListGeneratedMediaPageParams = {
  folderUuid?: string;
  cursor?: string | null;
  limit?: number;
};

export type ListGeneratedMediaPageResult = {
  items: ConvertedMediaItem[];
  nextCursor: string | null;
};

function isSupportedExt(ext: string): boolean {
  return VIDEO_EXTS.has(ext) || IMAGE_EXTS.has(ext) || AUDIO_EXTS.has(ext);
}

type GeneratedMediaCursor = {
  dateAddedMs: number;
  absPath: string;
};

function encodeGeneratedMediaCursor(cursor: GeneratedMediaCursor): string {
  try {
    return Buffer.from(JSON.stringify(cursor), "utf8").toString("base64url");
  } catch {
    // Best-effort fallback
    return Buffer.from(JSON.stringify(cursor), "utf8").toString("base64");
  }
}

function decodeGeneratedMediaCursor(encoded: string): GeneratedMediaCursor | null {
  try {
    const json = Buffer.from(encoded, "base64url").toString("utf8");
    const parsed = JSON.parse(json);
    if (
      parsed &&
      typeof parsed.dateAddedMs === "number" &&
      Number.isFinite(parsed.dateAddedMs) &&
      typeof parsed.absPath === "string" &&
      parsed.absPath.length > 0
    ) {
      return { dateAddedMs: parsed.dateAddedMs, absPath: parsed.absPath };
    }
    return null;
  } catch {
    try {
      const json = Buffer.from(encoded, "base64").toString("utf8");
      const parsed = JSON.parse(json);
      if (
        parsed &&
        typeof parsed.dateAddedMs === "number" &&
        Number.isFinite(parsed.dateAddedMs) &&
        typeof parsed.absPath === "string" &&
        parsed.absPath.length > 0
      ) {
        return { dateAddedMs: parsed.dateAddedMs, absPath: parsed.absPath };
      }
      return null;
    } catch {
      return null;
    }
  }
}

function compareGeneratedDesc(
  a: Pick<ConvertedMediaItem, "dateAddedMs" | "absPath">,
  b: Pick<ConvertedMediaItem, "dateAddedMs" | "absPath">,
): number {
  if (a.dateAddedMs !== b.dateAddedMs) return b.dateAddedMs - a.dateAddedMs;
  return b.absPath.localeCompare(a.absPath);
}

function isAfterCursor(
  item: Pick<ConvertedMediaItem, "dateAddedMs" | "absPath">,
  cursor: GeneratedMediaCursor | null,
): boolean {
  if (!cursor) return true;
  // We paginate in descending order; "after cursor" means strictly older.
  return compareGeneratedDesc(item, cursor) > 0;
}

async function removeIfBrokenSymlink(absPath: string): Promise<boolean> {
  try {
    const st = await fsp.lstat(absPath);
    if (!st.isSymbolicLink()) return false;

    try {
      // Follow the link; if target is missing, stat() throws ENOENT.
      await fsp.stat(absPath);
      return false;
    } catch (e: any) {
      const code = e?.code;
      if (code !== "ENOENT" && code !== "ENOTDIR") return false;
      try {
        await fsp.unlink(absPath);
      } catch {
        await fsp.rm(absPath, { force: true });
      }
      return true;
    }
  } catch {
    return false;
  }
}

async function loadProxyIndex(): Promise<ProxyIndex> {
  try {
    const root = getMediaRootAbsolute();
    const indexPath = join(root, ".proxy-index.json");
    const content = await fsp.readFile(indexPath, "utf8");
    return JSON.parse(content);
  } catch {
    return {};
  }
}

async function saveProxyIndex(index: ProxyIndex): Promise<void> {
  const root = getMediaRootAbsolute();
  const indexPath = join(root, ".proxy-index.json");
  await fsp.writeFile(indexPath, JSON.stringify(index, null, 2), "utf8");
}

async function ensureMediaDirs(): Promise<{
  root: string;
  symlinksAbs: string;
  proxyAbs: string;
}> {
  const root = getMediaRootAbsolute();
  const symlinksAbs = symlinksDir(root);
  const proxyAbs = proxyDir(root);
  if (!fs.existsSync(symlinksAbs))
    fs.mkdirSync(symlinksAbs, { recursive: true });
  if (!fs.existsSync(proxyAbs)) fs.mkdirSync(proxyAbs, { recursive: true });
  return { root, symlinksAbs, proxyAbs };
}

async function renameMediaPair(
  convertedOldName: string,
  convertedNewName: string,
  folderUuid?: string,
): Promise<any> {
  const baseRoot = getMediaRootAbsolute();
  const mediaRootAbs =
    typeof folderUuid === "string" && folderUuid.length > 0
      ? join(baseRoot, folderUuid)
      : baseRoot;
  return renameMediaPairInRoot(mediaRootAbs, convertedOldName, convertedNewName);
}

async function deleteMediaPair(
  convertedName: string,
  folderUuid?: string,
): Promise<any> {
  const baseRoot = getMediaRootAbsolute();
  const mediaRootAbs =
    typeof folderUuid === "string" && folderUuid.length > 0
      ? join(baseRoot, folderUuid)
      : baseRoot;
  return deleteMediaPairInRoot(mediaRootAbs, convertedName);
}

async function pickMediaPaths(options: {
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
  title?: string;
  defaultPath?: string;
}): Promise<string[]> {
  try {
    const paths = await ipcRenderer.invoke("dialog:pick-media", options ?? {});
    if (Array.isArray(paths)) return paths as string[];
    return [];
  } catch {
    return [];
  }
}

async function listConvertedMedia(
  folderUuid?: string,
): Promise<ConvertedMediaItem[]> {
  const baseRoot = getMediaRootAbsolute();

  let symlinksAbs: string;
  let proxyAbs: string;
  if (typeof folderUuid === "string" && folderUuid.length > 0) {
    const projectRoot = join(baseRoot, folderUuid);
    symlinksAbs = symlinksDir(projectRoot);
    proxyAbs = proxyDir(projectRoot);
    if (!fs.existsSync(symlinksAbs)) {
      fs.mkdirSync(symlinksAbs, { recursive: true });
    }
    if (!fs.existsSync(proxyAbs)) {
      fs.mkdirSync(proxyAbs, { recursive: true });
    }
  } else {
    const baseDirs = await ensureMediaDirs();
    symlinksAbs = baseDirs.symlinksAbs;
    proxyAbs = baseDirs.proxyAbs;
  }

  const proxyIndex = await loadProxyIndex();
  let proxyIndexDirty = false;

  let entries: import("node:fs").Dirent[] = [];
  try {
    entries = (await fsp.readdir(symlinksAbs, {
      withFileTypes: true,
    })) as unknown as import("node:fs").Dirent[];
  } catch {
    entries = [] as any;
  }

  const items: ConvertedMediaItem[] = [];
  for (const e of entries) {
    const name = e.name;
    const ext = getLowercaseExtension(name);
    if (!isSupportedExt(ext)) continue;

    const originalAbsPath = join(symlinksAbs, name);

    // If the media item is a symlink and its target has been deleted, remove
    // the broken symlink from disk (and any proxy mapping) and skip it.
    if (await removeIfBrokenSymlink(originalAbsPath)) {
      if (proxyIndex[name]) {
        const proxyPath = join(proxyAbs, proxyIndex[name]);
        try {
          await fsp.rm(proxyPath, { force: true });
        } catch {
          // ignore
        }
        delete proxyIndex[name];
        proxyIndexDirty = true;
      }
      continue;
    }

    let absPath = originalAbsPath;
    let hasProxy = false;

    if (proxyIndex[name]) {
      const proxyPath = join(proxyAbs, proxyIndex[name]);
      if (fs.existsSync(proxyPath)) {
        absPath = proxyPath;
        hasProxy = true;
      }
    }

    let dateAddedMs = Date.now();
    try {
      const st = await fsp.lstat(originalAbsPath);
      dateAddedMs =
        st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
    } catch {
      // If we can't stat it (race / deleted), don't return it.
      continue;
    }

    const originalAssetUrl = pathToFileURL(originalAbsPath).href;
    const assetUrl = pathToFileURL(absPath).href;
    const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
      ? "video"
      : IMAGE_EXTS.has(ext)
      ? "image"
      : AUDIO_EXTS.has(ext)
      ? "audio"
      : "video";

    items.push({
      name,
      absPath,
      assetUrl,
      originalAssetUrl,
      dateAddedMs,
      type,
      hasProxy,
    });
  }

  if (proxyIndexDirty) {
    try {
      await saveProxyIndex(proxyIndex);
    } catch {
      // ignore
    }
  }

  items.sort((a, b) =>
    a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
  );
  return items;
}

async function listGeneratedMedia(
  folderUuid?: string,
): Promise<ConvertedMediaItem[]> {
  try {
    // Back-compat helper: load all pages (used by older codepaths/tests).
    const all: ConvertedMediaItem[] = [];
    let cursor: string | null = null;
    for (let i = 0; i < 10_000; i++) {
      const page = await listGeneratedMediaPage({
        folderUuid,
        cursor,
        limit: 500,
      });
      all.push(...page.items);
      if (!page.nextCursor) break;
      cursor = page.nextCursor;
    }
    return all;
  } catch {
    return [];
  }
}

let generationRootsCache:
  | { atMs: number; roots: string[]; cachePath: string | null; userDataDir: string | null }
  | null = null;

async function getGenerationRoots(): Promise<string[]> {
  const now = Date.now();
  if (generationRootsCache && now - generationRootsCache.atMs < 30_000) {
    return generationRootsCache.roots;
  }

  // Prefer the cache path from the backend when available, but gracefully
  // fall back to the local default so generations still show up when the
  // API server is offline or unreachable.
  let cachePath: string | null = null;
    try {
      const envPath = process.env.APEX_CACHE_PATH;
      if (typeof envPath === "string" && envPath.length > 0) {
        cachePath = envPath;
      } else {
        const home = os.homedir?.();
        if (home && typeof home === "string" && home.length > 0) {
          cachePath = join(home, "apex-diffusion", "cache");
        }
      }
    } catch {
      cachePath = null;
    }
  
  let userDataDir: string | null = null;
  try {
    const userDataRes = await getUserDataPath();
    if (userDataRes?.success && userDataRes.data?.user_data) {
      userDataDir = userDataRes.data.user_data;
    }
  } catch {
    userDataDir = null;
  }

  const roots = new Set<string>();
  if (cachePath) roots.add(cachePath);
  if (userDataDir) {
    roots.add(join(userDataDir, "apex-cache"));
    roots.add(join(userDataDir, "apex-cache-remote"));
  }

  const rootsArr = Array.from(roots.values());
  generationRootsCache = { atMs: now, roots: rootsArr, cachePath, userDataDir };
  return rootsArr;
}

async function listGeneratedMediaPage(
  params: ListGeneratedMediaPageParams = {},
): Promise<ListGeneratedMediaPageResult> {
  try {
    const folderUuid = params.folderUuid;
    const limit =
      typeof params.limit === "number" && Number.isFinite(params.limit)
        ? Math.max(1, Math.min(500, Math.floor(params.limit)))
        : 60;

    const cursorObj =
      typeof params.cursor === "string" && params.cursor.length > 0
        ? decodeGeneratedMediaCursor(params.cursor)
        : null;

    const roots = await getGenerationRoots();
    if (!roots || roots.length === 0) return { items: [], nextCursor: null };

    type Candidate = {
      kind: "file" | "dir";
      baseName: string;
      absPath: string;
      dateKeyMs: number;
    };

    const candidates: Candidate[] = [];
    for (const root of roots) {
      // IMPORTANT:
      // - The backend historically writes to: <cache-root>/engine_results/<jobId>/...
      // - The renderer *may* pass folderUuid to scope generations to a project.
      // - On some platforms (notably Windows without Developer Mode/admin),
      //   creating symlinks/junctions under engine_results/<folderUuid>/... may fail,
      //   which would make generations appear "missing" even though outputs exist.
      //
      // To make generations reliably appear, when folderUuid is provided we scan BOTH:
      //   <root>/engine_results/<folderUuid>  (preferred when present)
      //   <root>/engine_results              (global fallback)
      //
      // Results are deduped by absolute path downstream.
      const engineResultsDirs = new Set<string>();
      if (typeof folderUuid === "string" && folderUuid.length > 0) {
        engineResultsDirs.add(join(root, "engine_results", folderUuid));
      }
      engineResultsDirs.add(join(root, "engine_results"));

      for (const engineResultsAbs of engineResultsDirs) {
        if (!fs.existsSync(engineResultsAbs)) continue;

        let entries: import("node:fs").Dirent[] = [];
        try {
          entries = (await fsp.readdir(engineResultsAbs, {
            withFileTypes: true,
          })) as unknown as import("node:fs").Dirent[];
        } catch {
          entries = [] as any;
        }
        if (!entries || entries.length === 0) continue;

        for (const e of entries) {
          const entryName = e.name;
          const entryPath = join(engineResultsAbs, entryName);

          if (!e.isDirectory()) {
            const ext = getLowercaseExtension(entryName);
            if (!isSupportedExt(ext)) continue;
            const stem = entryName.replace(/\.[^.]+$/, "");
            if (!stem.toLowerCase().startsWith("result")) continue;
          }

          let dateKeyMs = Date.now();
          try {
            const st = await fsp.lstat(entryPath);
            dateKeyMs =
              st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateKeyMs;
          } catch {
            // ignore
          }

          candidates.push({
            kind: e.isDirectory() ? "dir" : "file",
            baseName: entryName,
            absPath: entryPath,
            dateKeyMs,
          });
        }
      }
    }

    if (candidates.length === 0) return { items: [], nextCursor: null };

    // Newest candidates first (dirs are treated like batches of results).
    candidates.sort((a, b) => {
      if (a.dateKeyMs !== b.dateKeyMs) return b.dateKeyMs - a.dateKeyMs;
      return b.absPath.localeCompare(a.absPath);
    });

    const items: ConvertedMediaItem[] = [];
    const seen = new Set<string>();

    // Key used for stable dedupe + pagination across multiple roots (cache/user-data/remote).
    // If the same engine output is mirrored under different roots, absPath differs but this
    // relative engine_results key stays consistent.
    const generationKeyFromAbsPath = (absPath: string): string => {
      const normalized = absPath.replace(/\\/g, "/");
      const marker = "/engine_results/";
      const idx = normalized.lastIndexOf(marker);
      if (idx >= 0) return normalized.slice(idx + 1); // "engine_results/..."
      return normalized;
    };

    const canonicalizeAbsPath = async (absPath: string): Promise<string> => {
      // Resolve any symlink in the path (including parent dirs). This helps collapse
      // symlinked roots and junctions; if not resolvable, fall back to the original.
      try {
        return await fsp.realpath(absPath);
      } catch {
        return absPath;
      }
    };

    type Keyed = { item: ConvertedMediaItem; key: string };
    const keyed: Keyed[] = [];

    const compareKeyedDesc = (a: Keyed, b: Keyed): number => {
      if (a.item.dateAddedMs !== b.item.dateAddedMs)
        return b.item.dateAddedMs - a.item.dateAddedMs;
      return b.key.localeCompare(a.key);
    };

    const isAfterCursorKeyed = (it: Keyed, cursor: any): boolean => {
      if (!cursor) return true;
      const cursorKey = String(cursor.absPath ?? "");
      const cursorDate = Number(cursor.dateAddedMs ?? 0);
      if (!Number.isFinite(cursorDate) || !cursorKey) return true;
      if (it.item.dateAddedMs !== cursorDate) return it.item.dateAddedMs < cursorDate;
      // Descending order; "after cursor" means strictly older / smaller tie-break.
      return it.key < cursorKey;
    };

    const pushItem = (it: ConvertedMediaItem) => {
      const key = generationKeyFromAbsPath(it.absPath);
      if (seen.has(key)) return;
      const k: Keyed = { item: it, key };
      if (!isAfterCursorKeyed(k, cursorObj)) return;
      seen.add(key);
      keyed.push(k);
      items.push(it);
    };

    for (const c of candidates) {
      if (items.length >= limit) break;

      if (c.kind === "file") {
        const name = c.baseName;
        if (await removeIfBrokenSymlink(c.absPath)) continue;
        const ext = getLowercaseExtension(name);
        const canonicalAbsPath = await canonicalizeAbsPath(c.absPath);
        const assetUrl = pathToFileURL(canonicalAbsPath).href;
        const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
          ? "video"
          : IMAGE_EXTS.has(ext)
          ? "image"
          : AUDIO_EXTS.has(ext)
          ? "audio"
          : "video";
        pushItem({
          name,
          absPath: canonicalAbsPath,
          assetUrl,
          dateAddedMs: c.dateKeyMs,
          type,
        });
        continue;
      }

      // Directory candidate: scan for "result*" media, newest first.
      let jobEntries: import("node:fs").Dirent[] = [];
      try {
        jobEntries = (await fsp.readdir(c.absPath, {
          withFileTypes: true,
        })) as unknown as import("node:fs").Dirent[];
      } catch {
        jobEntries = [] as any;
      }
      if (!jobEntries || jobEntries.length === 0) continue;

      const jobItems: ConvertedMediaItem[] = [];
      for (const je of jobEntries) {
        if (items.length + jobItems.length >= limit * 2) {
          // Avoid runaway work in huge job dirs; we only need enough to fill the page.
          break;
        }

        const fileName = je.name;
        const ext = getLowercaseExtension(fileName);
        if (!isSupportedExt(ext)) continue;
        const stem = fileName.replace(/\.[^.]+$/, "");
        if (!stem.toLowerCase().startsWith("result")) continue;

        const absPath = join(c.absPath, fileName);
        if (await removeIfBrokenSymlink(absPath)) continue;
        const canonicalAbsPath = await canonicalizeAbsPath(absPath);
        let dateAddedMs = c.dateKeyMs;
        try {
          const st = await fsp.lstat(absPath);
          dateAddedMs =
            st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
        } catch {
          // ignore
        }

        const assetUrl = pathToFileURL(canonicalAbsPath).href;
        const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
          ? "video"
          : IMAGE_EXTS.has(ext)
          ? "image"
          : AUDIO_EXTS.has(ext)
          ? "audio"
          : "video";
        const displayName = `${basename(c.absPath)}/${fileName}`;
        jobItems.push({
          name: displayName,
          absPath: canonicalAbsPath,
          assetUrl,
          dateAddedMs,
          type,
        });
      }

      if (jobItems.length === 0) continue;
      jobItems.sort((a, b) => {
        if (a.dateAddedMs !== b.dateAddedMs) return b.dateAddedMs - a.dateAddedMs;
        const ak = generationKeyFromAbsPath(a.absPath);
        const bk = generationKeyFromAbsPath(b.absPath);
        return bk.localeCompare(ak);
      });

      for (const it of jobItems) {
        if (items.length >= limit) break;
        pushItem(it);
      }
    }

    if (items.length === 0) return { items: [], nextCursor: null };
    items.sort(compareGeneratedDesc);
    const sliced = items.slice(0, limit);
    // Next cursor uses the same stable key so duplicates can't "slip" onto later pages.
    const last = sliced[sliced.length - 1];
    const nextCursor =
      sliced.length < limit
        ? null
        : encodeGeneratedMediaCursor({
            dateAddedMs: last.dateAddedMs,
            absPath: generationKeyFromAbsPath(last.absPath),
          });
    return { items: sliced, nextCursor };
  } catch {
    return { items: [], nextCursor: null };
  }
}

async function importMediaPaths(
  inputAbsPaths: string[],
  _resolution?: string,
  folderUuid?: string,
): Promise<number> {
  const baseRoot = getMediaRootAbsolute();

  let symlinksAbs: string;
  if (typeof folderUuid === "string" && folderUuid.length > 0) {
    const projectRoot = join(baseRoot, folderUuid);
    symlinksAbs = symlinksDir(projectRoot);
    if (!fs.existsSync(symlinksAbs)) {
      fs.mkdirSync(symlinksAbs, { recursive: true });
    }
  } else {
    const { symlinksAbs: globalSymlinks } = await ensureMediaDirs();
    symlinksAbs = globalSymlinks;
  }

  let count = 0;
  for (const srcAbs of inputAbsPaths) {
    const fileName = srcAbs.split(/[/\\]/g).pop();
    if (!fileName) continue;
    const ext = getLowercaseExtension(fileName);
    if (!isSupportedExt(ext)) continue;
    const uniqueName = ensureUniqueNameSync(symlinksAbs, fileName);
    const dstAbs = join(symlinksAbs, uniqueName);
    try {
      await fsp.rm(dstAbs, { force: true });
      await fsp.symlink(srcAbs, dstAbs);
    } catch (error) {
      console.error("Error creating symlink:", error);
      try {
        await fsp.copyFile(srcAbs, dstAbs);
      } catch (copyError) {
        console.error("Error copying file:", copyError);
        continue;
      }
    }
    count += 1;
  }
  return count;
}

async function ensureUniqueConvertedName(desiredName: string): Promise<string> {
  const { symlinksAbs } = await ensureMediaDirs();
  return ensureUniqueNameSync(symlinksAbs, desiredName);
}

async function revealMediaItemInFolder(fileName: string): Promise<void> {
  try {
    if (!fileName) return;
    const { symlinksAbs } = await ensureMediaDirs();
    const symlinkPath = join(symlinksAbs, fileName);
    let targetPath = symlinkPath;
    try {
      const st = await fsp.lstat(symlinkPath);
      if (st.isSymbolicLink()) {
        targetPath = await fsp.realpath(symlinkPath);
      }
    } catch {
      // ignore
    }

    await ipcRenderer.invoke("files:reveal-in-folder", targetPath);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error("revealMediaItemInFolder failed", e);
  }
}

async function revealPathInFolder(absPath: string): Promise<void> {
  try {
    if (!absPath) return;
    await ipcRenderer.invoke("files:reveal-in-folder", absPath);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error("revealPathInFolder failed", e);
  }
}

async function createProxy(
  fileName: string,
  resolution: string = "480p",
  folderUuid?: string,
): Promise<void> {
  const baseRoot = getMediaRootAbsolute();

  let symlinksAbs: string;
  let proxyAbs: string;
  if (typeof folderUuid === "string" && folderUuid.length > 0) {
    const projectRoot = join(baseRoot, folderUuid);
    symlinksAbs = symlinksDir(projectRoot);
    proxyAbs = proxyDir(projectRoot);
    if (!fs.existsSync(symlinksAbs))
      fs.mkdirSync(symlinksAbs, { recursive: true });
    if (!fs.existsSync(proxyAbs)) fs.mkdirSync(proxyAbs, { recursive: true });
  } else {
    const dirs = await ensureMediaDirs();
    symlinksAbs = dirs.symlinksAbs;
    proxyAbs = dirs.proxyAbs;
  }
  const symlinkPath = join(symlinksAbs, fileName);

  if (!fs.existsSync(symlinkPath)) {
    throw new Error("Source file not found");
  }

  const realPath = await fsp.realpath(symlinkPath);

  const baseName = basename(fileName, extname(fileName));
  const proxyFileName = `${baseName}_proxy.mp4`;
  const proxyPath = join(proxyAbs, proxyFileName);

  const parseHeight = (res: string): number => {
    const s = res.trim().toLowerCase();
    if (s === "720" || s === "720p" || s === "hd") return 720;
    if (s === "1080" || s === "1080p") return 1080;
    const raw = s.endsWith("p") ? s.slice(0, -1) : s;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : 480;
  };

  const targetHeight = parseHeight(resolution);

  await new Promise<void>((resolve, reject) => {
    const args = [
      "-i",
      realPath,
      "-vf",
      `scale=-2:${targetHeight}`,
      "-r",
      "24",
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "23",
      "-c:a",
      "aac",
      "-b:a",
      "128k",
      "-ar",
      "48000",
      "-y",
      proxyPath,
    ];

    const ffmpeg = spawn(resolveFfmpegCommand("ffmpeg"), args);

    let stderr = "";
    ffmpeg.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    ffmpeg.on("close", async (code) => {
      if (code === 0) {
        const proxyIndex = await loadProxyIndex();
        proxyIndex[fileName] = proxyFileName;
        await saveProxyIndex(proxyIndex);
        resolve();
      } else {
        reject(new Error(`FFmpeg failed with code ${code}: ${stderr}`));
      }
    });

    ffmpeg.on("error", (err) => {
      reject(err);
    });
  });
}

async function removeProxy(
  fileName: string,
  folderUuid?: string,
): Promise<void> {
  const baseRoot = getMediaRootAbsolute();

  let proxyAbs: string;
  if (typeof folderUuid === "string" && folderUuid.length > 0) {
    const projectRoot = join(baseRoot, folderUuid);
    proxyAbs = proxyDir(projectRoot);
    if (!fs.existsSync(proxyAbs)) {
      const proxyIndex = await loadProxyIndex();
      if (proxyIndex[fileName]) {
        delete proxyIndex[fileName];
        await saveProxyIndex(proxyIndex);
      }
      return;
    }
  } else {
    const dirs = await ensureMediaDirs();
    proxyAbs = dirs.proxyAbs;
  }
  const proxyIndex = await loadProxyIndex();

  if (proxyIndex[fileName]) {
    const proxyPath = join(proxyAbs, proxyIndex[fileName]);
    if (fs.existsSync(proxyPath)) {
      await fsp.rm(proxyPath, { force: true });
    }
    delete proxyIndex[fileName];
    await saveProxyIndex(proxyIndex);
  }
}

async function resolveOriginalPath(path: string): Promise<string> {
  try {
    // Check if it's likely a proxy path
    if (!path.includes("proxy")) return path;

    const proxyIndex = await loadProxyIndex();
    const fileName = basename(path);
    
    // Find key for this proxy file
    const originalName = Object.keys(proxyIndex).find(
      (key) => proxyIndex[key] === fileName
    );

    if (originalName) {
      // Reconstruct original path assuming standard structure
      const dir = dirname(path);
      // Check if we are in a 'proxy' directory
      if (basename(dir) === "proxy") {
        const root = dirname(dir);
        const symlinks = symlinksDir(root);
        const originalPath = join(symlinks, originalName);
        
        // Verify existence
        if (fs.existsSync(originalPath)) {
          return originalPath;
        }
      }
    }
  } catch (e) {
    console.warn("Failed to resolve original path for:", path, e);
  }
  return path;
}

export {
  renameMediaPair,
  deleteMediaPair,
  pickMediaPaths,
  listConvertedMedia,
  listGeneratedMedia,
  listGeneratedMediaPage,
  importMediaPaths,
  ensureUniqueConvertedName,
  revealMediaItemInFolder,
  revealPathInFolder,
  createProxy,
  removeProxy,
  ensureMediaDirs,
  resolveOriginalPath,
};


