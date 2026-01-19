import fs from "node:fs";
import { promises as fsp } from "node:fs";
import { basename, extname, join, dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { ipcRenderer } from "electron";
import { spawn } from "node:child_process";
import { resolveFfmpegCommand } from "./ffmpegBin.js";
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

export type ListServerMediaPageParams = {
  folderUuid?: string;
  cursor?: string | null;
  limit?: number;
  type?: "generations" | "processors";
  sortKey?: "date" | "name";
  sortOrder?: "asc" | "desc";
};

export type ListServerMediaPageResult = {
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
    const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
    const root = join(userDataDir, "media");
    const indexPath = join(root, ".proxy-index.json");
    const content = await fsp.readFile(indexPath, "utf8");
    return JSON.parse(content);
  } catch {
    return {};
  }
}

async function saveProxyIndex(index: ProxyIndex): Promise<void> {
  const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
  const root = join(userDataDir, "media");
  const indexPath = join(root, ".proxy-index.json");
  await fsp.writeFile(indexPath, JSON.stringify(index, null, 2), "utf8");
}

async function ensureMediaDirs(): Promise<{
  root: string;
  symlinksAbs: string;
  proxyAbs: string;
}> {
  const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
  const root = join(userDataDir, "media");
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
  const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
  const baseRoot = join(userDataDir, "media");
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
  const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
  const baseRoot = join(userDataDir, "media");
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
  const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
  const baseRoot = join(userDataDir, "media");

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

async function listServerMedia(
  folderUuid?: string,
  type: "generations" | "processors" = "generations",
): Promise<ConvertedMediaItem[]> {
  try {
    // Back-compat helper: load all pages (used by older codepaths/tests).
    const all: ConvertedMediaItem[] = [];
    let cursor: string | null = null;
    for (let i = 0; i < 10_000; i++) {
      const page = await listServerMediaPage({
        folderUuid,
        cursor,
        limit: 500,
        type,
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


async function listServerMediaPage(
  params: ListServerMediaPageParams = { type: "generations" },
): Promise<ListServerMediaPageResult> {
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

    type Candidate = {
      kind: "file" | "dir";
      baseName: string;
      absPath: string;
      dateKeyMs: number;
    };


    const userDataDir = (await getUserDataPath())?.data?.user_data ?? "";
    const serverPath = folderUuid ? join(userDataDir, "media", folderUuid, "server") : join(userDataDir, "media", "server");
    const mediaPath = join(serverPath, params.type ?? "generations");
    // get all directories in generationsPath and get the files with result in their names
    
    const candidates: Candidate[] = [];
    let directories = await fsp.readdir(mediaPath, { withFileTypes: true })
    if (directories.length > 0) {
      for (const directory of directories) {
        if (directory.isDirectory()) {
          const dirPath = join(mediaPath, directory.name);
          try {
            const files = await fsp.readdir(dirPath, { withFileTypes: true });
            for (const file of files) {
              if (file.isFile()) {
                const fileName = file.name;
                const baseNameWithoutExt = basename(fileName, extname(fileName));
                // Check if the file's base name (without extension) is "result"
                if (baseNameWithoutExt === "result") {
                  const resultFile = join(dirPath, fileName);
                  const st = await fsp.stat(resultFile);
                  candidates.push({ kind: "file", baseName: directory.name, absPath: resultFile, dateKeyMs: st.birthtime?.getTime?.() ?? 0 });
                }
              }
            }
          } catch {
            // Skip directories that can't be read
            continue;
          }
        }
      }
    }


    if (candidates.length === 0) return { items: [], nextCursor: null };

    // Newest candidates first (dirs are treated like batches of results).
    const sortKey = params.sortKey ?? "date";
    const sortOrder = params.sortOrder ?? (sortKey === "name" ? "asc" : "desc");

    candidates.sort((a, b) => {
      if (sortKey === "name") {
        const cmp = a.baseName.localeCompare(b.baseName);
        return sortOrder === "asc" ? cmp : -cmp;
      }

      if (a.dateKeyMs !== b.dateKeyMs) {
        return sortOrder === "asc"
          ? a.dateKeyMs - b.dateKeyMs
          : b.dateKeyMs - a.dateKeyMs;
      }
      return b.absPath.localeCompare(a.absPath);
    });

    // Find starting index based on cursor
    let startIndex = 0;
    if (cursorObj) {
      startIndex = candidates.findIndex(
        (candidate) =>
          candidate.dateKeyMs === cursorObj.dateAddedMs &&
          candidate.absPath === cursorObj.absPath
      );
      // If cursor not found, start from beginning (could happen if item was deleted)
      // Otherwise, start after the cursor item
      if (startIndex >= 0) {
        startIndex += 1;
      } else {
        startIndex = 0;
      }
    }

    // Slice candidates based on cursor position and limit
    const paginatedCandidates = candidates.slice(startIndex, startIndex + limit);

    // Convert candidates to items
    const items: ConvertedMediaItem[] = [];
    for (const candidate of paginatedCandidates) {
      const ext = getLowercaseExtension(candidate.absPath);
      if (!isSupportedExt(ext)) continue;

      const fileName = basename(candidate.absPath);
      const assetUrl = pathToFileURL(candidate.absPath).href;
      const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
        ? "video"
        : IMAGE_EXTS.has(ext)
        ? "image"
        : AUDIO_EXTS.has(ext)
        ? "audio"
        : "video";

      items.push({
        name: fileName,
        absPath: candidate.absPath,
        assetUrl,
        dateAddedMs: candidate.dateKeyMs,
        type,
        hasProxy: false,
      });
    }


    // Generate next cursor if there are more items
    const nextCursor =
      startIndex + limit < candidates.length && items.length > 0
        ? encodeGeneratedMediaCursor({
            dateAddedMs: items[items.length - 1].dateAddedMs,
            absPath: items[items.length - 1].absPath,
          })
        : null;

    return { items, nextCursor };
} catch (e) {
  console.error("Error listing generated media page", e);
  return { items: [], nextCursor: null };
}
}


async function importMediaPaths(
  inputAbsPaths: string[],
  _resolution?: string,
  folderUuid?: string,
): Promise<number> {
  const userDataPath = (await getUserDataPath())?.data?.user_data ?? "";
  const baseRoot = join(userDataPath, "media");

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
      console.log("symlink created", dstAbs, srcAbs);
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
  const userDataPath = (await getUserDataPath())?.data?.user_data ?? "";
  const baseRoot = join(userDataPath, "media");

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
      "-g",
      "1",
      "-keyint_min",
      "1",
      "-sc_threshold",
      "0",
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
  const userDataPath = (await getUserDataPath())?.data?.user_data ?? "";
  const baseRoot = join(userDataPath, "media");

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
  listServerMedia,
  listServerMediaPage,
  importMediaPaths,
  ensureUniqueConvertedName,
  revealMediaItemInFolder,
  revealPathInFolder,
  createProxy,
  removeProxy,
  ensureMediaDirs,
  resolveOriginalPath,
};


