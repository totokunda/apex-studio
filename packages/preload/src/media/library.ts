import fs from "node:fs";
import { promises as fsp } from "node:fs";
import { basename, extname, join, dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { ipcRenderer } from "electron";
import { spawn } from "node:child_process";
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
import { getCachePath, getUserDataPath } from "../config.js";
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

function isSupportedExt(ext: string): boolean {
  return VIDEO_EXTS.has(ext) || IMAGE_EXTS.has(ext) || AUDIO_EXTS.has(ext);
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
      const st = await fsp.lstat(join(symlinksAbs, name));
      dateAddedMs =
        st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
    } catch {
      // ignore
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

  items.sort((a, b) =>
    a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
  );
  return items;
}

async function listGeneratedMedia(
  folderUuid?: string,
): Promise<ConvertedMediaItem[]> {
  try {
    const cacheRes = await getCachePath();
    const cachePath =
      cacheRes?.success && cacheRes.data?.cache_path
        ? cacheRes.data.cache_path
        : null;

    const userDataRes = await getUserDataPath();
    const userDataDir =
      userDataRes?.success && userDataRes.data?.user_data
        ? userDataRes.data.user_data
        : null;

    const roots = new Set<string>();
    if (cachePath) {
      roots.add(cachePath);
    }
    if (userDataDir) {
      roots.add(join(userDataDir, "apex-cache"));
      roots.add(join(userDataDir, "apex-cache-remote"));
    }

    const collected: ConvertedMediaItem[] = [];

    for (const root of roots) {
      let engineResultsAbs: string;
      if (typeof folderUuid === "string" && folderUuid.length > 0) {
        engineResultsAbs = join(root, "engine_results", folderUuid);
      } else {
        engineResultsAbs = join(root, "engine_results");
      }
      if (!fs.existsSync(engineResultsAbs)) {
        continue;
      }

      let entries: import("node:fs").Dirent[] = [];
      try {
        entries = (await fsp.readdir(engineResultsAbs, {
          withFileTypes: true,
        })) as unknown as import("node:fs").Dirent[];
      } catch {
        entries = [] as any;
      }

      if (entries.length === 0) continue;

      for (const e of entries) {
        const entryName = e.name;
        const entryPath = join(engineResultsAbs, entryName);

        if (e.isDirectory()) {
          let jobEntries: import("node:fs").Dirent[] = [];
          try {
            jobEntries = (await fsp.readdir(entryPath, {
              withFileTypes: true,
            })) as unknown as import("node:fs").Dirent[];
          } catch {
            jobEntries = [] as any;
          }

          for (const je of jobEntries) {
            const name = je.name;
            const ext = getLowercaseExtension(name);
            if (!isSupportedExt(ext)) continue;

            const stem = name.replace(/\.[^.]+$/, "");
            if (!stem.toLowerCase().startsWith("result")) continue;

            const absPath = join(entryPath, name);

            let dateAddedMs = Date.now();
            try {
              const st = await fsp.lstat(absPath);
              dateAddedMs =
                st.birthtime?.getTime?.() ??
                st.mtime?.getTime?.() ??
                dateAddedMs;
            } catch {
              // ignore
            }

            const assetUrl = pathToFileURL(absPath).href;
            const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
              ? "video"
              : IMAGE_EXTS.has(ext)
              ? "image"
              : AUDIO_EXTS.has(ext)
              ? "audio"
              : "video";

            const displayName = `${entryName}/${name}`;
            collected.push({
              name: displayName,
              absPath,
              assetUrl,
              dateAddedMs,
              type,
            });
          }
        } else {
          const name = entryName;
          const ext = getLowercaseExtension(name);
          if (!isSupportedExt(ext)) continue;

          const stem = name.replace(/\.[^.]+$/, "");
          if (!stem.toLowerCase().startsWith("result")) continue;

          const absPath = entryPath;

          let dateAddedMs = Date.now();
          try {
            const st = await fsp.lstat(absPath);
            dateAddedMs =
              st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
          } catch {
            // ignore
          }

          const assetUrl = pathToFileURL(absPath).href;
          const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
            ? "video"
            : IMAGE_EXTS.has(ext)
            ? "image"
            : AUDIO_EXTS.has(ext)
            ? "audio"
            : "video";

          collected.push({ name, absPath, assetUrl, dateAddedMs, type });
        }
      }
    }

    if (collected.length === 0) return [];

    const byPath = new Map<string, ConvertedMediaItem>();
    for (const it of collected) {
      if (!byPath.has(it.absPath)) {
        byPath.set(it.absPath, it);
      }
    }

    const unique = Array.from(byPath.values());
    unique.sort((a, b) => (b.dateAddedMs ?? 0) - (a.dateAddedMs ?? 0));
    return unique;
  } catch {
    return [];
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

    const ffmpeg = spawn("ffmpeg", args);

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
  importMediaPaths,
  ensureUniqueConvertedName,
  revealMediaItemInFolder,
  revealPathInFolder,
  createProxy,
  removeProxy,
  ensureMediaDirs,
  resolveOriginalPath,
};


