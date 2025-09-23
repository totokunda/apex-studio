import {sha256sum} from './nodeCrypto.js';
import {versions} from './versions.js';
import {processMediaTo24} from './media/process.js';
import {renameMediaPairInRoot, deleteMediaPairInRoot} from './media/renameDelete.js';
import {ipcRenderer} from 'electron';
import {getMediaRootAbsolute} from './media/root.js';
import {converted24Dir, originalDir} from './media/paths.js';
import {promises as fsp} from 'node:fs';
import fs from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, getLowercaseExtension} from './media/fileExts.js';
import {ensureUniqueNameSync} from './media/links.js';

function send(channel: string, message: string) {
  return ipcRenderer.invoke(channel, message);
}

async function renameMediaPair(convertedOldName: string, convertedNewName: string) {
  return renameMediaPairInRoot(getMediaRootAbsolute(), convertedOldName, convertedNewName);
}

async function deleteMediaPair(convertedName: string) {
  return deleteMediaPairInRoot(getMediaRootAbsolute(), convertedName);
}

async function pickMediaPaths(options: { directory?: boolean, filters?: { name: string, extensions: string[] }[], title?: string }): Promise<string[]> {
  try {
    const paths = await ipcRenderer.invoke('dialog:pick-media', options ?? {});
    if (Array.isArray(paths)) return paths as string[];
    return [];
  } catch {
    return [];
  }
}

const readFileBuffer = async (path: string) => {
  const buffer = await fsp.readFile(fileURLToPath(path));
  return buffer;
}

export type ConvertedMediaItem = {
  name: string;
  absPath: string;
  assetUrl: string;
  dateAddedMs: number;
  type: 'video' | 'image' | 'audio' | 'other';
};

function isSupportedExt(ext: string): boolean {
  return VIDEO_EXTS.has(ext) || IMAGE_EXTS.has(ext) || AUDIO_EXTS.has(ext);
}

async function ensureMediaDirs(): Promise<{root: string, originalAbs: string, convertedAbs: string}> {
  const root = getMediaRootAbsolute();
  const originalAbs = originalDir(root);
  const convertedAbs = converted24Dir(root);
  if (!fs.existsSync(originalAbs)) fs.mkdirSync(originalAbs, {recursive: true});
  if (!fs.existsSync(convertedAbs)) fs.mkdirSync(convertedAbs, {recursive: true});
  return {root, originalAbs, convertedAbs};
}

async function listConvertedMedia(): Promise<ConvertedMediaItem[]> {
  
  const {convertedAbs} = await ensureMediaDirs();
  let entries: import('node:fs').Dirent[] = [];
  try {
    entries = await fsp.readdir(convertedAbs, {withFileTypes: true}) as unknown as import('node:fs').Dirent[];
  } catch {
    entries = [] as any;
  }
  const items: ConvertedMediaItem[] = [];
  for (const e of entries) {
    if (!e.isFile()) continue;
    const name = e.name;
    const ext = getLowercaseExtension(name);
    if (!isSupportedExt(ext)) continue;
    const absPath = join(convertedAbs, name);
    let dateAddedMs = Date.now();
    try {
      const st = await fsp.stat(absPath);
      dateAddedMs = st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
    } catch {}
    const assetUrl = pathToFileURL(absPath).href;
    const type: ConvertedMediaItem['type'] = VIDEO_EXTS.has(ext) ? 'video' : IMAGE_EXTS.has(ext) ? 'image' : AUDIO_EXTS.has(ext) ? 'audio' : 'other';
    items.push({ name, absPath, assetUrl, dateAddedMs, type });
  }
  items.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
  return items;
}

async function importMediaPaths(inputAbsPaths: string[], resolution?: string): Promise<number> {
  const {originalAbs} = await ensureMediaDirs();
  let count = 0;
  for (const srcAbs of inputAbsPaths) {
    const fileName = srcAbs.split(/[/\\]/g).pop();
    if (!fileName) continue;
    const ext = getLowercaseExtension(fileName);
    if (!isSupportedExt(ext)) continue;
    const desired = fileName;
    const uniqueName = ensureUniqueNameSync(originalAbs, desired);
    const dstAbs = join(originalAbs, uniqueName);
    await fsp.copyFile(srcAbs, dstAbs);
    try {
      await processMediaTo24({ inputAbsPath: dstAbs, resolution });
    } catch {
      // ignore per-file errors, let caller report
    }
    count += 1;
  }
  return count;
}

async function ensureUniqueConvertedName(desiredName: string): Promise<string> {
  const {convertedAbs} = await ensureMediaDirs();
  return ensureUniqueNameSync(convertedAbs, desiredName);
}

export {
  sha256sum,
  versions,
  processMediaTo24,
  renameMediaPair,
  deleteMediaPair,
  send,
  getMediaRootAbsolute,
  listConvertedMedia,
  importMediaPaths,
  ensureUniqueConvertedName,
  pickMediaPaths,
  getLowercaseExtension,
  readFileBuffer
};
