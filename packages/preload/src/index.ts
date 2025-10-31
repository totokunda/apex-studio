import {sha256sum} from './nodeCrypto.js';
import {versions} from './versions.js';
import {processMediaTo24} from './media/process.js';
import {renameMediaPairInRoot, deleteMediaPairInRoot} from './media/renameDelete.js';
import {ipcRenderer, webUtils} from 'electron';
import {getMediaRootAbsolute} from './media/root.js';
import {symlinksDir, proxyDir} from './media/paths.js';
import {promises as fsp} from 'node:fs';
import fs from 'node:fs';
import {join, basename, extname} from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, getLowercaseExtension} from './media/fileExts.js';
import {ensureUniqueNameSync} from './media/links.js';
import { ClipType } from '../../renderer/src/lib/types.js';
import { fetchFilters } from './filters/fetch.js';
import {spawn} from 'node:child_process';  

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
  // check if path is a file url
  if (path.startsWith('file://')) {
    path = fileURLToPath(path);
  }
  const buffer = await fsp.readFile(path);
  return buffer;
}

const readFileStream = async (path: string) => {
  const stream = await fs.createReadStream(fileURLToPath(path));
  return stream;
}

export type ConvertedMediaItem = {
  name: string;
  absPath: string;
  assetUrl: string;
  dateAddedMs: number;
  type: ClipType;
  hasProxy?: boolean;
};

type ProxyIndex = {
  [symlinkName: string]: string; // maps symlink name to proxy filename
};

function isSupportedExt(ext: string): boolean {
  return VIDEO_EXTS.has(ext) || IMAGE_EXTS.has(ext) || AUDIO_EXTS.has(ext);
}

async function loadProxyIndex(): Promise<ProxyIndex> {
  try {
    const root = getMediaRootAbsolute();
    const indexPath = join(root, '.proxy-index.json');
    const content = await fsp.readFile(indexPath, 'utf8');
    return JSON.parse(content);
  } catch {
    return {};
  }
}

async function saveProxyIndex(index: ProxyIndex): Promise<void> {
  const root = getMediaRootAbsolute();
  const indexPath = join(root, '.proxy-index.json');
  await fsp.writeFile(indexPath, JSON.stringify(index, null, 2), 'utf8');
}

async function ensureMediaDirs(): Promise<{root: string, symlinksAbs: string, proxyAbs: string}> {
  const root = getMediaRootAbsolute();
  const symlinksAbs = symlinksDir(root);
  const proxyAbs = proxyDir(root);
  if (!fs.existsSync(symlinksAbs)) fs.mkdirSync(symlinksAbs, {recursive: true});
  if (!fs.existsSync(proxyAbs)) fs.mkdirSync(proxyAbs, {recursive: true});
  return {root, symlinksAbs, proxyAbs};
}

async function listConvertedMedia(): Promise<ConvertedMediaItem[]> {
  
  const {symlinksAbs, proxyAbs} = await ensureMediaDirs();
  const proxyIndex = await loadProxyIndex();
  
  let entries: import('node:fs').Dirent[] = [];
  try {
    entries = await fsp.readdir(symlinksAbs, {withFileTypes: true}) as unknown as import('node:fs').Dirent[];
  } catch {
    entries = [] as any;
  }
  const items: ConvertedMediaItem[] = [];
  for (const e of entries) {
    const name = e.name;
    const ext = getLowercaseExtension(name);
    if (!isSupportedExt(ext)) continue;
    
    let absPath = join(symlinksAbs, name);
    let hasProxy = false;
    
    // Check if this file has a proxy
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
      dateAddedMs = st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
    } catch {}
    const assetUrl = pathToFileURL(absPath).href;
    const type: ConvertedMediaItem['type'] = VIDEO_EXTS.has(ext) ? 'video' : IMAGE_EXTS.has(ext) ? 'image' : AUDIO_EXTS.has(ext) ? 'audio':'video';
    items.push({ name, absPath, assetUrl, dateAddedMs, type, hasProxy });
  }
  items.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
  return items;
}

async function importMediaPaths(inputAbsPaths: string[], resolution?: string): Promise<number> {
  const {symlinksAbs} = await ensureMediaDirs();
  let count = 0;
  for (const srcAbs of inputAbsPaths) {
    const fileName = srcAbs.split(/[/\\]/g).pop();
    if (!fileName) continue;
    const ext = getLowercaseExtension(fileName);
    if (!isSupportedExt(ext)) continue;
    const uniqueName = ensureUniqueNameSync(symlinksAbs, fileName);
    const dstAbs = join(symlinksAbs, uniqueName);
    try {
      await fsp.rm(dstAbs, {force: true});
      await fsp.symlink(srcAbs, dstAbs);
    } catch (error) {
      console.error('Error creating symlink:', error);
      try {
        await fsp.copyFile(srcAbs, dstAbs);
      } catch (copyError) {
        console.error('Error copying file:', copyError);
        continue;
      }
    }
    count += 1;
  }
  return count;
}

async function ensureUniqueConvertedName(desiredName: string): Promise<string> {
  const {symlinksAbs} = await ensureMediaDirs();
  return ensureUniqueNameSync(symlinksAbs, desiredName);
}

function getPathForFile(file: File): string {
  return webUtils.getPathForFile(file);
}

async function createProxy(fileName: string, resolution: string = '480p'): Promise<void> {
  const {symlinksAbs, proxyAbs} = await ensureMediaDirs();
  const symlinkPath = join(symlinksAbs, fileName);
  
  if (!fs.existsSync(symlinkPath)) {
    throw new Error('Source file not found');
  }
  
  // Resolve the actual file path from the symlink
  const realPath = await fsp.realpath(symlinkPath);
  
  // Generate proxy filename
  const baseName = basename(fileName, extname(fileName));
  const proxyFileName = `${baseName}_proxy.mp4`;
  const proxyPath = join(proxyAbs, proxyFileName);
  
  // Parse target height
  const parseHeight = (res: string): number => {
    const s = res.trim().toLowerCase();
    if (s === '720' || s === '720p' || s === 'hd') return 720;
    if (s === '1080' || s === '1080p') return 1080;
    const raw = s.endsWith('p') ? s.slice(0, -1) : s;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : 480;
  };
  
  const targetHeight = parseHeight(resolution);
  
  return new Promise((resolve, reject) => {
    const args = [
      '-i', realPath,
      '-vf', `scale=-2:${targetHeight}`,
      '-r', '24',
      '-c:v', 'libx264',
      '-preset', 'medium',
      '-crf', '23',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-ar', '48000',
      '-y',
      proxyPath
    ];
    
    const ffmpeg = spawn('ffmpeg', args);
    
    let stderr = '';
    ffmpeg.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    ffmpeg.on('close', async (code) => {
      if (code === 0) {
        const proxyIndex = await loadProxyIndex();
        proxyIndex[fileName] = proxyFileName;
        await saveProxyIndex(proxyIndex);
        resolve();
      } else {
        reject(new Error(`FFmpeg failed with code ${code}: ${stderr}`));
      }
    });
    
    ffmpeg.on('error', (err) => {
      reject(err);
    });
  });
}

async function removeProxy(fileName: string): Promise<void> {
  const {proxyAbs} = await ensureMediaDirs();
  const proxyIndex = await loadProxyIndex();
  
  if (proxyIndex[fileName]) {
    const proxyPath = join(proxyAbs, proxyIndex[fileName]);
    if (fs.existsSync(proxyPath)) {
      await fsp.rm(proxyPath, {force: true});
    }
    delete proxyIndex[fileName];
    await saveProxyIndex(proxyIndex);
  }
}



// Config API functions
interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

async function getBackendUrl(): Promise<ConfigResponse<{url: string}>> {
  return await ipcRenderer.invoke('backend:get-url');
}

async function setBackendUrl(url: string): Promise<ConfigResponse<{url: string}>> {
  return await ipcRenderer.invoke('backend:set-url', url);
}

async function getHomeDir(): Promise<ConfigResponse<{home_dir: string}>> {
  return await ipcRenderer.invoke('config:get-home-dir');
}

async function setHomeDir(homeDir: string): Promise<ConfigResponse<{home_dir: string}>> {
  return await ipcRenderer.invoke('config:set-home-dir', homeDir);
}

async function getTorchDevice(): Promise<ConfigResponse<{device: string}>> {
  return await ipcRenderer.invoke('config:get-torch-device');
}

async function setTorchDevice(device: string): Promise<ConfigResponse<{device: string}>> {
  return await ipcRenderer.invoke('config:set-torch-device', device);
}

async function getCachePath(): Promise<ConfigResponse<{cache_path: string}>> {
  return await ipcRenderer.invoke('config:get-cache-path');
}

async function setCachePath(cachePath: string): Promise<ConfigResponse<{cache_path: string}>> {
  return await ipcRenderer.invoke('config:set-cache-path', cachePath);
}

// Preprocessor API functions
async function listPreprocessors(checkDownloaded: boolean = true): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('preprocessor:list', checkDownloaded);
}

async function deletePreprocessor(name: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('preprocessor:delete', name);
}

// Components API functions
async function downloadComponents(paths: string[], savePath?: string, jobId?: string): Promise<ConfigResponse<{job_id: string; status: string; message?: string}>> {
  return await ipcRenderer.invoke('components:download', paths, savePath, jobId);
}

async function deleteComponent(targetPath: string): Promise<ConfigResponse<{status: string; path: string}>> {
  return await ipcRenderer.invoke('components:delete', targetPath);
}

async function getComponentsStatus(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('jobs:status', jobId);
}

async function cancelComponents(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('jobs:cancel', jobId);
}

async function connectComponentsWebSocket(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('components:connect-ws', jobId);
}

async function disconnectComponentsWebSocket(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('components:disconnect-ws', jobId);
}

function onComponentsWebSocketUpdate(jobId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`components:ws-update:${jobId}`, listener);
  return () => ipcRenderer.removeListener(`components:ws-update:${jobId}`, listener);
}

function onComponentsWebSocketStatus(jobId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`components:ws-status:${jobId}`, listener);
  return () => ipcRenderer.removeListener(`components:ws-status:${jobId}`, listener);
}

function onComponentsWebSocketError(jobId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`components:ws-error:${jobId}`, listener);
  return () => ipcRenderer.removeListener(`components:ws-error:${jobId}`, listener);
}

async function getPreprocessor(name: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('preprocessor:get', name);
}

async function downloadPreprocessor(name: string, jobId?: string): Promise<ConfigResponse<{job_id: string; status: string; message?: string}>> {
  return await ipcRenderer.invoke('preprocessor:download', name, jobId);
}

async function createMask(request: {
  input_path: string;
  frame_number?: number;
  tool: string;
  points?: Array<{x: number, y: number}>;
  point_labels?: Array<number>;
  box?: {x1: number, y1: number, x2: number, y2: number};
  multimask_output?: boolean;
  simplify_tolerance?: number;
  model_type?: string;
  id?: string;
}): Promise<ConfigResponse<{status: string; contours?: Array<Array<number>>; message?: string}>> {
  // check if input_path is a file url
  if (request.input_path.startsWith('file://')) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke('mask:create', request);
}

type MaskTrackStreamEvent =
  | { frame_number: number; contours: Array<Array<number>> }
  | { status: 'error'; error: string };

async function trackMask(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: 'forward' | 'backward' | 'both';
  model_type?: string;
}): Promise<ReadableStream<MaskTrackStreamEvent>> {
  if (request.input_path.startsWith('file://')) {
    request.input_path = fileURLToPath(request.input_path);
  }
  const { streamId } = await ipcRenderer.invoke('mask:track', request) as { streamId: string };
  const parsedStream = new ReadableStream<MaskTrackStreamEvent>({
    start(controller) {
      const onChunk = (_e: unknown, data: any) => {
        controller.enqueue(data as MaskTrackStreamEvent);
      };
      const onError = (_e: unknown, err: any) => {
        controller.enqueue({ status: 'error', error: err?.message || 'Unknown error' });
      };
      const onEnd = () => {
        controller.close();
        ipcRenderer.removeAllListeners(`mask:track:chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track:error:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track:end:${streamId}`);
      };
      ipcRenderer.on(`mask:track:chunk:${streamId}`, onChunk);
      ipcRenderer.on(`mask:track:error:${streamId}`, onError);
      ipcRenderer.on(`mask:track:end:${streamId}`, onEnd);
    },
    cancel() {
      ipcRenderer.invoke('mask:track:cancel', streamId).catch(() => {});
      ipcRenderer.removeAllListeners(`mask:track:chunk:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track:error:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track:end:${streamId}`);
    },
  });

  return parsedStream;
}

// Lower-level helpers to assemble stream from renderer (avoid returning ReadableStream across contextBridge)
async function startMaskTrack(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: 'forward' | 'backward' | 'both';
  model_type?: string;
}): Promise<{ streamId: string }> {
  if (request.input_path.startsWith('file://')) {
    request.input_path = fileURLToPath(request.input_path);
  }
  
  return await ipcRenderer.invoke('mask:track', request);
}

function onMaskTrackChunk(streamId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track:chunk:${streamId}`, listener);
  return () => ipcRenderer.removeListener(`mask:track:chunk:${streamId}`, listener);
}

function onMaskTrackError(streamId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track:error:${streamId}`, listener);
  return () => ipcRenderer.removeListener(`mask:track:error:${streamId}`, listener);
}

function onMaskTrackEnd(streamId: string, callback: () => void): () => void {
  const listener = () => callback();
  ipcRenderer.on(`mask:track:end:${streamId}`, listener);
  return () => ipcRenderer.removeListener(`mask:track:end:${streamId}`, listener);
}

async function cancelMaskTrack(streamId: string): Promise<void> {
  await ipcRenderer.invoke('mask:track:cancel', streamId);
}


// Shapes tracking (streaming shapeBounds)
type ShapeTrackStreamEvent =
  | { frame_number: number; shapeBounds?: any }
  | { status: 'error'; error: string };

async function trackShapes(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: 'forward' | 'backward' | 'both';
  model_type?: string;
}): Promise<ReadableStream<ShapeTrackStreamEvent>> {
  if (request.input_path.startsWith('file://')) {
    request.input_path = fileURLToPath(request.input_path);
  }
  const { streamId } = await ipcRenderer.invoke('mask:track-shapes', request) as { streamId: string };
  const parsedStream = new ReadableStream<ShapeTrackStreamEvent>({
    start(controller) {
      const onChunk = (_e: unknown, data: any) => {
        controller.enqueue(data as ShapeTrackStreamEvent);
      };
      const onError = (_e: unknown, err: any) => {
        controller.enqueue({ status: 'error', error: err?.message || 'Unknown error' });
      };
      const onEnd = () => {
        controller.close();
        ipcRenderer.removeAllListeners(`mask:track-shapes:chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track-shapes:error:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track-shapes:end:${streamId}`);
      };
      ipcRenderer.on(`mask:track-shapes:chunk:${streamId}`, onChunk);
      ipcRenderer.on(`mask:track-shapes:error:${streamId}`, onError);
      ipcRenderer.on(`mask:track-shapes:end:${streamId}`, onEnd);
    },
    cancel() {
      ipcRenderer.invoke('mask:track:cancel', streamId).catch(() => {});
      ipcRenderer.removeAllListeners(`mask:track-shapes:chunk:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track-shapes:error:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track-shapes:end:${streamId}`);
    },
  });
  return parsedStream;
}

async function startMaskTrackShapes(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: 'forward' | 'backward' | 'both';
  model_type?: string;
  shape_type?: string;
}): Promise<{ streamId: string }> {
  if (request.input_path.startsWith('file://')) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke('mask:track-shapes', request);
}

function onMaskTrackShapesChunk(streamId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track-shapes:chunk:${streamId}`, listener);
  return () => ipcRenderer.removeListener(`mask:track-shapes:chunk:${streamId}`, listener);
}

function onMaskTrackShapesError(streamId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track-shapes:error:${streamId}`, listener);
  return () => ipcRenderer.removeListener(`mask:track-shapes:error:${streamId}`, listener);
}

function onMaskTrackShapesEnd(streamId: string, callback: () => void): () => void {
  const listener = () => callback();
  ipcRenderer.on(`mask:track-shapes:end:${streamId}`, listener);
  return () => ipcRenderer.removeListener(`mask:track-shapes:end:${streamId}`, listener);
}


async function runPreprocessor(request: {
  preprocessor_name: string;
  input_path: string;
  job_id?: string;
  download_if_needed?: boolean;
  params?: Record<string, any>;
  start_frame?: number;
  end_frame?: number;
}): Promise<ConfigResponse<{job_id: string; status: string; message?: string}>> {
  // check if input_path is a file url
  if (request.input_path.startsWith('file://')) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke('preprocessor:run', request);
}



async function getPreprocessorStatus(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('jobs:status', jobId);
}

async function getPreprocessorResult(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('preprocessor:result', jobId);
}

async function connectPreprocessorWebSocket(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('preprocessor:connect-ws', jobId);
}

async function disconnectPreprocessorWebSocket(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('preprocessor:disconnect-ws', jobId);
}

function onPreprocessorWebSocketUpdate(jobId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-update:${jobId}`, listener);
  return () => ipcRenderer.removeListener(`preprocessor:ws-update:${jobId}`, listener);
}

function onPreprocessorWebSocketStatus(jobId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-status:${jobId}`, listener);
  return () => ipcRenderer.removeListener(`preprocessor:ws-status:${jobId}`, listener);
}

function onPreprocessorWebSocketError(jobId: string, callback: (data: any) => void): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-error:${jobId}`, listener);
  return () => ipcRenderer.removeListener(`preprocessor:ws-error:${jobId}`, listener);
}

async function cancelPreprocessor(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('jobs:cancel', jobId);
}

// Unified job helpers
async function jobStatus(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('jobs:status', jobId);
}

async function jobCancel(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('jobs:cancel', jobId);
}

// Unified WebSocket helpers (renderer-wide API)
async function wsConnect(key: string, pathOrUrl: string): Promise<ConfigResponse<{ key: string }>> {
  return await ipcRenderer.invoke('ws:connect', { key, pathOrUrl });
}

async function wsDisconnect(key: string): Promise<ConfigResponse<{ message: string }>> {
  return await ipcRenderer.invoke('ws:disconnect', key);
}

async function wsStatus(key: string): Promise<ConfigResponse<{ key: string; connected: boolean }>> {
  return await ipcRenderer.invoke('ws:status', key);
}

function onWsUpdate(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-update:${key}`;
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function onWsStatus(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-status:${key}`;
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function onWsError(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-error:${key}`;
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function pathToFileURLString(path: string): string {
  const fileUrl = pathToFileURL(path);
  return fileUrl.href;
}

// System API functions
async function getSystemMemory(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('system:memory');
}

// Manifest API functions
async function listManifestModelTypes(): Promise<ConfigResponse<any>> {
  const res = await ipcRenderer.invoke('manifest:types');
  return res;
}

async function listManifests(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('manifest:list');
}

async function listManifestsByModel(model: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('manifest:list-by-model', model);
}

async function listManifestsByType(modelType: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('manifest:list-by-type', modelType);
}

async function listManifestsByModelAndType(model: string, modelType: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('manifest:list-by-model-and-type', model, modelType);
}

async function getManifest(manifestId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke('manifest:get', manifestId);
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
  readFileBuffer,
  readFileStream,
  pathToFileURL,
  fileURLToPath,
  fetchFilters,
  getPathForFile,
  createProxy,
  removeProxy,
  wsConnect,
  wsDisconnect,
  wsStatus,
  onWsUpdate,
  onWsStatus,
  onWsError,
  getBackendUrl,
  setBackendUrl,
  getHomeDir,
  setHomeDir,
  getTorchDevice,
  setTorchDevice,
  getCachePath,
  setCachePath,
  getSystemMemory,
  listPreprocessors,
  getPreprocessor,
  downloadPreprocessor,
  runPreprocessor,
  getPreprocessorStatus,
  getPreprocessorResult,
  connectPreprocessorWebSocket,
  disconnectPreprocessorWebSocket,
  onPreprocessorWebSocketUpdate,
  onPreprocessorWebSocketStatus,
  onPreprocessorWebSocketError,
  downloadComponents,
  deleteComponent,
  getComponentsStatus,
  cancelComponents,
  connectComponentsWebSocket,
  disconnectComponentsWebSocket,
  onComponentsWebSocketUpdate,
  onComponentsWebSocketStatus,
  onComponentsWebSocketError,
  pathToFileURLString,
  cancelPreprocessor,
  createMask,
  trackMask,
  startMaskTrack,
  onMaskTrackChunk,
  onMaskTrackError,
  onMaskTrackEnd,
  cancelMaskTrack,
  trackShapes,
  startMaskTrackShapes,
  onMaskTrackShapesChunk,
  onMaskTrackShapesError,
  onMaskTrackShapesEnd,
  listManifestModelTypes,
  listManifests,
  listManifestsByModel,
  listManifestsByType,
  listManifestsByModelAndType,
  getManifest,
  deletePreprocessor
};
