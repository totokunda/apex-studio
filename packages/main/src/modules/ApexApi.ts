import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import {  App, ipcMain } from 'electron';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import inspector from 'node:inspector';
import { WebSocketManager } from './WebSocketManager.js';

const session = new inspector.Session();
session.connect();

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8765';

interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

interface ApexSettings {
  backendUrl: string;
}

export class ApexApi implements AppModule {
  private app: App | null = null;
  private settingsPath: string = '';
  private backendUrl: string = DEFAULT_BACKEND_URL;
  private wsManager: WebSocketManager | null = null;
  private activeMaskStreams: Map<string, { controller: AbortController; reader: ReadableStreamDefaultReader<Uint8Array> | null; id: string } > = new Map();

  constructor(backendUrl: string = DEFAULT_BACKEND_URL) {
    this.backendUrl = backendUrl;
    this.wsManager = new WebSocketManager(this.backendUrl);
  }


  async enable(_context: ModuleContext): Promise<void> {
    this.app = _context.app;
    await this.app.whenReady();
    
    this.settingsPath = path.join(this.app.getPath('userData'), 'apex-settings.json');
    await this.loadSettings();
    this.wsManager?.setBaseUrl(this.backendUrl);
    
    this.registerConfigHandlers();
    this.registerBackendUrlHandlers();
    this.registerWebSocketHandlers();
    this.registerJobHandlers();
    this.registerPreprocessorHandlers();
    this.registerComponentsHandlers();
    this.registerMaskHandlers();
    this.registerManifestHandlers();
  }

  private async loadSettings(): Promise<void> {
    try {
      if (fs.existsSync(this.settingsPath)) {
        const data = await fs.promises.readFile(this.settingsPath, 'utf-8');
        const settings: ApexSettings = JSON.parse(data);
        this.backendUrl = settings.backendUrl || DEFAULT_BACKEND_URL;
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
      this.backendUrl = DEFAULT_BACKEND_URL;
    }
  }

  private async saveSettings(): Promise<void> {
    try {
      const settings: ApexSettings = {
        backendUrl: this.backendUrl,
      };
      await fs.promises.writeFile(this.settingsPath, JSON.stringify(settings, null, 2), 'utf-8');
    } catch (error) {
      console.error('Failed to save settings:', error);
      throw error;
    }
  }

  private registerBackendUrlHandlers(): void {
    ipcMain.handle('backend:get-url', async () => {
      return {
        success: true,
        data: { url: this.backendUrl },
      };
    });

    ipcMain.handle('backend:set-url', async (_event, url: string) => {
      try {
        // Validate URL format
        new URL(url);
        this.backendUrl = url;
        this.wsManager?.setBaseUrl(url);
        await this.saveSettings();
        return {
          success: true,
          data: { url: this.backendUrl },
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Invalid URL format',
        };
      }
    });
  }

  private registerConfigHandlers(): void {
    // Home directory handlers
    ipcMain.handle('config:get-home-dir', async () => {
      return this.makeRequest<{home_dir: string}>('GET', '/config/home-dir');
    });

    ipcMain.handle('config:set-home-dir', async (_event, homeDir: string) => {
      return this.makeRequest<{home_dir: string}>('POST', '/config/home-dir', { home_dir: homeDir });
    });

    // Components path handlers
    ipcMain.handle('config:get-components-path', async () => {
      return this.makeRequest<{components_path: string}>('GET', '/config/components-path');
    });

    ipcMain.handle('config:set-components-path', async (_event, componentsPath: string) => {
      return this.makeRequest<{components_path: string}>('POST', '/config/components-path', { components_path: componentsPath });
    });

    // Torch device handlers
    ipcMain.handle('config:get-torch-device', async () => {
      return this.makeRequest<{device: string}>('GET', '/config/torch-device');
    });

    ipcMain.handle('config:set-torch-device', async (_event, device: string) => {
      return this.makeRequest<{device: string}>('POST', '/config/torch-device', { device });
    });

    // Cache path handlers
    ipcMain.handle('config:get-cache-path', async () => {
      return this.makeRequest<{cache_path: string}>('GET', '/config/cache-path');
    });

    ipcMain.handle('config:set-cache-path', async (_event, cachePath: string) => {
      return this.makeRequest<{cache_path: string}>('POST', '/config/cache-path', { cache_path: cachePath });
    });
  }

  private registerMaskHandlers(): void {
    ipcMain.handle('mask:create', async (_event, request: {
      input_path: string;
      frame_number?: number;
      tool: string;
      points?: Array<{x: number, y: number}>;
      point_labels?: Array<number>;
      box?: {x1: number, y1: number, x2: number, y2: number};
      multimask_output?: boolean;
      simplify_tolerance?: number;
      model_type?: string;
    }) => {
      return this.makeRequest<{status: string; contours?: Array<Array<number>>; message?: string}>('POST', '/mask/create', request);
    });

    ipcMain.handle('mask:track', async (event, request: {
      id: string;
      input_path: string;
      frame_start: number;
      anchor_frame?: number;
      frame_end: number;
      direction?: 'forward' | 'backward' | 'both';
      model_type?: string;
    }) => {
      const streamId = crypto.randomUUID();

      const controller = new AbortController();
      const response = await fetch(`${this.backendUrl}/mask/track`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/x-ndjson',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      this.activeMaskStreams.set(streamId, { controller, reader, id: request.id });

      const decoder = new TextDecoder();
      let buffered = '';

      const pump = (): void => {
        reader.read().then(({ value, done }) => {
          try {
            if (done) {
              const last = buffered.trim();
              if (last) {
                const line = last.startsWith('data:') ? last.slice(5) : last;
                const data = JSON.parse(line);
                event.sender.send(`mask:track:chunk:${streamId}`, data);
              }
              event.sender.send(`mask:track:end:${streamId}`);
              this.activeMaskStreams.delete(streamId);
              return;
            }

            if (value) {
              buffered += decoder.decode(value, { stream: true });
            }

            let newlineIdx = buffered.indexOf('\n');
            while (newlineIdx !== -1) {
              const rawLine = buffered.slice(0, newlineIdx);
              buffered = buffered.slice(newlineIdx + 1);
              const trimmed = rawLine.trim();
              if (trimmed && trimmed !== '[DONE]') {
                const line = trimmed.startsWith('data:') ? trimmed.slice(5) : trimmed;
                const data = JSON.parse(line);
                event.sender.send(`mask:track:chunk:${streamId}`, data);
              }
              newlineIdx = buffered.indexOf('\n');
            }
          } catch (err) {
            event.sender.send(`mask:track:error:${streamId}`, { message: err instanceof Error ? err.message : 'Unknown error' });
            try { reader.cancel().catch(() => {}); } catch {}
            this.activeMaskStreams.delete(streamId);
            return;
          }
          pump();
        }).catch((err) => {
          event.sender.send(`mask:track:error:${streamId}`, { message: err instanceof Error ? err.message : 'Unknown error' });
          this.activeMaskStreams.delete(streamId);
        });
      };
      pump();

      return { streamId };

    });

    ipcMain.handle('mask:track:cancel', async (_event, streamId: string) => {
      const entry = this.activeMaskStreams.get(streamId);
      if (entry) {
        // Signal backend to stop processing this mask id, then abort local stream
        try {
          await this.makeRequest<'ok' | any>('POST', `/mask/track/cancel/${encodeURIComponent(entry.id)}`);
        } catch {}
        try { entry.controller.abort(); } catch {}
        try { entry.reader?.cancel().catch(() => {}); } catch {}
        this.activeMaskStreams.delete(streamId);
      }
      return { success: true };
    });

    // Shapes tracking stream
    ipcMain.handle('mask:track-shapes', async (event, request: {
      id: string;
      input_path: string;
      frame_start: number;
      anchor_frame?: number;
      frame_end: number;
      direction?: 'forward' | 'backward' | 'both';
      model_type?: string;
    }) => {
      const streamId = crypto.randomUUID();

      const controller = new AbortController();
      const response = await fetch(`${this.backendUrl}/mask/track/shapes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/x-ndjson',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      this.activeMaskStreams.set(streamId, { controller, reader, id: request.id });

      const decoder = new TextDecoder();
      let buffered = '';

      const pump = (): void => {
        reader.read().then(({ value, done }) => {
          try {
            if (done) {
              const last = buffered.trim();
              if (last) {
                const line = last.startsWith('data:') ? last.slice(5) : last;
                const data = JSON.parse(line);
                event.sender.send(`mask:track-shapes:chunk:${streamId}`, data);
              }
              event.sender.send(`mask:track-shapes:end:${streamId}`);
              this.activeMaskStreams.delete(streamId);
              return;
            }

            if (value) {
              buffered += decoder.decode(value, { stream: true });
            }

            let newlineIdx = buffered.indexOf('\n');
            while (newlineIdx !== -1) {
              const rawLine = buffered.slice(0, newlineIdx);
              buffered = buffered.slice(newlineIdx + 1);
              const trimmed = rawLine.trim();
              if (trimmed && trimmed !== '[DONE]') {
                const line = trimmed.startsWith('data:') ? trimmed.slice(5) : trimmed;
                const data = JSON.parse(line);
                event.sender.send(`mask:track-shapes:chunk:${streamId}`, data);
              }
              newlineIdx = buffered.indexOf('\n');
            }
          } catch (err) {
            event.sender.send(`mask:track-shapes:error:${streamId}`, { message: err instanceof Error ? err.message : 'Unknown error' });
            try { reader.cancel().catch(() => {}); } catch {}
            this.activeMaskStreams.delete(streamId);
            return;
          }
          pump();
        }).catch((err) => {
          event.sender.send(`mask:track-shapes:error:${streamId}`, { message: err instanceof Error ? err.message : 'Unknown error' });
          this.activeMaskStreams.delete(streamId);
        });
      };
      pump();

      return { streamId };
    });
    
  }

  private registerPreprocessorHandlers(): void {
    // List all preprocessors
    ipcMain.handle('preprocessor:list', async (_event, checkDownloaded: boolean = true) => {
      return this.makeRequest<any>('GET', `/preprocessor/list?check_downloaded=${checkDownloaded}`);
    });

    // Get preprocessor details
    ipcMain.handle('preprocessor:get', async (_event, name: string) => {
      return this.makeRequest<any>('GET', `/preprocessor/get/${encodeURIComponent(name)}`);
    });

    // Download preprocessor
    ipcMain.handle('preprocessor:download', async (_event, name: string, jobId?: string) => {
      const body = {
        preprocessor_name: name,
        job_id: jobId,
      };
      return this.makeRequest<{job_id: string; status: string; message?: string}>('POST', '/preprocessor/download', body);
    });

    // Run preprocessor
    ipcMain.handle('preprocessor:run', async (_event, request: {
      preprocessor_name: string;
      input_path: string;
      job_id?: string;
      download_if_needed?: boolean;
      params?: Record<string, any>;
      start_frame?: number;
      end_frame?: number;
    }) => {
      return this.makeRequest<{job_id: string; status: string; message?: string}>('POST', '/preprocessor/run', request);
    });

    // Cancel preprocessor
    ipcMain.handle('preprocessor:cancel', async (_event, jobId: string) => {
      return this.makeRequest<{job_id: string; status: string; message?: string}>('POST', `/preprocessor/cancel/${encodeURIComponent(jobId)}`);
    });

    // Get job status
    ipcMain.handle('preprocessor:status', async (_event, jobId: string) => {
      return this.makeRequest<any>('GET', `/preprocessor/status/${encodeURIComponent(jobId)}`);
    });

    // Get job result
    ipcMain.handle('preprocessor:result', async (_event, jobId: string) => {
      return this.makeRequest<any>('GET', `/preprocessor/result/${encodeURIComponent(jobId)}`);
    });

    // Delete preprocessor (remove downloaded files and unmark)
    ipcMain.handle('preprocessor:delete', async (_event, name: string) => {
      return this.makeRequest<any>('DELETE', `/preprocessor/delete/${encodeURIComponent(name)}`);
    });

    // WebSocket handlers migrated to unified ws:* IPC
  }

  private registerManifestHandlers(): void {
    // List model types
    ipcMain.handle('manifest:types', async () => {
      return this.makeRequest<any>('GET', '/manifest/types');
    });

    // List all manifests
    ipcMain.handle('manifest:list', async () => {
      return this.makeRequest<any>('GET', '/manifest/list');
    });

    // List manifests by model
    ipcMain.handle('manifest:list-by-model', async (_event, model: string) => {
      return this.makeRequest<any>('GET', `/manifest/list/model/${encodeURIComponent(model)}`);
    });

    // List manifests by model type
    ipcMain.handle('manifest:list-by-type', async (_event, modelType: string) => {
      return this.makeRequest<any>('GET', `/manifest/list/type/${encodeURIComponent(modelType)}`);
    });

    // List manifests by model and type
    ipcMain.handle('manifest:list-by-model-and-type', async (_event, model: string, modelType: string) => {
      return this.makeRequest<any>('GET', `/manifest/list/model/${encodeURIComponent(model)}/model_type/${encodeURIComponent(modelType)}`);
    });

    // Get manifest content by id
    ipcMain.handle('manifest:get', async (_event, manifestId: string) => {
      return this.makeRequest<any>('GET', `/manifest/${encodeURIComponent(manifestId)}`);
    });
  }

  private registerComponentsHandlers(): void {
    // Start components download
    ipcMain.handle('components:download', async (_event, paths: string[], savePath?: string, jobId?: string) => {
      const body = { paths, save_path: savePath, job_id: jobId };
      return this.makeRequest<{job_id: string; status: string; message?: string}>('POST', '/components/download', body);
    });

    // Delete a downloaded component (file or directory under components path)
    ipcMain.handle('components:delete', async (_event, targetPath: string) => {
      return this.makeRequest<{status: string; path: string}>('DELETE', '/components/delete', { path: targetPath });
    });

    // Poll components download status
    ipcMain.handle('components:status', async (_event, jobId: string) => {
      return this.makeRequest<any>('GET', `/jobs/status/${encodeURIComponent(jobId)}`);
    });

    // Cancel components download
    ipcMain.handle('components:cancel', async (_event, jobId: string) => {
      return this.makeRequest<{job_id: string; status: string; message?: string}>('POST', `/jobs/cancel/${encodeURIComponent(jobId)}`);
    });

    // WebSocket handlers migrated to unified ws:* IPC
  }

  private registerJobHandlers(): void {
    // Unified job status
    ipcMain.handle('jobs:status', async (_event, jobId: string) => {
      return this.makeRequest<any>('GET', `/jobs/status/${encodeURIComponent(jobId)}`);
    });

    // Unified job cancel
    ipcMain.handle('jobs:cancel', async (_event, jobId: string) => {
      return this.makeRequest<{job_id: string; status: string; message?: string}>('POST', `/jobs/cancel/${encodeURIComponent(jobId)}`);
    });
  }

  private registerWebSocketHandlers(): void {
    // Unified connect: expects { key, pathOrUrl }
    ipcMain.handle('ws:connect', async (event, request: { key: string; pathOrUrl: string }) => {
      const { key, pathOrUrl } = request;
      const result = this.wsManager!.connect(key, pathOrUrl, {
        onOpen: () => {
          event.sender.send(`ws-status:${key}`, { status: 'connected' });
        },
        onMessage: (data) => {
          try {
            const message = JSON.parse(data);
            event.sender.send(`ws-update:${key}`, message);
          } catch (error) {
            // Forward as error if message is not JSON
            event.sender.send(`ws-error:${key}`, { error: 'Invalid JSON message', raw: data });
          }
        },
        onError: (error) => {
          event.sender.send(`ws-error:${key}`, { error: error.message });
        },
        onClose: () => {
          event.sender.send(`ws-status:${key}`, { status: 'disconnected' });
        }
      });
      if (result.success) {
        return { success: true, data: { key } };
      }
      return { success: false, error: result.error || 'Failed to connect to WebSocket' };
    });

    // Unified disconnect: expects key
    ipcMain.handle('ws:disconnect', async (_event, key: string) => {
      const result = this.wsManager!.disconnect(key);
      if (result.success) {
        return { success: true, data: { message: 'WebSocket disconnected' } };
      }
      return { success: false, error: result.error || 'WebSocket connection not found' };
    });

    // Optional: status check
    ipcMain.handle('ws:status', async (_event, key: string) => {
      const connected = this.wsManager?.has(key) ?? false;
      return { success: true, data: { key, connected } };
    });
  }


  private async makeRequest<T>(method: 'GET' | 'POST' | 'DELETE', endpoint: string, body?: any): Promise<ConfigResponse<T>> {
    try {
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
      };


      if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${this.backendUrl}${endpoint}`, options);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        return {
          success: false,
          error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      const data = await response.json();
      return {
        success: true,
        data: data as T,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

}

export function apexApi(...args: ConstructorParameters<typeof ApexApi>) {
  return new ApexApi(...args);
}