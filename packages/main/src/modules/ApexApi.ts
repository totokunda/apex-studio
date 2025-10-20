import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import { BrowserWindow, globalShortcut, App, contentTracing, ipcMain } from 'electron';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import inspector from 'node:inspector';
import WebSocket from 'ws';

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
  private wsConnections: Map<string, WebSocket> = new Map();
  private activeMaskStreams: Map<string, { controller: AbortController; reader: ReadableStreamDefaultReader<Uint8Array> | null; id: string } > = new Map();

  constructor(backendUrl: string = DEFAULT_BACKEND_URL) {
    this.backendUrl = backendUrl;
  }


  async enable(_context: ModuleContext): Promise<void> {
    this.app = _context.app;
    await this.app.whenReady();
    
    this.settingsPath = path.join(this.app.getPath('userData'), 'apex-settings.json');
    await this.loadSettings();
    
    this.registerConfigHandlers();
    this.registerBackendUrlHandlers();
    this.registerPreprocessorHandlers();
    this.registerMaskHandlers();
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

    // WebSocket connection for job updates
    ipcMain.handle('preprocessor:connect-ws', async (event, jobId: string) => {
      const connectionKey = `${jobId}`;
      
      // Close existing connection if any
      if (this.wsConnections.has(connectionKey)) {
        const existingWs = this.wsConnections.get(connectionKey);
        if (existingWs && existingWs.readyState === WebSocket.OPEN) {
          existingWs.close();
        }
        this.wsConnections.delete(connectionKey);
      }

      try {
        const wsUrl = this.backendUrl.replace('http://', 'ws://').replace('https://', 'wss://');
        const ws = new WebSocket(`${wsUrl}/ws/job/${jobId}`);
        
        ws.on('open', () => {
          event.sender.send(`preprocessor:ws-status:${jobId}`, { status: 'connected' });
        });

        ws.on('message', (data) => {
          try {
            const message = JSON.parse(data.toString());
            event.sender.send(`preprocessor:ws-update:${jobId}`, message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        });

        ws.on('error', (error) => {
          event.sender.send(`preprocessor:ws-error:${jobId}`, { error: error.message });
        });

        ws.on('close', () => {
          this.wsConnections.delete(connectionKey);
          event.sender.send(`preprocessor:ws-status:${jobId}`, { status: 'disconnected' });
        });

        this.wsConnections.set(connectionKey, ws);
        
        return {
          success: true,
          data: { connectionKey, jobId },
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to connect to WebSocket',
        };
      }
    });

    // Disconnect WebSocket
    ipcMain.handle('preprocessor:disconnect-ws', async (_event, jobId: string) => {
      const connectionKey = `${jobId}`;
      
      if (this.wsConnections.has(connectionKey)) {
        const ws = this.wsConnections.get(connectionKey);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
        this.wsConnections.delete(connectionKey);
        
        return {
          success: true,
          data: { message: 'WebSocket disconnected' },
        };
      }
      
      return {
        success: false,
        error: 'WebSocket connection not found',
      };
    });
  }

  private async makeRequest<T>(method: 'GET' | 'POST', endpoint: string, body?: any): Promise<ConfigResponse<T>> {
    try {
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
      };


      if (body && method === 'POST') {
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

  private async makeStreamingRequest(method: 'GET' | 'POST', endpoint: string, body?: any): Promise<ReadableStream<Uint8Array<ArrayBuffer>>> {
    try {
      const controller = new AbortController();
      console.log('body', body);
      const response = await fetch(`${this.backendUrl}${endpoint}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/x-ndjson',
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      } 

      // return stream back to the caller
      console.log('response.body', response.body);
      return response.body;

    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Unknown error occurred');
    }
  }
}

export function apexApi(...args: ConstructorParameters<typeof ApexApi>) {
  return new ApexApi(...args);
}