import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import { BrowserWindow, globalShortcut, App, contentTracing, ipcMain } from 'electron';
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
}

export function apexApi(...args: ConstructorParameters<typeof ApexApi>) {
  return new ApexApi(...args);
}