import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { App, ipcMain, shell } from "electron";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import inspector from "node:inspector";
import { fileURLToPath } from "node:url";
import { WebSocketManager } from "./WebSocketManager.js";
import os from "node:os";
import { FormData as NodeFormData } from "formdata-node";
import { fileFromPath } from "formdata-node/file-from-path";
import { FormDataEncoder } from "form-data-encoder";
import { Readable } from "node:stream";
import { getSettingsModule } from "./SettingsModule.js";

const session = new inspector.Session();
session.connect();

const MAX_UPLOAD_BYTES = Number(
  process.env.APEX_MAX_UPLOAD_BYTES || 1_073_741_824,
); // 1 GiB default

const DEFAULT_REQUEST_TIMEOUT_MS = Number(
  process.env.APEX_API_REQUEST_TIMEOUT_MS || 30_000,
);
const PROBE_TIMEOUT_MS = Number(process.env.APEX_API_PROBE_TIMEOUT_MS || 10_000);
const STREAM_CONNECT_TIMEOUT_MS = Number(
  process.env.APEX_API_STREAM_CONNECT_TIMEOUT_MS || 15_000,
); 
const UPLOAD_TIMEOUT_MS = Number(process.env.APEX_API_UPLOAD_TIMEOUT_MS || 60_000);
const PROBE_TTL_OK_MS = Number(process.env.APEX_API_PROBE_TTL_OK_MS || 30_000);
const PROBE_TTL_FAIL_MS = Number(process.env.APEX_API_PROBE_TTL_FAIL_MS || 5_000);

const REMOTE_FILE_EXISTS_TIMEOUT_MS = Number(
  process.env.APEX_API_REMOTE_FILE_EXISTS_TIMEOUT_MS || 1000,
);
const REMOTE_FILE_MATCH_TIMEOUT_MS = Number(
  process.env.APEX_API_REMOTE_FILE_MATCH_TIMEOUT_MS || 1_200,
);

interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

interface UploadCacheEntry {
  remoteAbs: string;
  sha256: string;
  size: number;
  mtimeMs: number;
}

export class ApexApi implements AppModule {
  private app: App | null = null;
  private backendUrl: string = "http://127.0.0.1:8765";
  private wsManager: WebSocketManager | null = null;
  private activeMaskStreams: Map<
    string,
    {
      controller: AbortController;
      reader: ReadableStreamDefaultReader<Uint8Array> | null;
      id: string;
    }
  > = new Map();
  private uploadCache: Map<string, UploadCacheEntry> = new Map(); // localAbsPath -> uploaded file metadata
  private remoteCachePath: string | null = null; // backend-reported cache path
  private loopbackAppearsRemote: boolean = false; // true when localhost is actually a tunnel to a remote backend
  private probeInFlight: Promise<ConfigResponse<null>> | null = null;
  private probeLastAt: number = 0;
  private probeLastRes: ConfigResponse<null> | null = null;

  constructor(backendUrl?: string) {
    if (backendUrl) {
      this.backendUrl = backendUrl;
    }
    this.wsManager = new WebSocketManager(this.backendUrl);
  }

  async enable(_context: ModuleContext): Promise<void> {
    this.app = _context.app;
    
    // Connect to settings module
    const settings = getSettingsModule();
    this.backendUrl = settings.getBackendUrl();
    
    settings.on("backend-url-changed", (newUrl: string) => {
      console.log(`[ApexApi] Backend URL changed to ${newUrl}`);
      this.backendUrl = newUrl;
      this.wsManager?.setBaseUrl(newUrl);
      // Backend locality & cache roots depend on backend URL; reset derived state.
      this.remoteCachePath = null;
      this.loopbackAppearsRemote = false;
      this.probeLastAt = 0;
      this.probeLastRes = null;
     
      void this.#probeBackendLocality();
    });

    // Register IPC handlers immediately so renderer can invoke them as soon as it loads
    this.registerConfigHandlers();
    this.registerBackendUrlHandlers();
    this.registerWebSocketHandlers();
    this.registerJobHandlers();
    this.registerPreprocessorHandlers();
    this.registerPostprocessorHandlers();
    this.registerEngineHandlers();
    this.registerDownloadHandlers();
    this.registerComponentsHandlers();
    this.registerMaskHandlers();
    this.registerManifestHandlers();
    this.registerSystemHandlers();

    // Defer app-dependent initialization until the app is ready,
    // but do not block startup on backend/network probes.
    await this.app.whenReady();

    // Ensure we have the latest settings after app is ready (handles race where settings loaded late)
    this.backendUrl = settings.getBackendUrl();
  
    this.wsManager?.setBaseUrl(this.backendUrl);
    void this.#probeBackendLocality();

    ipcMain.handle(
      "apexapi:path-exists",
      async (_event, pathOrUrl: string): Promise<{ exists: boolean }> => {
        return this.pathExists(pathOrUrl);
      },
    );
  }

  private registerBackendUrlHandlers(): void {
    ipcMain.handle("backend:get-url", async () => {
      return {
        success: true,
        data: { url: this.backendUrl },
      };
    });

    ipcMain.handle("backend:set-url", async (_event, url: string) => {
      try {
        // Validate URL format
        new URL(url);
        // Delegate storage to settings module
        await getSettingsModule().setBackendUrl(url);
        // Local state is updated via event listener
        
        return {
          success: true,
          data: { url },
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Invalid URL format",
        };
      }
    });

    // Preview (non-persisted) backend URL switch, used by Settings Verify.
    ipcMain.handle("backend:preview-url", async (_event, url: string) => {
      try {
        new URL(url);
        getSettingsModule().previewBackendUrl(url);
        return { success: true, data: { url } };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Invalid URL format",
        };
      }
    });

    // Expose a definitive remote/loopback-tunneled status computed in main
    ipcMain.handle("backend:is-remote", async () => {
      try {
        // Cached probe: avoids spamming /config/hostname.
        await this.#probeBackendLocality();
      } catch {}
      try {
        const isRemote = this.#isRemoteBackend();
        return { success: true, data: { isRemote } };
      } catch (error) {
        return {
          success: false,
          error:
            error instanceof Error
              ? error.message
              : "Failed to determine remote status",
        };
      }
    });

    // Determine if a given input path would trigger an upload when sent to backend
    ipcMain.handle("files:should-upload", async (_event, inputPath: string) => {
      try {
        if (!this.#isRemoteBackend())
          return { success: true, data: { shouldUpload: false } };
        const abs = this.#resolveLocalPath(String(inputPath || ""));
        if (!abs) return { success: true, data: { shouldUpload: false } };
        const isCached = await this.#hasValidUploadedCache(abs);
        return { success: true, data: { shouldUpload: !isCached } };
      } catch (error) {
        // Sensible default on timeout/errors: do not auto-upload.
        return { success: true, data: { shouldUpload: false } };
      }
    });

    // Check if the given local file has already been uploaded this session (cache hit)
    ipcMain.handle("files:is-uploaded", async (_event, inputPath: string) => {
      try {
        const abs = this.#resolveLocalPath(String(inputPath || ""));
        if (!abs) return { success: true, data: { isUploaded: false } };
        const isUploaded = await this.#hasValidUploadedCache(abs);
        return { success: true, data: { isUploaded } };
      } catch (error) {
        // Sensible default on timeout/errors: treat as not uploaded.
        return { success: true, data: { isUploaded: false } };
      }
    });
  }

  async #hasValidUploadedCache(absPath: string): Promise<boolean> {
    const cached = this.uploadCache.get(absPath);
    if (!cached?.sha256) return false;
    try {
      const st = await fs.promises.stat(absPath);
      if (st.size !== cached.size || st.mtimeMs !== cached.mtimeMs) return false;
      const shaNow = await this.#sha256File(absPath);
      return shaNow === cached.sha256;
    } catch {
      return false;
    }
  }

  private registerConfigHandlers(): void {
    // Home directory handlers
    ipcMain.handle("config:get-home-dir", async () => {
      return this.makeRequest<{ home_dir: string }>("GET", "/config/home-dir");
    });

    ipcMain.handle("config:set-home-dir", async (_event, homeDir: string) => {
      return this.makeRequest<{ home_dir: string }>(
        "POST",
        "/config/home-dir",
        { home_dir: homeDir },
      );
    });

    // Components path handlers
    ipcMain.handle("config:get-components-path", async () => {
      return this.makeRequest<{ components_path: string }>(
        "GET",
        "/config/components-path",
      );
    });

    ipcMain.handle(
      "config:set-components-path",
      async (_event, componentsPath: string) => {
        return this.makeRequest<{ components_path: string }>(
          "POST",
          "/config/components-path",
          { components_path: componentsPath },
        );
      },
    );

    // Torch device handlers
    ipcMain.handle("config:get-torch-device", async () => {
      return this.makeRequest<{ device: string }>(
        "GET",
        "/config/torch-device",
      );
    });

    ipcMain.handle(
      "config:set-torch-device",
      async (_event, device: string) => {
        return this.makeRequest<{ device: string }>(
          "POST",
          "/config/torch-device",
          { device },
        );
      },
    );

    // Cache path handlers
    ipcMain.handle("config:get-cache-path", async () => {
      return this.makeRequest<{ cache_path: string }>(
        "GET",
        "/config/cache-path",
      );
    });

    ipcMain.handle(
      "config:set-cache-path",
      async (_event, cachePath: string) => {
        return this.makeRequest<{ cache_path: string }>(
          "POST",
          "/config/cache-path",
          { cache_path: cachePath },
        );
      },
    );

    // Electron userData path (for local cache directories, etc.)
    ipcMain.handle("config:get-user-data-path", async () => {
      try {
        const dir = this.app ? this.app.getPath("userData") : "";
        const res: ConfigResponse<{ user_data: string }> = {
          success: true,
          data: { user_data: dir },
        };
        return res;
      } catch (e: any) {
        const res: ConfigResponse<{ user_data: string }> = {
          success: false,
          error: e?.message || "Failed to resolve userData path",
        };
        return res;
      }
    });
  }

  private registerMaskHandlers(): void {
    ipcMain.handle(
      "mask:create",
      async (
        _event,
        request: {
          input_path: string;
          frame_number?: number;
          tool: string;
          points?: Array<{ x: number; y: number }>;
          point_labels?: Array<number>;
          box?: { x1: number; y1: number; x2: number; y2: number };
          multimask_output?: boolean;
          simplify_tolerance?: number;
        },
      ) => {
        const payload = await this.#prepareMaskRequest(request);
        return this.makeRequest<{
          status: string;
          contours?: Array<Array<number>>;
          message?: string;
        }>("POST", "/mask/create", payload);
      },
    );

    ipcMain.handle(
      "mask:track",
      async (
        event,
        request: {
          id: string;
          input_path: string;
          frame_start: number;
          anchor_frame?: number;
          frame_end: number;
          direction?: "forward" | "backward" | "both";
        },
      ) => {
        // Ensure backend is healthy before starting streaming mask tracking
        const probe = await this.#probeBackendLocality();
        if (!probe.success) {
          throw new Error(
            probe.error ||
              "Cannot complete the request because the backend API is unavailable.",
          );
        }

        const streamId = crypto.randomUUID();

        const controller = new AbortController();
        const payload = await this.#prepareMaskRequest(request);
        const connectTimeoutId = setTimeout(() => {
          try {
            controller.abort();
          } catch {}
        }, Math.max(1, STREAM_CONNECT_TIMEOUT_MS));

        let response: Response;
        try {
          response = await fetch(`${this.backendUrl}/mask/track`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/x-ndjson",
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
          });
        } catch (err) {
          if (controller.signal.aborted) {
            throw new Error(
              `Mask track request timed out after ${STREAM_CONNECT_TIMEOUT_MS}ms`,
            );
          }
          throw err;
        } finally {
          clearTimeout(connectTimeoutId);
        }

        if (!response.ok || !response.body) {
          const errorData = await response
            .json()
            .catch(() => ({ detail: "Unknown error" }));
          throw new Error(
            errorData.detail ||
              `HTTP ${response.status}: ${response.statusText}`,
          );
        }

        const reader = response.body.getReader();
        this.activeMaskStreams.set(streamId, {
          controller,
          reader,
          id: (payload as any).id,
        });

        const decoder = new TextDecoder();
        let buffered = "";

        const pump = (): void => {
          reader
            .read()
            .then(({ value, done }) => {
              try {
                if (done) {
                  const last = buffered.trim();
                  if (last) {
                    const line = last.startsWith("data:")
                      ? last.slice(5)
                      : last;
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

                let newlineIdx = buffered.indexOf("\n");
                while (newlineIdx !== -1) {
                  const rawLine = buffered.slice(0, newlineIdx);
                  buffered = buffered.slice(newlineIdx + 1);
                  const trimmed = rawLine.trim();
                  if (trimmed && trimmed !== "[DONE]") {
                    const line = trimmed.startsWith("data:")
                      ? trimmed.slice(5)
                      : trimmed;
                    const data = JSON.parse(line);
                    event.sender.send(`mask:track:chunk:${streamId}`, data);
                  }
                  newlineIdx = buffered.indexOf("\n");
                }
              } catch (err) {
                event.sender.send(`mask:track:error:${streamId}`, {
                  message: err instanceof Error ? err.message : "Unknown error",
                });
                try {
                  reader.cancel().catch(() => {});
                } catch {}
                this.activeMaskStreams.delete(streamId);
                return;
              }
              pump();
            })
            .catch((err) => {
              event.sender.send(`mask:track:error:${streamId}`, {
                message: err instanceof Error ? err.message : "Unknown error",
              });
              this.activeMaskStreams.delete(streamId);
            });
        };
        pump();

        return { streamId };
      },
    );

    ipcMain.handle("mask:track:cancel", async (_event, streamId: string) => {
      const entry = this.activeMaskStreams.get(streamId);
      if (entry) {
        // Signal backend to stop processing this mask id, then abort local stream
        try {
          await this.makeRequest<"ok" | any>(
            "POST",
            `/mask/track/cancel/${encodeURIComponent(entry.id)}`,
          );
        } catch {}
        try {
          entry.controller.abort();
        } catch {}
        try {
          entry.reader?.cancel().catch(() => {});
        } catch {}
        this.activeMaskStreams.delete(streamId);
      }
      return { success: true };
    });

    // Shapes tracking stream
    ipcMain.handle(
      "mask:track-shapes",
      async (
        event,
        request: {
          id: string;
          input_path: string;
          frame_start: number;
          anchor_frame?: number;
          frame_end: number;
          direction?: "forward" | "backward" | "both";
        },
      ) => {
        // Ensure backend is healthy before starting streaming mask tracking
        const probe = await this.#probeBackendLocality();
        if (!probe.success) {
          throw new Error(
            probe.error ||
              "Cannot complete the request because the backend API is unavailable.",
          );
        }

        const streamId = crypto.randomUUID();

        const controller = new AbortController();
        const payload = await this.#prepareMaskRequest(request);
        const connectTimeoutId = setTimeout(() => {
          try {
            controller.abort();
          } catch {}
        }, Math.max(1, STREAM_CONNECT_TIMEOUT_MS));

        let response: Response;
        try {
          response = await fetch(`${this.backendUrl}/mask/track/shapes`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/x-ndjson",
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
          });
        } catch (err) {
          if (controller.signal.aborted) {
            throw new Error(
              `Mask track(shapes) request timed out after ${STREAM_CONNECT_TIMEOUT_MS}ms`,
            );
          }
          throw err;
        } finally {
          clearTimeout(connectTimeoutId);
        }

        if (!response.ok || !response.body) {
          const errorData = await response
            .json()
            .catch(() => ({ detail: "Unknown error" }));
          throw new Error(
            errorData.detail ||
              `HTTP ${response.status}: ${response.statusText}`,
          );
        }

        const reader = response.body.getReader();
        this.activeMaskStreams.set(streamId, {
          controller,
          reader,
          id: (payload as any).id,
        });

        const decoder = new TextDecoder();
        let buffered = "";

        const pump = (): void => {
          reader
            .read()
            .then(({ value, done }) => {
              try {
                if (done) {
                  const last = buffered.trim();
                  if (last) {
                    const line = last.startsWith("data:")
                      ? last.slice(5)
                      : last;
                    const data = JSON.parse(line);
                    event.sender.send(
                      `mask:track-shapes:chunk:${streamId}`,
                      data,
                    );
                  }
                  event.sender.send(`mask:track-shapes:end:${streamId}`);
                  this.activeMaskStreams.delete(streamId);
                  return;
                }

                if (value) {
                  buffered += decoder.decode(value, { stream: true });
                }

                let newlineIdx = buffered.indexOf("\n");
                while (newlineIdx !== -1) {
                  const rawLine = buffered.slice(0, newlineIdx);
                  buffered = buffered.slice(newlineIdx + 1);
                  const trimmed = rawLine.trim();
                  if (trimmed && trimmed !== "[DONE]") {
                    const line = trimmed.startsWith("data:")
                      ? trimmed.slice(5)
                      : trimmed;
                    const data = JSON.parse(line);
                    event.sender.send(
                      `mask:track-shapes:chunk:${streamId}`,
                      data,
                    );
                  }
                  newlineIdx = buffered.indexOf("\n");
                }
              } catch (err) {
                event.sender.send(`mask:track-shapes:error:${streamId}`, {
                  message: err instanceof Error ? err.message : "Unknown error",
                });
                try {
                  reader.cancel().catch(() => {});
                } catch {}
                this.activeMaskStreams.delete(streamId);
                return;
              }
              pump();
            })
            .catch((err) => {
              event.sender.send(`mask:track-shapes:error:${streamId}`, {
                message: err instanceof Error ? err.message : "Unknown error",
              });
              this.activeMaskStreams.delete(streamId);
            });
        };
        pump();

        return { streamId };
      },
    );
  }

  private registerPreprocessorHandlers(): void {
    // List all preprocessors
    ipcMain.handle(
      "preprocessor:list",
      async (_event, checkDownloaded: boolean = true) => {
        return this.makeRequest<any>(
          "GET",
          `/preprocessor/list?check_downloaded=${checkDownloaded}`,
        );
      },
    );

    // Get preprocessor details
    ipcMain.handle("preprocessor:get", async (_event, name: string) => {
      return this.makeRequest<any>(
        "GET",
        `/preprocessor/get/${encodeURIComponent(name)}`,
      );
    });

    // Download preprocessor
    ipcMain.handle(
      "preprocessor:download",
      async (_event, name: string, jobId?: string) => {
        const body = {
          preprocessor_name: name,
          job_id: jobId,
        };
        return this.makeRequest<{
          job_id: string;
          status: string;
          message?: string;
        }>("POST", "/preprocessor/download", body);
      },
    );

    // Run preprocessor
    ipcMain.handle(
      "preprocessor:run",
      async (
        _event,
        request: {
          preprocessor_name: string;
          input_path: string;
          job_id?: string;
          download_if_needed?: boolean;
          params?: Record<string, any>;
          start_frame?: number;
          end_frame?: number;
        },
      ) => {
        const payload = await this.#preparePreprocessorRunRequest(request);
        return this.makeRequest<{
          job_id: string;
          status: string;
          message?: string;
        }>("POST", "/preprocessor/run", payload);
      },
    );

    // Cancel preprocessor
    ipcMain.handle("preprocessor:cancel", async (_event, jobId: string) => {
      return this.makeRequest<{
        job_id: string;
        status: string;
        message?: string;
      }>("POST", `/preprocessor/cancel/${encodeURIComponent(jobId)}`);
    });

    // Get job status
    ipcMain.handle("preprocessor:status", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/preprocessor/status/${encodeURIComponent(jobId)}`,
      );
    });

    // Get job result
    ipcMain.handle("preprocessor:result", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/preprocessor/result/${encodeURIComponent(jobId)}`,
      );
    });

    // Delete preprocessor (remove downloaded files and unmark)
    ipcMain.handle("preprocessor:delete", async (_event, name: string) => {
      return this.makeRequest<any>(
        "DELETE",
        `/preprocessor/delete/${encodeURIComponent(name)}`,
      );
    });

    // WebSocket handlers migrated to unified ws:* IPC
  }

  async #preparePostprocessorRequest<T extends { input_path: string }>(
    request: T,
  ): Promise<T> {
    if (!this.#isRemoteBackend()) return request;
    const uploaded = await this.#uploadLocalFileIfNeeded(request.input_path);
    if (uploaded) {
      return { ...request, input_path: uploaded };
    }
    return request;
  }

  private registerPostprocessorHandlers(): void {
    // Generic postprocessor runner based on method/task
    ipcMain.handle("postprocessor:run", async (_event, request: any) => {
      try {
        const { method } = request || {};
        if (!method || typeof method !== "string") {
          return { success: false, error: "Missing method" };
        }

        // Normalize file URL for local requests; remote upload handled in prepare step
        if (request?.input_path?.startsWith?.("file://")) {
          try {
            request.input_path = fileURLToPath(request.input_path);
          } catch {}
        }

        // Map based on method
        if (method === "frame-interpolate") {
          const payload = await this.#preparePostprocessorRequest({
            input_path: String(request.input_path || ""),
            target_fps: Number(request.target_fps || 0),
            job_id: request.job_id,
            exp: request.exp,
            scale: request.scale,
          });
          return this.makeRequest<{
            job_id: string;
            status: string;
            message?: string;
          }>("POST", "/postprocessor/frame-interpolate", payload);
        }

        return {
          success: false,
          error: `Unknown postprocessor method: ${method}`,
        };
      } catch (error) {
        return {
          success: false,
          error:
            error instanceof Error
              ? error.message
              : "Failed to run postprocessor",
        };
      }
    });
  }

  private registerEngineHandlers(): void {
    // Run engine manifest
    ipcMain.handle(
      "engine:run",
      async (
        _event,
        request: {
          manifest_id?: string;
          yaml_path?: string;
          inputs: Record<string, any>;
          selected_components?: Record<string, any>;
          job_id?: string;
          folder_uuid?: string;
        },
      ) => {
        const payload = await this.#prepareEngineRunRequest(request);
        return this.makeRequest<{
          job_id: string;
          status: string;
          message?: string;
        }>("POST", "/engine/run", payload);
      },
    );

    // Cancel engine job
    ipcMain.handle("engine:cancel", async (_event, jobId: string) => {
      return this.makeRequest<{
        job_id: string;
        status: string;
        message?: string;
      }>("POST", `/engine/cancel/${encodeURIComponent(jobId)}`);
    });

    // Engine job status
    ipcMain.handle("engine:status", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/engine/status/${encodeURIComponent(jobId)}`,
      );
    });

    // Engine job result
    ipcMain.handle("engine:result", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/engine/result/${encodeURIComponent(jobId)}`,
      );
    });
  }

  private registerManifestHandlers(): void {
    // List model types
    ipcMain.handle("manifest:types", async () => {
      return this.makeRequest<any>("GET", "/manifest/types");
    });

    // List all manifests
    ipcMain.handle("manifest:list", async () => {
      return this.makeRequest<any>("GET", "/manifest/list");
    });

    // List manifests by model
    ipcMain.handle("manifest:list-by-model", async (_event, model: string) => {
      return this.makeRequest<any>(
        "GET",
        `/manifest/list/model/${encodeURIComponent(model)}`,
      );
    });

    // List manifests by model type
    ipcMain.handle(
      "manifest:list-by-type",
      async (_event, modelType: string) => {
        return this.makeRequest<any>(
          "GET",
          `/manifest/list/type/${encodeURIComponent(modelType)}`,
        );
      },
    );

    // List manifests by model and type
    ipcMain.handle(
      "manifest:list-by-model-and-type",
      async (_event, model: string, modelType: string) => {
        return this.makeRequest<any>(
          "GET",
          `/manifest/list/model/${encodeURIComponent(model)}/model_type/${encodeURIComponent(modelType)}`,
        );
      },
    );

    // Get manifest content by id
    ipcMain.handle("manifest:get", async (_event, manifestId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/manifest/${encodeURIComponent(manifestId)}`,
      );
    });

    // Get a specific part of a manifest by dot-path (e.g., spec.loras or spec.components.0.model_path)
    ipcMain.handle(
      "manifest:get-part",
      async (_event, manifestId: string, pathDot?: string) => {
        const qp = pathDot ? `?path=${encodeURIComponent(pathDot)}` : "";
        return this.makeRequest<any>(
          "GET",
          `/manifest/${encodeURIComponent(manifestId)}/part${qp}`,
        );
      },
    );

    // Validate and register a custom model path for a manifest component
    ipcMain.handle(
      "manifest:validate-custom-model-path",
      async (
        _event,
        request: {
          manifest_id: string;
          component_index: number;
          name?: string;
          path: string;
        },
      ) => {
        const body = {
          manifest_id: String(request.manifest_id || ""),
          component_index: Number(request.component_index),
          name: request.name,
          path: String(request.path || ""),
        };
        return this.makeRequest<any>(
          "POST",
          "/manifest/custom-model-path",
          body,
        );
      },
    );

    ipcMain.handle(
      "manifest:delete-custom-model-path",
      async (
        _event,
        request: {
          manifest_id: string;
          component_index: number;
          path: string;
        },
      ) => {
        const body = {
          manifest_id: String(request.manifest_id || ""),
          component_index: Number(request.component_index),
          path: String(request.path || ""),
        };
        return this.makeRequest<any>(
          "DELETE",
          "/manifest/custom-model-path",
          body,
        );
      },
    );

    // Update a single LoRA entry's scale inside a manifest YAML
    ipcMain.handle(
      "manifest:update-lora-scale",
      async (
        _event,
        request: {
          manifest_id: string;
          lora_index: number;
          scale: number;
        },
      ) => {
        const body = {
          manifest_id: String(request.manifest_id || ""),
          lora_index: Number(request.lora_index),
          scale: Number(request.scale),
        };
        return this.makeRequest<any>("POST", "/manifest/lora/scale", body);
      },
    );

    // Update a LoRA entry's name/label inside a manifest YAML
    ipcMain.handle(
      "manifest:update-lora-name",
      async (
        _event,
        request: {
          manifest_id: string;
          lora_index: number;
          name: string;
        },
      ) => {
        const body = {
          manifest_id: String(request.manifest_id || ""),
          lora_index: Number(request.lora_index),
          name: String(request.name || "").trim(),
        };
        return this.makeRequest<any>("POST", "/manifest/lora/name", body);
      },
    );

    // Delete a LoRA entry (and best-effort remove its local path)
    ipcMain.handle(
      "manifest:delete-lora",
      async (
        _event,
        request: {
          manifest_id: string;
          lora_index: number;
        },
      ) => {
        const body = {
          manifest_id: String(request.manifest_id || ""),
          lora_index: Number(request.lora_index),
        };
        return this.makeRequest<any>("DELETE", "/manifest/lora", body);
      },
    );
  }

  private registerComponentsHandlers(): void {
    // Start components download
    ipcMain.handle(
      "components:download",
      async (_event, paths: string[], savePath?: string, jobId?: string) => {
        const body = { paths, save_path: savePath, job_id: jobId };
        return this.makeRequest<{
          job_id: string;
          status: string;
          message?: string;
        }>("POST", "/components/download", body);
      },
    );

    // Delete a downloaded component (file or directory under components path)
    ipcMain.handle("components:delete", async (_event, targetPath: string) => {
      return this.makeRequest<{ status: string; path: string }>(
        "DELETE",
        "/components/delete",
        { path: targetPath },
      );
    });

    // Poll components download status
    ipcMain.handle("components:status", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/jobs/status/${encodeURIComponent(jobId)}`,
      );
    });

    // Cancel components download
    ipcMain.handle("components:cancel", async (_event, jobId: string) => {
      return this.makeRequest<{
        job_id: string;
        status: string;
        message?: string;
      }>("POST", `/jobs/cancel/${encodeURIComponent(jobId)}`);
    });

    // WebSocket handlers migrated to unified ws:* IPC
  }

  private registerDownloadHandlers(): void {
    // Start unified download
    ipcMain.handle(
      "download:start",
      async (
        _event,
        request: {
          item_type: "component" | "lora" | "preprocessor";
          source: string | string[];
          save_path?: string;
          job_id?: string;
          manifest_id?: string;
          lora_name?: string;
          component?: string;
        },
      ) => {
        const body = {
          item_type: request.item_type,
          source: request.source,
          save_path: request.save_path,
          job_id: request.job_id,
          manifest_id: request.manifest_id,
          lora_name: request.lora_name,
          component: request.component,
        };
        return this.makeRequest<{
          job_id: string;
          status: string;
          message?: string;
        }>("POST", "/download", body);
      },
    );

    // Resolve job id or check if already downloaded
    ipcMain.handle(
      "download:resolve",
      async (
        _event,
        request: {
          item_type: "component" | "lora" | "preprocessor";
          source: string | string[];
          save_path?: string;
        },
      ) => {
        const body = {
          item_type: request.item_type,
          source: request.source,
          save_path: request.save_path,
        };
        return this.makeRequest<{
          job_id: string;
          exists: boolean;
          running: boolean;
          downloaded: boolean;
          bucket: string;
          save_dir: string;
        }>("POST", "/download/resolve", body);
      },
    );

    // Resolve multiple sources at once
    ipcMain.handle(
      "download:resolve-batch",
      async (
        _event,
        request: {
          item_type: "component" | "lora" | "preprocessor";
          sources: Array<string | string[]>;
          save_path?: string;
        },
      ) => {
        const body = {
          item_type: request.item_type,
          sources: request.sources,
          save_path: request.save_path,
        };
        return this.makeRequest<{
          results: Array<{
            job_id: string;
            exists: boolean;
            running: boolean;
            downloaded: boolean;
            bucket: string;
            save_dir: string;
          }>;
        }>("POST", "/download/resolve/batch", body);
      },
    );

    // Poll unified download status
    ipcMain.handle("download:status", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/download/status/${encodeURIComponent(jobId)}`,
      );
    });

    // Cancel unified download
    ipcMain.handle("download:cancel", async (_event, jobId: string) => {
      return this.makeRequest<{
        job_id: string;
        status: string;
        message?: string;
      }>("POST", `/download/cancel/${encodeURIComponent(jobId)}`);
    });

    // Delete a downloaded path (file or directory) from filesystem
    ipcMain.handle(
      "download:delete",
      async (
        _event,
        request: {
          path: string;
          item_type?: "component" | "lora" | "preprocessor";
          source?: string | string[];
          save_path?: string;
        },
      ) => {
        const body = {
          path: String(request.path || ""),
          item_type: request.item_type,
          source: request.source,
          save_path: request.save_path,
        };
        return this.makeRequest<{
          path: string;
          status: string;
          removed_mapping?: boolean;
          unmarked?: boolean;
        }>("DELETE", "/download/delete", body);
      },
    );
  }

  private registerSystemHandlers(): void {
    // System memory usage
    ipcMain.handle("system:memory", async () => {
      return this.makeRequest<any>("GET", "/system/memory");
    });

    // Request backend to free/reclaim memory (best-effort)
    ipcMain.handle("system:free-memory", async () => {
      // This endpoint is action-oriented; prefer POST.
      return this.makeRequest<any>("POST", "/system/free-memory", {
        target: "disk"
      });
    });

    // Reveal a file or folder in the OS file manager
    ipcMain.handle(
      "files:reveal-in-folder",
      async (_event, absPath: string) => {
        try {
          if (typeof absPath !== "string" || !absPath) {
            return { success: false, error: "Invalid path" };
          }
          shell.showItemInFolder(absPath);
          return { success: true, data: { path: absPath } };
        } catch (error) {
          return {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to reveal item in folder",
          };
        }
      },
    );

    // Folder size (bytes) for a given path on the local filesystem
    ipcMain.handle(
      "system:get-folder-size",
      async (_event, targetPath: string) => {
        try {
          const root = String(targetPath || "").trim();
          if (!root) {
            return {
              success: false,
              error: "Invalid path",
            } as ConfigResponse<{ size_bytes: number }>;
          }

          const totalBytes = await this.#computeFolderSize(root);
          const res: ConfigResponse<{ size_bytes: number }> = {
            success: true,
            data: { size_bytes: totalBytes },
          };
          return res;
        } catch (error) {
          const res: ConfigResponse<{ size_bytes: number }> = {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to compute folder size",
          };
          return res;
        }
      },
    );
  }

  private registerJobHandlers(): void {
    // Unified job status
    ipcMain.handle("jobs:status", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/jobs/status/${encodeURIComponent(jobId)}`,
      );
    });

    // Unified job cancel
    ipcMain.handle("jobs:cancel", async (_event, jobId: string) => {
      return this.makeRequest<{
        job_id: string;
        status: string;
        message?: string;
      }>("POST", `/jobs/cancel/${encodeURIComponent(jobId)}`);
    });

    // Ray job listing and management (aggregated view across subsystems)
    ipcMain.handle("ray:jobs:list", async () => {
      return this.makeRequest<any>("GET", "/ray/jobs");
    });

    ipcMain.handle("ray:jobs:get", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "GET",
        `/ray/jobs/${encodeURIComponent(jobId)}`,
      );
    });

    ipcMain.handle("ray:jobs:cancel", async (_event, jobId: string) => {
      return this.makeRequest<any>(
        "POST",
        `/ray/jobs/${encodeURIComponent(jobId)}/cancel`,
      );
    });

    ipcMain.handle("ray:jobs:cancel-all", async () => {
      return this.makeRequest<any>("POST", "/ray/jobs/cancel_all");
    });
  }

  private registerWebSocketHandlers(): void {
    // Unified connect: expects { key, pathOrUrl }
    ipcMain.handle(
      "ws:connect",
      async (event, request: { key: string; pathOrUrl: string }) => {
        const { key, pathOrUrl } = request;
        const result = this.wsManager!.connect(key, pathOrUrl, {
          onOpen: () => {
            event.sender.send(`ws-status:${key}`, { status: "connected" });
          },
          onMessage: (data) => {
            try {
              const message = JSON.parse(data);
              event.sender.send(`ws-update:${key}`, message);
            } catch (error) {
              // Forward as error if message is not JSON
              event.sender.send(`ws-error:${key}`, {
                error: "Invalid JSON message",
                raw: data,
              });
            }
          },
          onError: (error) => {
            event.sender.send(`ws-error:${key}`, { error: error.message });
          },
          onClose: () => {

            event.sender.send(`ws-status:${key}`, { status: "disconnected" });
          },
        });
        if (result.success) {
          return { success: true, data: { key } };
        }
        return {
          success: false,
          error: result.error || "Failed to connect to WebSocket",
        };
      },
    );

    // Unified disconnect: expects key
    ipcMain.handle("ws:disconnect", async (_event, key: string) => {
      const result = this.wsManager!.disconnect(key);
      if (result.success) {
        return { success: true, data: { message: "WebSocket disconnected" } };
      }
      return {
        success: false,
        error: result.error || "WebSocket connection not found",
      };
    });

    // Optional: status check
    ipcMain.handle("ws:status", async (_event, key: string) => {
      const connected = this.wsManager?.has(key) ?? false;
      return { success: true, data: { key, connected } };
    });
  }


  async #fetchWithTimeout(
    url: string,
    init: RequestInit = {},
    timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS,
  ): Promise<Response> {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    
    try {
      return await fetch(url, { ...init, signal: controller.signal });
    } catch (err) {
      console.log("Failed to fetch with timeout", err, url, timeoutMs);
      throw err;
    } finally {
      clearTimeout(id);
    }
  }

  async #remoteApexCacheFileExists(relPath: string): Promise<boolean> {
    try {
      if (!this.#isRemoteBackend()) return false;
      const url = `${this.backendUrl}/files?scope=apex-cache&path=${encodeURIComponent(relPath)}`;
      // Prefer a tiny range request to avoid downloading large files.
      const resp = await this.#fetchWithTimeout(
        url,
        {
          method: "GET",
          headers: { Range: "bytes=0-0" },
        },
        REMOTE_FILE_EXISTS_TIMEOUT_MS,
      );
      return resp.ok;
    } catch {
      return false;
    }
  }

  async #matchRemoteFile(params: {
    scope: string;
    path: string;
    sha256: string;
    size?: number;
  }): Promise<{ exists: boolean; matches: boolean; sha256?: string; size?: number }> {
    const qp = new URLSearchParams();
    qp.set("scope", params.scope);
    qp.set("path", params.path);
    qp.set("sha256", params.sha256);
    if (typeof params.size === "number" && Number.isFinite(params.size)) {
      qp.set("size", String(params.size));
    }

    const url = `${this.backendUrl}/files/match?${qp.toString()}`;
    const resp = await this.#fetchWithTimeout(
      url,
      { method: "GET", headers: { "Content-Type": "application/json" } },
      REMOTE_FILE_MATCH_TIMEOUT_MS,
    );
    if (!resp.ok) {
      const t = await resp.text().catch(() => "");
      throw new Error(
        `Remote match failed (HTTP ${resp.status}: ${t || resp.statusText})`,
      );
    }
    const j = (await resp.json().catch(() => ({}))) as any;
    return {
      exists: !!j?.exists,
      matches: !!j?.matches,
      sha256: typeof j?.sha256 === "string" ? j.sha256 : undefined,
      size: typeof j?.size === "number" ? j.size : undefined,
    };
  }

  async #sha256File(absPath: string): Promise<string> {
    return await new Promise<string>((resolve, reject) => {
      const hash = crypto.createHash("sha256");
      const stream = fs.createReadStream(absPath);
      stream.on("error", reject);
      stream.on("data", (chunk) => hash.update(chunk));
      stream.on("end", () => resolve(hash.digest("hex")));
    });
  }

  private async makeRequest<T>(
    method: "GET" | "POST" | "DELETE",
    endpoint: string,
    body?: any,
  ): Promise<ConfigResponse<T>> {
    try {
      const baseUrl = this.backendUrl;

      const options: RequestInit = {
        method,
        headers: {
          "Content-Type": "application/json",
        },
      };

      if (body && method !== "GET") {
        options.body = JSON.stringify(body);
      }

      const run = (async (): Promise<ConfigResponse<T>> => {
        const response = await this.#fetchWithTimeout(
          `${baseUrl}${endpoint}`,
          options,
          DEFAULT_REQUEST_TIMEOUT_MS,
        );

        if (!response.ok) {
          console.log("Failed to make request", response.statusText, endpoint);
          const errorData = await response
            .json()
            .catch(() => ({ detail: "Unknown error" }));
          return {
            success: false,
            error:
              errorData.detail ||
              `HTTP ${response.status}: ${response.statusText}`,
          };
        }

        const data = await response.json();
        const res: ConfigResponse<T> = {
          success: true,
          data: data as T,
        };


        return res;
      })();


      return await run
    } catch (error) {
      // Best-effort background locality refresh on failures; do not block API calls.
      void this.#probeBackendLocality().catch(() => {});
      return {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }

  // ===== Helpers for remote file handling =====
  async #probeBackendLocality(opts?: { force?: boolean }): Promise<ConfigResponse<null>> {
    const now = Date.now();

    // Serve cached result if still fresh
    if (!opts?.force && this.probeLastRes) {
      const ttl = this.probeLastRes.success ? PROBE_TTL_OK_MS : PROBE_TTL_FAIL_MS;
      if (now - this.probeLastAt < ttl) return this.probeLastRes;
    }

    // De-dupe concurrent callers
    if (!opts?.force && this.probeInFlight) return this.probeInFlight;

    const run = (async (): Promise<ConfigResponse<null>> => {
      try {
        const u = new URL(this.backendUrl);
        const host = (u.hostname || "").toLowerCase();

        const resp = await this.#fetchWithTimeout(
          `${this.backendUrl}/config/hostname`,
          {
            method: "GET",
            headers: { "Content-Type": "application/json" },
          },
          PROBE_TIMEOUT_MS,
        );

        if (!resp.ok) {
          let detail = "";
          try {
            const data = await resp.json();
            if (data && typeof (data as any).detail === "string") {
              detail = ` ${(data as any).detail}`;
            }
          } catch {
            // ignore JSON parse errors from hostname endpoint
          }

          return {
            success: false,
            error:
              `Cannot complete the request: backend hostname probe failed (HTTP ${resp.status}: ${resp.statusText}).` +
              detail,
          };
        }

        const data = (await resp.json().catch(() => ({}))) as any;
        const remoteHostname =
          typeof data?.hostname === "string" ? data.hostname : "";
        const localHostname = os.hostname();

        if (host === "localhost" || host === "127.0.0.1" || host === "::1") {
          // For loopback, treat it as remote if the reported hostname differs
          this.loopbackAppearsRemote =
            !!remoteHostname && remoteHostname !== localHostname;
        } else {
          // Non-loopback hosts are definitely remote
          this.loopbackAppearsRemote = true;
        }

        return {
          success: true,
          data: null,
        };
      } catch (error) {
        return {
          success: false,
          error:
            error instanceof Error
              ? `Cannot complete the request because the backend API is unreachable: ${error.message}`
              : "Cannot complete the request because the backend API is unreachable.",
        };
      }
    })();

    this.probeInFlight = run;
    const res = await run.finally(() => {
      this.probeInFlight = null;
    });
    this.probeLastAt = Date.now();
    this.probeLastRes = res;
    return res;
  }

  #isRemoteBackend(): boolean {
    try {
      const u = new URL(this.backendUrl);
      const host = (u.hostname || "").toLowerCase();
      if (host === "localhost" || host === "127.0.0.1" || host === "::1")
        return this.loopbackAppearsRemote;
      return true;
    } catch {
      // Fall back to last known probe result
      return this.loopbackAppearsRemote;
    }
  }

  async #ensureRemoteCachePath(): Promise<string | null> {
    if (!this.#isRemoteBackend()) return null;
    if (this.remoteCachePath) return this.remoteCachePath;
    const res = await this.makeRequest<{ cache_path: string }>(
      "GET",
      "/config/cache-path",
    );
    if (res.success && res.data?.cache_path) {
      this.remoteCachePath = res.data.cache_path;
      return this.remoteCachePath;
    }
    return null;
  }

  #looksLikeHttp(urlOrPath: string): boolean {
    return /^https?:\/\//i.test(urlOrPath);
  }

  #looksLikeAppUrl(urlOrPath: string): boolean {
    return /^app:\/\//i.test(urlOrPath);
  }

  #looksLikeFileUrl(urlOrPath: string): boolean {
    return /^file:\/\//i.test(urlOrPath);
  }

  #resolveLocalPath(maybePath: string): string | null {
    try {
      if (!maybePath || typeof maybePath !== "string") return null;
      if (this.#looksLikeHttp(maybePath)) return null;
      if (this.#looksLikeFileUrl(maybePath)) {
        const p = fileURLToPath(maybePath);
        return fs.existsSync(p) ? p : null;
      }
      if (this.#looksLikeAppUrl(maybePath)) {
        // Support app://user-data/... and app://apex-cache/...
        const u = new URL(maybePath);
        const pathname = decodeURIComponent(u.pathname || "");
        const norm = pathname.startsWith("/") ? pathname.slice(1) : pathname;
        if (u.hostname === "user-data" && this.app) {
          const base = this.app.getPath("userData");
          const candidate = norm.startsWith(base)
            ? norm
            : path.join(base, norm);
          return fs.existsSync(candidate) ? candidate : null;
        }
        // For apex-cache, we do not know the local cache root reliably in remote mode; skip
        return null;
      }
      // Absolute/relative paths
      if (fs.existsSync(maybePath)) return maybePath;
      return null;
    } catch {
      return null;
    }
  }

  async #uploadLocalFileIfNeeded(
    localPathOrUrl: string,
  ): Promise<string | null> {
    if (!this.#isRemoteBackend()) return null;
    const abs = this.#resolveLocalPath(localPathOrUrl);
    if (!abs) return null;
    const st = await fs.promises.stat(abs);
    const size = st.size;
    const mtimeMs = st.mtimeMs;

    let sha: string | null = null;
    const cached = this.uploadCache.get(abs);
    // Only reuse cached uploads if we can prove the local content is identical.
    if (cached && cached.sha256 && cached.size === size && cached.mtimeMs === mtimeMs) {
      try {
        sha = await this.#sha256File(abs);
        if (sha === cached.sha256) return cached.remoteAbs;
      } catch {
        // If hashing fails, do not assume cache hit.
      }
    }
    // Ensure backend is reachable before attempting to upload, and update
    // our remote/local detection at the same time.
    const probe = await this.#probeBackendLocality();
    if (!probe.success) {
      throw new Error(
        probe.error ||
          "Cannot complete the request because the backend API is unavailable.",
      );
    }

    const remoteCacheBase = await this.#ensureRemoteCachePath();
    if (!remoteCacheBase) throw new Error("Remote cache path unavailable");

    const fileName = path.basename(abs);

    if (size > MAX_UPLOAD_BYTES) {
      const fileGiB = (size / 1024 ** 3).toFixed(2);
      const limitGiB = (MAX_UPLOAD_BYTES / 1024 ** 3).toFixed(2);
      throw new Error(
        `File is too large to upload automatically (${fileGiB} GiB > ${limitGiB} GiB). Place the file on the backend or raise APEX_MAX_UPLOAD_BYTES.`,
      );
    }

    // Use a deterministic destination so we can cheaply skip uploads when the
    // backend already has the same content.
    // - Path is inside the apex-cache scope (server-side).
    // - Name is content-addressed to dedupe across repeated uploads.
    let destRel = `uploads/${crypto.randomUUID()}-${fileName}`;
    try {
      sha = sha || (await this.#sha256File(abs));
      const ext = path.extname(fileName || "");
      destRel = `uploads/by-hash/${sha}${ext}`;
    } catch {
      // If hashing fails, fall back to randomized dest.
    }

    // If the backend already has this exact dest path, skip uploading.
    try {
      if (sha) {
        const match = await this.#matchRemoteFile({
          scope: "apex-cache",
          path: destRel,
          sha256: sha,
          size,
        });
        if (match.exists && match.matches) {
          const remoteAbs = this.#joinRemote(remoteCacheBase, destRel);
          if (sha) {
            this.uploadCache.set(abs, { remoteAbs, sha256: sha, size, mtimeMs });
          }
          return remoteAbs;
        }
        // If the deterministic path is occupied by different content, avoid
        // overwriting and choose an alternate unique destination.
        if (match.exists && !match.matches) {
          const ext = path.extname(fileName || "");
          destRel = `uploads/by-hash/${sha}-${crypto.randomUUID()}${ext}`;
        }
      } else {
        const exists = await this.#remoteApexCacheFileExists(destRel);
        if (exists) {
          const remoteAbs = this.#joinRemote(remoteCacheBase, destRel);
          // Without a sha, we can't safely reuse this mapping if the local file changes.
          return remoteAbs;
        }
      }
    } catch {
      // Ignore preflight probe errors and proceed with upload.
    }

    // Use query params for scope/dest to avoid multipart field name disagreements
    const url = `${this.backendUrl}/files/ingest?scope=apex-cache&dest=${encodeURIComponent(destRel)}`;
    const form = new NodeFormData();
    form.set("file", await fileFromPath(abs, fileName));
    const encoder = new FormDataEncoder(form);

    const uploadController = new AbortController();
    let uploadTimedOut = false;
    const uploadTimeoutId = setTimeout(() => {
      uploadTimedOut = true;
      try {
        uploadController.abort();
      } catch {}
    }, Math.max(1, UPLOAD_TIMEOUT_MS));

    let resp: Response;
    try {
      resp = await fetch(url, {
        method: "POST",
        headers: encoder.headers as any,
        body: Readable.from(encoder) as any,
        signal: uploadController.signal,
        // Required by Node's fetch for streaming request bodies
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        duplex: "half",
      });
    } catch (err) {
      if (uploadTimedOut) {
        throw new Error(`Upload timed out after ${UPLOAD_TIMEOUT_MS}ms`);
      }
      throw err;
    } finally {
      clearTimeout(uploadTimeoutId);
    }
    if (!resp.ok) {
      const t = await resp.text().catch(() => "");
      throw new Error(
        `Upload failed (${resp.status}): ${t || resp.statusText}`,
      );
    }
    // Expect { path: string } relative to cache
    let relReturned: string = destRel;
    try {
      const j = await resp.json();
      relReturned = j && typeof j.path === "string" ? j.path : destRel;
    } catch {
      relReturned = destRel;
    }
    const remoteAbs = this.#joinRemote(
      remoteCacheBase as string,
      relReturned as string,
    );
    if (sha) {
      this.uploadCache.set(abs, { remoteAbs, sha256: sha, size, mtimeMs });
    }
    return remoteAbs;
  }

  #joinRemote(base: string, rel: string): string {
    const b = base.replace(/\\/g, "/").replace(/\/$/, "");
    const r = rel.replace(/\\/g, "/").replace(/^\//, "");
    return `${b}/${r}`;
  }

  async #prepareMaskRequest<T extends { input_path: string }>(
    request: T,
  ): Promise<T> {
    if (!this.#isRemoteBackend()) return request;
    const uploaded = await this.#uploadLocalFileIfNeeded(request.input_path);
    if (uploaded) {
      return { ...request, input_path: uploaded };
    }
    return request;
  }

  async #preparePreprocessorRunRequest(request: {
    preprocessor_name: string;
    input_path: string;
    job_id?: string;
    download_if_needed?: boolean;
    params?: Record<string, any>;
    start_frame?: number;
    end_frame?: number;
  }): Promise<{
    preprocessor_name: string;
    input_path: string;
    job_id?: string;
    download_if_needed?: boolean;
    params?: Record<string, any>;
    start_frame?: number;
    end_frame?: number;
  }> {
    // Ensure our remote/local detection is up-to-date before deciding whether to
    // auto-upload local paths. (Cached probe; non-fatal if hostname endpoint is missing.)
    try {
      await this.#probeBackendLocality();
    } catch {}
    if (!this.#isRemoteBackend()) return request;
    const uploaded = await this.#uploadLocalFileIfNeeded(request.input_path);
    if (uploaded) {
      return { ...request, input_path: uploaded };
    }
    return request;
  }

  async #prepareEngineRunRequest(request: {
    manifest_id?: string;
    yaml_path?: string;
    inputs: Record<string, any>;
    selected_components?: Record<string, any>;
    job_id?: string;
    folder_uuid?: string;
  }): Promise<{
    manifest_id?: string;
    yaml_path?: string;
    inputs: Record<string, any>;
    selected_components?: Record<string, any>;
    job_id?: string;
    folder_uuid?: string;
  }> {
    // Ensure our remote/local detection is up-to-date before deciding whether to
    // auto-upload local paths. (Cached probe; non-fatal if hostname endpoint is missing.)
    try {
      await this.#probeBackendLocality();
    } catch {}
    if (!this.#isRemoteBackend()) return request;
    const transformed: Record<string, any> = {};
    for (const [k, v] of Object.entries(request.inputs || {})) {
      transformed[k] = await this.#transformValueForUpload(v);
    }
    return { ...request, inputs: transformed };
  }

  async #transformValueForUpload(value: any): Promise<any> {
    if (value == null) return value;
    if (typeof value === "string") {
      const uploaded = await this.#uploadLocalFileIfNeeded(value);
      return uploaded || value;
    }
    if (Array.isArray(value)) {
      const out = [] as any[];
      for (const item of value) {
        // Only map strings or one-level nested arrays of strings
        if (typeof item === "string") {
          out.push((await this.#uploadLocalFileIfNeeded(item)) || item);
        } else if (Array.isArray(item)) {
          const inner: any[] = [];
          for (const s of item) {
            inner.push(
              typeof s === "string"
                ? (await this.#uploadLocalFileIfNeeded(s)) || s
                : s,
            );
          }
          out.push(inner);
        } else {
          out.push(item);
        }
      }
      return out;
    }
    if (typeof value === "object") {
      const keys = Object.keys(value);
      const onlySrcKey = keys.length === 1 && keys[0] === "src";
      if (onlySrcKey && typeof value.src === "string") {
        const uploaded = await this.#uploadLocalFileIfNeeded(value.src);
        return uploaded || value.src;
      }
      let mutated = false;
      let nextValue: any = value;
      if (typeof value.input_path === "string") {
        const uploaded = await this.#uploadLocalFileIfNeeded(value.input_path);
        if (uploaded) {
          nextValue = { ...nextValue, input_path: uploaded };
          mutated = true;
        }
      }
      if (typeof value.src === "string") {
        const uploaded = await this.#uploadLocalFileIfNeeded(value.src);
        if (uploaded) {
          if (!mutated) {
            nextValue = { ...nextValue };
            mutated = true;
          }
          nextValue.src = uploaded;
        }
      }
      return mutated ? nextValue : value;
    }
    return value;
  }

  async #computeFolderSize(root: string): Promise<number> {
    try {
      const stat = await fs.promises.stat(root);
      if (!stat.isDirectory()) {
        return stat.size;
      }
    } catch {
      return 0;
    }

    let total = 0;
    const stack: string[] = [root];

    while (stack.length > 0) {
      const dir = stack.pop() as string;
      let entries: fs.Dirent[] = [];
      try {
        entries = await fs.promises.readdir(dir, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          stack.push(fullPath);
        } else if (entry.isFile()) {
          try {
            const fileStat = await fs.promises.stat(fullPath);
            total += fileStat.size;
          } catch {
            // ignore individual file errors
          }
        }
      }
    }

    return total;
  }

  async pathExists(pathOrUrl: string): Promise<{ exists: boolean }> {
    try {
      const resolved = this.#resolveLocalPath(pathOrUrl);
      if (resolved) return { exists: true };
      // If this looks like an HTTP URL, do a lightweight HEAD probe.
      if (this.#looksLikeHttp(pathOrUrl)) {
        try {
          const resp = await this.#fetchWithTimeout(
            pathOrUrl,
            { method: "HEAD" },
            PROBE_TIMEOUT_MS,
          );
          return { exists: resp.ok };
        } catch {
          return { exists: false };
        }
      }
      return { exists: false };
    } catch {
      return { exists: false };
    }
  }
}

export function apexApi(...args: ConstructorParameters<typeof ApexApi>) {
  return new ApexApi(...args);
}
