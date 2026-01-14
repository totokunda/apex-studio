import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { getSettingsModule } from "./SettingsModule.js";
import { protocol, ipcMain } from "electron";
import fs from "node:fs";
import path from "node:path";
import mime from "mime";
import os from "node:os";
import { Readable } from "node:stream";
import { createRequire } from "node:module";

// Register 'app://' as a privileged scheme as early as possible (before 'ready')
protocol.registerSchemesAsPrivileged([
  {
    scheme: "app",
    privileges: {
      standard: true,
      secure: true,
      stream: true,
      supportFetchAPI: true, // important for window.fetch
      corsEnabled: true,
    },
  },
]);

// Helper to convert Node.js stream to Web ReadableStream with proper error handling
function nodeStreamToWebStream(
  nodeStream: fs.ReadStream,
): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      nodeStream.on("data", (chunk: string | Buffer) => {
        try {
          const uint8Array =
            typeof chunk === "string"
              ? new TextEncoder().encode(chunk)
              : new Uint8Array(chunk);
          controller.enqueue(uint8Array);
        } catch (err) {
          // Stream might be closed, ignore
        }
      });

      nodeStream.on("end", () => {
        try {
          controller.close();
        } catch (err) {
          // Already closed, ignore
        }
      });

      nodeStream.on("error", (err) => {
        try {
          controller.error(err);
        } catch {
          // Already errored, ignore
        }
        nodeStream.destroy();
      });
    },
    cancel() {
      nodeStream.destroy();
    },
  });
}

class AppDirProtocol implements AppModule {
  private electronApp: Electron.App | null = null;
  private cachePath: string | null = null; // local base used for serving
  private remoteCacheBasePath: string | null = null; // absolute base path on remote machine
  private backendUrl: string = "http://127.0.0.1:8765";
  private loopbackAppearsRemote: boolean = false;
  private inflightDownloads: Map<string, Promise<boolean>> = new Map();
  private inflightRemoteReprobe: Promise<boolean> | null = null;
  private lastRemoteReprobeAtMs: number = 0;
  private rendererDistPath: string | null = null;

  private backendUrlIsLoopbackHost(): boolean {
    try {
      const u = new URL(this.backendUrl);
      const host = (u.hostname || "").toLowerCase();
      return host === "localhost" || host === "127.0.0.1" || host === "::1";
    } catch {
      return false;
    }
  }

  private appDirBackendRequestTimeoutMs(): number {
    // Keep this aligned with `packages/main/src/modules/ApexApi.ts` defaults.
    // Prefer an AppDirProtocol-specific override, but fall back to the general API timeout.
    const raw =
      process.env.APEX_APPDIR_REQUEST_TIMEOUT_MS ??
      process.env.APEX_API_REQUEST_TIMEOUT_MS;
    const n = Number(raw);
    if (Number.isFinite(n)) {
      // Allow 0/negative to mean "no timeout" if explicitly configured.
      return n;
    }
    return 30_000;
  }

  private onBackendRequestTimeout(context: string, timeoutMs: number): void {
    // Only "downgrade" remote-mode when the backend URL is loopback.
    // If the backend host is truly remote (non-loopback), forcing local-mode breaks remote setups.
    if (this.loopbackAppearsRemote && this.backendUrlIsLoopbackHost()) {
      console.warn(
        `[AppDirProtocol] Backend request timed out after ${timeoutMs}ms (${context}); setting loopbackAppearsRemote=false`,
      );
      this.loopbackAppearsRemote = false;
    }
  }

  private async fetchBackendWithTimeout(
    url: string,
    init?: RequestInit,
    context: string = "backend-request",
  ): Promise<Response> {
    const timeoutMs = this.appDirBackendRequestTimeoutMs();
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      return await fetch(url, init);
    }

    const controller = new AbortController();
    let didTimeout = false;
    const timer = setTimeout(() => {
      didTimeout = true;
      controller.abort();
    }, timeoutMs);

    try {
      return await fetch(url, { ...(init || {}), signal: controller.signal });
    } catch (err) {
      if (didTimeout) {
        this.onBackendRequestTimeout(context, timeoutMs);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  private stripLeadingSlashes(p: string): string {
    return (p || "").replace(/^\/+/, "");
  }

  private stripTrailingSlashes(p: string): string {
    return (p || "").replace(/\/+$/, "");
  }

  private localUserDataBaseCandidates(app: Electron.App): string[] {
    // Keep this aligned with preload's `guessUserDataDir()` (packages/preload/src/media/root.ts).
    // Electron's `app.getPath("userData")` is often ".../Application Support/Electron" in dev,
    // while Apex Studio stores its own data under ".../Application Support/apex-studio".
    const appData = app.getPath("appData");
    const candidates: string[] = [];
    candidates.push(app.getPath("userData"));
    candidates.push(path.join(appData, "apex-studio"));
    candidates.push(path.join(appData, "Apex Studio"));
    const explicit = process.env.APEX_USER_DATA_DIR;
    if (typeof explicit === "string" && explicit.length > 0) {
      candidates.push(explicit);
    }
    return Array.from(new Set(candidates.map((p) => path.resolve(p))));
  }

  private isUnderAnyBase(childAbs: string, basesAbs: string[]): boolean {
    const child = path.resolve(childAbs);
    for (const b of basesAbs) {
      if (this.isSubPath(path.resolve(b), child)) return true;
    }
    return false;
  }

  /**
   * Resolve a relative-ish path under a base directory and reject traversal.
   * Returns an absolute path under `basePath`, or null if it escapes.
   */
  private resolveUnderBase(basePath: string, relOrAbs: string): string | null {
    const baseAbs = path.resolve(basePath);
    const abs = path.resolve(baseAbs, relOrAbs);
    const rel = path.relative(baseAbs, abs);
    // If rel starts with ".." or is absolute, it escaped the base.
    if (rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel))) {
      return abs;
    }
    return null;
  }

  private isSubPath(parentAbs: string, childAbs: string): boolean {
    const rel = path.relative(parentAbs, childAbs);
    return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
  }

  async enable({ app }: ModuleContext): Promise<void> {
    await app.whenReady();

    // Subscribe to backend URL updates from SettingsModule
    const settings = getSettingsModule();
    settings.on("backend-url-changed", (newUrl: string) => {
      void this.onBackendUrlChanged(newUrl);
    });

    this.electronApp = app;
    // Resolve renderer dist path once so we can serve it via `app://renderer/...`
    // with proper Content-Type headers (important for module scripts/workers).
    this.rendererDistPath = null;
    const rendererCandidates: string[] = [];
    try {
      // Best-case: node resolution works (dev + some packaged layouts).
      const require = createRequire(import.meta.url);
      const indexHtml = require.resolve("@app/renderer/dist/index.html");
      rendererCandidates.push(path.dirname(indexHtml));
    } catch {
      // ignore
    }
    try {
      // Packaged app layouts:
      // - process.resourcesPath usually ".../Contents/Resources" (macOS)
      // - app.getAppPath() usually ".../Resources/app.asar" (asar enabled)
      const appPath = app.getAppPath();
      rendererCandidates.push(
        path.join(appPath, "node_modules", "@app", "renderer", "dist"),
      );
      // When asar is enabled, some artifacts can be unpacked:
      rendererCandidates.push(
        path.join(
          process.resourcesPath,
          "app.asar.unpacked",
          "node_modules",
          "@app",
          "renderer",
          "dist",
        ),
      );
      // Non-asar packaging (or dev-ish layouts):
      rendererCandidates.push(
        path.join(process.resourcesPath, "app", "node_modules", "@app", "renderer", "dist"),
      );
      rendererCandidates.push(
        path.join(process.resourcesPath, "node_modules", "@app", "renderer", "dist"),
      );
    } catch {
      // ignore
    }
    // Pick first candidate that actually contains index.html
    for (const cand of rendererCandidates) {
      try {
        const idx = path.join(cand, "index.html");
        if (fs.existsSync(idx)) {
          this.rendererDistPath = cand;
          break;
        }
      } catch {
        // ignore
      }
    }
    if (!this.rendererDistPath) {
      console.warn(
        "[AppDirProtocol] Could not resolve renderer dist path; app://renderer/* will 404. Candidates:",
        rendererCandidates,
      );
    } else {
      console.log(
        `[AppDirProtocol] rendererDistPath: ${this.rendererDistPath}`,
      );
    }

    ipcMain.handle(
      "appdir:resolve-path",
      async (_event, appUrl: string): Promise<string | null> => {
        try {
          const u = new URL(appUrl);
          return await this.resolveAppUrlToFilePathWithRemoteFallback(u);
        } catch {
          return null;
        }
      },
    );
    ipcMain.handle("appdir:renderer-dist-path", async (): Promise<string | null> => {
      return this.rendererDistPath;
    });

    // Kick off non-blocking initialization (backend locality, remote cache path, etc.)
    // Do NOT block app startup on backend/network availability.
    void this.initializeAsync(app);

    protocol.handle("app", async (request) => {
      // CORS/preflight handling for renderer -> app:// fetches
      const origin = request.headers.get("origin") || "*";
      const baseCors = {
        "Access-Control-Allow-Origin": origin,
        Vary: "Origin",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Range",
        "Cross-Origin-Resource-Policy": "cross-origin",
      } as Record<string, string>;
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: baseCors });
      }
      const u = new URL(request.url);
      const rawFolderUuid = u.searchParams.get("folderUuid");
      const folderUuid =
        typeof rawFolderUuid === "string" && rawFolderUuid.length > 0
          ? rawFolderUuid
          : undefined;

      let basePath: string | null = null;
      if (u.hostname === "user-data") {
        basePath = app.getPath("userData");
      } else if (u.hostname === "apex-cache") {
        basePath = this.cachePath;
      }
 

      const filePath = await this.resolveAppUrlToFilePathWithRemoteFallback(u);
      if (!filePath) {
        if (u.hostname === "renderer") {
          console.warn(
            `[AppDirProtocol] 404 for ${request.url} (rendererDistPath=${this.rendererDistPath ?? "null"})`,
          );
        }
        return new Response(null, { status: 404, headers: baseCors });
      }

      let stat: fs.Stats;
      try {
        stat = fs.statSync(filePath);
      } catch (e) {
        console.warn(
          `[AppDirProtocol] stat failed for ${request.url} -> ${filePath}`,
          e,
        );
        return new Response(null, { status: 404, headers: baseCors });
      }
      const fileSize = stat.size;
      const contentType = mime.getType(filePath) || "application/octet-stream";

      // Fast-path for HEAD: only return headers, never open a read stream.
      if (request.method === "HEAD") {
        return new Response(null, {
          status: 200,
          headers: {
            "Content-Type": contentType,
            "Content-Length": fileSize.toString(),
            "Accept-Ranges": "bytes",
            ...baseCors,
          },
        });
      }

      // Handle Range requests
      const rangeHeader = request.headers.get("range");
      if (rangeHeader) {
        const parts = rangeHeader.replace(/bytes=/, "").split("-");
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
        const chunkSize = end - start + 1;

        const nodeStream = fs.createReadStream(filePath, { start, end });
        const webStream = nodeStreamToWebStream(nodeStream);

        return new Response(webStream, {
          status: 206,
          headers: {
            "Content-Type": contentType,
            "Content-Length": chunkSize.toString(),
            "Content-Range": `bytes ${start}-${end}/${fileSize}`,
            "Accept-Ranges": "bytes",
            ...baseCors,
          },
        });
      }

      // Full content response
      const nodeStream = fs.createReadStream(filePath);
      const webStream = nodeStreamToWebStream(nodeStream);

      return new Response(webStream, {
        status: 200,
        headers: {
          "Content-Type": contentType,
          "Content-Length": fileSize.toString(),
          "Accept-Ranges": "bytes",
          ...baseCors,
        },
      });
    });
  }

  private async resolveAppUrlToFilePathWithRemoteFallback(
    u: URL,
  ): Promise<string | null> {
    const first = await this.resolveAppUrlToFilePath(u);
    if (first) return first;

    // Only attempt remote fallback for apex-cache.
    if (u.hostname !== "apex-cache") return null;
    if (this.loopbackAppearsRemote) return null;

    const decodedPathname = this.safeDecodeURIComponent(u.pathname);
    const candidates = this.remoteRelCandidatesFromPath(decodedPathname);
    if (candidates.length === 0) return null;

    const switched = await this.maybeReprobeRemoteAndSwitch(candidates);
    if (!switched) return null;

    // Retry once with updated remote settings.
    return await this.resolveAppUrlToFilePath(u);
  }

  private async resolveAppUrlToFilePath(u: URL): Promise<string | null> {
    let basePath: string | null = null;
    const app = this.electronApp;

    if (u.hostname === "user-data") {
      basePath = app ? app.getPath("userData") : null;
    } else if (u.hostname === "apex-cache") {
      basePath = this.cachePath;
    } else if (u.hostname === "renderer") {
      basePath = this.rendererDistPath;
    }


    if (!basePath) {
      return null;
    }

    const decodedPathname = this.safeDecodeURIComponent(u.pathname);
    const rawFolderUuid = u.searchParams.get("folderUuid");
    const folderUuid =
      typeof rawFolderUuid === "string" && rawFolderUuid.length > 0
        ? rawFolderUuid
        : undefined;

    let filePath: string;


    if (u.hostname === "apex-cache" && this.loopbackAppearsRemote) {
      // In remote mode, apex-cache usually refers to a remote cache mirrored locally under `this.cachePath`.
      // However, some callers pass absolute *local* paths (e.g. appData/apex-studio/media/...).
      // If it's a local absolute path under our allowed roots, serve it directly.
      const candidateAbs =
        this.coerceWindowsAbsolutePathFromUrlPathname(decodedPathname) ??
        (path.isAbsolute(decodedPathname) ? path.normalize(decodedPathname) : null);
      if (candidateAbs && app) {
        // Construct broad list of allowed local bases to prevent unnecessary remote fetches
        // for files that are definitely present locally.
        const allowedLocalBases = [
          ...(this.cachePath ? [this.cachePath] : []),
          ...this.localUserDataBaseCandidates(app),
        ];
        
        // Also explicitly allow standard local cache locations even if not currently configured
        // as the "active" cache (since active cache might be the remote mirror).
        try {
          allowedLocalBases.push(path.join(app.getPath("userData"), "apex-cache"));
          const home = os.homedir();
          if (home) {
            allowedLocalBases.push(path.join(home, "apex-diffusion", "cache"));
          }
          // Best-effort: read persistent settings for a custom local cache path
          const settingsPath = path.join(app.getPath("userData"), "apex-settings.json");
          if (fs.existsSync(settingsPath)) {
            const raw = fs.readFileSync(settingsPath, "utf-8");
            const parsed = JSON.parse(raw);
            if (parsed.cachePath && typeof parsed.cachePath === "string") {
              allowedLocalBases.push(path.resolve(parsed.cachePath));
            }
          }
        } catch {
          // ignore
        }

        if (this.isUnderAnyBase(candidateAbs, allowedLocalBases)) {
          filePath = candidateAbs;
        } else {
          const rel = this.remoteRelFromNormalized(
            this.stripLeadingSlashes(decodedPathname),
          );
          const localPath = this.resolveUnderBase(basePath!, rel);
          if (!localPath) return null;
          try {
            if (!fs.existsSync(localPath) || !fs.statSync(localPath).isFile()) {
              const ok = await this.ensureLocalFromRemote(rel, localPath);
              if (!ok) {
                return null;
              }
            }
            await this.ensureProjectSymlinkForCache(rel, localPath, folderUuid);
          } catch {
            const ok = await this.ensureLocalFromRemote(rel, localPath).catch(
              () => false,
            );
            if (!ok) {
              return null;
            }
            await this.ensureProjectSymlinkForCache(
              rel,
              localPath,
              folderUuid,
            ).catch(() => {
              /* ignore */
            });
          }
          filePath = localPath;
        }
      } else {
        const rel = this.remoteRelFromNormalized(
          this.stripLeadingSlashes(decodedPathname),
        );
        const localPath = this.resolveUnderBase(basePath!, rel);
        if (!localPath) return null;
        try {
          if (!fs.existsSync(localPath) || !fs.statSync(localPath).isFile()) {
            const ok = await this.ensureLocalFromRemote(rel, localPath);
            if (!ok) {
              return null;
            }
          }
          await this.ensureProjectSymlinkForCache(rel, localPath, folderUuid);
        } catch {
          const ok = await this.ensureLocalFromRemote(rel, localPath).catch(
            () => false,
          );
          if (!ok) {
            return null;
          }
          await this.ensureProjectSymlinkForCache(
            rel,
            localPath,
            folderUuid,
          ).catch(() => {
            /* ignore */
          });
        }
        filePath = localPath;
      }
    } else {
      // URL pathnames always start with "/". Treat them as paths *relative to basePath*,
      // except when the decoded pathname is an absolute filesystem path already under basePath.
      const candidateAbs =
        this.coerceWindowsAbsolutePathFromUrlPathname(decodedPathname) ??
        (path.isAbsolute(decodedPathname) ? path.normalize(decodedPathname) : null);

      // If the renderer passed an absolute filesystem path, accept it only if it falls under our allowed roots.
      // Otherwise, treat it as a normal URL path relative to basePath (e.g. "/index.html" -> "index.html").
      //
      // This is important because URL pathnames always start with "/" (e.g. "/index.html"), which is NOT
      // necessarily an absolute *filesystem* path.
      if (candidateAbs) {
        const allowedBases: string[] = [];
        // Always allow the protocol host's basePath (e.g. rendererDistPath) as a valid root.
        allowedBases.push(basePath);
        if (u.hostname === "apex-cache" && this.cachePath) {
          allowedBases.push(this.cachePath);
        }
        if (app) {
          allowedBases.push(...this.localUserDataBaseCandidates(app));
        }

        if (this.isUnderAnyBase(candidateAbs, allowedBases)) {
          // Crucially: never "re-base" an absolute path under basePath (that causes double-concat bugs).
          filePath = candidateAbs;
        } else {
          const rel = this.stripLeadingSlashes(decodedPathname);
          const resolved = this.resolveUnderBase(basePath, rel);
          if (!resolved) return null;
          filePath = resolved;
        }
      } else {
        const rel = this.stripLeadingSlashes(decodedPathname);
        const resolved = this.resolveUnderBase(basePath, rel);
        if (!resolved) return null;
        filePath = resolved;
      }

      if (u.hostname === "apex-cache") {
        const cacheBase = this.cachePath;
        if (cacheBase && filePath.startsWith(cacheBase)) {
          try {
            const relFromCache = path
              .relative(cacheBase, filePath)
              .replace(/\\/g, "/");
            await this.ensureProjectSymlinkForCache(
              relFromCache,
              filePath,
              folderUuid,
            );
          } catch {
            // Best-effort only; ignore failures
          }
        }
      }
    }

    try {
      if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
        return null;
      }
    } catch {
      return null;
    }

    return filePath;
  }

  private safeDecodeURIComponent(value: string): string {
    try {
      return decodeURIComponent(value);
    } catch {
      return value;
    }
  }

  /**
   * On Windows, URL pathnames for drive-letter paths commonly appear as:
   *   "/C:/Users/..."
   * or occasionally:
   *   "C:/Users/..."
   * UNC paths may appear as:
   *   "//server/share/..."
   *
   * Convert these into real Windows absolute paths (e.g. "C:\\Users\\..." or "\\\\server\\share\\...").
   */
  private coerceWindowsAbsolutePathFromUrlPathname(
    decodedPathname: string,
  ): string | null {
    if (process.platform !== "win32") return null;
    const n = String(decodedPathname || "").replace(/\\/g, "/");

    // Drive letter: "/C:/foo" or "C:/foo"
    const m = n.match(/^\/?([a-zA-Z]):\/(.*)$/);
    if (m) {
      const drive = m[1]!;
      const rest = m[2] ?? "";
      return path.win32.normalize(`${drive}:/${rest}`);
    }

    // UNC: "//server/share/..."
    if (n.startsWith("//") && n.length > 2) {
      return path.win32.normalize(`\\\\${n.slice(2)}`);
    }

    return null;
  }

  private remoteRelCandidatesFromPath(decodedPathname: string): string[] {
    const n = (decodedPathname || "").replace(/\\/g, "/");
    const candidates: string[] = [];

    const noLead = n.startsWith("/") ? n.slice(1) : n;
    if (noLead) candidates.push(noLead);

    // Heuristic: strip up to and including 'apex-diffusion/cache' segment if present.
    const marker = "apex-diffusion/cache/";
    const markerIdx = noLead.indexOf(marker);
    if (markerIdx !== -1) {
      const after = noLead.slice(markerIdx + marker.length);
      if (after) candidates.push(after.startsWith("/") ? after.slice(1) : after);
    }

    // Dedupe while preserving order
    return Array.from(new Set(candidates));
  }

  private async fetchExistsWithTimeout(
    url: string,
    timeoutMs: number,
  ): Promise<boolean> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      // Prefer a tiny range request to avoid downloading large files.
      const resp = await fetch(url, {
        method: "GET",
        headers: { Range: "bytes=0-0" },
        signal: controller.signal,
      });
      return resp.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  private async remoteApexCacheFileExists(relPath: string): Promise<boolean> {
    try {
      const url = `${this.backendUrl}/files?scope=apex-cache&path=${encodeURIComponent(relPath)}`;
      // Keep this probe quick; we only use it to detect remote mode.
      return await this.fetchExistsWithTimeout(url, 450);
    } catch {
      return false;
    }
  }

  private async maybeReprobeRemoteAndSwitch(
    relCandidates: string[],
  ): Promise<boolean> {
    // Debounce + dedupe concurrent probes.
    const now = Date.now();
    if (now - this.lastRemoteReprobeAtMs < 1500) {
      return this.loopbackAppearsRemote;
    }
    if (this.inflightRemoteReprobe) {
      return await this.inflightRemoteReprobe;
    }

    const probePromise = (async (): Promise<boolean> => {
      this.lastRemoteReprobeAtMs = Date.now();

      // First, check if the requested path exists via the backend file endpoint.
      let exists = false;
      for (const rel of relCandidates) {
        // eslint-disable-next-line no-await-in-loop
        exists = await this.remoteApexCacheFileExists(rel);
        if (exists) break;
      }
      if (!exists) return false;

      // If it exists remotely, switch to remote mode and initialize mirror/cache base.
      this.loopbackAppearsRemote = true;
      const app = this.electronApp;
      if (app) {
        try {
          await this.fetchCachePath(app);
        } catch {}
      }
      return true;
    })();

    this.inflightRemoteReprobe = probePromise;
    try {
      return await probePromise;
    } finally {
      this.inflightRemoteReprobe = null;
    }
  }

  private async initializeAsync(app: Electron.App): Promise<void> {
    try {
      await this.initBackendUrl(app);
    } catch {}
    try {
      await this.probeBackendLocality();
    } catch {}
    try {
      await this.fetchCachePath(app);
    } catch {}
  }

  private async fetchCachePath(app: Electron.App): Promise<void> {
    try {
      const response = await this.fetchBackendWithTimeout(
        `${this.backendUrl}/config/cache-path`,
        { method: "GET", headers: { "Content-Type": "application/json" } },
        "config/cache-path",
      );
      const ok = response.ok;
      let data: { cache_path?: string } = {};
      if (ok) {
        try {
          data = (await response.json()) as { cache_path?: string };
        } catch {}
      }

      // When remote, always create and use a local mirror path under userData
      if (this.loopbackAppearsRemote) {
        if (typeof data.cache_path === "string" && data.cache_path) {
          this.remoteCacheBasePath = data.cache_path;
        }
        const mirror = path.join(app.getPath("userData"), "apex-cache-remote");
        try {
          fs.mkdirSync(mirror, { recursive: true });
        } catch {}
        this.cachePath = mirror;
        return;
      }

      // Local backend: prefer backend-provided cache path; fallback to userData
      if (typeof data.cache_path === "string" && data.cache_path) {
        this.cachePath = data.cache_path;
      } else {
        const fallback = path.join(app.getPath("userData"), "apex-cache");
        try {
          fs.mkdirSync(fallback, { recursive: true });
        } catch {}
        this.cachePath = fallback;
      }
    } catch (error) {
      console.error("Failed to fetch cache path:", error);
      // Ensure we still have a usable cache path even on failure
      try {
        if (this.loopbackAppearsRemote) {
          // Remote: ensure mirror exists
          const fallback = path.join(
            app.getPath("userData"),
            "apex-cache-remote",
          );
          try {
            fs.mkdirSync(fallback, { recursive: true });
          } catch {}
          this.cachePath = fallback;
        } else {
          const fallback = path.join(app.getPath("userData"), "apex-cache");
          try {
            fs.mkdirSync(fallback, { recursive: true });
          } catch {}
          this.cachePath = fallback;
        }
      } catch {}
    }
  }

  private async onBackendUrlChanged(newUrl: string): Promise<void> {
    console.log(`[AppDirProtocol] Backend URL changed to ${newUrl}`);
    this.backendUrl = newUrl;
    // Reset state that depends on backend
    this.loopbackAppearsRemote = false;
    this.remoteCacheBasePath = null;

    // Re-probe environment
    await this.probeBackendLocality();
    if (this.electronApp) {
      await this.fetchCachePath(this.electronApp);
    }
  }

  private async initBackendUrl(app: Electron.App): Promise<void> {
    try {
      const settingsPath = path.join(
        app.getPath("userData"),
        "apex-settings.json",
      );
      if (fs.existsSync(settingsPath)) {
        const raw = await fs.promises.readFile(settingsPath, "utf-8");
        const j = JSON.parse(raw) as { backendUrl?: string };
        if (j.backendUrl && typeof j.backendUrl === "string") {
          // Validate
          new URL(j.backendUrl);
          this.backendUrl = j.backendUrl;
          return;
        }
      }
    } catch {}
    this.backendUrl = "http://127.0.0.1:8765";
  }

  private async probeBackendLocality(): Promise<void> {
    try {
      const u = new URL(this.backendUrl);
      const host = (u.hostname || "").toLowerCase();
      // Non-loopback hosts are always treated as remote
      if (!(host === "localhost" || host === "127.0.0.1" || host === "::1")) {
        this.loopbackAppearsRemote = true;
        return;
      }

      // For loopback hosts, rely solely on backend-reported hostname
      const resp = await this.fetchBackendWithTimeout(
        `${this.backendUrl}/config/hostname`,
        { method: "GET", headers: { "Content-Type": "application/json" } },
        "config/hostname",
      );
      if (!resp.ok) {
        // On failure, keep previous value (default false)
        return;
      }
      const data = await resp.json().catch(() => ({} as any));
      const remoteHostname =
        typeof data?.hostname === "string" ? data.hostname : "";
      const localHostname = os.hostname();
      this.loopbackAppearsRemote =
        !!remoteHostname && remoteHostname !== localHostname;
    } catch {
      // On failure to probe, keep previous value (default false)
    }
  }

  private remoteRelFromNormalized(normalizedPathname: string): string {
    const n = normalizedPathname.replace(/\\/g, "/");
    if (this.loopbackAppearsRemote) {
      // Known bases to strip
      const bases: string[] = [];
      // Also strip the local mirror path if the renderer accidentally sent it
      // (this prevents double-prefixing like: mirror + "/Users/.../mirror/...").
      if (this.cachePath) bases.push(this.cachePath);
      if (this.remoteCacheBasePath) bases.push(this.remoteCacheBasePath);
      const stripBaseOnce = (
        value: string,
        baseRaw: string,
      ): { changed: boolean; value: string } => {
        const base = this.stripTrailingSlashes(baseRaw.replace(/\\/g, "/"));
        if (!base) return { changed: false, value };
        const baseNoLead = base.startsWith("/") ? base.slice(1) : base;

        const tryStrip = (prefix: string): { changed: boolean; value: string } => {
          if (!prefix) return { changed: false, value };
          if (value === prefix) return { changed: true, value: "" };
          if (value.startsWith(prefix)) {
            const next = value.charAt(prefix.length);
            // Only strip on boundary (end or '/'), so we don't strip partial matches.
            if (next === "" || next === "/") {
              const rest = value.slice(prefix.length);
              return { changed: true, value: rest };
            }
          }
          return { changed: false, value };
        };

        const a = tryStrip(base);
        if (a.changed) return a;
        return tryStrip(baseNoLead);
      };

      // Normalize and attempt stripping repeatedly.
      // This is important because a common bug is doing:
      //   path.join(mirrorBase, absolutePathUnderMirrorBase)
      // which creates:
      //   mirrorBase + "/" + mirrorBase(no-leading-slash) + "/..."
      // We want to strip *all* repeated base prefixes until the remainder is truly relative.
      let cur = n;
      for (let i = 0; i < 6; i++) {
        let changed = false;
        for (const b of bases) {
          const res = stripBaseOnce(cur, b);
          if (res.changed) {
            cur = res.value;
            changed = true;
          }
        }
        if (!changed) break;
      }
      if (cur !== n) {
        return this.stripLeadingSlashes(cur);
      }

      // Heuristic: strip up to and including 'apex-diffusion/cache' segment if present
      const markerIdx = n.indexOf("apex-diffusion/cache/");
      if (markerIdx !== -1) {
        const rel = n.slice(markerIdx + "apex-diffusion/cache/".length);
        return rel.startsWith("/") ? rel.slice(1) : rel;
      }
      return n.startsWith("/") ? n.slice(1) : n;
    }
    return n.startsWith("/") ? n.slice(1) : n;
  }

  private async ensureLocalFromRemote(
    relPath: string,
    localPath: string,
  ): Promise<boolean> {
    try {
      // If the local file already exists and is non-empty, avoid refetching
      try {
        const stat = await fs.promises.stat(localPath);
        if (stat.isFile() && stat.size > 0) {
          return true;
        }
      } catch {}
      // Deduplicate concurrent downloads for the same relPath
      const existing = this.inflightDownloads.get(relPath);
      if (existing) {
        return await existing;
      }

      const downloadPromise = (async (): Promise<boolean> => {
        await fs.promises.mkdir(path.dirname(localPath), { recursive: true });
        const url = `${this.backendUrl}/files?scope=apex-cache&path=${encodeURIComponent(relPath)}`;
        // Retry a few times to handle race where backend publishes file moments later
        let resp: Response | null = null;
        const maxAttempts = 6;
        const baseDelayMs = 80;
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
          try {
            resp = await this.fetchBackendWithTimeout(
              url,
              { method: "GET" },
              "files(apex-cache)",
            );
            if (resp.ok && resp.body) break;
          } catch {
            // If a backend timeout flipped us out of remote-mode, stop retrying.
            if (!this.loopbackAppearsRemote && this.backendUrlIsLoopbackHost()) {
              return false;
            }
          }
          // If target got created during wait, bail out early as success
          try {
            const st = await fs.promises
              .stat(localPath)
              .catch(() => null as any);
            if (st && st.isFile() && st.size > 0) return true;
          } catch {}
          await new Promise((r) =>
            setTimeout(r, baseDelayMs * Math.pow(1.5, attempt)),
          );
        }
        if (!resp || !resp.ok || !resp.body) return false;
        const tmp = `${localPath}.part-${Date.now()}`;
        const out = fs.createWriteStream(tmp);
        try {
          const body = resp.body as any;
          const nodeReadable = (Readable as any).fromWeb
            ? (Readable as any).fromWeb(body)
            : null;
          if (!nodeReadable) {
            const buf = Buffer.from(await resp.arrayBuffer());
            await fs.promises.writeFile(tmp, buf);
          } else {
            await new Promise<void>((resolve, reject) => {
              nodeReadable.pipe(out);
              out.on("finish", () => resolve());
              out.on("error", reject);
            });
          }
          // If the target already exists (written by another concurrent download), treat as success
          try {
            await fs.promises.rename(tmp, localPath);
          } catch (err: any) {
            try {
              const st = await fs.promises
                .stat(localPath)
                .catch(() => null as any);
              if (st && st.isFile() && st.size > 0) {
                // Target exists and is valid; cleanup tmp and consider success
                try {
                  await fs.promises.unlink(tmp);
                } catch {}
                return true;
              }
              // Otherwise, attempt to overwrite by unlinking and renaming again
              try {
                await fs.promises.unlink(localPath);
              } catch {}
              await fs.promises.rename(tmp, localPath);
            } catch (e) {
              // Cleanup tmp on failure
              try {
                await fs.promises.unlink(tmp);
              } catch {}
              return false;
            }
          }
          return true;
        } finally {
          try {
            out.close();
          } catch {}
        }
      })();

      this.inflightDownloads.set(relPath, downloadPromise);
      try {
        return await downloadPromise;
      } finally {
        this.inflightDownloads.delete(relPath);
      }
    } catch {
      return false;
    }
  }

  /**
   * Ensure that each apex-cache download is also exposed under a per-project
   * folder UUID view:
   *
   *   apex-cache/<folder-uuid>/...
   *
   * We infer the folder UUID from engine_results paths of the form:
   *   engine_results/<folder-uuid>/...
   *
   * and create a symlink rooted at this.cachePath so that the app://apex-cache
   * protocol can also serve files from app://apex-cache/<folder-uuid>/...
   */
  private async ensureProjectSymlinkForCache(
    relPath: string,
    localPath: string,
    folderUuid?: string,
  ): Promise<void> {
    try {
      if (!this.cachePath) return;
      if (!folderUuid) return;

      const normalizedRel = relPath.replace(/\\/g, "/");
      const enginePrefix = "engine_results/";
      if (!normalizedRel.startsWith(enginePrefix)) return;

      // Strip the leading "engine_results/" so we get "<job-id>/..."
      const afterEngine = normalizedRel.slice(enginePrefix.length);
      if (!afterEngine || afterEngine.startsWith("/")) return;

      // If the path is already scoped under this folderUuid, nothing to do
      if (afterEngine.startsWith(`${folderUuid}/`)) return;

      // We want:
      //   cachePath/engine_results/<folderUuid>/<job-id>/result_*.*
      const symlinkPath = path.join(
        this.cachePath,
        enginePrefix,
        folderUuid,
        afterEngine,
      );

      // Ensure parent directory exists
      await fs.promises.mkdir(path.dirname(symlinkPath), { recursive: true });

      // If a symlink already exists, keep it when it points at the same target
      try {
        const st = await fs.promises.lstat(symlinkPath);
        if (st.isSymbolicLink()) {
          try {
            const existingTarget = await fs.promises.readlink(symlinkPath);
            if (existingTarget === localPath) {
              return;
            }
          } catch {
            // If we can't read the link, fall through and recreate it
          }
        }
        // Remove any stale file/symlink so we can recreate it
        try {
          await fs.promises.unlink(symlinkPath);
        } catch {
          // ignore unlink failures
        }
      } catch {
        // Path does not exist yet; that's fine
      }

      try {
        await fs.promises.symlink(localPath, symlinkPath);
      } catch {
        // Best-effort only; ignore failures
      }
    } catch {
      // Never let symlink setup break serving the main file
    }
  }
}

export function appDirProtocol(): AppDirProtocol {
  return new AppDirProtocol();
}
