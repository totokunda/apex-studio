import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { getSettingsModule } from "./SettingsModule.js";
import { protocol, ipcMain } from "electron";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { Readable } from "node:stream";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import fsp from "node:fs/promises";
import mime from "mime-types";


function parseRange(rangeHeader: string | null, size: number) {
  if (!rangeHeader) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader.trim());
  if (!m) return null;

  const startStr = m[1];
  const endStr = m[2];

  // bytes=-500 (last 500 bytes)
  if (startStr === "" && endStr !== "") {
    const suffixLen = Number(endStr);
    if (!Number.isFinite(suffixLen) || suffixLen <= 0) return null;
    const start = Math.max(0, size - suffixLen);
    const end = size - 1;
    return { start, end };
  }

  const start = Number(startStr);
  let end = endStr === "" ? size - 1 : Number(endStr);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start < 0 || end < start || start >= size) return null;

  end = Math.min(end, size - 1);
  return { start, end };
}



function hopHeaders(reqHeaders: Headers) {
  // forward range for immediate playback / seeking from upstream
  const out: Record<string, string> = {};
  const range = reqHeaders.get("range");
  if (range) out["range"] = range;
  const ifNoneMatch = reqHeaders.get("if-none-match");
  if (ifNoneMatch) out["if-none-match"] = ifNoneMatch;
  const ifModifiedSince = reqHeaders.get("if-modified-since");
  if (ifModifiedSince) out["if-modified-since"] = ifModifiedSince;
  return out;
}


function parseContentLength(headers: Headers): number | null {
  const raw = headers.get("content-length");
  if (!raw) return null;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return null;
  return Math.floor(n);
}

function parseExpectedBodyBytes(headers: Headers, status: number): number | null {
  // For stream integrity enforcement, only return a value when we can know
  // exactly how many bytes should arrive.
  //
  // - 200: use Content-Length
  // - 206: use Content-Length when present, else compute from Content-Range
  if (status === 200) {
    return parseContentLength(headers);
  }
  if (status === 206) {
    const cl = parseContentLength(headers);
    if (cl !== null) return cl;
    const cr = headers.get("content-range");
    if (!cr) return null;
    const m = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(cr.trim());
    if (!m) return null;
    const start = Number(m[1]);
    const end = Number(m[2]);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
    return end - start + 1;
  }
  return null;
}

function enforceExpectedByteLength(
  upstream: ReadableStream<Uint8Array>,
  expectedBytes: number,
): ReadableStream<Uint8Array> {
  let seen = 0;
  const reader = upstream.getReader();
  let finalized = false;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (finalized) return;
      let res: ReadableStreamReadResult<Uint8Array>;
      try {
        res = await reader.read();
      } catch (e) {
        // If the consumer cancelled while a pull was in-flight, undici may surface
        // "Invalid state: ReadableStream is already closed". Treat this as terminal.
        finalized = true;
        try {
          controller.error(e instanceof Error ? e : new Error(String(e)));
        } catch {
          // ignore
        }
        return;
      }

      const { done, value } = res;
      if (done) {
        if (seen !== expectedBytes) {
          finalized = true;
          try {
            controller.error(
              new Error(
                `Incomplete proxied stream: expected ${expectedBytes} bytes, got ${seen}`,
              ),
            );
          } catch {
            // ignore
          }
          return;
        }
        finalized = true;
        try {
          controller.close();
        } catch {
          // ignore (can happen if consumer cancelled)
        }
        return;
      }

      if (value) {
        seen += value.byteLength;
        try {
          controller.enqueue(value);
        } catch {
          // If the consumer cancelled while we were enqueuing, don't crash the process.
          finalized = true;
          try {
            void reader.cancel();
          } catch {
            // ignore
          }
        }
      }
    },
    cancel(reason) {
      finalized = true;
      try {
        void reader.cancel(reason);
      } catch {
        // ignore
      }
    },
  });
}

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

function isCachedServerMediaPath(absPath: string, userDataDir: string): boolean {
  // Only enforce cache integrity markers for the on-disk cache we control:
  //   <userData>/media/<folderUuid>/server/(generations|processors)/...
  try {
    const normalized = path.resolve(absPath);
    const base = path.resolve(path.join(userDataDir, "media"));
    if (!normalized.startsWith(base + path.sep) && normalized !== base) return false;
    return normalized.includes(`${path.sep}server${path.sep}`);
  } catch {
    return false;
  }
}

interface ServerDetails {
  folderUuid: string | null;
  folderName: string | null;
  fileName: string | null;
  type: "engine_results" | "preprocessor_results" | "postprocessor_results";
  localType: "generations" | "processors"
}


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


// For cover paths we need to convert the path to a local file path
const isCoverPath = (filePath: string) => {
  return filePath.startsWith("/projects-json/covers/");
};

const getCoverPath = (filePath: string, userDataDir: string) => {
  return path.join(userDataDir, filePath.slice(1));
};

const isServerPath = (filePath: string) => {
  // NOTE: `filePath` here is a URL pathname (decoded + normalized to POSIX separators).
  // It may be an absolute filesystem path like `/C:/.../cache/engine_results/...` or a
  // logical route like `/engine_results/<folderUuid>/<folder>/<file>`.
  //
  // Avoid substring matches (false positives like "my_engine_results_notes.png") by
  // matching path *segments*.
  const segs = String(filePath || "")
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean);

  // Skip Windows drive segment ("C:") when checking the first logical segment.
  const firstLogical = segs.length > 0 && /^[a-zA-Z]:$/.test(segs[0]!) ? segs[1] : segs[0];

  // Canonical server markers can appear anywhere in an absolute path.
  if (segs.includes("engine_results")) return true;
  if (segs.includes("preprocessor_results")) return true;
  if (segs.includes("postprocessor_results")) return true;

  // Legacy/logical aliases only count when they are the first logical segment.
  if (firstLogical === "generations") return true;
  if (firstLogical === "processors") return true;

  return false;
};


async function exists(p: string) {
  try {
    await fsp.access(p);
    return true;
  } catch {
    return false;
  }
}

class AppDirProtocol implements AppModule {
  private electronApp: Electron.App | null = null;
  private backendUrl: string = "http://127.0.0.1:8765";
  private activeProjectId: string | null = null;
  private activeFolderUuid: string | null = null;
  private rendererDistPath: string | null = null;
  // Cache in-flight remote file saves so we don't write the same file multiple times concurrently.
  private inflightRemoteFileSaves: Map<string, Promise<void>> = new Map();

  private stripLeadingSlashes(p: string): string {
    return (p || "").replace(/^\/+/, "");
  }

  /**
   * Resolve `relOrAbs` under `basePath` and reject traversal.
   * Returns an absolute path under `basePath`, or null if it escapes.
   */
  private resolveUnderBase(basePath: string, relOrAbs: string): string | null {
    const baseAbs = path.resolve(basePath);
    const abs = path.resolve(baseAbs, relOrAbs);
    const rel = path.relative(baseAbs, abs);
    if (rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel))) {
      return abs;
    }
    return null;
  }

  private tryResolveRendererDistPath(app: Electron.App): string | null {
    const candidates: string[] = [];
    try {
      const require = createRequire(import.meta.url);
      const indexHtml = require.resolve("@app/renderer/dist/index.html");
      candidates.push(path.dirname(indexHtml));
    } catch {
      // ignore
    }
    try {
      const appPath = app.getAppPath();
      candidates.push(path.join(appPath, "node_modules", "@app", "renderer", "dist"));
      candidates.push(
        path.join(
          process.resourcesPath,
          "app.asar.unpacked",
          "node_modules",
          "@app",
          "renderer",
          "dist",
        ),
      );
      candidates.push(
        path.join(
          process.resourcesPath,
          "app",
          "node_modules",
          "@app",
          "renderer",
          "dist",
        ),
      );
      candidates.push(path.join(process.resourcesPath, "node_modules", "@app", "renderer", "dist"));
    } catch {
      // ignore
    }

    for (const cand of candidates) {
      try {
        const idx = path.join(cand, "index.html");
        if (fs.existsSync(idx)) {
          return cand;
        }
      } catch {
        // ignore
      }
    }

    console.warn(
      "[AppDirProtocol] Could not resolve renderer dist path; app://renderer/* will 404. Candidates:",
      candidates,
    );
    return null;
  }

  private async returnResponseFromRenderer(request: Request): Promise<Response> {
    const u = new URL(request.url);
    const base = this.rendererDistPath;
    if (!base) {
      return new Response(null, { status: 404 });
    }

    let rel = this.stripLeadingSlashes(decodeURIComponent(u.pathname || ""));
    if (rel === "" || rel === "/") rel = "index.html";
    rel = path.posix.normalize(rel).replace(/\\/g, "/");
    rel = this.stripLeadingSlashes(rel);

    // If a directory is requested, serve its index.html.
    if (rel.endsWith("/")) rel = `${rel}index.html`;

    let filePath = this.resolveUnderBase(base, rel);
    if (!filePath) {
      return new Response(null, { status: 404 });
    }

    // SPA fallback: for routes without an extension, fall back to index.html.
    const hasExt = Boolean(path.extname(rel));
    try {
      const st = await fsp.stat(filePath);
      if (st.isDirectory()) {
        const idx = this.resolveUnderBase(base, path.join(rel, "index.html"));
        if (!idx) return new Response(null, { status: 404 });
        filePath = idx;
      }
    } catch {
      if (!hasExt) {
        const idx = this.resolveUnderBase(base, "index.html");
        if (!idx) return new Response(null, { status: 404 });
        filePath = idx;
      } else {
        return new Response(null, { status: 404 });
      }
    }

    let stat: fs.Stats;
    try {
      stat = await fsp.stat(filePath);
      if (!stat.isFile()) return new Response(null, { status: 404 });
    } catch {
      return new Response(null, { status: 404 });
    }

    const ct = mime.lookup(filePath) || "application/octet-stream";
    const headers = new Headers({
      "Content-Type": String(ct),
      "Content-Length": String(stat.size),
      "Cache-Control": "public, max-age=31536000, immutable",
    });

    if (request.method === "HEAD") {
      return new Response(null, { status: 200, headers });
    }

    return new Response(nodeStreamToWebStream(fs.createReadStream(filePath)), { status: 200, headers });
  }

  private getServerMetaPath(savePath: string): string {
    return `${savePath}.meta.json`;
  }

  private async hasVerifiedServerCache(savePath: string): Promise<boolean> {
    const metaPath = this.getServerMetaPath(savePath);
    try {
      const [stat, raw] = await Promise.all([
        fsp.stat(savePath),
        fsp.readFile(metaPath, "utf8"),
      ]);
      const meta = JSON.parse(raw) as { size?: number };
      return typeof meta?.size === "number" && meta.size === stat.size;
    } catch {
      return false;
    }
  }

  private getServerSavePath(serverDetails: ServerDetails): string | null {
    const userDataDir = this.electronApp?.getPath("userData") ?? "";
    const folderUuid = serverDetails.folderUuid ?? "";
    const folderName = serverDetails.folderName ?? "";
    const fileName = serverDetails.fileName ?? "";
    if (!userDataDir || !folderUuid || !folderName || !fileName) return null;

    return path.join(
      userDataDir,
      "media",
      folderUuid,
      "server",
      serverDetails.localType,
      folderName,
      fileName,
    );
  }

  private queueSaveRemoteFileToLocalFile(response: Response, serverDetails: ServerDetails): void {
    const savePath = this.getServerSavePath(serverDetails);
    if (!savePath) return;
    if (this.inflightRemoteFileSaves.has(savePath)) return;

    const p = this.saveRemoteFileToLocalFile(response, serverDetails).finally(() => {
      this.inflightRemoteFileSaves.delete(savePath);
    });
    this.inflightRemoteFileSaves.set(savePath, p);
  }

  private corsHeadersFor(request: Request): Record<string, string> {
    // Mirror the Origin if present so the renderer can fetch app:// resources in both dev (http://localhost)
    // and prod (app://renderer) contexts.
    const origin = request.headers.get("origin") || "*";
    return {
      "Access-Control-Allow-Origin": origin,
      Vary: "Origin",
      "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Range",
      "Cross-Origin-Resource-Policy": "cross-origin",
    };
  }

  async getActiveFolderUuid(): Promise<string | null> {
    if (!this.activeProjectId) return null;
    let projectFilesPath = path.join(this.electronApp?.getPath("userData") ?? "", "projects-json", `project-${this.activeProjectId}.json`);
    try {
      let projectFiles = await fsp.readFile(projectFilesPath, "utf8");
      let projectFilesJson = JSON.parse(projectFiles);
      return projectFilesJson?.meta?.id ?? null;  
    } catch {
      return null;
    }
  }

  private withCors(request: Request, response: Response): Response {
    const headers = new Headers(response.headers);
    const cors = this.corsHeadersFor(request);
    for (const [k, v] of Object.entries(cors)) headers.set(k, v);

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  }

  private async onActiveProjectIdChanged(newProjectId: string | null): Promise<void> {
    this.activeProjectId = newProjectId;
    this.activeFolderUuid = await this.getActiveFolderUuid();
  }

  private async onBackendUrlChanged(newUrl: string): Promise<void> {
    this.backendUrl = newUrl;
  }

  private async fetchRemoteFileIfExists(
    serverDetails: ServerDetails,
    requestHeaders: Headers,
    method: string = "GET",
  ): Promise<Response> {
    let url = `${this.backendUrl}/files`;
    if (serverDetails.type === "engine_results") {
      url += `/engine_results/${serverDetails.folderUuid}/${serverDetails.folderName}/${serverDetails.fileName}`;
    } else {
      url += `/${serverDetails.type}/${serverDetails.folderName}/${serverDetails.fileName}`;
    }
    const response = await fetch(url, {
      method: method === "HEAD" ? "HEAD" : "GET",
      headers: hopHeaders(requestHeaders),
    });
    return response;

  }

  private async saveLocalFileToServer(filePath: string, savePath: string): Promise<void> {
    // Copy a local file into our on-disk server cache and write the
    // corresponding verified meta marker so it won't be GC'd as "invalid cache".
    if (filePath === savePath) return;

    await fsp.mkdir(path.dirname(savePath), { recursive: true });

    // Copy the file to the cache location.
    await fsp.copyFile(filePath, savePath);

    // Write/refresh a verified meta marker (best-effort, atomic).
    try {
      const st = await fsp.stat(savePath);
      const metaPath = this.getServerMetaPath(savePath);
      const metaTmp = `${metaPath}.part`;
      const meta = {
        size: st.size,
        fetchedAtMs: Date.now(),
        source: "local",
      };
      await fsp.writeFile(metaTmp, JSON.stringify(meta), "utf8");
      try {
        await fsp.rm(metaPath, { force: true });
      } catch {
        // ignore
      }
      await fsp.rename(metaTmp, metaPath);
    } catch {
      // Best-effort only; never fail the main request due to meta creation issues.
    }
  }

  private async returnResponseFromLocalFile(request: Request): Promise<Response> {

      const urlObj = new URL(request.url);
      const userDataDir = this.electronApp?.getPath("userData") ?? "";
      let filePath = decodeURIComponent(urlObj.pathname);
      filePath = path.posix.normalize(filePath).replace(/\\/g, "/");
      let folderUuidFromUrl = urlObj.searchParams.get("folderUuid");
      if (!folderUuidFromUrl) {
        folderUuidFromUrl = this.activeFolderUuid;
      }

      if (isCoverPath(filePath)) {
        filePath = getCoverPath(filePath, userDataDir);
      }

      const serverDetails = this.parseServerDetails(filePath, folderUuidFromUrl);
      if (serverDetails) {
        if (!(await exists(filePath))) {
          if (
            serverDetails.folderUuid &&
            serverDetails.folderName &&
            serverDetails.fileName
          ) {
            filePath = path.join(
              userDataDir,
              "media",
              serverDetails.folderUuid,
              "server",
              serverDetails.localType,
              serverDetails.folderName,
              serverDetails.fileName,
            );
          } else {
            return new Response(null, { status: 404 });
          }
        }

        // Backfill the on-disk server cache from local files (generations + processors).
        const savePath = this.getServerSavePath(serverDetails);
        if (savePath && !(await exists(savePath))) {
          try {
            // Copy the file to the save path (+ write verified meta marker).
            void this.saveLocalFileToServer(filePath, savePath).catch((e) => {
              console.error("Error saving local file to server", e);
            });
          } catch (e) {
            console.error("Error saving local file to server", e);
          }
        }
      }

      const stat = await fsp.stat(filePath);
      if (!stat.isFile()) return new Response(null, { status: 404 });
      const size = stat.size;

      // For cached server media, require a verified marker so we never serve
      // a file that might have been truncated by a previous network hiccup.
      if (isCachedServerMediaPath(filePath, userDataDir)) {
        const metaPath = this.getServerMetaPath(filePath);
        let meta: { size?: number } | null = null;
        try {
          const raw = await fsp.readFile(metaPath, "utf8");
          meta = JSON.parse(raw) as { size?: number };
        } catch {
          meta = null;
        }
        if (!meta || typeof meta.size !== "number" || meta.size !== size) {
          // Best-effort cleanup: treat as cache miss and refetch.
          try {
            await fsp.rm(filePath, { force: true });
          } catch {
            // ignore
          }
          try {
            await fsp.rm(metaPath, { force: true });
          } catch {
            // ignore
          }
          throw new Error(`Cached server file missing/invalid meta: ${filePath}`);
        }
      }

      const ct = mime.lookup(filePath) || "application/octet-stream";
      const range = parseRange(request.headers.get("range"), size);

      if (!range) {
        const headers = new Headers({
          "Content-Type": String(ct),
          "Content-Length": String(size),
          "Accept-Ranges": "bytes",
          "Cache-Control": "public, max-age=31536000, immutable",
        });
        // HEAD must not include a body; serving a body here can lead to mid-stream
        // cancellations that trigger undici stream state errors.
        if (request.method === "HEAD") {
          return new Response(null, { status: 200, headers });
        }
        return new Response(nodeStreamToWebStream(fs.createReadStream(filePath)), { status: 200, headers });
      }
    
      const { start, end } = range;
      const chunkSize = end - start + 1;
    
      const headers = new Headers({
        "Content-Type": String(ct),
        "Content-Length": String(chunkSize),
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=31536000, immutable",
      });

      if (request.method === "HEAD") {
        return new Response(null, { status: 206, headers });
      }
      return new Response(nodeStreamToWebStream(fs.createReadStream(filePath, { start, end })), { status: 206, headers });
   
  }


  private async saveRemoteFileToLocalFile(response: Response, serverDetails: ServerDetails): Promise<void> {

    if (!response.ok) return;
    if (!response.body) return;
    // Only persist full responses. Range responses must not be published as "complete" files.
    if (response.status !== 200) return;

    const savePath = this.getServerSavePath(serverDetails);
    if (!savePath) return;
    const partPath = savePath + ".part";
    const metaPath = this.getServerMetaPath(savePath);

    // If we cannot verify completeness, do not cache to disk (better a cache miss than
    // persisting a potentially truncated file).
    const expectedBytes = parseExpectedBodyBytes(response.headers, response.status);
    if (expectedBytes === null) return;

    await fsp.mkdir(path.dirname(partPath), { recursive: true });

    const ws = fs.createWriteStream(partPath, { flags: "w" });
    ws.on("error", () => {
      // Best-effort: errors are handled by pipeline's rejection.
    });

    let written = 0;
    const counter = new Transform({
      transform(chunk: any, _enc, cb) {
        try {
          if (chunk) {
            if (Buffer.isBuffer(chunk)) {
              written += chunk.byteLength;
            } else if (chunk instanceof Uint8Array) {
              written += chunk.byteLength;
            } else if (typeof chunk?.length === "number") {
              written += Number(chunk.length) || 0;
            }
          }
        } catch {
          // ignore
        }
        cb(null, chunk);
      },
    });

    try {
      const body = Readable.fromWeb(response.body as any);
      await pipeline(body, counter, ws);

      if (written !== expectedBytes) {
        throw new Error(
          `Incomplete download for ${savePath}: expected ${expectedBytes} bytes, got ${written}`,
        );
      }

      // Windows rename fails if destination exists.
      try {
        await fsp.rm(savePath, { force: true });
      } catch (e) {

        // ignore
      }
      await fsp.rename(partPath, savePath);

      // Write a verified marker last (best-effort atomic).
      const metaTmp = `${metaPath}.part`;
      const meta = {
        size: expectedBytes,
        fetchedAtMs: Date.now(),
        etag: response.headers.get("etag") ?? undefined,
        lastModified: response.headers.get("last-modified") ?? undefined,
      };
      try {
        await fsp.writeFile(metaTmp, JSON.stringify(meta), "utf8");
        try {
          await fsp.rm(metaPath, { force: true });
        } catch {
          // ignore
        }
        await fsp.rename(metaTmp, metaPath);
      } catch {
        // If meta write fails, treat the cache entry as invalid and remove it.
        try {
          await fsp.rm(savePath, { force: true });
        } catch {
          // ignore
        }
        try {
          await fsp.rm(metaTmp, { force: true });
        } catch {
          // ignore
        }
      }

   
    } catch (e) {

      // Ensure partial downloads don't accumulate.
      try {
        ws.destroy();
      } catch (e) {

        // ignore
      }
      try {
        await fsp.rm(partPath, { force: true });
      } catch {
        // ignore
      }
      try {
        await fsp.rm(metaPath, { force: true });
      } catch {
        // ignore
      }
    }
  }

  private parseServerDetails(filePath: string, folderUuid?: string | null): ServerDetails | null {
    if (!isServerPath(filePath)) return null;

    const segs = String(filePath || "")
      .replace(/\\/g, "/")
      .split("/")
      .filter(Boolean);

    const startIdx = segs.length > 0 && /^[a-zA-Z]:$/.test(segs[0]!) ? 1 : 0;
    const firstLogical = segs[startIdx] ?? null;

    const getIdx = (needle: string): number => {
      // Prefer the last occurrence in case "cache/engine_results/..." appears in longer paths.
      for (let i = segs.length - 1; i >= 0; i--) {
        if (segs[i] === needle) return i;
      }
      return -1;
    };

    const fromTriplet = (idx: number): { folderUuid: string | null; folderName: string | null; fileName: string | null } => {
      const fu = segs[idx + 1] ?? null;
      const fn = segs[idx + 2] ?? null;
      const file = segs[idx + 3] ?? null;
      return { folderUuid: fu, folderName: fn, fileName: file };
    };

    const fromPair = (idx: number): { folderName: string | null; fileName: string | null } => {
      const fn = segs[idx + 1] ?? null;
      const file = segs[idx + 2] ?? null;
      return { folderName: fn, fileName: file };
    };

    // 1) Canonical markers (can appear anywhere in absolute paths).
    const engIdx = getIdx("engine_results");
    if (engIdx >= 0) {
      const { folderUuid: fu, folderName, fileName } = fromTriplet(engIdx);
      const finalFolderUuid = fu ?? folderUuid ?? null;
      if (!finalFolderUuid || !folderName || !fileName) return null;
      return {
        folderUuid: finalFolderUuid,
        folderName,
        fileName,
        type: "engine_results",
        localType: "generations",
      };
    }

    const preIdx = getIdx("preprocessor_results");
    if (preIdx >= 0) {
      const { folderName, fileName } = fromPair(preIdx);
      if (!folderName || !fileName) return null;
      return {
        folderUuid: folderUuid ?? null,
        folderName,
        fileName,
        type: "preprocessor_results",
        localType: "processors",
      };
    }

    const postIdx = getIdx("postprocessor_results");
    if (postIdx >= 0) {
      const { folderName, fileName } = fromPair(postIdx);
      if (!folderName || !fileName) return null;
      return {
        folderUuid: folderUuid ?? null,
        folderName,
        fileName,
        type: "postprocessor_results",
        localType: "processors",
      };
    }

    // 2) Legacy/logical aliases (only when they're the first logical segment).
    if (firstLogical === "generations") {
      const fu = segs[startIdx + 1] ?? null;
      const folderName = segs[startIdx + 2] ?? null;
      const fileName = segs[startIdx + 3] ?? null;
      const finalFolderUuid = fu ?? folderUuid ?? null;
      if (!finalFolderUuid || !folderName || !fileName) return null;
      return {
        folderUuid: finalFolderUuid,
        folderName,
        fileName,
        type: "engine_results",
        localType: "generations",
      };
    }
    if (firstLogical === "processors") {
      // We don't have enough information to distinguish pre vs post from this alias.
      // Default to preprocessor_results (historically the more common case).
      const folderName = segs[startIdx + 1] ?? null;
      const fileName = segs[startIdx + 2] ?? null;
      if (!folderName || !fileName) return null;
      return {
        folderUuid: folderUuid ?? null,
        folderName,
        fileName,
        type: "preprocessor_results",
        localType: "processors",
      };
    }

    return null;
  }

  private async fetchRemoteFile(request: Request): Promise<Response> {


    const urlObj = new URL(request.url);
    const filePath = decodeURIComponent(urlObj.pathname);
    let folderUuid = urlObj.searchParams.get("folderUuid");
    if (!folderUuid) {
      folderUuid = this.activeFolderUuid;
    }
    const serverDetails = this.parseServerDetails(filePath, folderUuid);
    if (!serverDetails) {
      return new Response(null, { status: 404 });
    }
    const response = await this.fetchRemoteFileIfExists(
      serverDetails,
      request.headers,
      request.method,
    );

    // Background caching:
    // - Never persist Range/206 responses as complete files.
    // - Avoid Response.clone()/tee to prevent undici stream state crashes when the renderer cancels mid-stream.
    // - Instead, kick off a separate local fetch (localhost) when we don't already have a verified cache entry.
    if (request.method === "GET") {
      const savePath = this.getServerSavePath(serverDetails);
      if (savePath) {
        const shouldFetchForCache = !(await this.hasVerifiedServerCache(savePath));
        const isRangeReq = Boolean(request.headers.get("range"));
        if (
          shouldFetchForCache &&
          (isRangeReq || (response.ok && response.status === 200))
        ) {
          void (async () => {
            try {
              const full = await this.fetchRemoteFileIfExists(
                serverDetails,
                new Headers(),
                "GET",
              );
              this.queueSaveRemoteFileToLocalFile(full, serverDetails);
            } catch {
              // ignore
            }
          })();
        }
      }
    }

    if (request.method === "HEAD") {
      return new Response(null, {
        status: response?.status ?? 404,
        headers: response?.headers ?? new Headers(),
      });
    }

    const upstream = response?.body ? (response.body as ReadableStream<Uint8Array>) : null;
    if (!upstream) {
      return new Response(null, {
        status: response?.status ?? 404,
        headers: response?.headers ?? new Headers(),
      });
    }

    // If we can determine the expected body size, enforce it so truncated upstream streams
    // surface as an error to the renderer (instead of silently "completing").
    const expected = parseExpectedBodyBytes(response.headers, response.status);
    const wrapped = expected !== null ? enforceExpectedByteLength(upstream, expected) : upstream;
    return new Response(wrapped as any, { status: response?.status ?? 404, headers: response?.headers ?? new Headers() });
  
}
  
  async enable({ app }: ModuleContext): Promise<void> {
    await app.whenReady();
    this.electronApp = app;
    // Resolve renderer dist once so `app://renderer/*` works in packaged builds.
    this.rendererDistPath = this.tryResolveRendererDistPath(app);
    const settings = getSettingsModule();
    this.backendUrl = settings.getBackendUrl();
    
    this.activeProjectId = settings.getActiveProjectId();

    try {
      this.activeFolderUuid = await this.getActiveFolderUuid();
    } catch {
      this.activeFolderUuid = null;
    }

    settings.on("backend-url-changed", (newUrl: string) => {
      void this.onBackendUrlChanged(newUrl);
    });

    settings.on("active-project-id-changed", (newProjectId: string | null) => {
      void this.onActiveProjectIdChanged(newProjectId);
    });


    ipcMain.handle("appdir:resolve-path", async (event, filePath: string) => {
      let url = new URL(filePath);
      let pathName = url.pathname;
      filePath = decodeURIComponent(pathName);
      filePath = path.posix.normalize(filePath).replace(/\\/g, "/");

      // Resolve packaged renderer assets to a real local file path.
      if (url.hostname === "renderer") {
        const base = this.rendererDistPath;
        if (!base) return null;
        let rel = this.stripLeadingSlashes(filePath);
        if (rel === "" || rel === "/") rel = "index.html";
        rel = path.posix.normalize(rel).replace(/\\/g, "/");
        rel = this.stripLeadingSlashes(rel);
        const resolved = this.resolveUnderBase(base, rel);
        if (!resolved) return null;
        try {
          const st = await fsp.stat(resolved);
          if (st.isFile()) return resolved;
        } catch {
          // ignore
        }
        return null;
      }

      const userDataDir = this.electronApp?.getPath("userData") ?? "";
      if (isCoverPath(filePath)) {
        filePath = getCoverPath(filePath, userDataDir);
      }
      const folderUuidFromUrl = url.searchParams.get("folderUuid") ?? this.activeFolderUuid;
      const serverDetails = this.parseServerDetails(filePath, folderUuidFromUrl);
      if (serverDetails) {
        // check if the file exists locally 
        if (!await exists(filePath)) {
          if (serverDetails.folderUuid && serverDetails.folderName && serverDetails.fileName) {
            filePath = path.join(
              userDataDir,
              "media",
              serverDetails.folderUuid,
              "server",
              serverDetails.localType,
              serverDetails.folderName,
              serverDetails.fileName,
            );
          }
          if (!await exists(filePath)) {
            const response = await this.fetchRemoteFileIfExists(
              serverDetails,
              new Headers(),
              "GET",
            );
            if (response.ok) {
              await this.saveRemoteFileToLocalFile(response, serverDetails);
              // file path should now exist
            }
          }
        }
      }

      return filePath;
    });
    

    protocol.handle("app", async (request) => {
       // CORS/preflight handling for renderer -> app:// fetches

       if (request.method === "OPTIONS") {
         return new Response(null, { status: 204, headers: this.corsHeadersFor(request) });
       }
       if (request.method !== "GET" && request.method !== "HEAD") {
         const headers = this.corsHeadersFor(request);
         headers["Allow"] = "GET, HEAD, OPTIONS";
         return new Response(null, { status: 405, headers });
       }
       try {
        const u = new URL(request.url);
        if (u.hostname === "renderer") {
          const r = await this.returnResponseFromRenderer(request);
          return this.withCors(request, r);
        }
       } catch {
        // ignore URL parse errors; fall back below
       }
       try {
        const r = await this.returnResponseFromLocalFile(request);
        return this.withCors(request, r);
       } catch (e) {
        try { 
          const r = await this.fetchRemoteFile(request);
          return this.withCors(request, r);
        } catch (e) {

          return new Response(null, { status: 404, headers: this.corsHeadersFor(request) });
        }
       }
    });

  }
}

export function appDirProtocol(): AppDirProtocol {
  return new AppDirProtocol();
}
