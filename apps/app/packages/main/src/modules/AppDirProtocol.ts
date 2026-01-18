import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { getSettingsModule } from "./SettingsModule.js";
import { protocol } from "electron";
import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
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
  return filePath.includes("engine_results") || filePath.includes("generations") || filePath.includes("processors") || filePath.includes("postprocessor_results") || filePath.includes("preprocessor_results")
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
  // Cache in-flight remote file saves so we don't write the same file multiple times concurrently.
  private inflightRemoteFileSaves: Map<string, Promise<void>> = new Map();

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
    let projectFilesPath = path.join(this.electronApp?.getPath("userData") ?? "", "projects-json", `project-${this.activeProjectId}.json`);
    let projectFiles = await fsp.readFile(projectFilesPath, "utf8");
    let projectFilesJson = JSON.parse(projectFiles);
    return projectFilesJson?.meta?.id ?? null;  
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

  private async fetchRemoteFileIfExists(serverDetails: ServerDetails, requestHeaders: Headers): Promise<Response> {
    let url = `${this.backendUrl}/files`;
    if (serverDetails.type === "engine_results") {
      url += `/engine_results/${serverDetails.folderUuid}/${serverDetails.folderName}/${serverDetails.fileName}`;
    } else {
      url += `/${serverDetails.type}/${serverDetails.folderName}/${serverDetails.fileName}`;
    }
    const response = await fetch(url, { headers: hopHeaders(requestHeaders) });
    return response;

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

  
      if (isServerPath(filePath)) {
        // check if the file exists locally 
        if (!await exists(filePath)) {
          const { folderUuid, folderName, fileName, localType  } = this.parseServerDetails(filePath, folderUuidFromUrl);
          if (folderUuid && folderName && fileName) {
            filePath = path.join(userDataDir, "media", folderUuid, "server", localType, folderName, fileName);
          } else {
            return new Response(null, { status: 404 });
          }
        }
      }

      const stat = await fsp.stat(filePath);
      const size = stat.size;

      const ct = mime.lookup(filePath) || "application/octet-stream";
      const range = parseRange(request.headers.get("range"), size);

      if (!range) {
        const headers = new Headers({
          "Content-Type": String(ct),
          "Content-Length": String(size),
          "Accept-Ranges": "bytes",
          "Cache-Control": "public, max-age=31536000, immutable",
        });
        return new Response(fs.createReadStream(filePath) as any, { status: 200, headers });
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
    
      return new Response(fs.createReadStream(filePath, { start, end }) as any, { status: 206, headers });
   
  }


  private async saveRemoteFileToLocalFile(response: Response, serverDetails: ServerDetails): Promise<void> {

    if (!response.ok) return;
    if (!response.body) return;

    const savePath = this.getServerSavePath(serverDetails);
    if (!savePath) return;
    const partPath = savePath + ".part";

    await fsp.mkdir(path.dirname(partPath), { recursive: true });

    const ws = fs.createWriteStream(partPath, { flags: "w" });
    ws.on("error", () => {
      // Best-effort: errors are handled by pipeline's rejection.
    });


    try {
      const body = Readable.fromWeb(response.body as any);
      await pipeline(body, ws);

      // Windows rename fails if destination exists.
      try {
        await fsp.rm(savePath, { force: true });
      } catch (e) {
        console.log("Error removing existing file", savePath, e);
        // ignore
      }
      await fsp.rename(partPath, savePath);
      console.log("Saved remote file to local file", savePath);
    } catch (e) {
      console.log("Error saving remote file to local file", savePath, e);
      // Ensure partial downloads don't accumulate.
      try {
        ws.destroy();
      } catch (e) {
        console.log("Error destroying write stream", partPath, e);
        // ignore
      }
      try {
        await fsp.rm(partPath, { force: true });
      } catch {
        console.log("Error removing partial file", partPath);
        // ignore
      }
    }
  }

  private parseServerDetails(filePath: string, folderUuid?: string | null): ServerDetails {
    // check if engine_results in the filePath or processors in the filePath
    const splitFilePath = filePath.split("/");
    const fileName = splitFilePath[splitFilePath.length - 1] ?? null;
    const folderName = splitFilePath[splitFilePath.length - 2] ?? null;
    folderUuid = folderUuid ?? splitFilePath[splitFilePath.length - 3] ?? null; 
    let type: "engine_results" | "postprocessor_results" | "preprocessor_results" = "engine_results";
    let localType: "generations" | "processors" = "generations";
    if (filePath.includes("preprocessor_results")) {
      type = "preprocessor_results";
      localType = "processors";
    } else if (filePath.includes("postprocessor_results")) {
      type = "postprocessor_results";
      localType = "processors";
    }
    return { folderUuid, folderName, fileName, type, localType };
  }

  private async fetchRemoteFile(request: Request): Promise<Response> {


    const urlObj = new URL(request.url);
    const filePath = decodeURIComponent(urlObj.pathname);
    let folderUuid = urlObj.searchParams.get("folderUuid");
    if (!folderUuid) {
      folderUuid = this.activeFolderUuid;
    }
    const serverDetails = this.parseServerDetails(filePath, folderUuid);
    const response = await this.fetchRemoteFileIfExists(serverDetails, request.headers);

    // The Response body can only be consumed once; clone for background caching.
    try {
      const clone = response.clone();
      this.queueSaveRemoteFileToLocalFile(clone, serverDetails);
    } catch {
      // Some bodies can't be cloned/teed; skip caching in that case.
    }

    const body = response?.body ? Readable.fromWeb(response.body as any) : null;
    return new Response(body as any, { status: response?.status ?? 404, headers: response?.headers ?? new Headers() });
  
}
  
  async enable({ app }: ModuleContext): Promise<void> {
    await app.whenReady();
    this.electronApp = app;
    const settings = getSettingsModule();
    this.backendUrl = settings.getBackendUrl();
    this.activeProjectId = settings.getActiveProjectId();
    this.activeFolderUuid = await this.getActiveFolderUuid();

    settings.on("backend-url-changed", (newUrl: string) => {
      void this.onBackendUrlChanged(newUrl);
    });

    settings.on("active-project-id-changed", (newProjectId: string | null) => {
      void this.onActiveProjectIdChanged(newProjectId);
    });

    protocol.handle("app", async (request) => {
       // CORS/preflight handling for renderer -> app:// fetches
       if (request.method === "OPTIONS") {
         return new Response(null, { status: 204, headers: this.corsHeadersFor(request) });
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
