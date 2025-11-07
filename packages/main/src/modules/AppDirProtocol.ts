import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import { protocol } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import mime from 'mime';
import os from 'node:os';
import { Readable } from 'node:stream';

// Helper to convert Node.js stream to Web ReadableStream with proper error handling
function nodeStreamToWebStream(nodeStream: fs.ReadStream): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      nodeStream.on('data', (chunk: string | Buffer) => {
        try {
          const uint8Array = typeof chunk === 'string' 
            ? new TextEncoder().encode(chunk) 
            : new Uint8Array(chunk);
          controller.enqueue(uint8Array);
        } catch (err) {
          // Stream might be closed, ignore
        }
      });

      nodeStream.on('end', () => {
        try {
          controller.close();
        } catch (err) {
          // Already closed, ignore
        }
      });

      nodeStream.on('error', (err) => {
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
    }
  });
}

class AppDirProtocol implements AppModule {
  private cachePath: string | null = null; // local base used for serving
  private remoteCacheBasePath: string | null = null; // absolute base path on remote machine
  private backendUrl: string = 'http://127.0.0.1:8765';
  private loopbackAppearsRemote: boolean = false;
  private inflightDownloads: Map<string, Promise<boolean>> = new Map();
  private remoteHomeDir: string | null = null;

  async enable({app}: ModuleContext): Promise<void> {
    protocol.registerSchemesAsPrivileged([{
        scheme: 'app',
        privileges: {
          standard: true,
          secure: true,
          stream: true,
          supportFetchAPI: true, // important for window.fetch
          corsEnabled: true
        }
      }]);
    await app.whenReady();
    
    // Initialize backend URL and locality
    await this.initBackendUrl(app);
    await this.probeBackendLocality();
    // Fetch cache path from backend and set local serving base accordingly
    await this.fetchCachePath(app);
    
    protocol.handle('app', async (request) => {
        const u = new URL(request.url);
        
        let basePath: string | null = null;
        if (u.hostname === 'user-data') {
          basePath = app.getPath('userData');
        } else if (u.hostname === 'apex-cache') {
          basePath = this.cachePath;
        }



        if (!basePath) {
          return new Response(null, { status: 404 });
        }
        
        const decodedPathname = decodeURIComponent(u.pathname);
        const normalizedPathname = decodedPathname.startsWith('/') ? decodedPathname.slice(1) : decodedPathname;
        let filePath: string;
        if (u.hostname === 'apex-cache' && this.loopbackAppearsRemote) {
          const rel = this.remoteRelFromNormalized(normalizedPathname);
          const localPath = rel.startsWith(basePath!) ? rel : path.join(basePath!, rel);
          try {
            if (!fs.existsSync(localPath) || !fs.statSync(localPath).isFile()) {
              const ok = await this.ensureLocalFromRemote(rel, localPath);
              if (!ok) {
                return new Response(null, { status: 404 });
              }
            }
          } catch {
            const ok = await this.ensureLocalFromRemote(rel, localPath).catch(() => false);
            if (!ok) {
              return new Response(null, { status: 404 });
            }
          }
          // Serve from the mirrored local path we just ensured
          filePath = localPath;
        } else {
          filePath = normalizedPathname.startsWith(basePath)
            ? normalizedPathname
            : path.join(basePath, normalizedPathname);
        }
        // Check if file exists and is readable
        try {
          if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            return new Response(null, { status: 404 });
          }
        } catch {
          return new Response(null, { status: 404 });
        }
        
        const contentType = mime.getType(filePath) || 'application/octet-stream';
        const fileSize = fs.statSync(filePath).size;
        
        // Handle Range requests
        const rangeHeader = request.headers.get('range');
        if (rangeHeader) {
          const parts = rangeHeader.replace(/bytes=/, '').split('-');
          const start = parseInt(parts[0], 10);
          const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
          const chunkSize = (end - start) + 1;

          const nodeStream = fs.createReadStream(filePath, { start, end });
          const webStream = nodeStreamToWebStream(nodeStream);
          console.log(webStream, "webStream");
          
          return new Response(webStream, {
            status: 206,
            headers: {
              'Content-Type': contentType,
              'Content-Length': chunkSize.toString(),
              'Content-Range': `bytes ${start}-${end}/${fileSize}`,
              'Accept-Ranges': 'bytes'
            }
          });
        }
        
        // Full content response
        const nodeStream = fs.createReadStream(filePath);
        const webStream = nodeStreamToWebStream(nodeStream);
        
        return new Response(webStream, {
          status: 200,
          headers: {
            'Content-Type': contentType,
            'Content-Length': fileSize.toString(),
            'Accept-Ranges': 'bytes'
          }
        });
      });
  }

  private async fetchCachePath(app: Electron.App): Promise<void> {
    try {
      const response = await fetch(`${this.backendUrl}/config/cache-path`);
      const ok = response.ok;
      let data: { cache_path?: string } = {};
      if (ok) {
        try { data = await response.json() as { cache_path?: string }; } catch {}
      }

      // When remote, always create and use a local mirror path under userData
      if (this.loopbackAppearsRemote) {
        if (typeof data.cache_path === 'string' && data.cache_path) {
          this.remoteCacheBasePath = data.cache_path;
        } else if (!this.remoteCacheBasePath && this.remoteHomeDir) {
          // Fallback guess when backend doesn't provide cache_path yet
          this.remoteCacheBasePath = path.join(this.remoteHomeDir, 'apex-diffusion', 'cache');
        }
        const mirror = path.join(app.getPath('userData'), 'apex-cache-remote');
        try { fs.mkdirSync(mirror, { recursive: true }); } catch {}
        this.cachePath = mirror;
        return;
      }

      // Local backend: prefer backend-provided cache path; fallback to userData
      if (typeof data.cache_path === 'string' && data.cache_path) {
        this.cachePath = data.cache_path;
      } else {
        const fallback = path.join(app.getPath('userData'), 'apex-cache');
        try { fs.mkdirSync(fallback, { recursive: true }); } catch {}
        this.cachePath = fallback;
      }
    } catch (error) {
      console.error('Failed to fetch cache path:', error);
      // Ensure we still have a usable cache path even on failure
      try {
        if (this.loopbackAppearsRemote) {
          // Remote: ensure mirror exists
          const fallback = path.join(app.getPath('userData'), 'apex-cache-remote');
          try { fs.mkdirSync(fallback, { recursive: true }); } catch {}
          this.cachePath = fallback;
        } else {
          const fallback = path.join(app.getPath('userData'), 'apex-cache');
          try { fs.mkdirSync(fallback, { recursive: true }); } catch {}
          this.cachePath = fallback;
        }
      } catch {}
    }
  }

  private async initBackendUrl(app: Electron.App): Promise<void> {
    try {
      const settingsPath = path.join(app.getPath('userData'), 'apex-settings.json');
      if (fs.existsSync(settingsPath)) {
        const raw = await fs.promises.readFile(settingsPath, 'utf-8');
        const j = JSON.parse(raw) as { backendUrl?: string };
        if (j.backendUrl && typeof j.backendUrl === 'string') {
          // Validate
          new URL(j.backendUrl);
          this.backendUrl = j.backendUrl;
          return;
        }
      }
    } catch {}
    this.backendUrl = 'http://127.0.0.1:8765';
  }

  private async probeBackendLocality(): Promise<void> {
    try {
      const u = new URL(this.backendUrl);
      const host = (u.hostname || '').toLowerCase();
      if (!(host === 'localhost' || host === '127.0.0.1' || host === '::1')) {
        this.loopbackAppearsRemote = true;
        return;
      }
      // Probe multiple signals concurrently; consider remote if any clear mismatch is found
      const [homeRes, memRes] = await Promise.allSettled([
        fetch(`${this.backendUrl}/config/home-dir`, { method: 'GET' }),
        fetch(`${this.backendUrl}/system/memory`, { method: 'GET' })
      ]);

      let appearsRemote = false;

      // Heuristic 1: Compare reported APEX home directory vs local home directory
      if (homeRes.status === 'fulfilled' && homeRes.value.ok) {
        try {
          const data = await homeRes.value.json().catch(() => ({} as any));
          const remoteHome = typeof data?.home_dir === 'string' ? data.home_dir : '';
          const localHome = os.homedir();
          const norm = (p: string) => p.replace(/\\/g, '/');
          if (remoteHome && norm(remoteHome) !== norm(localHome)) {
            appearsRemote = true;
          }
          if (remoteHome) {
            this.remoteHomeDir = remoteHome;
          }
        } catch {}
      }

      // Heuristic 2: Compare total system memory (very likely to differ across machines)
      if (!appearsRemote && memRes.status === 'fulfilled' && memRes.value.ok) {
        try {
          const m = await memRes.value.json().catch(() => ({} as any));
          const remoteTotal: number | null =
            (m && m.unified && typeof m.unified.total === 'number' && m.unified.total) ||
            (m && m.cpu && typeof m.cpu.total === 'number' && m.cpu.total) ||
            null;
          const localTotal = os.totalmem();
          if (remoteTotal && Number(remoteTotal) !== Number(localTotal)) {
            appearsRemote = true;
          }
        } catch {}
      }

      // Do not force false on probe failures; just set based on detected heuristics
      this.loopbackAppearsRemote = appearsRemote;
    } catch {}
  }

  private remoteRelFromNormalized(normalizedPathname: string): string {
    const n = normalizedPathname.replace(/\\/g, '/');
    if (this.loopbackAppearsRemote) {
      // Known bases to strip
      const bases: string[] = [];
      if (this.remoteCacheBasePath) bases.push(this.remoteCacheBasePath);
      if (this.remoteHomeDir) bases.push(path.join(this.remoteHomeDir, 'apex-diffusion', 'cache'));
      // Normalize and attempt stripping
      for (const b of bases) {
        const base = b.replace(/\\/g, '/');
        const baseNoLead = base.startsWith('/') ? base.slice(1) : base;
        if (n.startsWith(base)) {
          const rel = n.slice(base.length);
          return rel.startsWith('/') ? rel.slice(1) : rel;
        }
        if (n.startsWith(baseNoLead)) {
          const rel = n.slice(baseNoLead.length);
          return rel.startsWith('/') ? rel.slice(1) : rel;
        }
      }
      // Heuristic: strip up to and including 'apex-diffusion/cache' segment if present
      const markerIdx = n.indexOf('apex-diffusion/cache/');
      if (markerIdx !== -1) {
        const rel = n.slice(markerIdx + 'apex-diffusion/cache/'.length);
        return rel.startsWith('/') ? rel.slice(1) : rel;
      }
      return n.startsWith('/') ? n.slice(1) : n;
    }
    return n.startsWith('/') ? n.slice(1) : n;
  }

  private async ensureLocalFromRemote(relPath: string, localPath: string): Promise<boolean> {
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
            resp = await fetch(url);
            if (resp.ok && resp.body) break;
          } catch {}
          // If target got created during wait, bail out early as success
          try {
            const st = await fs.promises.stat(localPath).catch(() => null as any);
            if (st && st.isFile() && st.size > 0) return true;
          } catch {}
          await new Promise(r => setTimeout(r, baseDelayMs * Math.pow(1.5, attempt)));
        }
        if (!resp || !resp.ok || !resp.body) return false;
        const tmp = `${localPath}.part-${Date.now()}`;
        const out = fs.createWriteStream(tmp);
        try {
          const body = resp.body as any;
          const nodeReadable = (Readable as any).fromWeb ? (Readable as any).fromWeb(body) : null;
          if (!nodeReadable) {
            const buf = Buffer.from(await resp.arrayBuffer());
            await fs.promises.writeFile(tmp, buf);
          } else {
            await new Promise<void>((resolve, reject) => {
              nodeReadable.pipe(out);
              out.on('finish', () => resolve());
              out.on('error', reject);
            });
          }
          // If the target already exists (written by another concurrent download), treat as success
          try {
            await fs.promises.rename(tmp, localPath);
          } catch (err: any) {
            try {
              const st = await fs.promises.stat(localPath).catch(() => null as any);
              if (st && st.isFile() && st.size > 0) {
                // Target exists and is valid; cleanup tmp and consider success
                try { await fs.promises.unlink(tmp); } catch {}
                return true;
              }
              // Otherwise, attempt to overwrite by unlinking and renaming again
              try { await fs.promises.unlink(localPath); } catch {}
              await fs.promises.rename(tmp, localPath);
            } catch (e) {
              // Cleanup tmp on failure
              try { await fs.promises.unlink(tmp); } catch {}
              return false;
            }
          }
          return true;
        } finally {
          try { out.close(); } catch {}
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
}

export function appDirProtocol(): AppDirProtocol {
  return new AppDirProtocol();
}