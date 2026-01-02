import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { ipcMain } from "electron";
import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";

export type BundledUvPaths = {
  uv: string;
  uvx: string | null;
};

function resolveBundledUvPaths(isPackaged: boolean): BundledUvPaths {
  const platform = process.platform;
  const arch = process.arch;
  const exe = platform === "win32" ? ".exe" : "";

  if (!isPackaged) {
    return {
      uv: platform === "win32" ? `uv${exe}` : "uv",
      uvx: platform === "win32" ? `uvx${exe}` : "uvx",
    };
  }

  const base = path.join(process.resourcesPath, "uv", platform, arch);
  const uv = path.join(base, `uv${exe}`);
  const uvxCandidate = path.join(base, `uvx${exe}`);
  return {
    uv,
    uvx: fs.existsSync(uvxCandidate) ? uvxCandidate : null,
  };
}

export class UvTool implements AppModule {
  enable({ app }: ModuleContext): void {
    ipcMain.handle("uv:get-paths", async () => {
      return resolveBundledUvPaths(app.isPackaged);
    });

    ipcMain.handle("uv:get-version", async () => {
      const { uv } = resolveBundledUvPaths(app.isPackaged);
      const exists = app.isPackaged ? fs.existsSync(uv) : true;
      if (!exists) {
        return { ok: false, error: `Bundled uv not found at: ${uv}` };
      }
      return await new Promise((resolve) => {
        const p = spawn(uv, ["--version"], { windowsHide: true });
        let out = "";
        let err = "";
        p.stdout?.on("data", (d) => (out += d.toString()));
        p.stderr?.on("data", (d) => (err += d.toString()));
        p.on("close", (code) => {
          if (code === 0) resolve({ ok: true, version: out.trim() });
          else resolve({ ok: false, error: err.trim() || `uv exited with ${code}` });
        });
        p.on("error", (e) => resolve({ ok: false, error: e.message }));
      });
    });
  }
}

// Convenience factory to match other modules' style
export function uvTool(): UvTool {
  return new UvTool();
}


