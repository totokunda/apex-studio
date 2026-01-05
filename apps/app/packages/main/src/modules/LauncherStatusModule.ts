import type { AppModule } from "../AppModule.js";
import type { ModuleContext } from "../ModuleContext.js";
import { ipcMain } from "electron";
import { getSettingsModule } from "./SettingsModule.js";
import { pythonProcess } from "./PythonProcess.js";

type LauncherPythonState = {
  status: "stopped" | "starting" | "running" | "stopping" | "error";
  pid?: number;
  port: number;
  host: string;
  error?: string;
  restartCount: number;
};

export type LauncherStatus = {
  backendUrl: string | null;
  backendHealthy: boolean;
  pythonHealthy: boolean;
  pythonState: LauncherPythonState;
  runtimeInfo: {
    available: boolean;
    mode: "dev" | "bundled" | "installed" | "missing";
    installedApiPath: string | null;
    bundleRoot: string | null;
    pythonExe: string | null;
    reason?: string;
  };
  runtimeVerified: { ok: boolean; reason?: string };
  canStartLocal: boolean;
  hasBackend: boolean;
  backendStarting: boolean;
  shouldShowInstaller: boolean;
  lastError?: string | null;
};

function isLoopbackUrl(urlStr: string | null): boolean {
  if (!urlStr) return true;
  try {
    const u = new URL(urlStr);
    return (
      u.hostname === "127.0.0.1" ||
      u.hostname === "localhost" ||
      u.hostname === "[::1]"
    );
  } catch {
    return true;
  }
}

async function checkHttpHealth(urlStr: string | null): Promise<boolean> {
  if (!urlStr) return false;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${urlStr.replace(/\/$/, "")}/health`, {
      method: "GET",
      signal: controller.signal,
    }).finally(() => clearTimeout(timeout));
    return res.ok;
  } catch {
    return false;
  }
}

class LauncherStatusModule implements AppModule {
  #watchTimers = new Map<number, NodeJS.Timeout>();
  #lastSentSig = new Map<number, string>();
  #autoStartAttempted = false;
  #lastError: string | null = null;

  enable(_context: ModuleContext): void {
    this.#registerIpc();
  }

  #registerIpc(): void {
    if (ipcMain.listenerCount("launcher:status") === 0) {
      ipcMain.handle("launcher:status", async () => {
        const status = await this.#computeStatus();
        return { success: true, data: status };
      });
    }

    if (ipcMain.listenerCount("launcher:watch-start") === 0) {
      ipcMain.handle("launcher:watch-start", async (event, intervalMs?: number) => {
        const wcId = event.sender.id;
        const existing = this.#watchTimers.get(wcId);
        if (existing) {
          return { success: true };
        }

        const tick = async () => {
          try {
            const status = await this.#computeStatus();
            const sig = JSON.stringify({
              backendUrl: status.backendUrl,
              backendHealthy: status.backendHealthy,
              pythonHealthy: status.pythonHealthy,
              pythonState: status.pythonState,
              runtimeInfo: status.runtimeInfo,
              runtimeVerified: status.runtimeVerified,
              canStartLocal: status.canStartLocal,
              hasBackend: status.hasBackend,
              backendStarting: status.backendStarting,
              shouldShowInstaller: status.shouldShowInstaller,
              lastError: status.lastError,
            });
            if (this.#lastSentSig.get(wcId) !== sig) {
              this.#lastSentSig.set(wcId, sig);
              try {
                event.sender.send("launcher:status-changed", status);
              } catch {
                // ignore send failures
              }
            }
          } catch {
            // ignore tick errors
          }
        };

        // Send initial status immediately, then poll.
        await tick();
        const t = setInterval(() => {
          void tick();
        }, Math.max(500, Number(intervalMs) || 3000));
        this.#watchTimers.set(wcId, t);

        try {
          event.sender.once("destroyed", () => this.#stopWatch(wcId));
        } catch {
          // ignore
        }

        return { success: true };
      });
    }

    if (ipcMain.listenerCount("launcher:watch-stop") === 0) {
      ipcMain.handle("launcher:watch-stop", async (event) => {
        this.#stopWatch(event.sender.id);
        return { success: true };
      });
    }
  }

  #stopWatch(wcId: number): void {
    const t = this.#watchTimers.get(wcId);
    if (t) clearInterval(t);
    this.#watchTimers.delete(wcId);
    this.#lastSentSig.delete(wcId);
  }

  async #computeStatus(): Promise<LauncherStatus> {
    const settings = getSettingsModule();
    const backendUrlRaw = (() => {
      try {
        const v = settings.getBackendUrl();
        const s = (v ?? "").trim();
        return s ? s : null;
      } catch {
        return null;
      }
    })();

    const py = pythonProcess();
    const runtimeInfo = py.getRuntimeInfoSnapshot();
    const runtimeVerified = await py.verifyRuntime();
    // Health checks
    const backendHealthy = await checkHttpHealth(backendUrlRaw);
    const pythonHealthy = await py.checkApiHealth();

    // Auto-start local backend once when it looks like we're meant to use loopback.
    if (
      !backendHealthy &&
      !pythonHealthy &&
      runtimeVerified.ok &&
      !this.#autoStartAttempted &&
      isLoopbackUrl(backendUrlRaw)
    ) {
      this.#autoStartAttempted = true;
      this.#lastError = null;
      try {
        await py.start();
      } catch (e) {
        this.#lastError =
          e instanceof Error ? e.message : "Failed to start backend";
      }
    }

    const st = py.getState();
    const pythonState: LauncherPythonState = {
      status: st.status,
      pid: st.pid,
      port: st.port,
      host: st.host,
      error: st.error,
      restartCount: st.restartCount,
    };

    const hasBackend = Boolean(backendHealthy || pythonHealthy);
    const canStartLocal = Boolean(runtimeVerified.ok);
    const backendStarting = pythonState.status === "starting";
    const shouldShowInstaller = !hasBackend && !canStartLocal;

    return {
      backendUrl: backendUrlRaw,
      backendHealthy,
      pythonHealthy,
      pythonState,
      runtimeInfo,
      runtimeVerified,
      canStartLocal,
      hasBackend,
      backendStarting,
      shouldShowInstaller,
      lastError: this.#lastError,
    };
  }
}

// Singleton-ish module factory
let launcherStatusInstance: LauncherStatusModule | null = null;
export function launcherStatusModule(): LauncherStatusModule {
  if (!launcherStatusInstance) {
    launcherStatusInstance = new LauncherStatusModule();
  }
  return launcherStatusInstance;
}


