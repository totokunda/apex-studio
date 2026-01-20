import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { BrowserWindow, ipcMain } from "electron";
import Store from "electron-store";
import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import { pythonProcess } from "./PythonProcess.js";

type ApiUpdateEvent =
  | { type: "checking" }
  | { type: "available"; info: unknown }
  | { type: "not-available"; info: unknown }
  | { type: "updating" }
  | {
      type: "progress";
      stage: "stopping" | "downloading" | "applying" | "restarting";
      /**
       * 0..100 (best-effort; may be omitted when unknown).
       */
      percent?: number;
      message?: string;
    }
  | { type: "updated"; info?: unknown }
  | { type: "error"; message: string }
  | { type: "allow-nightly-changed"; allowNightly: boolean };

type ApiUpdateState = {
  status: "idle" | "checking" | "available" | "not-available" | "updating" | "updated" | "error";
  updateInfo?: unknown;
  errorMessage?: string;
  lastCheckedAt?: number;
  allowNightly: boolean;
  toastSuppressedUntil?: number;
};

const API_UPDATE_EVENT_CHANNEL = "api-update:event";
const CHECK_INTERVAL_MS = 4 * 60 * 60 * 1000; // 4 hours

let lastKnownState: ApiUpdateState = {
  status: "idle",
  allowNightly: false,
};

// In-memory toast suppression (clears on app restart).
let toastSuppressedUntil = 0;

function broadcastApiUpdateEvent(ev: ApiUpdateEvent) {
  // Best-effort broadcast to all windows. We intentionally avoid hard dependencies on WindowManager.
  for (const win of BrowserWindow.getAllWindows()) {
    try {
      win.webContents.send(API_UPDATE_EVENT_CHANNEL, ev);
    } catch {
      // ignore
    }
  }
}

function resolveApexEngineInvocation(pythonExe: string): { cmd: string; baseArgs: string[] } {
  const dir = path.dirname(pythonExe);
  const candidates =
    process.platform === "win32"
      ? [path.join(dir, "apex-engine.exe"), path.join(dir, "apex-engine.cmd"), path.join(dir, "apex-engine")]
      : [path.join(dir, "apex-engine")];

  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) {
        return { cmd: c, baseArgs: [] };
      }
    } catch {
      // ignore
    }
  }

  // Fallback: use the module entrypoint installed in the venv.
  return { cmd: pythonExe, baseArgs: ["-m", "src.__main__"] };
}

function resolveApexEngineFallback(pythonExe: string): { cmd: string; baseArgs: string[] } {
  return { cmd: pythonExe, baseArgs: ["-m", "src.__main__"] };
}

async function spawnCapture(opts: {
  cmd: string;
  args: string[];
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
}): Promise<{ code: number | null; stdout: string; stderr: string }> {
  const timeoutMs = typeof opts.timeoutMs === "number" ? opts.timeoutMs : 60_000;

  return await new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let settled = false;

    const child = spawn(opts.cmd, opts.args, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: opts.env || process.env,
    });

    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            try {
              child.kill();
            } catch {
              // ignore
            }
          }, timeoutMs)
        : null;

    child.stdout?.on("data", (b) => {
      stdout += b.toString("utf-8");
    });
    child.stderr?.on("data", (b) => {
      stderr += b.toString("utf-8");
    });

    child.on("error", (err) => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      settled = true;
      reject(err);
    });

    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      settled = true;
      resolve({ code, stdout, stderr });
    });
  });
}

async function spawnCaptureWithProgress(opts: {
  cmd: string;
  args: string[];
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
  onStdoutChunk?: (chunk: string) => void;
  onStderrChunk?: (chunk: string) => void;
}): Promise<{ code: number | null; stdout: string; stderr: string }> {
  const timeoutMs = typeof opts.timeoutMs === "number" ? opts.timeoutMs : 60_000;

  return await new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let settled = false;

    const child = spawn(opts.cmd, opts.args, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: opts.env || process.env,
    });

    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            try {
              child.kill();
            } catch {
              // ignore
            }
          }, timeoutMs)
        : null;

    child.stdout?.on("data", (b) => {
      const chunk = b.toString("utf-8");
      stdout += chunk;
      try {
        opts.onStdoutChunk?.(chunk);
      } catch {
        // ignore
      }
    });
    child.stderr?.on("data", (b) => {
      const chunk = b.toString("utf-8");
      stderr += chunk;
      try {
        opts.onStderrChunk?.(chunk);
      } catch {
        // ignore
      }
    });

    child.on("error", (err) => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      settled = true;
      reject(err);
    });

    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      settled = true;
      resolve({ code, stdout, stderr });
    });
  });
}

function safeJsonParseFromStdout(stdout: string): unknown {
  const s = String(stdout || "").trim();
  if (!s) return null;
  // Best-effort: find the first JSON object/array start.
  const idxObj = s.indexOf("{");
  const idxArr = s.indexOf("[");
  const idx =
    idxObj === -1 ? idxArr : idxArr === -1 ? idxObj : Math.min(idxObj, idxArr);
  const candidate = idx >= 0 ? s.slice(idx) : s;
  return JSON.parse(candidate);
}

export class ApexApiAutoUpdater implements AppModule {
  readonly #store: Store<{ apiAllowNightlyUpdates: boolean }>;
  #timer: NodeJS.Timeout | null = null;
  #checkInFlight: Promise<unknown | null> | null = null;
  #updateInFlight: Promise<{ ok: boolean }> | null = null;

  constructor() {
    this.#store = new Store<{ apiAllowNightlyUpdates: boolean }>({
      name: "apex-settings",
      defaults: { apiAllowNightlyUpdates: false },
    });
  }

  async enable({ app }: ModuleContext): Promise<void> {
    try {
      this.registerIpc();
      await app.whenReady();

      // Initialize persisted nightly setting into state.
      lastKnownState = { ...lastKnownState, allowNightly: this.getAllowNightly() };


      if (this.#timer) return;

      // Initial check + periodic checks.
      void this.checkForUpdates();
      this.#timer = setInterval(() => void this.checkForUpdates(), CHECK_INTERVAL_MS);
      app.on("before-quit", () => {
        if (this.#timer) clearInterval(this.#timer);
        this.#timer = null;
      });
    } catch (error) {
      console.warn("[ApexApiAutoUpdater] Failed to enable:", error);
    }
  }

  private getAllowNightly(): boolean {
    try {
      return Boolean(this.#store.get("apiAllowNightlyUpdates"));
    } catch {
      return false;
    }
  }

  private setAllowNightly(v: boolean): void {
    try {
      this.#store.set("apiAllowNightlyUpdates", Boolean(v));
    } catch {
      // ignore
    }
    lastKnownState = { ...lastKnownState, allowNightly: Boolean(v) };
    broadcastApiUpdateEvent({ type: "allow-nightly-changed", allowNightly: Boolean(v) });
    void this.checkForUpdates();
  }

  registerIpc() {
    if (ipcMain.listenerCount("api-update:get-state") > 0) return;

    ipcMain.handle("api-update:get-state", async () => {
      return {
        ...lastKnownState,
        toastSuppressedUntil: toastSuppressedUntil > 0 ? toastSuppressedUntil : undefined,
      } satisfies ApiUpdateState;
    });

    ipcMain.handle("api-update:set-allow-nightly", async (_event, allowNightly: boolean) => {
      this.setAllowNightly(Boolean(allowNightly));
      return { ok: true, allowNightly: lastKnownState.allowNightly };
    });

    ipcMain.handle(
      "api-update:suppress-toast",
      async (_event, payload?: { durationMs?: number }) => {
        const durationMs =
          typeof payload?.durationMs === "number" && Number.isFinite(payload.durationMs)
            ? Math.max(0, payload.durationMs)
            : 12 * 60 * 60 * 1000;
        toastSuppressedUntil = Date.now() + durationMs;
        return { ok: true, suppressedUntil: toastSuppressedUntil };
      },
    );

    ipcMain.handle("api-update:check", async () => {
      const res = await this.checkForUpdates();
      return res;
    });

    ipcMain.handle("api-update:apply", async () => {
      return await this.applyUpdate();
    });
  }

  private async checkForUpdates(): Promise<unknown | null> {
    if (this.#checkInFlight) return await this.#checkInFlight;

    this.#checkInFlight = (async () => {
      lastKnownState = { ...lastKnownState, status: "checking", errorMessage: undefined };
      broadcastApiUpdateEvent({ type: "checking" });

      const py = pythonProcess();
      const verified = await py.verifyRuntime();
      const runtime = py.getRuntimeInfoSnapshot();

      if (!verified.ok || !runtime.available || !runtime.pythonExe || !runtime.bundleRoot) {
        const message =
          verified.ok === false
            ? verified.reason || "Backend runtime not available"
            : "Backend runtime not available";
        lastKnownState = { ...lastKnownState, status: "error", errorMessage: message, lastCheckedAt: Date.now() };
        broadcastApiUpdateEvent({ type: "error", message });
        return null;
      }

      const { cmd, baseArgs } = resolveApexEngineInvocation(runtime.pythonExe);
      const fallback = resolveApexEngineFallback(runtime.pythonExe);

      const env = { ...process.env, PYTHONPATH: runtime.bundleRoot };
      const args = [
        ...baseArgs,
        "check-updates",
        "--format",
        "json",
        "--target-dir",
        runtime.bundleRoot,
        ...(lastKnownState.allowNightly ? ["--allow-nightly"] : []),
      ];

      try {
        const { code, stdout, stderr } = await (async () => {
          try {
            return await spawnCapture({
              cmd,
              args,
              timeoutMs: 90_000,
              env,
            });
          } catch (e) {
            // If the chosen launcher is not executable (e.g. broken shebang from a relocated bundle),
            // retry via `python -m src.__main__` so update-checking can self-heal.
            if (cmd !== fallback.cmd) {
              return await spawnCapture({
                cmd: fallback.cmd,
                args: [...fallback.baseArgs, ...args.slice(baseArgs.length)],
                timeoutMs: 90_000,
                env,
              });
            }
            throw e;
          }
        })();
        if (code !== 0) {
          const message = (stderr || stdout || "Failed to check for API updates").trim();
          lastKnownState = { ...lastKnownState, status: "error", errorMessage: message, lastCheckedAt: Date.now() };
          broadcastApiUpdateEvent({ type: "error", message });
          return null;
        }

        const parsed = safeJsonParseFromStdout(stdout);
        const updates = (parsed as any)?.updates;
        const hasUpdates = Array.isArray(updates) && updates.length > 0;

        lastKnownState = {
          ...lastKnownState,
          status: hasUpdates ? "available" : "not-available",
          updateInfo: parsed,
          lastCheckedAt: Date.now(),
          errorMessage: undefined,
        };
        broadcastApiUpdateEvent(
          hasUpdates ? { type: "available", info: parsed } : { type: "not-available", info: parsed },
        );
        return parsed;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to check for API updates";
        lastKnownState = { ...lastKnownState, status: "error", errorMessage: message, lastCheckedAt: Date.now() };
        broadcastApiUpdateEvent({ type: "error", message });
        return null;
      }
    })().finally(() => {
      this.#checkInFlight = null;
    });

    return await this.#checkInFlight;
  }

  private async applyUpdate(): Promise<{ ok: boolean; message?: string }> {
    if (this.#updateInFlight) return await this.#updateInFlight;

    this.#updateInFlight = (async () => {
      const py = pythonProcess();
      const runtime = py.getRuntimeInfoSnapshot();
      const verified = await py.verifyRuntime();

      if (!verified.ok || !runtime.available || !runtime.pythonExe || !runtime.bundleRoot) {
        const message =
          verified.ok === false
            ? verified.reason || "Backend runtime not available"
            : "Backend runtime not available";
        lastKnownState = { ...lastKnownState, status: "error", errorMessage: message };
        broadcastApiUpdateEvent({ type: "error", message });
        return { ok: false, message };
      }

      lastKnownState = { ...lastKnownState, status: "updating", errorMessage: undefined };
      broadcastApiUpdateEvent({ type: "updating" });
      broadcastApiUpdateEvent({ type: "progress", stage: "stopping", percent: 0, message: "Stopping engine…" });

      let updateOutput: unknown | null = null;
      let updateOk = false;
      try {
        try {
          py.setAutoRestartSuppressed(true, "api-update");
        } catch {
          // ignore
        }
        // Stop the API process before applying updates.
        await py.stop();

        const { cmd, baseArgs } = resolveApexEngineInvocation(runtime.pythonExe);
        const fallback = resolveApexEngineFallback(runtime.pythonExe);
        const env = { ...process.env, PYTHONPATH: runtime.bundleRoot };
        const args = [
          ...baseArgs,
          "update",
          "--target-dir",
          runtime.bundleRoot,
          ...(lastKnownState.allowNightly ? ["--allow-nightly"] : []),
        ];

        let rolling = "";
        let lastPct: number | null = null;
        const emitProgressFromText = (text: string) => {
          rolling = (rolling + text).slice(-4096);

          // Stage signals.
          if (/Downloading:/i.test(text) || /Downloading[.…\.]/u.test(text)) {
            broadcastApiUpdateEvent({ type: "progress", stage: "downloading", message: "Downloading update…" });
          }
          if (/Downloaded:/i.test(text) || /Update applied/i.test(text)) {
            broadcastApiUpdateEvent({ type: "progress", stage: "applying", message: "Applying update…" });
          }

          // Percent signal: handles "Downloading…  12.3% ..." (uses carriage returns).
          const m = rolling.match(/Downloading[.…\.]*\s*([0-9]{1,3}(?:\.[0-9]+)?)%/u);
          if (m?.[1]) {
            const pct = Number(m[1]);
            if (Number.isFinite(pct)) {
              const clamped = Math.max(0, Math.min(100, pct));
              // De-noise: only emit on meaningful change.
              if (lastPct === null || Math.abs(clamped - lastPct) >= 0.2) {
                lastPct = clamped;
                broadcastApiUpdateEvent({
                  type: "progress",
                  stage: "downloading",
                  percent: clamped,
                  message: `Downloading update… ${clamped.toFixed(1)}%`,
                });
              }
            }
          }
        };

        const { code, stdout, stderr } = await (async () => {
          try {
            return await spawnCaptureWithProgress({
              cmd,
              args,
              // Updates can take a while (download + extract).
              timeoutMs: 30 * 60 * 1000,
              env,
              onStdoutChunk: emitProgressFromText,
              onStderrChunk: emitProgressFromText,
            });
          } catch (e) {
            // Same fallback as checkForUpdates: allow older broken bundles to update themselves.
            if (cmd !== fallback.cmd) {
              return await spawnCaptureWithProgress({
                cmd: fallback.cmd,
                args: [...fallback.baseArgs, ...args.slice(baseArgs.length)],
                timeoutMs: 30 * 60 * 1000,
                env,
                onStdoutChunk: emitProgressFromText,
                onStderrChunk: emitProgressFromText,
              });
            }
            throw e;
          }
        })();

        if (code !== 0) {
          const message = (stderr || stdout || "Failed to update API").trim();
          lastKnownState = { ...lastKnownState, status: "error", errorMessage: message };
          broadcastApiUpdateEvent({ type: "error", message });
          return { ok: false, message };
        }

        // Best-effort: re-check after update to refresh info.
        try {
          updateOutput = safeJsonParseFromStdout(stdout);
        } catch {
          updateOutput = { stdout };
        }
        updateOk = true;
        return { ok: true };
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to update API";
        lastKnownState = { ...lastKnownState, status: "error", errorMessage: message };
        broadcastApiUpdateEvent({ type: "error", message });
        return { ok: false, message };
      } finally {
        // Restart the API process after updates (or failed attempt).
        try {
          broadcastApiUpdateEvent({ type: "progress", stage: "restarting", message: "Restarting engine…" });
          await py.start();
          if (updateOk) {
            lastKnownState = { ...lastKnownState, status: "updated", updateInfo: updateOutput ?? undefined };
            broadcastApiUpdateEvent({ type: "updated", ...(updateOutput ? { info: updateOutput } : {}) });
          }
        } catch (e) {
          const message =
            e instanceof Error ? e.message : "Failed to restart API after update";
          lastKnownState = { ...lastKnownState, status: "error", errorMessage: message };
          broadcastApiUpdateEvent({ type: "error", message });
        } finally {
          try {
            py.setAutoRestartSuppressed(false);
          } catch {
            // ignore
          }
        }
        // Refresh update state in the background.
        void this.checkForUpdates();
      }
    })().finally(() => {
      this.#updateInFlight = null;
    });

    return await this.#updateInFlight;
  }
}

export function apexApiAutoUpdater() {
  return new ApexApiAutoUpdater();
}

