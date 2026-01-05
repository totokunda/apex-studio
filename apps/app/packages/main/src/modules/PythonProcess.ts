/**
 * PythonProcess Module
 * 
 * Manages the bundled Python API lifecycle within the Electron app.
 * - Starts the Python API server when the app starts
 * - Monitors health and restarts if necessary
 * - Gracefully shuts down when the app closes
 * - Handles different platforms (macOS, Windows, Linux)
 */

import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { App, ipcMain, dialog } from "electron";
import { spawn, ChildProcess, exec, execFile } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { EventEmitter } from "node:events";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { getSettingsModule } from "./SettingsModule.js";

// Configuration constants
const DEFAULT_API_PORT = 8765;
const DEFAULT_API_HOST = "127.0.0.1";
// Keep this tight: we want near-immediate detection/restart if the backend dies.
const HEALTH_CHECK_INTERVAL_MS = 2000;
const HEALTH_CHECK_TIMEOUT_MS = 1500;
const MAX_STARTUP_WAIT_MS = 60000;
// Small delay to avoid hot-looping the event loop on repeated instant exits.
const RESTART_DELAY_MS = 500;
// Avoid flapping: require a few consecutive failures before restarting.
const HEALTH_FAILURES_BEFORE_RESTART = 3;
// If we intentionally stop a process (during restart/stop), suppress "unexpected exit" handling briefly.
const EXPECTED_EXIT_WINDOW_MS = 10_000;

interface PythonProcessConfig {
  port?: number;
  host?: string;
  autoStart?: boolean;
  devMode?: boolean;
  pythonPath?: string;
  /**
   * If true, stop the API when ALL windows are closed.
   * Default is false because the API should be monitored by the main process
   * across the whole application (launcher may close while the app stays alive).
   */
  stopOnAllWindowsClosed?: boolean;
}

interface ProcessState {
  status: "stopped" | "starting" | "running" | "stopping" | "error";
  pid?: number;
  port: number;
  host: string;
  error?: string;
  restartCount: number;
  lastHealthCheck?: Date;
}

export type PythonRuntimeInfo = {
  available: boolean;
  mode: "dev" | "bundled" | "installed" | "missing";
  installedApiPath: string | null;
  bundleRoot: string | null;
  pythonExe: string | null;
  reason?: string;
};

export class PythonProcessManager extends EventEmitter implements AppModule {
  private app: App | null = null;
  private process: ChildProcess | null = null;
  private lastSpawnedPid: number | null = null;
  private state: ProcessState;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private config: PythonProcessConfig;
  private logDir: string = "";
  private latestLogPath: string = "";
  private sessionLogPath: string = "";
  private bundledPythonPath: string = "";
  private bundledBundleRoot: string = "";
  private isShuttingDown: boolean = false;
  // Whether the user/app currently expects the backend to be running.
  // Used to decide if we should auto-restart after crashes/unhealthy signals.
  private desiredRunning: boolean = false;
  // Serialize start/stop/restart to avoid races during window transitions / IPC spam.
  private opQueue: Promise<void> = Promise.resolve();
  private restartTimer: NodeJS.Timeout | null = null;
  private consecutiveHealthFailures: number = 0;
  private expectedExitPid: number | null = null;
  private expectedExitUntilMs: number = 0;
  private runtimeVerifyCache:
    | { at: number; ok: boolean; reason?: string }
    | null = null;

  constructor(config: PythonProcessConfig = {}) {
    super();
    this.config = {
      port: config.port ?? DEFAULT_API_PORT,
      host: config.host ?? DEFAULT_API_HOST,
      autoStart: config.autoStart ?? true,
      devMode: config.devMode ?? false,
      pythonPath: config.pythonPath,
      stopOnAllWindowsClosed: config.stopOnAllWindowsClosed ?? false,
    };
    this.state = {
      status: "stopped",
      port: this.config.port!,
      host: this.config.host!,
      restartCount: 0,
    };
  }

  async enable(context: ModuleContext): Promise<void> {
    this.app = context.app;

    // Set up logging paths
    // In production, write logs into the OS-specific "logs" folder (AppData Logs on Windows).
    // In dev, keep it in userData so it's easy to find.
    this.resolveAndPrepareLogPaths();

    // Determine the bundled Python path
    this.bundledPythonPath = this.getBundledPythonPath();
    this.bundledBundleRoot = this.getBundledBundleRoot();

    // Register IPC handlers
    this.registerHandlers();

    // Set up app lifecycle hooks
    this.setupLifecycleHooks();

    // Auto-start if configured and not in dev mode
    if (this.config.autoStart && !this.config.devMode) {
      await this.app.whenReady();
      // Small delay to ensure window is ready
      setTimeout(() => {
        this.start().catch((err) => {
          console.error("Failed to auto-start Python API:", err);
          this.emit("error", err);
        });
      }, 1000);
    }
  }

  /**
   * Snapshot of runtime discovery used by the launcher to decide whether the backend
   * can be started locally (vs. installer required).
   */
  public getRuntimeInfoSnapshot(): PythonRuntimeInfo {
    return this.getRuntimeInfo();
  }

  /**
   * Strong runtime verification: ensures the backend runtime/code exist and can execute.
   * This prevents false "healthy" when the backend code has been deleted.
   */
  public async verifyRuntime(): Promise<{ ok: boolean; reason?: string }> {
    const now = Date.now();
    const cached = this.runtimeVerifyCache;
    // 30s TTL avoids repeated exec while still reacting to reinstall/delete quickly.
    if (cached && now - cached.at < 30_000) {
      return {
        ok: cached.ok,
        ...(cached.reason ? { reason: cached.reason } : {}),
      };
    }

    const info = this.getRuntimeInfo();
    if (!info.available) {
      const reason = info.reason || "Runtime not available";
      this.runtimeVerifyCache = { at: now, ok: false, reason };
      return { ok: false, reason };
    }

    // Dev mode: ensure the API code directory exists (repo layout / apps/api).
    if (info.mode === "dev") {
      try {
        const devApiPath = this.#resolveDevApiPath();
        const marker = path.join(devApiPath, "pyproject.toml");
        if (!fs.existsSync(marker)) {
          const reason = `Backend code not found (missing ${marker})`;
          this.runtimeVerifyCache = { at: now, ok: false, reason };
          return { ok: false, reason };
        }
        this.runtimeVerifyCache = { at: now, ok: true };
        return { ok: true };
      } catch (e) {
        const reason =
          e instanceof Error ? e.message : "Failed to verify dev backend code";
        this.runtimeVerifyCache = { at: now, ok: false, reason };
        return { ok: false, reason };
      }
    }

    const pythonExe = info.pythonExe || this.bundledPythonPath || null;
    if (!pythonExe) {
      const reason = "Python executable not resolved";
      this.runtimeVerifyCache = { at: now, ok: false, reason };
      return { ok: false, reason };
    }
    try {
      if (!fs.existsSync(pythonExe)) {
        const reason = `Python runtime not found: ${pythonExe}`;
        this.runtimeVerifyCache = { at: now, ok: false, reason };
        return { ok: false, reason };
      }
    } catch {
      const reason = `Python runtime not accessible: ${pythonExe}`;
      this.runtimeVerifyCache = { at: now, ok: false, reason };
      return { ok: false, reason };
    }

    const execFileAsync = promisify(execFile);
    try {
      // Validate python can run AND backend code is present.
      await execFileAsync(
        pythonExe,
        [
          "-c",
          "import sys; print(sys.version)",
        ],
        { timeout: 5000, windowsHide: true },
      );
      this.runtimeVerifyCache = { at: now, ok: true };
      return { ok: true };
    } catch (e) {
      const reason =
        e instanceof Error
          ? `Backend runtime verification failed: ${e.message}`
          : "Backend runtime verification failed";
      this.runtimeVerifyCache = { at: now, ok: false, reason };
      return { ok: false, reason };
    }
  }

  private getBundledPythonPath(): string {
    if (this.config.pythonPath) {
      return this.config.pythonPath;
    }

    // Check if we're in development or production
    const isPackaged = this.app?.isPackaged ?? false;
    const platform = process.platform;

    if (isPackaged) {
      // Production: look for bundled Python in resources
      const resourcesPath = process.resourcesPath!;
      
      if (platform === "darwin") {
        // macOS: venv-based bundle
        return path.join(
          resourcesPath,
          "python-api",
          "apex-engine",
          "apex-studio",
          "bin",
          "python"
        );
      } else if (platform === "win32") {
        return path.join(
          resourcesPath,
          "python-api",
          "apex-engine",
          "apex-studio",
          "Scripts",
          "python.exe"
        );
      } else {
        return path.join(
          resourcesPath,
          "python-api",
          "apex-engine",
          "apex-studio",
          "bin",
          "python"
        );
      }
    } else {
      // Development: use system Python or virtual environment
      const devApiPath = this.#resolveDevApiPath();
      
      // Check for conda/venv
      const condaEnv = process.env.CONDA_PREFIX;
      if (condaEnv) {
        return platform === "win32"
          ? path.join(condaEnv, "python.exe")
          : path.join(condaEnv, "bin", "python");
      }

      const venvPath = path.join(devApiPath, ".venv", platform === "win32" ? "Scripts" : "bin", "python");
      if (fs.existsSync(venvPath)) {
        return venvPath;
      }

      // Fallback to system Python
      return platform === "win32" ? "python" : "python3";
    }
  }

  private getBundledBundleRoot(): string {
    const isPackaged = this.app?.isPackaged ?? false;
    if (!isPackaged) return "";
    const resourcesPath = process.resourcesPath!;
    return path.join(resourcesPath, "python-api", "apex-engine");
  }

  private resolveInstalledBundleRoot(apiInstallDir: string): string | null {
    /**
     * We want the directory that contains:
     *   <bundleRoot>/apex-studio/(bin|Scripts)/python
     *
     * Supported layouts (apiInstallDir is user-chosen install dir / extraction dir):
     * - <apiInstallDir>/python-api/apex-engine/apex-studio/...
     * - <apiInstallDir>/apex-engine/apex-studio/...
     * - <apiInstallDir> (already points at apex-engine root)
     */
    const root = String(apiInstallDir || "").trim();
    if (!root) return null;

    const candidates = [
      path.join(root, "python-api", "apex-engine"),
      path.join(root, "apex-engine"),
      root,
    ];

    for (const p of candidates) {
      try {
        if (!fs.existsSync(p) || !fs.statSync(p).isDirectory()) continue;
        const marker = path.join(p, "apex-studio");
        if (fs.existsSync(marker) && fs.statSync(marker).isDirectory()) return p;
      } catch {
        // ignore
      }
    }
    return null;
  }

  private resolveInstalledPythonExe(bundleRoot: string): string | null {
    const base = path.join(bundleRoot, "apex-studio");
    const candidates =
      process.platform === "win32"
        ? [path.join(base, "Scripts", "python.exe")]
        : [path.join(base, "bin", "python"), path.join(base, "bin", "python3")];
    for (const p of candidates) {
      try {
        if (fs.existsSync(p)) return p;
      } catch {
        // ignore
      }
    }
    return null;
  }

  private getRuntimeInfo(): PythonRuntimeInfo {
    const installedApiPath = (() => {
      try {
        return getSettingsModule().getApiPath();
      } catch {
        return null;
      }
    })();

    // check if path exists 
    if (installedApiPath && !fs.existsSync(installedApiPath)) {
      return {
        available: false,
        mode: "missing",
        installedApiPath,
        bundleRoot: null,
        pythonExe: null,
        reason: "Installed API path does not exist",
      };
    }


    // Packaged: prefer installed bundle if valid.
    if (installedApiPath) {
      const installedBundleRoot = this.resolveInstalledBundleRoot(installedApiPath);
      const installedPythonExe = installedBundleRoot
        ? this.resolveInstalledPythonExe(installedBundleRoot)
        : null;
      if (installedBundleRoot && installedPythonExe) {
        return {
          available: true,
          mode: "installed",
          installedApiPath,
          bundleRoot: installedBundleRoot,
          pythonExe: installedPythonExe,
        };
      }
    }

    // Otherwise fall back to the app-bundled runtime.
    try {
      if (this.bundledPythonPath && fs.existsSync(this.bundledPythonPath)) {
        return {
          available: true,
          mode: "bundled",
          installedApiPath,
          bundleRoot: this.bundledBundleRoot || null,
          pythonExe: this.bundledPythonPath,
        };
      }
    } catch {
      // ignore
    }

    return {
      available: false,
      mode: "missing",
      installedApiPath,
      bundleRoot: null,
      pythonExe: null,
      reason:
        "No bundled runtime found and no valid installed runtime detected from settings apiPath",
    };
  }

  #resolveDevApiPath(): string {
    // In ESM, __dirname is not available. Resolve relative to import.meta.url and cwd.
    const moduleDir = path.dirname(fileURLToPath(import.meta.url));
    const cwd = process.cwd();

    // We want the directory that contains `pyproject.toml` for the API package.
    // In this repo, that is: <repo>/apps/api
    const candidates = [
      // Common when running `cd apps/app && npm run start`
      path.resolve(cwd, "..", "api"),
      // Common when running from repo root
      path.resolve(cwd, "apps", "api"),
      // Relative to the compiled main package output (apps/app/packages/main/dist)
      path.resolve(moduleDir, "..", "..", "..", "..", "api"),
      path.resolve(moduleDir, "..", "..", "..", "..", "..", "apps", "api"),
    ];

    for (const p of candidates) {
      try {
        const marker = path.join(p, "pyproject.toml");
        if (fs.existsSync(marker)) return p;
      } catch {
        // ignore
      }
    }

    // Best-effort fallback: repo layout assumption
    return path.resolve(cwd, "..", "api");
  }

  private registerHandlers(): void {
    ipcMain.handle("python:start", async () => {
      try {
        await this.start();
        return { success: true, data: this.state };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Failed to start Python API",
        };
      }
    });

    ipcMain.handle("python:stop", async () => {
      try {
        await this.stop();
        return { success: true, data: this.state };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Failed to stop Python API",
        };
      }
    });

    ipcMain.handle("python:restart", async () => {
      try {
        await this.restart();
        return { success: true, data: this.state };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Failed to restart Python API",
        };
      }
    });

    ipcMain.handle("python:status", async () => {
      return { success: true, data: this.state };
    });

    ipcMain.handle("python:health", async () => {
      const healthy = await this.checkApiHealth();
      return { success: true, data: { healthy, state: this.state } };
    });

    ipcMain.handle("python:runtime-info", async () => {
      try {
        const info = this.getRuntimeInfo();
        return { success: true, data: info };
      } catch (error) {
        return {
          success: false,
          error:
            error instanceof Error
              ? error.message
              : "Failed to determine Python runtime availability",
        };
      }
    });

    ipcMain.handle("python:logs", async () => {
      try {
        const p = this.latestLogPath || this.sessionLogPath;
        if (p && fs.existsSync(p)) {
          const logs = fs.readFileSync(p, "utf-8");
          // Return last 1000 lines
          const lines = logs.split("\n").slice(-1000).join("\n");
          return {
            success: true,
            data: {
              logs: lines,
              logDir: this.logDir,
              latestLogPath: this.latestLogPath,
              sessionLogPath: this.sessionLogPath,
            },
          };
        }
        return {
          success: true,
          data: {
            logs: "",
            logDir: this.logDir,
            latestLogPath: this.latestLogPath,
            sessionLogPath: this.sessionLogPath,
          },
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Failed to read logs",
        };
      }
    });
  }

  private resolveAndPrepareLogPaths(): void {
    if (!this.app) return;
    const isPackaged = this.app.isPackaged;
    const appNameSafe = (this.app.getName?.() || "apex-studio")
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-");

    // Electron provides a dedicated "logs" path in production; fall back safely if missing.
    const baseDir = (() => {
      if (isPackaged) {
        try {
          // Available in modern Electron; OS-specific Logs folder (AppData/Logs on Windows).
          return this.app!.getPath("logs");
        } catch {
          // Some environments may not expose "logs" path; fall back to userData.
          return this.app!.getPath("userData");
        }
      }
      return this.app!.getPath("userData");
    })();

    this.logDir = path.join(baseDir, appNameSafe);

    try {
      fs.mkdirSync(this.logDir, { recursive: true });
    } catch {
      // If directory creation fails, fall back to userData root.
      const fallback = this.app.getPath("userData");
      this.logDir = path.join(fallback, appNameSafe);
      try {
        fs.mkdirSync(this.logDir, { recursive: true });
      } catch {
        // Last resort: keep logs in userData root.
        this.logDir = fallback;
      }
    }

    // Keep a "latest" file for quick access, plus a session file per start for full history.
    this.latestLogPath = path.join(this.logDir, "apex-api-latest.log");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    this.sessionLogPath = path.join(this.logDir, `apex-api-${stamp}.log`);
  }

  private appendSupervisorLog(line: string): void {
    // Best-effort: never crash the app due to logging.
    const msg = line.endsWith("\n") ? line : `${line}\n`;
    const latest = this.latestLogPath;
    const session = this.sessionLogPath;
    if (latest) {
      try {
        fs.appendFileSync(latest, msg, { encoding: "utf-8" });
      } catch {
        // ignore
      }
    }
    if (session) {
      try {
        fs.appendFileSync(session, msg, { encoding: "utf-8" });
      } catch {
        // ignore
      }
    }
  }

  private markExpectedExit(pid: number | null, reason: string): void {
    if (!pid || !Number.isFinite(pid) || pid <= 0) return;
    this.expectedExitPid = pid;
    this.expectedExitUntilMs = Date.now() + EXPECTED_EXIT_WINDOW_MS;
    this.appendSupervisorLog(
      `[PythonProcess] Marking PID=${pid} as expected-to-exit (${reason}) for ${EXPECTED_EXIT_WINDOW_MS}ms`,
    );
  }

  private isExpectedExit(pid: number | null): boolean {
    if (!pid || !Number.isFinite(pid) || pid <= 0) return false;
    if (this.expectedExitPid !== pid) return false;
    if (Date.now() > this.expectedExitUntilMs) return false;
    return true;
  }

  private clearExpectedExit(): void {
    this.expectedExitPid = null;
    this.expectedExitUntilMs = 0;
  }

  private setupLifecycleHooks(): void {
    if (!this.app) return;

    // Stop Python when app is quitting
    this.app.on("before-quit", async (event) => {
      if (this.state.status === "running" && !this.isShuttingDown) {
        event.preventDefault();
        this.isShuttingDown = true;
        try {
          await this.stop();
        } finally {
          this.app?.quit();
        }
      }
    });

    // Handle window-all-closed
    // IMPORTANT: do NOT stop the backend on window-all-closed by default.
    // The Python process must be monitored across the whole application lifetime,
    // not just a specific window (e.g. launcher can close while app continues).
    if (this.config.stopOnAllWindowsClosed) {
      this.app.on("window-all-closed", async () => {
        await this.stop();
      });
    }

    // Cleanup on unexpected exit
    process.on("exit", () => {
      this.forceKill();
    });

    process.on("SIGINT", async () => {
      await this.stop();
      process.exit(0);
    });

    process.on("SIGTERM", async () => {
      await this.stop();
      process.exit(0);
    });
  }

  async start(): Promise<void> {
    return await this.enqueue(() => this.startInternal());
  }

  private async startInternal(): Promise<void> {
    // The moment start is requested, we consider the backend "desired".
    // This allows fast auto-restart if it crashes during startup.
    this.desiredRunning = true;

    if (this.state.status === "running") {
      console.log("Python API is already running");
      return;
    }

    if (this.state.status === "starting") {
      console.log("Python API is already starting");
      return;
    }

    this.state.status = "starting";
    this.emit("status", this.state);

    // In production, we may be using either:
    // - a python-api shipped inside app resources, OR
    // - a user-installed server bundle path (settings apiPath) downloaded via the installer.
    // Resolve the best candidate early so our existence check is correct.
    const isPackaged = this.app?.isPackaged ?? false;
    if (isPackaged && !this.config.devMode) {
      try {
        const settings = getSettingsModule();
        const installedApiPath = settings.getApiPath();
        const installedBundleRoot = installedApiPath
          ? this.resolveInstalledBundleRoot(installedApiPath)
          : null;
        const installedPythonExe = installedBundleRoot
          ? this.resolveInstalledPythonExe(installedBundleRoot)
          : null;
        if (installedBundleRoot && installedPythonExe) {
          this.bundledBundleRoot = installedBundleRoot;
          this.bundledPythonPath = installedPythonExe;
        }
      } catch {
        // ignore; fall back to app-bundled resources path
      }
    }

    if (!this.config.devMode && !fs.existsSync(this.bundledPythonPath)) {
      this.state.status = "error";
      this.state.error = `Python runtime not found: ${this.bundledPythonPath}`;
      const err = new Error(this.state.error);
      this.emit("error", err);
      throw err;
    }

    try {
      await this.spawnProcess();
      await this.waitForHealthy();
      this.startHealthChecks();

      this.state.status = "running";
      this.state.error = undefined;
      this.emit("status", this.state);
      console.log(`Python API started on ${this.state.host}:${this.state.port}`);
    } catch (error) {
      // If startup fails, keep desiredRunning=true so the watchdog can retry.
      this.state.status = "error";
      this.state.error = error instanceof Error ? error.message : "Unknown error";
      this.emit("error", error);
      // If the process died very quickly during startup, schedule a restart.
      this.scheduleRestart("startup failure");
      throw error;
    }
  }

  private async spawnProcess(): Promise<void> {
    const isPackaged = this.app?.isPackaged ?? false;
    const mirrorPythonLogsToStdout = !isPackaged || this.config.devMode;

    // Prepare log file
    if (!this.latestLogPath || !this.sessionLogPath) {
      this.resolveAndPrepareLogPaths();
    }

    const latestStream = fs.createWriteStream(this.latestLogPath, { flags: "a" });
    const sessionStream = fs.createWriteStream(this.sessionLogPath, { flags: "a" });

    let logClosed = false;
    const canWrite = (s: fs.WriteStream) =>
      !logClosed && !s.destroyed && !s.writableEnded && !s.writableFinished;

    const writeLog = (msg: string) => {
      if (canWrite(latestStream)) {
        try {
          latestStream.write(msg);
        } catch {
          // ignore
        }
      }
      if (canWrite(sessionStream)) {
        try {
          sessionStream.write(msg);
        } catch {
          // ignore
        }
      }
    };

    // Never crash the app due to logging failures.
    latestStream.on("error", () => {});
    sessionStream.on("error", () => {});

    writeLog(`\n\n=== Starting Python API at ${new Date().toISOString()} ===\n`);
    // Ensure the target port is free before spawning. This prevents "address already in use"
    // situations when a previous backend crashed or a stale process is still listening.
    await this.freePortIfOccupied(this.config.port!, (msg) => {
      writeLog(`${msg}\n`);
      if (mirrorPythonLogsToStdout) process.stdout.write(`${msg}\n`);
    });

    let cmd: string;
    let args: string[];

      // Production: prefer user-installed server bundle (settings apiPath) if present;
      // otherwise fall back to the app-bundled python-api.
      const settings = getSettingsModule();
      const installedApiPath = settings.getApiPath();
      const installedBundleRoot = installedApiPath
        ? this.resolveInstalledBundleRoot(installedApiPath)
        : null;
      const installedPythonExe = installedBundleRoot
        ? this.resolveInstalledPythonExe(installedBundleRoot)
        : null;

      const useInstalled = Boolean(installedBundleRoot && installedPythonExe);
      const bundleRoot = useInstalled ? installedBundleRoot! : this.bundledBundleRoot;
      const pythonExe = useInstalled ? installedPythonExe! : this.bundledPythonPath;

      // Persist for env builder + logs
      this.bundledBundleRoot = bundleRoot;
      this.bundledPythonPath = pythonExe;

      cmd = pythonExe;
      args = [
        "-m",
        "src",
        // Avoid Procfile/Honcho indirection; run the API server directly so the
        // Electron-spawned process remains the true owner of the server tree.
        "serve",
      ];
    
    const env = this.buildEnvironment();

    this.process = spawn(cmd, args, {
      env,
      // Always run from the resolved bundle root so `-m src ...` resolves.
      cwd: this.bundledBundleRoot || undefined,
      stdio: ["ignore", "pipe", "pipe"],
      // Run in its own process group on Unix so we can reliably kill the whole tree.
      detached: process.platform !== "win32",
    });

    this.state.pid = this.process.pid;
    this.lastSpawnedPid = this.process.pid ?? null;
    const spawnedPid = this.process.pid ?? null;

    writeLog(
      `[PythonProcess] Spawned PID=${this.process.pid} cmd=${cmd} args=${JSON.stringify(
        args,
      )} cwd=${this.bundledBundleRoot || ""}\n`,
    );

    // Handle stdout
    const onStdout = (data: Buffer) => {
      const text = data.toString();
      writeLog(text);
      if (mirrorPythonLogsToStdout) process.stdout.write(text);
      this.emit("log", { type: "stdout", data: text });
    };
    this.process.stdout?.on("data", onStdout);

    // Handle stderr
    const onStderr = (data: Buffer) => {
      const text = data.toString();
      writeLog(text);
      if (mirrorPythonLogsToStdout) process.stderr.write(text);
      this.emit("log", { type: "stderr", data: text });
    };
    this.process.stderr?.on("data", onStderr);

    // Handle exit
    this.process.on("exit", (code, signal) => {
      writeLog(`\n=== Process exited with code ${code}, signal ${signal} ===\n`);
      // Prevent any late stdout/stderr chunks from trying to write after streams close.
      logClosed = true;
      try {
        this.process?.stdout?.off("data", onStdout);
      } catch {
        // ignore
      }
      try {
        this.process?.stderr?.off("data", onStderr);
      } catch {
        // ignore
      }
      try {
        latestStream.end();
      } catch {
        // ignore
      }
      try {
        sessionStream.end();
      } catch {
        // ignore
      }
      // Ensure our state transitions even if exit happens during startup.
      this.state.status = "stopped";
      this.state.pid = undefined;
      this.emit("status", this.state);
      // The process is gone; drop our handle so stop()/start() don't reason on a stale object.
      this.process = null;

      // Fast-restart if the backend is supposed to be running.
      this.handleProcessExit(spawnedPid, code, signal);
    });

    // Handle errors
    this.process.on("error", (error) => {
      writeLog(`\n=== Process error: ${error.message} ===\n`);
      this.state.status = "error";
      this.state.error = error.message;
      this.emit("error", error);
    });
  }

  /**
   * If anything is currently LISTENING on `port`, attempt to terminate it.
   * This is intentionally best-effort and time-bounded; failures won't prevent
   * the spawn attempt (but will likely surface as "port already in use").
   */
  private async freePortIfOccupied(
    port: number,
    log?: (msg: string) => void,
  ): Promise<void> {
    const logger = log ?? (() => {});
    if (!Number.isFinite(port) || port <= 0) return;

    const execAsync = promisify(exec);

    const listListeningPids = async (): Promise<Set<number>> => {
      const pids = new Set<number>();
      if (process.platform === "win32") {
        // Example:
        // TCP    0.0.0.0:8765           0.0.0.0:0              LISTENING       1234
        // TCP    [::]:8765              [::]:0                 LISTENING       1234
        try {
          const { stdout } = await execAsync(`netstat -ano -p tcp`);
          for (const line of stdout.split(/\r?\n/)) {
            if (!line.includes(`:${port}`)) continue;
            if (!/LISTENING/i.test(line)) continue;
            const parts = line.trim().split(/\s+/);
            const pid = Number(parts[parts.length - 1]);
            if (Number.isFinite(pid) && pid > 0 && pid !== process.pid) pids.add(pid);
          }
        } catch {
          // ignore
        }
        return pids;
      }

      // macOS/Linux: lsof is the simplest reliable way to find the LISTEN pid(s).
      try {
        const { stdout } = await execAsync(
          `lsof -nP -iTCP:${port} -sTCP:LISTEN -t`,
        );
        for (const line of stdout.split(/\r?\n/)) {
          const pid = Number(line.trim());
          if (Number.isFinite(pid) && pid > 0 && pid !== process.pid) pids.add(pid);
        }
      } catch {
        // No listeners (or lsof not available). Ignore.
      }
      return pids;
    };

    const terminatePid = async (pid: number, force: boolean): Promise<void> => {
      if (!Number.isFinite(pid) || pid <= 0) return;
      if (pid === process.pid) return;
      try {
        if (process.platform === "win32") {
          // /F if force; without /F is effectively a polite request.
          await execAsync(`taskkill /pid ${pid} /T ${force ? "/F" : ""}`.trim());
          return;
        }
        process.kill(pid, force ? "SIGKILL" : "SIGTERM");
      } catch {
        // ignore
      }
    };

    try {
      const found = await listListeningPids();
      if (!found.size) return;

      logger(
        `[PythonProcess] Port ${port} is in use by PID(s): ${Array.from(found).join(", ")}. Attempting to terminate...`,
      );

      // First pass: SIGTERM / graceful kill.
      for (const pid of found) await terminatePid(pid, false);
      await this.sleep(500);

      // Second pass: if still listening, force kill.
      const still = await listListeningPids();
      if (!still.size) {
        logger(`[PythonProcess] Port ${port} freed successfully.`);
        return;
      }

      logger(
        `[PythonProcess] Port ${port} still in use by PID(s): ${Array.from(still).join(", ")}. Forcing termination...`,
      );
      for (const pid of still) await terminatePid(pid, true);
      await this.sleep(500);
    } catch (e) {
      // Best-effort: don't block spawn on cleanup failures.
      const msg =
        e instanceof Error
          ? `[PythonProcess] Failed to free port ${port}: ${e.message}`
          : `[PythonProcess] Failed to free port ${port}`;
      logger(msg);
    }
  }

  private buildEnvironment(): NodeJS.ProcessEnv {
    const env = { ...process.env };
    
    // Set API configuration
    env.APEX_HOST = this.config.host;
    env.APEX_PORT = String(this.config.port);
    // Ownership: allow the server to self-terminate if the Electron parent dies.
    env.APEX_PARENT_PID = String(process.pid);
    
    // Set Python paths for bundled deployment
    if (this.app?.isPackaged) {
      // The bundle root contains `src/`, `assets/`, etc.
      const bundleRoot = this.bundledBundleRoot;
      env.PYTHONPATH = bundleRoot;
      
      // Platform-specific library paths
      if (process.platform === "darwin") {
        // Don't set PYTHONHOME for venv Python; it can break venv isolation.
        // Leave DYLD_LIBRARY_PATH untouched unless you have a specific need.
      } else if (process.platform === "linux") {
        // Leave LD_LIBRARY_PATH untouched unless you have a specific need.
      }
    }

    // Disable Python buffering for real-time logs
    env.PYTHONUNBUFFERED = "1";

    // GPU configuration
    if (process.platform === "darwin") {
      // Enable Metal/MPS for macOS
      env.PYTORCH_ENABLE_MPS_FALLBACK = "1";
    }

    return env;
  }

  private async waitForHealthy(): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < MAX_STARTUP_WAIT_MS) {
      if (await this.checkHealth()) {
        return;
      }
      await this.sleep(500);
    }

    throw new Error(`Python API did not become healthy within ${MAX_STARTUP_WAIT_MS}ms`);
  }

  private async checkHealth(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);

      const response = await fetch(
        `http://${this.config.host}:${this.config.port}/health`,
        { 
          method: "GET",
          signal: controller.signal,
        }
      );

      clearTimeout(timeoutId);
      this.state.lastHealthCheck = new Date();
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Public health check used by IPC and launcher gating.
   * Ensures we never report "healthy" if the backend runtime/code is missing.
   */
  public async checkApiHealth(): Promise<boolean> {
    const v = await this.verifyRuntime();
    if (!v.ok) return false;
    return await this.checkHealth();
  }

  private startHealthChecks(): void {
    this.stopHealthChecks();

    this.healthCheckInterval = setInterval(async () => {
      if (this.state.status !== "running") return;
      if (!this.desiredRunning) return;

      // Strong check: includes runtime verification so we don't "monitor" a missing backend.
      const healthy = await this.checkApiHealth();
      if (healthy) {
        this.consecutiveHealthFailures = 0;
        return;
      }

      console.warn("Python API health check failed (unhealthy or unreachable)");
      this.emit("unhealthy");
      this.consecutiveHealthFailures++;
      this.appendSupervisorLog(
        `[PythonProcess] Health failure ${this.consecutiveHealthFailures}/${HEALTH_FAILURES_BEFORE_RESTART}`,
      );
      if (this.consecutiveHealthFailures < HEALTH_FAILURES_BEFORE_RESTART) return;
      this.consecutiveHealthFailures = 0;
      this.scheduleRestart("health check failed");
    }, HEALTH_CHECK_INTERVAL_MS);
  }

  private stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  private handleUnexpectedExit(code: number | null, signal: NodeJS.Signals | null): void {
    console.warn(`Python API exited unexpectedly: code=${code}, signal=${signal}`);
    
    this.state.status = "stopped";
    this.state.pid = undefined;
    this.emit("status", this.state);

    // Attempt restart
    if (!this.desiredRunning) return;
    this.scheduleRestart("process exit");
  }

  async stop(): Promise<void> {
    return await this.enqueue(() =>
      this.stopInternal({ markUndesired: true, suppressAutoRestartOnExit: true }),
    );
  }

  private async stopInternal(opts: {
    markUndesired: boolean;
    suppressAutoRestartOnExit: boolean;
  }): Promise<void> {
    // Explicit stop means the backend is no longer desired; suppress auto-restarts.
    if (opts.markUndesired) this.desiredRunning = false;
    // If we're stopping intentionally, cancel any pending restart timer.
    if (opts.markUndesired) this.clearRestartTimer();
    this.consecutiveHealthFailures = 0;

    if (opts.suppressAutoRestartOnExit) {
      this.markExpectedExit(this.process?.pid ?? null, opts.markUndesired ? "stop" : "restart");
    }

    // Always stop checks even if state already says "stopped" (it might be stale).
    this.stopHealthChecks();

    if (this.state.status === "stopped") {
      // Still ensure we don't hold onto a dead process handle.
      if (this.process) {
        try {
          // If it is somehow still alive, request shutdown anyway.
          await this.gracefulShutdown();
        } catch {
          // ignore
        }
      }
      this.process = null;
      return;
    }

    this.state.status = "stopping";
    this.emit("status", this.state);

    if (this.process) {
      await this.gracefulShutdown();
    }

    this.state.status = "stopped";
    this.state.pid = undefined;
    this.state.restartCount = 0;
    this.emit("status", this.state);

    console.log("Python API stopped");
  }

  private async gracefulShutdown(): Promise<void> {
    if (!this.process) return;
    const pid = this.process.pid;
    const ownedPids = pid ? await this.collectDescendantPids(pid) : new Set<number>();
    if (pid) ownedPids.add(pid);

    // Try graceful shutdown first
    try {
      // Send shutdown request to API
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 5000);
      
      await fetch(`http://${this.config.host}:${this.config.port}/shutdown`, {
        method: "POST",
        signal: controller.signal,
      }).catch(() => {});
    } catch {
      // Ignore errors
    }

    // Send SIGTERM
    this.killProcessTree("SIGTERM");
    // Also SIGTERM all known descendants (ownership-safe backstop, in case process groups aren't shared).
    this.killPidSet(ownedPids, "SIGTERM");

    // Wait for process to exit
    const exitPromise = new Promise<void>((resolve) => {
      if (!this.process) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        // Force kill if not exited
        this.forceKill();
        resolve();
      }, 10000);

      this.process.once("exit", () => {
        clearTimeout(timeout);
        resolve();
      });
    });

    await exitPromise;
    this.process = null;
  }

  private forceKill(): void {
  
    if (!this.process) return;
    const pid = this.process.pid;
    // Capture descendants before killing the parent so we can still target them if they orphan.
    const ownedPidsPromise =
      pid ? this.collectDescendantPids(pid).then((s) => (s.add(pid), s)) : Promise.resolve(new Set<number>());

    try {
      if (process.platform === "win32") {
        // Windows: use taskkill
        exec(`taskkill /pid ${this.process.pid} /T /F`);
      } else {
        // Unix: kill the process group if possible, otherwise just the child process.
        this.killProcessTree("SIGKILL");
      }
    } catch (error) {
      console.error("Failed to force kill Python process:", error);
    }

    void ownedPidsPromise.then((owned) => this.killPidSet(owned, "SIGKILL"));
    this.process = null;
  }

  private killPidSet(pids: Set<number>, signal: NodeJS.Signals): void {
    for (const pid of pids) {
      if (!Number.isFinite(pid) || pid <= 0) continue;
      try {
        process.kill(pid, signal);
      } catch {
        // ignore
      }
    }
  }

  private async collectDescendantPids(rootPid: number): Promise<Set<number>> {
    const out = new Set<number>();
    if (!Number.isFinite(rootPid) || rootPid <= 0) return out;
    if (process.platform === "win32") return out; // taskkill /T handles tree on Windows.

    const execAsync = promisify(exec);
    const queue: number[] = [rootPid];
    while (queue.length) {
      const pid = queue.pop()!;
      try {
        const { stdout } = await execAsync(`pgrep -P ${pid}`);
        const children = stdout
          .split(/\r?\n/)
          .map((s) => Number(s.trim()))
          .filter((n) => Number.isFinite(n) && n > 0);
        for (const c of children) {
          if (!out.has(c)) {
            out.add(c);
            queue.push(c);
          }
        }
      } catch {
        // pgrep returns non-zero when there are no children; ignore
      }
    }
    return out;
  }

  private killProcessTree(signal: NodeJS.Signals): void {
    if (!this.process?.pid) return;
    const pid = this.process.pid;
    try {
      if (process.platform === "win32") {
        // Windows: use taskkill to terminate the full tree.
        exec(`taskkill /pid ${pid} /T /F`);
        return;
      }
      // Unix: prefer killing the whole process group (requires detached: true)
      try {
        process.kill(-pid, signal);
      } catch {
        this.process.kill(signal);
      }
    } catch (error) {
      console.error(`Failed to kill Python process tree with ${signal}:`, error);
    }
  }

  async restart(): Promise<void> {
    return await this.enqueue(() => this.restartInternal());
  }

  private async restartInternal(): Promise<void> {
    // Restart is still "desired running".
    this.desiredRunning = true;
    this.clearRestartTimer();
    this.state.restartCount++;

    // IMPORTANT: call internal stop/start to avoid queue re-entrancy deadlocks.
    await this.stopInternal({ markUndesired: false, suppressAutoRestartOnExit: true });
    await this.sleep(RESTART_DELAY_MS);
    await this.startInternal();
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.opQueue.then(fn, fn);
    // Ensure the queue continues regardless of success/failure.
    this.opQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private handleProcessExit(
    pid: number | null,
    code: number | null,
    signal: NodeJS.Signals | null,
  ): void {
    // Ignore exits during intentional shutdown or when the backend isn't desired.
    if (this.isShuttingDown) return;
    if (!this.desiredRunning) return;
    if (this.isExpectedExit(pid)) {
      this.appendSupervisorLog(
        `[PythonProcess] Observed expected exit for PID=${pid} (code=${code}, signal=${signal})`,
      );
      this.clearExpectedExit();
      return;
    }
    this.handleUnexpectedExit(code, signal);
  }

  private scheduleRestart(reason: string): void {
    if (this.isShuttingDown) return;
    if (!this.desiredRunning) return;

    // If we're already starting, let that attempt complete.
    if (this.state.status === "starting") return;

    // Avoid scheduling overlapping restarts (health check + exit can both trigger).
    if (this.restartTimer) return;

    console.log(`[PythonProcess] Scheduling restart: ${reason}`);
    this.appendSupervisorLog(`[PythonProcess] Scheduling restart: ${reason}`);
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      // Use restart() so we always cleanly tear down any partial tree.
      this.restart().catch((err) => {
        console.error("Failed to restart Python API:", err);
        this.appendSupervisorLog(
          `[PythonProcess] Restart failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      });
    }, RESTART_DELAY_MS);
  }

  private clearRestartTimer(): void {
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  getState(): ProcessState {
    return { ...this.state };
  }

  getApiUrl(): string {
    return `http://${this.config.host}:${this.config.port}`;
  }
}

// Singleton instance
let pythonProcessInstance: PythonProcessManager | null = null;

export function pythonProcess(config?: PythonProcessConfig): PythonProcessManager {
  if (!pythonProcessInstance) {
    pythonProcessInstance = new PythonProcessManager(config);
  }
  return pythonProcessInstance;
}

export function getPythonProcess(): PythonProcessManager | null {
  return pythonProcessInstance;
}

