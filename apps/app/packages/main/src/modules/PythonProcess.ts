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
const HEALTH_CHECK_INTERVAL_MS = 5000;
const HEALTH_CHECK_TIMEOUT_MS = 3000;
const MAX_STARTUP_WAIT_MS = 60000;
const MAX_RESTART_ATTEMPTS = 3;
const RESTART_DELAY_MS = 2000;

interface PythonProcessConfig {
  port?: number;
  host?: string;
  autoStart?: boolean;
  devMode?: boolean;
  pythonPath?: string;
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
  private logPath: string = "";
  private bundledPythonPath: string = "";
  private bundledBundleRoot: string = "";
  private isShuttingDown: boolean = false;
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

    // Set up paths
    const userData = this.app.getPath("userData");
    this.logPath = path.join(userData, "apex-api.log");

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
          "import importlib; importlib.import_module('apex_engine'); print('ok')",
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

    const isPackaged = this.app?.isPackaged ?? false;

    // Dev: assume available (system python / venv).
    if (this.config.devMode || !isPackaged) {
      return {
        available: true,
        mode: "dev",
        installedApiPath,
        bundleRoot: null,
        pythonExe: this.bundledPythonPath || null,
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
        if (fs.existsSync(this.logPath)) {
          const logs = fs.readFileSync(this.logPath, "utf-8");
          // Return last 1000 lines
          const lines = logs.split("\n").slice(-1000).join("\n");
          return { success: true, data: { logs: lines } };
        }
        return { success: true, data: { logs: "" } };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : "Failed to read logs",
        };
      }
    });
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
    this.app.on("window-all-closed", async () => {
      // On macOS the app may remain running after the last window closes.
      // The launcher owns the API lifecycle, so if there are no windows, stop the API too.
      await this.stop();
    });

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
      this.state.status = "error";
      this.state.error = error instanceof Error ? error.message : "Unknown error";
      this.emit("error", error);
      throw error;
    }
  }

  private async spawnProcess(): Promise<void> {
    const isPackaged = this.app?.isPackaged ?? false;

    // Prepare log file
    const logStream = fs.createWriteStream(this.logPath, { flags: "a" });
    logStream.write(`\n\n=== Starting Python API at ${new Date().toISOString()} ===\n`);

    let cmd: string;
    let args: string[];

    if (isPackaged) {
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
    } else {
      // Development: run via Python module
      const apiPath = this.#resolveDevApiPath();
      cmd = this.bundledPythonPath;
      args = ["-m", "src", "serve"];
      // Ensure the Python process runs from the backend repo root (contains `src/`)
      // so module imports resolve consistently in dev.
      this.bundledBundleRoot = apiPath;
    }

    console.log(`Spawning: ${cmd} ${args.join(" ")}`);

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

    // Handle stdout
    this.process.stdout?.on("data", (data) => {
      const text = data.toString();
      logStream.write(text);
      this.emit("log", { type: "stdout", data: text });
    });

    // Handle stderr
    this.process.stderr?.on("data", (data) => {
      const text = data.toString();
      logStream.write(text);
      this.emit("log", { type: "stderr", data: text });
    });

    // Handle exit
    this.process.on("exit", (code, signal) => {
      logStream.write(`\n=== Process exited with code ${code}, signal ${signal} ===\n`);
      logStream.end();

      if (!this.isShuttingDown && this.state.status === "running") {
        this.handleUnexpectedExit(code, signal);
      } else {
        this.state.status = "stopped";
        this.state.pid = undefined;
        this.emit("status", this.state);
      }
    });

    // Handle errors
    this.process.on("error", (error) => {
      logStream.write(`\n=== Process error: ${error.message} ===\n`);
      this.state.status = "error";
      this.state.error = error.message;
      this.emit("error", error);
    });
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

      const healthy = await this.checkHealth();
      
      if (!healthy) {
        console.warn("Python API health check failed");
        this.emit("unhealthy");
        
        // Attempt restart if configured
        if (this.state.restartCount < MAX_RESTART_ATTEMPTS) {
          console.log(`Attempting restart (${this.state.restartCount + 1}/${MAX_RESTART_ATTEMPTS})`);
          await this.restart();
        } else {
          this.state.status = "error";
          this.state.error = "API became unhealthy and max restart attempts exceeded";
          this.emit("error", new Error(this.state.error));
        }
      }
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
    if (this.state.restartCount < MAX_RESTART_ATTEMPTS) {
      this.state.restartCount++;
      console.log(`Attempting restart (${this.state.restartCount}/${MAX_RESTART_ATTEMPTS})`);
      
      setTimeout(() => {
        this.start().catch((err) => {
          console.error("Failed to restart Python API:", err);
        });
      }, RESTART_DELAY_MS);
    } else {
      this.state.status = "error";
      this.state.error = "Process exited unexpectedly and max restart attempts exceeded";
      this.emit("error", new Error(this.state.error));
    }
  }

  async stop(): Promise<void> {
    if (this.state.status === "stopped") {
      return;
    }

    this.state.status = "stopping";
    this.emit("status", this.state);

    this.stopHealthChecks();

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
    this.state.restartCount++;
    await this.stop();
    await this.sleep(RESTART_DELAY_MS);
    await this.start();
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

