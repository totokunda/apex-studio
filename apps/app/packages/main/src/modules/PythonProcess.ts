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
import { spawn, ChildProcess, exec } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { EventEmitter } from "node:events";
import { fileURLToPath } from "node:url";

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

export class PythonProcessManager extends EventEmitter implements AppModule {
  private app: App | null = null;
  private process: ChildProcess | null = null;
  private state: ProcessState;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private config: PythonProcessConfig;
  private logPath: string = "";
  private bundledPythonPath: string = "";
  private isShuttingDown: boolean = false;

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
        // macOS: Check for .app bundle or standalone
        const appBundle = path.join(resourcesPath, "python-api", "apex-engine.app", "Contents", "MacOS", "apex-engine");
        const standalone = path.join(resourcesPath, "python-api", "apex-engine", "apex-engine");
        return fs.existsSync(appBundle) ? appBundle : standalone;
      } else if (platform === "win32") {
        return path.join(resourcesPath, "python-api", "apex-engine", "apex-engine.exe");
      } else {
        return path.join(resourcesPath, "python-api", "apex-engine", "apex-engine");
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
      const healthy = await this.checkHealth();
      return { success: true, data: { healthy, state: this.state } };
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
      if (process.platform !== "darwin") {
        await this.stop();
      }
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

    // Check if bundled Python exists
    if (!this.config.devMode && !fs.existsSync(this.bundledPythonPath)) {
      // In production, if bundled Python doesn't exist, show error
      this.state.status = "error";
      this.state.error = `Bundled Python not found: ${this.bundledPythonPath}`;
      this.emit("error", new Error(this.state.error));
      return;
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
    const env = this.buildEnvironment();

    // Prepare log file
    const logStream = fs.createWriteStream(this.logPath, { flags: "a" });
    logStream.write(`\n\n=== Starting Python API at ${new Date().toISOString()} ===\n`);

    let cmd: string;
    let args: string[];

    if (isPackaged) {
      // Production: run bundled executable
      cmd = this.bundledPythonPath;
      args = ["start", "--daemon", "--port", String(this.config.port)];
    } else {
      // Development: run via Python module
      const apiPath = path.resolve(__dirname, "../../../../..", "api");
      cmd = this.bundledPythonPath;
      args = ["-m", "src", "start", "--cwd", apiPath];
    }

    console.log(`Spawning: ${cmd} ${args.join(" ")}`);

    this.process = spawn(cmd, args, {
      env,
      cwd: isPackaged ? path.dirname(this.bundledPythonPath) : undefined,
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
    });

    this.state.pid = this.process.pid;

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
    
    // Set Python paths for bundled deployment
    if (this.app?.isPackaged) {
      const bundleDir = path.dirname(this.bundledPythonPath);
      env.PYTHONPATH = bundleDir;
      env.PYTHONHOME = bundleDir;
      
      // Platform-specific library paths
      if (process.platform === "darwin") {
        env.DYLD_LIBRARY_PATH = path.join(bundleDir, "lib");
      } else if (process.platform === "linux") {
        env.LD_LIBRARY_PATH = path.join(bundleDir, "lib");
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
    this.process.kill("SIGTERM");

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

    try {
      if (process.platform === "win32") {
        // Windows: use taskkill
        exec(`taskkill /pid ${this.process.pid} /T /F`);
      } else {
        // Unix: send SIGKILL
        this.process.kill("SIGKILL");
      }
    } catch (error) {
      console.error("Failed to force kill Python process:", error);
    }

    this.process = null;
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

