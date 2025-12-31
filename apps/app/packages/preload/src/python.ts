/**
 * Python Process Management Preload API
 * 
 * Provides IPC bridge for managing the bundled Python API process
 * from the renderer process.
 */

import { ipcRenderer } from "electron";

export interface PythonProcessState {
  status: "stopped" | "starting" | "running" | "stopping" | "error";
  pid?: number;
  port: number;
  host: string;
  error?: string;
  restartCount: number;
  lastHealthCheck?: Date;
}

export interface PythonResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

/**
 * Start the bundled Python API server
 */
export async function startPythonApi(): Promise<PythonResponse<PythonProcessState>> {
  return ipcRenderer.invoke("python:start");
}

/**
 * Stop the Python API server
 */
export async function stopPythonApi(): Promise<PythonResponse<PythonProcessState>> {
  return ipcRenderer.invoke("python:stop");
}

/**
 * Restart the Python API server
 */
export async function restartPythonApi(): Promise<PythonResponse<PythonProcessState>> {
  return ipcRenderer.invoke("python:restart");
}

/**
 * Get the current status of the Python API server
 */
export async function getPythonStatus(): Promise<PythonResponse<PythonProcessState>> {
  return ipcRenderer.invoke("python:status");
}

/**
 * Check if the Python API is healthy and responding
 */
export async function checkPythonHealth(): Promise<PythonResponse<{ healthy: boolean; state: PythonProcessState }>> {
  return ipcRenderer.invoke("python:health");
}

/**
 * Get recent Python API logs
 */
export async function getPythonLogs(): Promise<PythonResponse<{ logs: string }>> {
  return ipcRenderer.invoke("python:logs");
}

/**
 * Subscribe to Python process status changes
 * 
 * @param callback - Function to call when status changes
 * @returns Unsubscribe function
 */
export function onPythonStatusChange(callback: (state: PythonProcessState) => void): () => void {
  const handler = (_event: Electron.IpcRendererEvent, state: PythonProcessState) => {
    callback(state);
  };
  
  ipcRenderer.on("python:status-changed", handler);
  
  return () => {
    ipcRenderer.off("python:status-changed", handler);
  };
}

/**
 * Subscribe to Python process log output
 * 
 * @param callback - Function to call when new log data is available
 * @returns Unsubscribe function
 */
export function onPythonLog(callback: (log: { type: "stdout" | "stderr"; data: string }) => void): () => void {
  const handler = (_event: Electron.IpcRendererEvent, log: { type: "stdout" | "stderr"; data: string }) => {
    callback(log);
  };
  
  ipcRenderer.on("python:log", handler);
  
  return () => {
    ipcRenderer.off("python:log", handler);
  };
}

/**
 * Subscribe to Python process errors
 * 
 * @param callback - Function to call when an error occurs
 * @returns Unsubscribe function
 */
export function onPythonError(callback: (error: Error) => void): () => void {
  const handler = (_event: Electron.IpcRendererEvent, error: Error) => {
    callback(error);
  };
  
  ipcRenderer.on("python:error", handler);
  
  return () => {
    ipcRenderer.off("python:error", handler);
  };
}

/**
 * Check if the app is running in bundled/production mode
 * (bundled Python API should be available)
 */
export async function isBundledMode(): Promise<boolean> {
  const status = await getPythonStatus();
  // In dev mode, the Python process module is disabled
  // So if we get a valid status, we're in bundled mode
  return status.success;
}

