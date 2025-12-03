import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

// Projects API functions
async function listProjects<T = any>(): Promise<ConfigResponse<T[]>> {
  return await ipcRenderer.invoke("projects:list-json");
}

async function createProject<T = any>(payload: {
  name: string;
  fps: number;
}): Promise<ConfigResponse<T>> {
  return await ipcRenderer.invoke("projects:create-json", payload);
}

async function deleteProject(
  projectId: string | number,
): Promise<ConfigResponse<{ id: string | number }>> {
  return await ipcRenderer.invoke("projects:delete-json", projectId);
}

async function updateProjectAspectRatio(
  projectId: number,
  aspectRatioWidth: number,
  aspectRatioHeight: number,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke(
    "projects:update-aspect-ratio",
    projectId,
    aspectRatioWidth,
    aspectRatioHeight,
  );
}

async function saveProjectJson(
  projectId: string | number,
  data: any,
): Promise<ConfigResponse<{ path: string }>> {
  return await ipcRenderer.invoke("projects:save-json", projectId, data);
}

async function loadProjectJson(
  projectId: string | number,
): Promise<ConfigResponse<any | null>> {
  return await ipcRenderer.invoke("projects:load-json", projectId);
}

// Timelines API functions
async function createTimeline(payload: {
  projectId: number;
  timelineId: string;
  type: string;
  sortOrder?: number;
  timeline_width?: number;
  timelineWidth?: number;
  timelineY: number;
  timelineHeight: number;
  timelinePadding?: number;
  muted?: boolean;
  hidden?: boolean;
}): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("timelines:create", payload);
}

async function listTimelines(
  projectId: number,
): Promise<ConfigResponse<any[]>> {
  return await ipcRenderer.invoke("timelines:list", projectId);
}

async function deleteTimeline(
  timelineId: number | string,
): Promise<ConfigResponse<{ id: number }>> {
  return await ipcRenderer.invoke("timelines:delete", timelineId);
}

async function updateTimeline(
  timelineId: number | string,
  partial: Record<string, unknown>,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("timelines:update", timelineId, partial);
}

// Clips API functions
async function listClips(
  projectId: number,
): Promise<
  ConfigResponse<
    {
      clip: any;
      masks: any[];
      preprocessors: any[];
    }[]
  >
> {
  return await ipcRenderer.invoke("clips:list", projectId);
}

async function createClip(payload: {
  timelineId: number;
  type: string;
  groupId?: string | null;
  startTick: number | string | bigint;
  endTick: number | string | bigint;
  trimStartTick: number;
  trimEndTick: number;
  clipPadding?: number;
  width: number;
  height: number;
  transform?: unknown | null;
  originalTransform?: unknown | null;
  props?: unknown | null;
}): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("clips:create", payload);
}

async function updateClip(
  clipId: number | string,
  partial: Record<string, unknown>,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("clips:update", clipId, partial);
}

async function updateClipsBatch(
  updates: Array<{
    clipId: number | string;
    partial: Record<string, unknown>;
  }>,
): Promise<ConfigResponse<any[]>> {
  return await ipcRenderer.invoke("clips:update-batch", updates);
}

async function deleteClip(
  clipId: number | string,
): Promise<ConfigResponse<{ id: number }>> {
  return await ipcRenderer.invoke("clips:delete", clipId);
}

// Preprocessor clip API functions
async function createPreprocessorClip(payload: {
  id: string;
  clipId: number;
  src?: string | null;
  preprocessor: string;
  startFrame?: number | null;
  endFrame?: number | null;
  values: Record<string, unknown>;
  status?: "running" | "complete" | "failed" | null;
  activeJobId?: string | null;
  jobIds?: string[] | null;
}): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor-clips:create", payload);
}

async function updatePreprocessorClip(
  id: string,
  partial: Record<string, unknown>,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor-clips:update", id, partial);
}

async function deletePreprocessorClip(
  id: string,
): Promise<ConfigResponse<{ id: string }>> {
  return await ipcRenderer.invoke("preprocessor-clips:delete", id);
}

// Mask clip API functions
async function createMaskClip(payload: {
  id: string;
  clipId: number;
  tool: string;
  featherAmount: number;
  brushSize?: number | null;
  keyframes: Record<string | number, unknown>;
  isTracked?: boolean;
  trackingDirection?: string | null;
  confidenceThreshold?: number | null;
  transform?: unknown | null;
  inverted?: boolean;
  backgroundColor?: string | null;
  backgroundOpacity?: number | null;
  backgroundColorEnabled?: boolean;
  maskColor?: string | null;
  maskOpacity?: number | null;
  maskColorEnabled?: boolean;
  maxTrackingFrames?: number | null;
}): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("mask-clips:create", payload);
}

async function updateMaskClip(
  id: string,
  partial: Record<string, unknown>,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("mask-clips:update", id, partial);
}

async function deleteMaskClip(
  id: string,
): Promise<ConfigResponse<{ id: string }>> {
  return await ipcRenderer.invoke("mask-clips:delete", id);
}

export {
  listProjects,
  createProject,
  deleteProject,
  updateProjectAspectRatio,
  saveProjectJson,
  loadProjectJson,
  createTimeline,
  listTimelines,
  deleteTimeline,
  updateTimeline,
  listClips,
  createClip,
  updateClip,
  updateClipsBatch,
  deleteClip,
  createPreprocessorClip,
  updatePreprocessorClip,
  deletePreprocessorClip,
  createMaskClip,
  updateMaskClip,
  deleteMaskClip,
};


