import { ipcRenderer } from "electron";
import { fileURLToPath } from "node:url";
import type { ConfigResponse } from "./types.js";

async function createMask(request: {
  input_path: string;
  frame_number?: number;
  tool: string;
  points?: Array<{ x: number; y: number }>;
  point_labels?: Array<number>;
  box?: { x1: number; y1: number; x2: number; y2: number };
  multimask_output?: boolean;
  simplify_tolerance?: number;
  model_type?: string;
  id?: string;
}): Promise<
  ConfigResponse<{
    status: string;
    contours?: Array<Array<number>>;
    message?: string;
  }>
> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke("mask:create", request);
}

type MaskTrackStreamEvent =
  | { frame_number: number; contours: Array<Array<number>> }
  | { status: "error"; error: string };

async function trackMask(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
}): Promise<ReadableStream<MaskTrackStreamEvent>> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  const { streamId } = (await ipcRenderer.invoke("mask:track", request)) as {
    streamId: string;
  };
  const parsedStream = new ReadableStream<MaskTrackStreamEvent>({
    start(controller) {
      const onChunk = (_e: unknown, data: any) => {
        controller.enqueue(data as MaskTrackStreamEvent);
      };
      const onError = (_e: unknown, err: any) => {
        controller.enqueue({
          status: "error",
          error: err?.message || "Unknown error",
        });
      };
      const onEnd = () => {
        controller.close();
        ipcRenderer.removeAllListeners(`mask:track:chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track:error:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track:end:${streamId}`);
      };
      ipcRenderer.on(`mask:track:chunk:${streamId}`, onChunk);
      ipcRenderer.on(`mask:track:error:${streamId}`, onError);
      ipcRenderer.on(`mask:track:end:${streamId}`, onEnd);
    },
    cancel() {
      ipcRenderer.invoke("mask:track:cancel", streamId).catch(() => {});
      ipcRenderer.removeAllListeners(`mask:track:chunk:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track:error:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track:end:${streamId}`);
    },
  });

  return parsedStream;
}

async function startMaskTrack(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
}): Promise<{ streamId: string }> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }

  return await ipcRenderer.invoke("mask:track", request);
}

function onMaskTrackChunk(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`mask:track:chunk:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track:chunk:${streamId}`, listener);
}

function onMaskTrackError(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`mask:track:error:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track:error:${streamId}`, listener);
}

function onMaskTrackEnd(streamId: string, callback: () => void): () => void {
  const listener = () => callback();
  ipcRenderer.on(`mask:track:end:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track:end:${streamId}`, listener);
}

async function cancelMaskTrack(streamId: string): Promise<void> {
  await ipcRenderer.invoke("mask:track:cancel", streamId);
}

type ShapeTrackStreamEvent =
  | { frame_number: number; shapeBounds?: any }
  | { status: "error"; error: string };

async function trackShapes(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
}): Promise<ReadableStream<ShapeTrackStreamEvent>> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  const { streamId } = (await ipcRenderer.invoke(
    "mask:track-shapes",
    request,
  )) as { streamId: string };
  const parsedStream = new ReadableStream<ShapeTrackStreamEvent>({
    start(controller) {
      const onChunk = (_e: unknown, data: any) => {
        controller.enqueue(data as ShapeTrackStreamEvent);
      };
      const onError = (_e: unknown, err: any) => {
        controller.enqueue({
          status: "error",
          error: err?.message || "Unknown error",
        });
      };
      const onEnd = () => {
        controller.close();
        ipcRenderer.removeAllListeners(`mask:track-shapes:chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track-shapes:error:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track-shapes:end:${streamId}`);
      };
      ipcRenderer.on(`mask:track-shapes:chunk:${streamId}`, onChunk);
      ipcRenderer.on(`mask:track-shapes:error:${streamId}`, onError);
      ipcRenderer.on(`mask:track-shapes:end:${streamId}`, onEnd);
    },
    cancel() {
      ipcRenderer.invoke("mask:track:cancel", streamId).catch(() => {});
      ipcRenderer.removeAllListeners(`mask:track-shapes:chunk:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track-shapes:error:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track-shapes:end:${streamId}`);
    },
  });
  return parsedStream;
}

async function startMaskTrackShapes(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
  shape_type?: string;
}): Promise<{ streamId: string }> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke("mask:track-shapes", request);
}

function onMaskTrackShapesChunk(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`mask:track-shapes:chunk:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track-shapes:chunk:${streamId}`, listener);
}

function onMaskTrackShapesError(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`mask:track-shapes:error:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track-shapes:error:${streamId}`, listener);
}

function onMaskTrackShapesEnd(
  streamId: string,
  callback: () => void,
): () => void {
  const listener = () => callback();
  ipcRenderer.on(`mask:track-shapes:end:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track-shapes:end:${streamId}`, listener);
}

export {
  createMask,
  trackMask,
  startMaskTrack,
  onMaskTrackChunk,
  onMaskTrackError,
  onMaskTrackEnd,
  cancelMaskTrack,
  trackShapes,
  startMaskTrackShapes,
  onMaskTrackShapesChunk,
  onMaskTrackShapesError,
  onMaskTrackShapesEnd,
};


