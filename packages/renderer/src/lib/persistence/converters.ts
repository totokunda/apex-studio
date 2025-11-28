import { DEFAULT_FPS, TIMELINE_DURATION_SECONDS } from "../settings";
import type { AnyClipProps, TimelineProps, Point } from "../types";
import { useClipStore } from "../clip";
import { useControlsStore } from "../control";
import { useViewportStore } from "../viewport";
import { TICKS_PER_SECOND } from "./constants";
import type {
  PersistedClipState,
  PersistedControlState,
  PersistedViewportState,
  SerializedClip,
  SerializedMask,
  SerializedPreprocessor,
} from "./types";

const coerceNumber = (value: unknown, fallback = 0) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

export const framesToTicks = (
  frame: number,
  fps: number,
  tickRate: number = TICKS_PER_SECOND,
) =>
  Math.round(
    ((Number.isFinite(frame) ? frame : 0) / Math.max(1, fps)) * tickRate,
  );

export const ticksToFrames = (
  ticks: number,
  fps: number,
  tickRate: number = TICKS_PER_SECOND,
) =>
  Math.round(
    ((Number.isFinite(ticks) ? ticks : 0) / Math.max(1, tickRate)) *
      Math.max(1, fps),
  );

const normalizeMaskKeyframes = (mask: SerializedMask) => {
  if (!mask.keyframes) return undefined;
  const normalized: Record<number, any> = {};
  for (const [rawKey, value] of Object.entries(mask.keyframes)) {
    const keyNum = Number(rawKey);
    if (!Number.isNaN(keyNum)) {
      normalized[keyNum] = value;
    }
  }
  return normalized;
};

export const serializeClipState = (
  clipState: Pick<
    ReturnType<typeof useClipStore.getState>,
    "clips" | "timelines"
  >,
  fps: number,
  tickRate: number = TICKS_PER_SECOND,
): PersistedClipState => {
  const clips: SerializedClip[] = (clipState.clips || []).map((clip) => {
    const startTicks = framesToTicks(clip.startFrame ?? 0, fps, tickRate);
    const endTicks = framesToTicks(clip.endFrame ?? 0, fps, tickRate);
    const trimStartTicks =
      typeof clip.trimStart === "number"
        ? framesToTicks(clip.trimStart, fps, tickRate)
        : undefined;
    const trimEndTicks =
      typeof clip.trimEnd === "number"
        ? framesToTicks(clip.trimEnd, fps, tickRate)
        : undefined;

    const base: any = {
      ...clip,
    };
    delete base.startFrame;
    delete base.endFrame;
    delete base.trimStart;
    delete base.trimEnd;

    let preprocessors: SerializedPreprocessor[] | undefined;
    let masks: SerializedMask[] | undefined;

    if (clip.type === "video" || clip.type === "image") {
      preprocessors = (clip.preprocessors || []).map((p) => {
        const serialized: SerializedPreprocessor = {
          ...p,
          startTicks:
            typeof p.startFrame === "number"
              ? framesToTicks(p.startFrame, fps, tickRate)
              : undefined,
          endTicks:
            typeof p.endFrame === "number"
              ? framesToTicks(p.endFrame, fps, tickRate)
              : undefined,
        };
        delete (serialized as any).startFrame;
        delete (serialized as any).endFrame;
        return serialized;
      });

      masks = (clip.masks || []).map((m) => {
        const serialized: SerializedMask = {
          ...m,
          keyframes:
            m.keyframes instanceof Map
              ? Object.fromEntries(
                  Array.from(m.keyframes.entries()).map(([k, v]) => [
                    String(k),
                    v,
                  ]),
                )
              : (m.keyframes as any),
        };
        return serialized;
      });
    }

    return {
      ...base,
      startTicks,
      endTicks,
      trimStartTicks,
      trimEndTicks,
      preprocessors,
      masks,
    };
  });

  return {
    clips,
    timelines: clipState.timelines || [],
  };
};

export const deserializeClipState = (
  persisted: PersistedClipState | undefined,
  fps: number,
  tickRate: number = TICKS_PER_SECOND,
): { clips: AnyClipProps[]; timelines: TimelineProps[] } => {
  if (!persisted) return { clips: [], timelines: [] };
  const clips: AnyClipProps[] = (persisted.clips || []).map((clip) => {
    const base: any = {
      ...clip,
      startFrame: ticksToFrames(clip.startTicks ?? 0, fps, tickRate),
      endFrame: ticksToFrames(clip.endTicks ?? 0, fps, tickRate),
    };
    if (typeof clip.trimStartTicks === "number") {
      base.trimStart = ticksToFrames(clip.trimStartTicks, fps, tickRate);
    }
    if (typeof clip.trimEndTicks === "number") {
      base.trimEnd = ticksToFrames(clip.trimEndTicks, fps, tickRate);
    }

    delete base.startTicks;
    delete base.endTicks;
    delete base.trimStartTicks;
    delete base.trimEndTicks;

    if (clip.preprocessors) {
      base.preprocessors = clip.preprocessors.map((p) => {
        const next: any = {
          ...p,
        };
        if (typeof p.startTicks === "number") {
          next.startFrame = ticksToFrames(p.startTicks, fps, tickRate);
        }
        if (typeof p.endTicks === "number") {
          next.endFrame = ticksToFrames(p.endTicks, fps, tickRate);
        }
        delete next.startTicks;
        delete next.endTicks;
        return next;
      });
    }

    if (clip.masks) {
      base.masks = clip.masks.map((m) => ({
        ...m,
        keyframes: normalizeMaskKeyframes(m),
      }));
    }

    return base as AnyClipProps;
  });

  return {
    clips,
    timelines: persisted.timelines || [],
  };
};

export const serializeControlState = (
  controlState: ReturnType<typeof useControlsStore.getState>,
  tickRate: number = TICKS_PER_SECOND,
): PersistedControlState => {
  const fps = Math.max(1, Math.round(controlState.fps || DEFAULT_FPS));
  const [timelineStart, timelineEnd] = controlState.timelineDuration || [
    0,
    controlState.totalTimelineFrames,
  ];
  return {
    fps,
    maxZoomLevel: controlState.maxZoomLevel,
    minZoomLevel: controlState.minZoomLevel,
    zoomLevel: controlState.zoomLevel,
    totalTicks: framesToTicks(
      controlState.totalTimelineFrames || 0,
      fps,
      tickRate,
    ),
    timelineStartTicks: framesToTicks(timelineStart || 0, fps, tickRate),
    timelineEndTicks: framesToTicks(timelineEnd || 0, fps, tickRate),
    focusTicks: framesToTicks(controlState.focusFrame || 0, fps, tickRate),
    focusAnchorRatio: coerceNumber(controlState.focusAnchorRatio, 0.5),
    selectedClipIds: controlState.selectedClipIds || [],
    isFullscreen: !!controlState.isFullscreen,
  };
};

export const deserializeControlState = (
  persisted: PersistedControlState | undefined,
  tickRate: number = TICKS_PER_SECOND,
) => {
  const fps = Math.max(1, Math.round(persisted?.fps || DEFAULT_FPS));
  const totalTimelineFrames = ticksToFrames(
    persisted?.totalTicks ?? 0,
    fps,
    tickRate,
  );
  const timelineStart = ticksToFrames(
    persisted?.timelineStartTicks ?? 0,
    fps,
    tickRate,
  );
  const timelineEnd = ticksToFrames(
    persisted?.timelineEndTicks ?? totalTimelineFrames ?? 0,
    fps,
    tickRate,
  );

  const duration: [number, number] = [
    timelineStart,
    Math.max(timelineStart + 1, timelineEnd),
  ];

  return {
    fps,
    maxZoomLevel: persisted?.maxZoomLevel ?? 10,
    minZoomLevel: persisted?.minZoomLevel ?? 1,
    zoomLevel: persisted?.zoomLevel ?? 1,
    totalTimelineFrames: totalTimelineFrames || TIMELINE_DURATION_SECONDS * fps,
    timelineDuration: duration,
    focusFrame: ticksToFrames(persisted?.focusTicks ?? 0, fps, tickRate),
    focusAnchorRatio: coerceNumber(persisted?.focusAnchorRatio, 0.5),
    selectedClipIds: persisted?.selectedClipIds ?? [],
    isFullscreen: !!persisted?.isFullscreen,
  };
};

export const serializeViewportState = (
  viewportState: ReturnType<typeof useViewportStore.getState>,
): PersistedViewportState => ({
  scale: viewportState.scale,
  minScale: viewportState.minScale,
  maxScale: viewportState.maxScale,
  position: viewportState.position as Point,
  tool: viewportState.tool,
  shape: viewportState.shape,
  clipPositions: viewportState.clipPositions,
  viewportSize: viewportState.viewportSize,
  contentBounds: viewportState.contentBounds,
  aspectRatio: viewportState.aspectRatio,
  isAspectEditing: viewportState.isAspectEditing,
});

export const deserializeViewportState = (
  persisted: PersistedViewportState | undefined,
) => {
  if (!persisted) {
    return {
      scale: 0.75,
      minScale: 0.1,
      maxScale: 4,
      position: { x: 0, y: 0 },
      tool: "pointer" as const,
      shape: "rectangle" as const,
      clipPositions: {},
      viewportSize: { width: 0, height: 0 },
      contentBounds: null,
      aspectRatio: { width: 16, height: 9, id: "16:9" },
      isAspectEditing: false,
    };
  }

  return {
    scale: coerceNumber(persisted.scale, 0.75),
    minScale: coerceNumber(persisted.minScale, 0.1),
    maxScale: coerceNumber(persisted.maxScale, 4),
    position: persisted.position ?? { x: 0, y: 0 },
    tool: persisted.tool ?? ("pointer" as const),
    shape: persisted.shape ?? ("rectangle" as const),
    clipPositions: persisted.clipPositions ?? {},
    viewportSize: persisted.viewportSize ?? { width: 0, height: 0 },
    contentBounds: persisted.contentBounds ?? null,
    aspectRatio: persisted.aspectRatio ?? { width: 16, height: 9, id: "16:9" },
    isAspectEditing: !!persisted.isAspectEditing,
  };
};
