import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  MediaInfo,
  ImageClipProps,
  MaskClipProps,
  PreprocessorClipProps,
  PreprocessorClipType,
  VideoClipProps,
} from "@/lib/types";
import { getClipWidth, getTimelineHeightForClip, useClipStore } from "@/lib/clip";
import { Rect, Image } from "react-konva";
import { Text } from "react-konva";
import { Group } from "react-konva";
import { Html } from "react-konva-utils";
import { FaPlay, FaStop } from "react-icons/fa6";
import { FaRegSquare, FaCheckSquare } from "react-icons/fa";
import { LuTrash } from "react-icons/lu";
import { useControlsStore } from "@/lib/control";
import { useAssetControlsStore } from "@/lib/assetControl";
import { useInputControlsStore } from "@/lib/inputControl";
import Konva from "konva";
import {
  calculateFrameFromX as calcFrameFromX,
  getOtherPreprocessors as getOthers,
  detectCollisions as detectColls,
  findGapAfterBlock as findGap,
} from "@/lib/preprocessorHelpers";
import {
  runPreprocessor,
  usePreprocessorJob,
  usePreprocessorJobActions,
  getPreprocessorResult,
  cancelPreprocessor,
} from "@/lib/preprocessor/api";
import { toast } from "sonner";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import { pathToFileURLString } from "@app/preload";
import { generateTimelineSamples } from "@/lib/media/timeline";
import { getNearestCachedCanvasSamples } from "@/lib/media/canvas";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { toFrameRange } from "@/lib/media/fps";
import { cn } from "@/lib/utils";
import { useWebGLMask } from "@/components/preview/mask/useWebGLMask";
import { useViewportStore } from "@/lib/viewport";
import { v4 as uuidv4 } from "uuid";
const THUMBNAIL_TILE_SIZE = 36;

const rangesOverlap = (aStart: number, aEnd: number, bStart: number, bEnd: number) => {
  const as = Math.min(aStart, aEnd);
  const ae = Math.max(aStart, aEnd);
  const bs = Math.min(bStart, bEnd);
  const be = Math.max(bStart, bEnd);
  return Math.max(as, bs) < Math.min(ae, be);
};

interface PropsPreprocessorClip {
  preprocessor: PreprocessorClipProps;
  currentStartFrame: number;
  currentEndFrame: number;
  timelineWidth: number;
  clipPosition: { x: number; y: number };
  timelineHeight: number;
  isDragging: boolean;
  clipId: string;
  cornerRadius: number;
  assetMode?: boolean;
  inputMode?: boolean;
  timelinePadding: number;
  clipOverride?: PreprocessorClipType;
  clipWidthOverride?: number;
  inputId?: string;
  timelineDuration?: [number, number];
}

export const PreprocessorClip: React.FC<PropsPreprocessorClip> = ({
  preprocessor: inputPreprocessor,
  currentStartFrame,
  currentEndFrame,
  timelineWidth,
  clipPosition,
  timelineHeight,
  cornerRadius,
  clipId,
  timelinePadding,
  timelineDuration: timelineDurationOverride,
  isDragging,
  assetMode,
  inputMode = false,
  clipOverride,
  clipWidthOverride,
  inputId,
}) => {
  const ctrlTimelineDuration = useControlsStore((s) => s.timelineDuration);
  const ctrlSetSelectedClipIds = useControlsStore((s) => s.setSelectedClipIds);
  const assetTimelineDuration = useAssetControlsStore(
    (s) => s.timelineDuration,
  );
  const { timelineDurationByInputId } = useInputControlsStore();
  const inputTimelineDuration = timelineDurationByInputId[inputId ?? ""] ?? [
    0, 10,
  ];
  const setSelectedAssetClipId = useAssetControlsStore(
    (s) => s.setSelectedAssetClipId,
  );
  let timelineDuration = useMemo(() => {
    if (timelineDurationOverride) {
      return timelineDurationOverride;
    }
    let duration = inputMode
      ? inputTimelineDuration
      : assetMode
        ? assetTimelineDuration
        : ctrlTimelineDuration;
    return duration;
  }, [
    inputMode,
    assetMode,
    inputTimelineDuration,
    assetTimelineDuration,
    ctrlTimelineDuration,
    clipOverride,
    currentStartFrame,
    currentEndFrame,
    timelineDurationOverride,
  ]);
  const removePreprocessorFromClip = useClipStore(
    (s) => s.removePreprocessorFromClip,
  );
  const getPreprocessorsForClip = useClipStore(
    (s) => s.getPreprocessorsForClip,
  );
  const getClipFromPreprocessorId = useClipStore(
    (s) => s.getClipFromPreprocessorId,
  );
  const getPreprocessorById = useClipStore((s) => s.getPreprocessorById);
  const isDraggingGlobal = useClipStore((s) => s.isDragging);
  const preprocessor = useMemo(
    () => getPreprocessorById(inputPreprocessor.id) ?? inputPreprocessor,
    [inputPreprocessor.id, getPreprocessorById, inputPreprocessor],
  );
  const clip = clipOverride ?? getClipFromPreprocessorId(preprocessor.id);
  // Preprocessor frames are stored relative to the parent clip (0 = clip start)
  const preprocessorStartFrame = preprocessor.startFrame ?? 0;
  const preprocessorEndFrame =
    preprocessor.endFrame ?? currentEndFrame - currentStartFrame;
  const textRef = useRef<Konva.Text>(null);
  const addAsset = useClipStore((s) => s.addAsset);

  // Calculate parent clip dimensions
  const clipDuration = currentEndFrame - currentStartFrame;
  const clipWidth =
    clipWidthOverride ??
    Math.max(
      getClipWidth(
        currentStartFrame,
        currentEndFrame,
        timelineWidth,
        timelineDuration,
      ),
      3,
    );

  // Calculate preprocessor position and size as proportion of parent clip
  const preprocessorDuration = useMemo(() => {
    if (clip?.type === "image") {
      return clipDuration;
    }
    return preprocessorEndFrame - preprocessorStartFrame;
  }, [preprocessorEndFrame, preprocessorStartFrame, clip?.type]);
  const preprocessorX = useMemo(
    () => (preprocessorStartFrame / clipDuration) * clipWidth,
    [preprocessorStartFrame, clipDuration, clipWidth],
  );
  const preprocessorWidth = useMemo(() => {
    if (clip?.type === "image") {
      return clipWidth;
    }
    return Math.max((preprocessorDuration / clipDuration) * clipWidth, 3);
  }, [preprocessorDuration, clipDuration, clipWidth, clip?.type]);
  const selectedPreprocessorId = useClipStore((s) => s.selectedPreprocessorId);
  const setSelectedPreprocessorId = useClipStore(
    (s) => s.setSelectedPreprocessorId,
  );
  const updatePreprocessor = useClipStore((s) => s.updatePreprocessor);
  const [resizingPreprocessor, setResizingPreprocessor] = useState<{
    id: string;
    side: "left" | "right";
  } | null>(null);
  const preprocessorRef = useRef<Konva.Group>(null);
  const prevClipBounds = useRef({
    startFrame: currentStartFrame,
    endFrame: currentEndFrame,
  });
  const didCreateNewClipRef = useRef(false);
  const [isCtrlPressed, setIsCtrlPressed] = useState(false);
  const previousMouseX = useRef<number | null>(null);
  const dragOffsetX = useRef<number>(0);
  const preprocessorJobId = useMemo(
    () =>
      preprocessor.status === "running"
        ? (preprocessor.activeJobId ?? null)
        : null,
    [preprocessor.activeJobId, preprocessor.status],
  );
  const { isProcessing, progress, result } =
    usePreprocessorJob(preprocessorJobId);
  const { clearJob, stopTracking } = usePreprocessorJobActions();
  const mediaInfoRef = useRef<MediaInfo | null>(
    getMediaInfoCached(preprocessor.assetId ?? "") ?? null,
  );
  const [imageCanvas] = useState<HTMLCanvasElement>(() =>
    document.createElement("canvas"),
  );
  const { applyFilters } = useWebGLFilters();
  const exactVideoUpdateTimerRef = useRef<number | null>(null);
  const exactVideoUpdateSeqRef = useRef<number>(0);
  const lastExactRequestKeyRef = useRef<string | null>(null);
  const [forceRerenderCounter, setForceRerenderCounter] = useState(0);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const ctrlFps = useControlsStore((s) => s.fps);
  const ctrlFocusFrame = useControlsStore((s) => s.focusFrame);
  const assetFps = useAssetControlsStore((s) => s.fps);
  const assetFocusFrame = useAssetControlsStore((s) => s.focusFrame);
  const inputFps = useInputControlsStore((s) => s.getFps(inputId ?? ""));
  const inputFocusFrame = useInputControlsStore((s) => s.getFocusFrame(inputId ?? ""));
  const clipAsset = clip?.assetId ? getAssetById(clip.assetId) : null;
  const fps = inputMode ? inputFps : assetMode ? assetFps : ctrlFps;
  const focusFrame = inputMode
    ? inputFocusFrame
    : assetMode
      ? assetFocusFrame
      : ctrlFocusFrame;
  const { tool } = useViewportStore();

  const { applyMask } = useWebGLMask({
    focusFrame,
    masks: clip?.masks ?? [],
    disabled:
      (clip?.type !== "video" && clip?.type !== "image") || tool === "mask",
    clip: clip || undefined,
  });

  const showProgress = useMemo(() => {
    if (preprocessor.status === "running") {
      return true;
    }
    if (result !== null && result.result_path !== null) {
      return false;
    }
    if (isProcessing || progress > 0) {
      return true;
    }
    return false;
  }, [
    isProcessing,
    progress,
    result,
    inputPreprocessor.id,
    preprocessor.status,
  ]);

  const canResize = useMemo(() => {
    return preprocessor.status !== "running" && clip?.type === "video";
  }, [preprocessor.status, clip?.type]);

  const preprocessorXPosition = useMemo(() => {
    const timelineSpan = timelineDuration[1] - timelineDuration[0];
    const pxPerFrame = timelineWidth / timelineSpan;
    const startSpanWidth = preprocessorStartFrame * pxPerFrame;
    return clipPosition.x + startSpanWidth;
  }, [clipPosition.x, preprocessorWidth, timelineDuration]);

  const createNewClipFromResultAsset = useCallback(
    async (resultAssetId: string, mediaInfo?: MediaInfo | null) => {
      if (didCreateNewClipRef.current) return;

      const state = useClipStore.getState();
      const parentClip = state.getClipById(clipId);
      if (!parentClip) return;
      const parentTimelineId = parentClip.timelineId;
      if (!parentTimelineId) return;

      const asset = state.getAssetById(resultAssetId);
      if (!asset?.path) return;

      // Build absolute frame range aligned to the parent clip timeline position
      const relStart = Math.max(0, Math.round(preprocessor.startFrame ?? 0));
      const relEnd = Math.max(
        relStart + 1,
        Math.round(
          preprocessor.endFrame ??
            (parentClip.endFrame - parentClip.startFrame),
        ),
      );
      const absStart = Math.max(
        0,
        Math.round((parentClip.startFrame ?? 0) + relStart),
      );
      const absEnd = Math.max(
        absStart + 1,
        Math.round((parentClip.startFrame ?? 0) + relEnd),
      );

      const chooseTimelineAbove = (
        desiredType: "media",
        startFrame: number,
        endFrame: number,
      ) => {
        const timelines = state.timelines || [];
        const clips = state.clips || [];
        const parentIdx = timelines.findIndex(
          (t) => t.timelineId === parentTimelineId,
        );

        // Find the closest compatible timeline above that has no overlap in the desired range.
        for (let i = parentIdx - 1; i >= 0; i--) {
          const t = timelines[i];
          if (!t || t.type !== desiredType) continue;
          const hasOverlap = clips.some((c) => {
            if (!c || c.hidden) return false;
            if (c.timelineId !== t.timelineId) return false;
            return rangesOverlap(
              startFrame,
              endFrame,
              c.startFrame ?? 0,
              c.endFrame ?? 0,
            );
          });
          if (!hasOverlap) return t.timelineId;
        }

        // Otherwise insert a new compatible timeline directly above the parent.
        const newTimelineId = uuidv4();
        const parentTimeline = timelines.find(
          (t) => t.timelineId === parentTimelineId,
        );
        state.addTimeline(
          {
            timelineId: newTimelineId,
            type: desiredType,
            timelineHeight: getTimelineHeightForClip(desiredType),
            timelineWidth:
              parentTimeline?.timelineWidth ??
              timelines[timelines.length - 1]?.timelineWidth ??
              0,
            timelinePadding:
              parentTimeline?.timelinePadding ??
              timelines[timelines.length - 1]?.timelinePadding ??
              24,
          },
          parentIdx - 1,
        );
        return newTimelineId;
      };

      let mi = mediaInfo ?? (getMediaInfoCached(asset.path));
      if (!mi) {
        mi = await getMediaInfo(asset.path, { sourceDir: "apex-cache" });
      }
      const isVideo = !!mi?.video;

      const timelineId = chooseTimelineAbove("media", absStart, absEnd);
      const width = isVideo ? mi?.video?.displayWidth : mi?.image?.width;
      const height = isVideo ? mi?.video?.displayHeight : mi?.image?.height;
      const newClipId = uuidv4();
      const base = {
        clipId: newClipId,
        timelineId,
        startFrame: absStart,
        endFrame: absEnd,
        trimStart: 0,
        trimEnd: 0,
        mediaWidth: width,
        mediaHeight: height,
        assetId: resultAssetId,
        assetIdHistory: [resultAssetId],
        preprocessors: [] as PreprocessorClipProps[],
        masks: [] as MaskClipProps[],
        transform: clip?.transform ?? undefined,
        originalTransform: clip?.originalTransform ?? undefined,
      };

      const newClip: VideoClipProps | ImageClipProps = isVideo
        ? ({
            ...base,
            type: "video",
          } as VideoClipProps)
        : ({
            ...base,
            type: "image",
          } as ImageClipProps);

      // Important ordering: add the clip first so the asset won't be pruned
      // when the preprocessor is removed from the parent clip.
      didCreateNewClipRef.current = true;
      state.addClip(newClip);
      state.removePreprocessorFromClip(clipId, preprocessor.id);
    },
    [clipId, preprocessor.id, preprocessor.startFrame, preprocessor.endFrame, preprocessor.createNewClip],
  );

  useEffect(() => {
    if (result?.result_path) {
      const resultPath = result.result_path;
      // convert to file url
      const fileUrl = pathToFileURLString(resultPath);
      const asset = addAsset({ path: fileUrl });
      getMediaInfo(fileUrl, { sourceDir: "apex-cache" })
        .then((mediaInfo) => {
          mediaInfoRef.current = mediaInfo;
          // If createNewClip is enabled (default), create a new clip from the result
          // and remove the preprocessor WITHOUT attaching the result asset to the parent clip.
          if (preprocessor.createNewClip !== false) {
            createNewClipFromResultAsset(asset.id, mediaInfo);
            return;
          }
          // Otherwise, attach the result to the preprocessor as before.
          updatePreprocessor(clipId, preprocessor.id, {
            assetId: asset.id,
            status: "complete",
          });
        })
        .catch(() => {
          updatePreprocessor(clipId, preprocessor.id, {
            status: "failed",
          });
        });
    }
  }, [result]);

  // Fallback mechanism: if progress reaches 100% but result doesn't come through, manually fetch it
  useEffect(() => {
    if (preprocessor.status !== "running" || progress < 99.9) return;

    const timeoutId = setTimeout(async () => {
      // Check if we still don't have a result after waiting
      if (!result?.result_path && preprocessor.status === "running") {
        try {
          const response = await getPreprocessorResult(
            preprocessor.activeJobId || preprocessor.id,
          );
          if (response.success && response.data?.result_path) {
            const resultPath = response.data.result_path;
            const fileUrl = pathToFileURLString(resultPath);
            const mediaInfo = await getMediaInfo(fileUrl, {
              sourceDir: "apex-cache",
            });

            mediaInfoRef.current = mediaInfo;
            const asset = addAsset({ path: fileUrl });
            if (preprocessor.createNewClip !== false) {
              await createNewClipFromResultAsset(asset.id, mediaInfo as any);
            } else {
              updatePreprocessor(clipId, preprocessor.id, {
                assetId: asset.id,
                status: "complete",
              });
            }
          } else if (response.data?.status === "failed") {
            updatePreprocessor(clipId, preprocessor.id, {
              status: "failed",
            });
          }
        } catch (error) {
          console.error("Failed to fetch preprocessor result:", error);
        }
      }
    }, 5000); // Wait 8 seconds after reaching 100% before trying to fetch

    return () => clearTimeout(timeoutId);
  }, [
    preprocessor.status,
    progress,
    result,
    preprocessor.id,
    clipId,
    updatePreprocessor,
  ]);

  useEffect(() => {
    mediaInfoRef.current = getMediaInfoCached(preprocessor.assetId ?? "") ?? null;
  }, [preprocessor.assetId]);

  // Set canvas dimensions based on preprocessor width and height
  useEffect(() => {
    imageCanvas.width = Math.min(preprocessorWidth, timelineWidth);
    imageCanvas.height = timelineHeight;
  }, [preprocessorWidth, timelineHeight, imageCanvas, timelineWidth]);

  const imageWidth = useMemo(
    () => Math.min(preprocessorWidth, timelineWidth),
    [preprocessorWidth, timelineWidth],
  );

  useEffect(() => {
    if (isDragging) {
      setTimeout(() => {
        preprocessorRef.current?.moveToTop();
      }, 100);
    }
  }, [isDragging]);

  const imageX = useMemo(() => {
    let extraDist = 0;
    let positionX =
      preprocessorWidth <= imageWidth ? 0 : -preprocessorXPosition;

    const clipOffset = 0; // currentStartFrame - (isFinite(clip?.trimStart ?? 0) ? clip?.trimStart ?? 0 : 0);
    const timelineStartFrame = timelineDuration[0] - clipOffset;
    const timelineEndFrame = timelineDuration[1] - clipOffset;
    const pxPerFrame = timelineWidth / (timelineEndFrame - timelineStartFrame);
    const distFromStartFrames = preprocessorStartFrame - timelineStartFrame;
    const distFromStartPixels = distFromStartFrames * pxPerFrame;
    const distFromEndFrames = timelineEndFrame - preprocessorEndFrame;
    const distFromEndPixels = distFromEndFrames * pxPerFrame;
    // check if end is beyond timelineEndFrame

    const timelineSpan = timelineEndFrame - timelineStartFrame;
    const preprocessorSpan = preprocessorEndFrame - preprocessorStartFrame;
    if (
      preprocessorSpan > timelineSpan &&
      preprocessorStartFrame > timelineStartFrame
    ) {
      extraDist += distFromStartPixels;
    } else if (
      preprocessorEndFrame < timelineEndFrame &&
      preprocessorEndFrame < timelineStartFrame
    ) {
      extraDist += distFromEndPixels;
    } else if (
      preprocessorSpan <= timelineSpan &&
      preprocessorStartFrame < timelineStartFrame
    ) {
      extraDist +=
        (timelineStartFrame - preprocessorStartFrame) * pxPerFrame -
        timelinePadding;
    }

    if (timelineStartFrame === 0 && timelineSpan < preprocessorSpan) {
      extraDist += timelinePadding;
    }

    return positionX + extraDist;
  }, [
    preprocessorXPosition,
    imageWidth,
    preprocessorX,
    currentStartFrame,
    clip?.trimStart,
    timelineDuration,
  ]);

  // Track Alt key state globally
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Meta") {
        setIsCtrlPressed(true);
      }
      if (e.key === "Delete" && selectedPreprocessorId === preprocessor.id) {
        removePreprocessorFromClip(clipId, preprocessor.id);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Meta") {
        setIsCtrlPressed(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    // Handle case where window loses focus while Alt is pressed
    const handleBlur = () => {
      setIsCtrlPressed(false);
    };

    window.addEventListener("blur", handleBlur);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, [selectedPreprocessorId, preprocessor.id]);

  // Adjust preprocessor when parent clip is resized
  useEffect(() => {
    const prevStart = prevClipBounds.current.startFrame;
    const prevEnd = prevClipBounds.current.endFrame;
    const clipDurationNow = currentEndFrame - currentStartFrame;

    // Detect if clip was resized or moved
    const startDelta = currentStartFrame - prevStart;
    const endDelta = currentEndFrame - prevEnd;

    // If no change, just update ref and return
    if (startDelta === 0 && endDelta === 0) {
      prevClipBounds.current = {
        startFrame: currentStartFrame,
        endFrame: currentEndFrame,
      };
      return;
    }

    // If both start and end changed by the same amount, it's a drag/move, not a resize
    // In this case, preprocessor maintains its relative position (no update needed)
    if (startDelta === endDelta) {
      prevClipBounds.current = {
        startFrame: currentStartFrame,
        endFrame: currentEndFrame,
      };
      return;
    }

    let needsUpdate = false;
    let newStartFrame = preprocessorStartFrame;
    let newEndFrame = preprocessorEndFrame;
    const preprocessorDuration = preprocessorEndFrame - preprocessorStartFrame;

    // Clip was resized (not just moved)
    // If clip start moved (left resize), shift preprocessor to maintain absolute position
    if (startDelta !== 0) {
      // Shift preprocessor in opposite direction to maintain absolute position
      newStartFrame = preprocessorStartFrame - startDelta;
      newEndFrame = preprocessorEndFrame - startDelta;

      // If shifted too far left, clamp to start and shrink if needed
      if (newStartFrame < 0) {
        newStartFrame = 0;
        newEndFrame = Math.min(preprocessorDuration, clipDurationNow);
      }

      needsUpdate = true;
    }

    // If clip end moved (right resize), only adjust if preprocessor exceeds bounds
    if (endDelta !== 0 && newEndFrame > clipDurationNow) {
      // First try to shift left to keep full duration
      const overflow = newEndFrame - clipDurationNow;
      if (newStartFrame >= overflow) {
        newStartFrame -= overflow;
        newEndFrame = clipDurationNow;
      } else {
        // Not enough space to shift, shrink from the end
        newEndFrame = clipDurationNow;
      }
      needsUpdate = true;
    }

    // Final bounds check - ensure preprocessor stays within clip
    if (newStartFrame < 0) {
      newStartFrame = 0;
      needsUpdate = true;
    }

    if (newEndFrame > clipDurationNow) {
      newEndFrame = clipDurationNow;
      needsUpdate = true;
    }

    // If start is beyond clip duration, move it back
    if (newStartFrame >= clipDurationNow) {
      newStartFrame = Math.max(0, clipDurationNow - 1);
      needsUpdate = true;
    }

    // Ensure end is after start
    if (newEndFrame <= newStartFrame) {
      newEndFrame = Math.min(newStartFrame + 1, clipDurationNow);
      needsUpdate = true;
    }

    if (needsUpdate) {
      updatePreprocessor(clipId, preprocessor.id, {
        startFrame: newStartFrame,
        endFrame: newEndFrame,
      });
    }

    // Update previous bounds
    prevClipBounds.current = {
      startFrame: currentStartFrame,
      endFrame: currentEndFrame,
    };
  }, [
    currentStartFrame,
    currentEndFrame,
    preprocessorStartFrame,
    preprocessorEndFrame,
    clipId,
    preprocessor.id,
    updatePreprocessor,
  ]);

  const calculateFrameFromX = useCallback(
    (xPosition: number) => {
      return calcFrameFromX(
        xPosition,
        timelinePadding,
        timelineWidth,
        timelineDuration,
      );
    },
    [timelinePadding, timelineWidth, timelineDuration],
  );

  const getOtherPreprocessors = useCallback(() => {
    return getOthers(getPreprocessorsForClip(clipId), preprocessor.id);
  }, [getPreprocessorsForClip, clipId, preprocessor.id]);

  const detectCollisions = useCallback(
    (targetStart: number, targetEnd: number) => {
      const others = getOtherPreprocessors();
      const clipDurationNow = currentEndFrame - currentStartFrame;
      return detectColls(targetStart, targetEnd, others, clipDurationNow);
    },
    [getOtherPreprocessors, currentEndFrame, currentStartFrame],
  );

  const findGapAfterBlock = useCallback(
    (
      collidingPreprocessors: PreprocessorClipProps[],
      direction: "left" | "right",
      preprocessorDuration: number,
    ): number | null => {
      const clipDurationNow = currentEndFrame - currentStartFrame;
      return findGap(
        collidingPreprocessors,
        direction,
        preprocessorDuration,
        clipDurationNow,
      );
    },
    [currentEndFrame, currentStartFrame],
  );

  const handleDragStart = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      // Only allow preprocessor drag if Alt key is held OR dragging from header
      if (preprocessor.status === "running") return;
      const stage = e.target.getStage();
      if (!stage) return;
      stage.container().style.cursor = "grab";
      setSelectedPreprocessorId(preprocessor.id);
      e.target.moveToTop();

      // Calculate the offset between mouse position and clip start position
      const pointerPosition = stage.getPointerPosition();
      if (pointerPosition) {
        const [timelineStartFrame, timelineEndFrame] = timelineDuration;
        const absoluteCurrentStart = preprocessorStartFrame + currentStartFrame;
        const relativeToTimeline =
          (absoluteCurrentStart - timelineStartFrame) /
          (timelineEndFrame - timelineStartFrame);
        const clipStartX = timelinePadding + relativeToTimeline * timelineWidth;
        dragOffsetX.current = pointerPosition.x - clipStartX;
      }

      // Reset mouse tracking for clean drag direction detection
      previousMouseX.current = null;
    },
    [
      preprocessor.id,
      setSelectedPreprocessorId,
      preprocessor.status,
      timelineDuration,
      preprocessorStartFrame,
      currentStartFrame,
      timelinePadding,
      timelineWidth,
    ],
  );

  const handleDragMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      const stage = e.target.getStage();
      if (!stage) return;

      // Get actual mouse position from the stage pointer, not the element's position
      const pointerPosition = stage.getPointerPosition();
      if (!pointerPosition) return;

      const mouseX = pointerPosition.x;

      // Determine drag direction by comparing with previous position
      let dragDirection: "left" | "right" | null = null;
      if (previousMouseX.current !== null) {
        if (mouseX > previousMouseX.current) {
          dragDirection = "right";
        } else if (mouseX < previousMouseX.current) {
          dragDirection = "left";
        }
      }

      previousMouseX.current = mouseX;

      // Calculate desired position based on actual mouse position, applying the drag offset
      // This maintains the relative position where the user clicked
      const adjustedMouseX = mouseX - dragOffsetX.current;
      let absoluteStartFrame = calculateFrameFromX(adjustedMouseX);
      const preprocessorDuration =
        preprocessorEndFrame - preprocessorStartFrame;
      let absoluteEndFrame = absoluteStartFrame + preprocessorDuration;

      // Enforce bounds within the parent clip
      if (absoluteStartFrame < currentStartFrame) {
        absoluteStartFrame = currentStartFrame;
        absoluteEndFrame = absoluteStartFrame + preprocessorDuration;
      }

      if (absoluteEndFrame > currentEndFrame) {
        absoluteEndFrame = currentEndFrame;
        absoluteStartFrame = absoluteEndFrame - preprocessorDuration;
      }

      // Convert to relative frames (relative to clip start)
      let relativeStartFrame = absoluteStartFrame - currentStartFrame;
      let relativeEndFrame = absoluteEndFrame - currentStartFrame;

      // Check for collisions
      const collisions = detectCollisions(relativeStartFrame, relativeEndFrame);

      if (collisions.length > 0 && dragDirection) {
        // We have a collision - check if we can snap around it
        const snapPosition = findGapAfterBlock(
          collisions,
          dragDirection,
          preprocessorDuration,
        );

        if (snapPosition !== null) {
          // Check if mouse has reached the snap position before applying it
          if (dragDirection === "right" && relativeStartFrame >= snapPosition) {
            // Mouse is at or past the snap position - apply the snap
            relativeStartFrame = snapPosition;
            relativeEndFrame = snapPosition + preprocessorDuration;
          } else if (
            dragDirection === "left" &&
            relativeStartFrame <= snapPosition
          ) {
            // Mouse is at or past the snap position - apply the snap
            relativeStartFrame = snapPosition;
            relativeEndFrame = snapPosition + preprocessorDuration;
          } else {
            // Mouse hasn't reached snap position yet - move as close as possible to the blocking preprocessor
            const sorted = [...collisions].sort(
              (a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0),
            );
            if (dragDirection === "right") {
              // Find the leftmost edge of the blocking preprocessor(s)
              const blockStart = sorted[0].startFrame ?? 0;
              // Move as close as possible without overlapping
              relativeStartFrame = Math.min(
                relativeStartFrame,
                blockStart - preprocessorDuration,
              );
              relativeEndFrame = relativeStartFrame + preprocessorDuration;
            } else {
              // Find the rightmost edge of the blocking preprocessor(s)
              let blockEnd =
                sorted[0].endFrame ?? currentEndFrame - currentStartFrame;
              for (const p of sorted) {
                const pEnd = p.endFrame ?? currentEndFrame - currentStartFrame;
                if (pEnd > blockEnd) blockEnd = pEnd;
              }
              // Move as close as possible without overlapping
              relativeStartFrame = Math.max(relativeStartFrame, blockEnd);
              relativeEndFrame = relativeStartFrame + preprocessorDuration;
            }
          }
        } else {
          // No gap available - move as close as possible to the blocking preprocessor
          const sorted = [...collisions].sort(
            (a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0),
          );
          if (dragDirection === "right") {
            // Find the leftmost edge of the blocking preprocessor(s)
            const blockStart = sorted[0].startFrame ?? 0;
            // Move as close as possible without overlapping
            relativeStartFrame = Math.min(
              relativeStartFrame,
              blockStart - preprocessorDuration,
            );
            relativeEndFrame = relativeStartFrame + preprocessorDuration;
          } else {
            // Find the rightmost edge of the blocking preprocessor(s)
            let blockEnd =
              sorted[0].endFrame ?? currentEndFrame - currentStartFrame;
            for (const p of sorted) {
              const pEnd = p.endFrame ?? currentEndFrame - currentStartFrame;
              if (pEnd > blockEnd) blockEnd = pEnd;
            }
            // Move as close as possible without overlapping
            relativeStartFrame = Math.max(relativeStartFrame, blockEnd);
            relativeEndFrame = relativeStartFrame + preprocessorDuration;
          }
        }
      }

      // Final bounds check - ensure preprocessor stays within parent clip
      const clipDurationNow = currentEndFrame - currentStartFrame;
      relativeStartFrame = Math.max(
        0,
        Math.min(relativeStartFrame, clipDurationNow - preprocessorDuration),
      );
      relativeEndFrame = Math.max(
        preprocessorDuration,
        Math.min(relativeEndFrame, clipDurationNow),
      );

      // Ensure start is before end
      if (relativeStartFrame >= relativeEndFrame) {
        relativeStartFrame = Math.max(
          0,
          relativeEndFrame - preprocessorDuration,
        );
      }
      if (relativeEndFrame <= relativeStartFrame) {
        relativeEndFrame = Math.min(
          clipDurationNow,
          relativeStartFrame + preprocessorDuration,
        );
      }

      // Final collision check - if there's still a collision, don't update
      const finalCollisionCheck = detectCollisions(
        relativeStartFrame,
        relativeEndFrame,
      );
      if (finalCollisionCheck.length > 0) {
        // Collision detected - revert to current position and return
        const [timelineStartFrame, timelineEndFrame] = timelineDuration;
        const absoluteCurrentStart = preprocessorStartFrame + currentStartFrame;
        const relativeToTimeline =
          (absoluteCurrentStart - timelineStartFrame) /
          (timelineEndFrame - timelineStartFrame);
        const currentX = timelinePadding + relativeToTimeline * timelineWidth;
        e.target.x(currentX);
        return;
      }

      // Calculate the visual x position based on the collision-adjusted frames
      const [timelineStartFrame, timelineEndFrame] = timelineDuration;
      const absoluteAdjustedStart = relativeStartFrame + currentStartFrame;
      const relativeToTimeline =
        (absoluteAdjustedStart - timelineStartFrame) /
        (timelineEndFrame - timelineStartFrame);
      const adjustedX = timelinePadding + relativeToTimeline * timelineWidth;

      // Update the visual position immediately to avoid jitter
      e.target.x(adjustedX);

      updatePreprocessor(clipId, preprocessor.id, {
        startFrame: relativeStartFrame,
        endFrame: relativeEndFrame,
      });
    },
    [
      calculateFrameFromX,
      preprocessorEndFrame,
      preprocessorStartFrame,
      currentStartFrame,
      currentEndFrame,
      clipId,
      preprocessor.id,
      updatePreprocessor,
      timelineDuration,
      timelinePadding,
      timelineWidth,
      detectCollisions,
      findGapAfterBlock,
      preprocessor.status,
    ],
  );

  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (preprocessor.status === "running") return;
      e.target.getStage()!.container().style.cursor = "default";
    },
    [preprocessor.status],
  );

  const dragBoundFunc = useCallback(
    (pos: { x: number; y: number }) => {
      // Just constrain Y, let handleDragMove handle X collisions
      return {
        x: pos.x,
        y: clipPosition.y,
      };
    },
    [clipPosition.y],
  );

  useEffect(() => {
    if (!resizingPreprocessor) return;
    const stage = preprocessorRef.current?.getStage();
    if (!stage) return;

    const handleMouseMove = (e: MouseEvent) => {
      stage.container().style.cursor = "col-resize";
      const rect = stage.container().getBoundingClientRect();
      const stageX = e.clientX - rect.left;

      // Calculate absolute timeline frame from mouse position
      const absoluteFrame = calculateFrameFromX(stageX);

      // Convert to relative frame within the clip
      const relativeFrame = absoluteFrame - currentStartFrame;
      const clipDurationNow = currentEndFrame - currentStartFrame;

      if (!canResize) return;

      if (resizingPreprocessor.side === "right") {
        // Resizing right edge
        const minRelativeFrame = preprocessorStartFrame + 1;
        let maxRelativeFrame = clipDurationNow;

        // Check for collision with other preprocessors
        const otherPreprocessors = getOtherPreprocessors();
        for (const other of otherPreprocessors) {
          const otherStart = other.startFrame ?? 0;

          // If other preprocessor is to the right and would collide
          if (
            otherStart > preprocessorStartFrame &&
            otherStart < maxRelativeFrame
          ) {
            maxRelativeFrame = otherStart;
          }
        }

        const targetRelativeFrame = Math.max(
          minRelativeFrame,
          Math.min(maxRelativeFrame, relativeFrame),
        );
        updatePreprocessor(clipId, preprocessor.id, {
          endFrame: targetRelativeFrame,
        });
      } else if (resizingPreprocessor.side === "left") {
        // Resizing left edge
        let minRelativeFrame = 0;
        const maxRelativeFrame = preprocessorEndFrame - 1;

        // Check for collision with other preprocessors
        const otherPreprocessors = getOtherPreprocessors();
        for (const other of otherPreprocessors) {
          const otherEnd = other.endFrame ?? clipDurationNow;

          // If other preprocessor is to the left and would collide
          if (otherEnd < preprocessorEndFrame && otherEnd > minRelativeFrame) {
            minRelativeFrame = otherEnd;
          }
        }

        const targetRelativeFrame = Math.max(
          minRelativeFrame,
          Math.min(maxRelativeFrame, relativeFrame),
        );
        updatePreprocessor(clipId, preprocessor.id, {
          startFrame: targetRelativeFrame,
        });
      }
    };

    const handleMouseUp = () => {
      setResizingPreprocessor(null);
      stage.container().style.cursor = "default";
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [
    resizingPreprocessor,
    currentStartFrame,
    currentEndFrame,
    calculateFrameFromX,
    clipId,
    updatePreprocessor,
    preprocessor.id,
    preprocessorStartFrame,
    preprocessorEndFrame,
    getOtherPreprocessors,
    canResize,
  ]);

  // Preprocessor is interactive only when Alt is pressed OR when hovering/interacting with header
  const isListening = !isCtrlPressed && !assetMode && !inputMode;

  // Trigger thumbnail generation when preprocessor completes
  useEffect(() => {
    if (
      preprocessor.status !== "complete" ||
      !mediaInfoRef.current ||
      !preprocessor.assetId
    )
      return;

    const generateTimelineThumbnailImage = async () => {
      if (!mediaInfoRef.current?.image) return;

      const width = mediaInfoRef.current.image.width ?? 1;
      const height = mediaInfoRef.current.image.height ?? 1;
      const ratio = width / height;
      let thumbnailWidth = Math.max(
        timelineHeight * ratio,
        THUMBNAIL_TILE_SIZE,
      );
      const tClipWidth = Math.min(clipWidth, timelineWidth);
      const asset = getAssetById(preprocessor.assetId!);
      if (!asset) return;

      const samples = await generateTimelineSamples(
        preprocessor.id,
        asset.path,
        [0],
        thumbnailWidth,
        timelineHeight,
        tClipWidth,
        {
          mediaInfo: mediaInfoRef.current,
        },
      );

      if (samples?.[0]?.canvas) {
        const inputCanvas = samples[0].canvas as HTMLCanvasElement;
        const canvasToTile = applyMask(inputCanvas);
        const ctx = imageCanvas.getContext("2d");
        if (ctx) {
          const targetWidth = Math.max(1, imageCanvas.width);
          const targetHeight = Math.max(1, imageCanvas.height);
          ctx.clearRect(0, 0, targetWidth, targetHeight);

          // Determine tile dimensions from the input canvas/image
          const tileWidth = Math.max(
            1,
            (canvasToTile as any).width ||
              (canvasToTile as any).naturalWidth ||
              1,
          );
          const tileHeight = Math.max(
            1,
            (canvasToTile as any).height ||
              (canvasToTile as any).naturalHeight ||
              1,
          );
          const sourceHeight = Math.min(tileHeight, targetHeight);

          // Repeat the inputCanvas horizontally until we fill the target width
          let x = 0;
          while (x < targetWidth) {
            const remaining = targetWidth - x;
            const drawWidth = Math.min(tileWidth, remaining);

            if (x + drawWidth > 0) {
              ctx.drawImage(canvasToTile, x, 0, drawWidth, sourceHeight);
            }
            x += drawWidth;
          }

          // Apply WebGL filters to image thumbnails
          applyFilters(imageCanvas, {
            brightness: clip?.brightness,
            contrast: clip?.contrast,
            hue: clip?.hue,
            saturation: clip?.saturation,
            blur: clip?.blur,
            sharpness: clip?.sharpness,
            noise: clip?.noise,
            vignette: clip?.vignette,
          });
        }
      }
      preprocessorRef.current?.getLayer()?.batchDraw();
    };

    const generateTimelineThumbnailVideo = async () => {
      if (!mediaInfoRef.current?.video) return;

      const clipFps =
        mediaInfoRef.current?.stats.video?.averagePacketRate || fps;

      const width = mediaInfoRef.current.video.displayWidth ?? 1;
      const height = mediaInfoRef.current.video.displayHeight ?? 1;
      const ratio = width / height;
      const thumbnailWidth = Math.max(
        timelineHeight * ratio,
        THUMBNAIL_TILE_SIZE,
      );

      const mediaStartFrame = mediaInfoRef.current.startFrame
        ? Math.round(((mediaInfoRef.current.startFrame ?? 0) / fps) * clipFps)
        : 0;
      const mediaEndFrame = mediaInfoRef.current.endFrame
        ? Math.round(((mediaInfoRef.current.endFrame ?? 0) / fps) * clipFps)
        : undefined;
      const timelineStartFrame = timelineDuration[0];
      const timelineEndFrame = timelineDuration[1];
      const timelineSpan = timelineEndFrame - timelineStartFrame;
      const speed = Math.max(
        0.1,
        Math.min(5, Number((clip as any)?.speed ?? 1)),
      );

      const renderStartFrame = Math.max(
        timelineStartFrame,
        preprocessorStartFrame,
      );
      const renderEndFrame = Math.min(timelineEndFrame, preprocessorEndFrame);
      const renderSpan = renderEndFrame - renderStartFrame;
      const pxPerFrame = timelineWidth / timelineSpan;
      const renderWidth = renderSpan * pxPerFrame;
      const numColumns = Math.ceil(renderWidth / thumbnailWidth) + 1;

      let frameIndices: number[] = [];
      if (renderSpan >= numColumns && numColumns > 1) {
        frameIndices = Array.from({ length: numColumns }, (_, i) => {
          const progress = i / (numColumns - 1);
          const frameIndex = Math.round(
            renderStartFrame + progress * renderSpan,
          );
          return frameIndex;
        });
      } else if (numColumns > 1) {
        frameIndices = Array.from({ length: numColumns }, (_, i) => {
          const frameIndex = Math.floor(
            i / Math.ceil(numColumns / (renderSpan + 1)),
          );
          const clampedIndex = Math.min(frameIndex, renderSpan);
          return renderStartFrame + clampedIndex;
        });
      } else {
        frameIndices = [renderStartFrame];
      }

      frameIndices = frameIndices.filter(
        (frameIndex) => isNaN(frameIndex) === false && isFinite(frameIndex),
      );

      const projectFps =
        (inputMode
          ? useInputControlsStore.getState().getFps(inputId ?? "")
          : assetMode
            ? useAssetControlsStore.getState().fps
            : useControlsStore.getState().fps) || 30;
      const fpsAdjustment = projectFps / clipFps;

      // frameIndices are in clip-relative coordinates, so shift by preprocessor start to get preprocessor-relative frames
      const frameShift = preprocessorStartFrame;
      frameIndices = frameIndices.map((frameIndex) => {
        const local = frameIndex - frameShift;
        const speedAdjusted = local * speed;
        // Map from project fps space to native clip fps space
        const nativeFpsFrame = Math.round((speedAdjusted / fps) * clipFps);
        let sourceFrame = nativeFpsFrame + mediaStartFrame;
        if (mediaEndFrame !== undefined) {
          sourceFrame = Math.min(sourceFrame, mediaEndFrame);
        }
        return Math.max(mediaStartFrame, sourceFrame);
      });

      if (frameIndices.length === 0) {
        return;
      }

      const asset = getAssetById(preprocessor.assetId!);
      if (!asset) return;

      // Immediate draw using nearest cached frames
      const nearest = getNearestCachedCanvasSamples(
        asset.path,
        frameIndices,
        thumbnailWidth,
        timelineHeight,
        { mediaInfo: mediaInfoRef.current },
      );

      const hasCachedSamples = nearest.some(
        (sample) => sample !== null && sample !== undefined,
      );
      // get the sum width of all the samples
      const ctx = imageCanvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
        let x = 0;
        const targetWidth = Math.max(1, imageCanvas.width);
        const targetHeight = Math.max(1, imageCanvas.height);

        for (let i = 0; i < nearest.length && x < targetWidth; i++) {
          const sample = nearest[i];
          if (!sample) continue;
          const inputCanvas = sample.canvas as HTMLCanvasElement;
          const canvasToTile = applyMask(
            inputCanvas,
            Math.round(frameIndices[i] * fpsAdjustment),
          );
          const anyCanvas = inputCanvas as any;
          const tileWidth = Math.max(
            1,
            anyCanvas.width || anyCanvas.naturalWidth || 1,
          );
          const tileHeight = Math.max(
            1,
            anyCanvas.height || anyCanvas.naturalHeight || 1,
          );
          const sourceHeight = Math.min(tileHeight, targetHeight);

          const remaining = targetWidth - x;
          if (remaining <= 0) break;
          const drawWidth = Math.min(tileWidth, remaining);
          if (drawWidth <= 0) break;
          ctx.drawImage(
            canvasToTile,
            0,
            0,
            drawWidth,
            sourceHeight,
            x,
            0,
            drawWidth,
            sourceHeight,
          );
          x += drawWidth;
        }

        // Apply WebGL filters
        applyFilters(imageCanvas, {
          brightness: clip?.brightness,
          contrast: clip?.contrast,
          hue: clip?.hue,
          saturation: clip?.saturation,
          blur: clip?.blur,
          sharpness: clip?.sharpness,
          noise: clip?.noise,
          vignette: clip?.vignette,
        });
      }
      preprocessorRef.current?.getLayer()?.batchDraw();

      // Debounced fetch of exact frames
      if (exactVideoUpdateTimerRef.current != null) {
        window.clearTimeout(exactVideoUpdateTimerRef.current);
        exactVideoUpdateTimerRef.current = null;
      }
      const DEBOUNCE_MS = hasCachedSamples ? 100 : 0;
      const requestKey = `${preprocessor.id}|${frameIndices.join(",")}|${thumbnailWidth}x${timelineHeight}`;
      exactVideoUpdateTimerRef.current = window.setTimeout(async () => {
        const mySeq = ++exactVideoUpdateSeqRef.current;
        try {
          if (lastExactRequestKeyRef.current === requestKey) {
            return;
          }
          const exactSamples = await generateTimelineSamples(
            preprocessor.id,
            asset.path,
            frameIndices,
            thumbnailWidth,
            timelineHeight,
            preprocessorWidth,
            {},
          );

          if (mySeq !== exactVideoUpdateSeqRef.current) {
            return;
          }
          const ctx2 = imageCanvas.getContext("2d");

          if (ctx2 && exactSamples) {
            ctx2.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
            let x2 = 0;
            const targetWidth2 = Math.max(1, imageCanvas.width);
            const targetHeight2 = Math.max(1, imageCanvas.height);

            for (let i = 0; i < exactSamples.length && x2 < targetWidth2; i++) {
              const sample = exactSamples[i];
              const inputCanvas = sample.canvas as HTMLCanvasElement;
              const canvasToTile = applyMask(
                inputCanvas,
                Math.round(frameIndices[i] * fpsAdjustment),
              );
              const anyCanvas = inputCanvas as any;
              const tileWidth = Math.max(
                1,
                anyCanvas.width || anyCanvas.naturalWidth || 1,
              );
              const tileHeight = Math.max(
                1,
                anyCanvas.height || anyCanvas.naturalHeight || 1,
              );
              const sourceHeight = Math.min(tileHeight, targetHeight2);

              const remaining2 = targetWidth2 - x2;
              if (remaining2 <= 0) break;
              const drawWidth2 = Math.min(tileWidth, remaining2);
              if (drawWidth2 <= 0) break;
              ctx2.drawImage(
                canvasToTile,
                0,
                0,
                drawWidth2,
                sourceHeight,
                x2,
                0,
                drawWidth2,
                sourceHeight,
              );
              x2 += drawWidth2;
            }

            // Apply WebGL filters
            applyFilters(imageCanvas, {
              brightness: clip?.brightness,
              contrast: clip?.contrast,
              hue: clip?.hue,
              saturation: clip?.saturation,
              blur: clip?.blur,
              sharpness: clip?.sharpness,
              noise: clip?.noise,
              vignette: clip?.vignette,
            });
          }
        } finally {
          if (mySeq === exactVideoUpdateSeqRef.current) {
            preprocessorRef.current?.getLayer()?.batchDraw();
            lastExactRequestKeyRef.current = requestKey;

            if (!hasCachedSamples) {
              setForceRerenderCounter((prev) => prev + 1);
            }
          }
        }
      }, DEBOUNCE_MS);
    };

    // Determine type from mediaInfo
    if (mediaInfoRef.current.video) {
      generateTimelineThumbnailVideo();
    } else if (mediaInfoRef.current.image) {
      generateTimelineThumbnailImage();
    }
  }, [
    preprocessor.status,
    preprocessor.assetId,
    applyMask,
    preprocessorWidth,
    clipWidth,
    timelineHeight,
    timelineWidth,
    timelineDuration,
    forceRerenderCounter,
    clip,
  ]);

  const handleRunPreprocessor = useCallback(async () => {
    if (preprocessor.status === "running") return;
    const clip = getClipFromPreprocessorId(preprocessor.id);
    if (!clip || !clip.assetId) return;

    // Clear any existing job data for previous active job
    if (preprocessor.activeJobId) {
      clearJob(preprocessor.activeJobId);
    }

    // If backend is remote and src is local-like, inform user about upload delay
    try {
      const { getBackendIsRemote, getFileShouldUpload } =
        await import("@app/preload");
      const remoteRes = await getBackendIsRemote();
      const isRemote = !!(
        remoteRes &&
        remoteRes.success &&
        remoteRes.data?.isRemote
      );
      if (isRemote) {
        const su = await getFileShouldUpload(String(clipAsset?.path || ""));
        const shouldUpload = !!(su && su.success && su.data?.shouldUpload);
        if (shouldUpload) {
          toast.info("Uploading source media to server…");
        }
      }
    } catch {}

    // need to convert our startFrame and endFrame back to where it would be with real FPS
    const clipMediaInfo = getMediaInfoCached(clipAsset?.path ?? "");
    const clipFps = clipMediaInfo?.stats.video?.averagePacketRate ?? 24;

    if (
      preprocessor.startFrame === undefined ||
      preprocessor.endFrame === undefined
    )
      return;

    const { start: startFrameReal, end: endFrameReal } = toFrameRange(
      preprocessor.startFrame,
      preprocessor.endFrame,
      fps,
      clipFps,
      clipMediaInfo?.duration ?? 0,
    );

    // Generate a unique job id for this run and store on the preprocessor
    const newJobId = uuidv4();
    updatePreprocessor(clipId, preprocessor.id, {
      status: "running",
      activeJobId: newJobId,
      jobIds: [...(preprocessor.jobIds || []), newJobId],
    });

    const response = await runPreprocessor({
      preprocessor_name: preprocessor.preprocessor.id,
      input_path: clipAsset?.path ?? "",
      start_frame: startFrameReal,
      end_frame: endFrameReal,
      job_id: newJobId,
      download_if_needed: true,
      params: preprocessor.values,
    });

    if (response.success) {
      updatePreprocessor(clipId, preprocessor.id, { status: "running" });
      toast.success(
        `Preprocessor ${preprocessor.preprocessor.name} run started successfully`,
      );
    } else {
      toast.error(
        `Failed to run preprocessor ${preprocessor.preprocessor.name}`,
      );
    }
  }, [
    canResize,
    preprocessor.preprocessor.id,
    clipId,
    updatePreprocessor,
    getClipFromPreprocessorId,
    preprocessor.status,
    clearJob,
    fps,
    preprocessor.id,
    preprocessor.startFrame,
    preprocessor.endFrame,
    preprocessor.values,
  ]);

  const handleStopPreprocessor = useCallback(async () => {
    if (preprocessor.status !== "running") return;
    const jobId = preprocessor.activeJobId || preprocessor.id;

    // Stop tracking the job and clear it
    try {
      await cancelPreprocessor(jobId);
    } catch {}
    await stopTracking(jobId);
    clearJob(jobId);

    // Update preprocessor status to idle and clear active job id
    updatePreprocessor(clipId, preprocessor.id, {
      status: undefined,
      activeJobId: undefined,
    });

    toast.info(`Preprocessor ${preprocessor.preprocessor.name} stopped`);
  }, [
    preprocessor.status,
    preprocessor.activeJobId,
    preprocessor.id,
    preprocessor.preprocessor.name,
    stopTracking,
    clearJob,
    updatePreprocessor,
    clipId,
  ]);

  if (preprocessor.status !== "complete" && assetMode) return null;

  
  return (
    <>
      <Group
        id={preprocessor.id}
        x={
          isDragging
            ? clipPosition.x + preprocessorX + 1.5
            : clipPosition.x + preprocessorX
        }
        y={isDragging ? clipPosition.y + 1.5 : clipPosition.y}
        clipX={0}
        clipY={0}
        draggable={
          preprocessor.status !== "running" &&
          preprocessor.status !== "complete"
        }
        listening={preprocessor.status === "complete" ? false : isListening}
        key={preprocessor.id}
        onDragStart={handleDragStart}
        onDragMove={handleDragMove}
        onDragEnd={handleDragEnd}
        dragBoundFunc={dragBoundFunc}
        clipWidth={preprocessorWidth}
        ref={preprocessorRef}
        clipFunc={(ctx) => {
          ctx.rect(-4, -4, preprocessorWidth + 6, timelineHeight + 8);
        }}
      >
        <React.Fragment key={preprocessor.id}>
          {/* Background Rect - visible when no thumbnail or during progress */}
          {(!mediaInfoRef.current || preprocessor.status !== "complete") && (
            <Rect
              x={0}
              y={0}
              width={preprocessorWidth}
              onClick={() => {
                setSelectedPreprocessorId(preprocessor.id);
                if (assetMode) {
                  setSelectedAssetClipId(null);
                } else {
                  ctrlSetSelectedClipIds([]);
                }
              }}
              height={timelineHeight}
              fill={
                showProgress
                  ? "rgb(34, 33, 36)"
                  : selectedPreprocessorId === preprocessor.id
                    ? "rgba(0, 0, 85, 0.85)"
                    : "rgba(0, 0, 85, 0.7)"
              }
              stroke={
                showProgress
                  ? "rgb(174, 129, 206)"
                  : selectedPreprocessorId === preprocessor.id
                    ? "rgba(255, 255, 255, 1)"
                    : "rgba(255, 255, 255, 0.1)"
              }
              strokeWidth={
                showProgress
                  ? 2
                  : selectedPreprocessorId === preprocessor.id
                    ? preprocessor.status === "complete"
                      ? 1
                      : 2
                    : 0
              }
              cornerRadius={cornerRadius}
              onMouseOver={(e) => {
                e.target.getStage()!.container().style.cursor = "grab";
              }}
              onMouseLeave={(e) => {
                if (!resizingPreprocessor) {
                  e.target.getStage()!.container().style.cursor = "default";
                }
              }}
            />
          )}

          {/* Image thumbnail - visible when preprocessor is complete and has valid mediaInfo */}
          {mediaInfoRef.current && preprocessor.status === "complete" && (
            <>
              <Image
                x={imageX}
                y={0}
                image={imageCanvas}
                width={imageWidth}
                height={timelineHeight}
                cornerRadius={cornerRadius}
                fill={"black"}
                listening={false}
              />
              {/* Border/stroke for the image */}
              <Rect
                x={0}
                y={0}
                width={preprocessorWidth}
                height={timelineHeight}
                fill={"transparent"}
                stroke={
                  selectedPreprocessorId === preprocessor.id
                    ? "rgba(255, 255, 255, 1)"
                    : "rgba(255, 255, 255, 0.1)"
                }
                strokeWidth={selectedPreprocessorId === preprocessor.id ? 1 : 0}
                cornerRadius={cornerRadius}
                listening={false}
              />
            </>
          )}
          {/* Progress fill animation - fills like a battery as job progresses */}
          {showProgress && (
            <Rect
              x={0}
              y={0}
              width={(preprocessorWidth * (progress ?? 0)) / 100}
              height={timelineHeight}
              fill={"rgb(174, 129, 206)"}
              cornerRadius={cornerRadius}
              listening={false}
            />
          )}

          <Text
            x={4}
            y={timelineHeight - 16}
            text={preprocessor.preprocessor.name}
            fontSize={9.5}
            fontStyle={"500"}
            visible={preprocessor.status !== "complete"}
            fontFamily={"Poppins"}
            listening={true}
            fill={"white"}
            ref={textRef}
          />
          {/* Only show icons if there's enough space (at least 35px wide) */}
          {preprocessorWidth >= (textRef.current?.width() ?? 0) + 32 &&
            !isDraggingGlobal && (
              <Group
                x={Math.max(preprocessorWidth - 28, 4)}
                y={preprocessor.status === "complete" ? 4 : timelineHeight - 16}
                listening={true}
              >
                <Html>
                  <div
                    className="flex w-32 items-center gap-x-1 justify-end"
                    style={{ overflow: "hidden", maxWidth: "24px" }}
                  >
                    {preprocessor.status !== "running" &&
                      preprocessor.status !== "complete" && (
                        <FaPlay
                          size={10}
                          fill={"white"}
                          className="cursor-pointer"
                          onClick={() => handleRunPreprocessor()}
                        />
                      )}
                    {preprocessor.status === "running" && (
                      <FaStop
                        size={10}
                        fill={"white"}
                        className="cursor-pointer"
                        onClick={() => handleStopPreprocessor()}
                      />
                    )}
                    <div
                      onClick={() =>
                        removePreprocessorFromClip(clipId, preprocessor.id)
                      }
                      className={cn(
                        "relative rounded cursor-pointer transition-colors duration-200 group",
                        {
                          hidden:
                            preprocessor.status === "complete" &&
                            selectedPreprocessorId !== preprocessor.id,
                          "bg-brand/90  p-1 border border-brand-light/20":
                            preprocessor.status === "complete" &&
                            selectedPreprocessorId === preprocessor.id,
                          "hover:bg-brand-background":
                            preprocessor.status === "complete" &&
                            selectedPreprocessorId === preprocessor.id,
                        },
                      )}
                    >
                      <LuTrash
                        size={10}
                        className={cn(
                          " text-white group-hover:fill-white transition-colors duration-200",
                        )}
                      />
                    </div>
                  </div>
                </Html>
              </Group>
            )}
          {/* Left drag handle for preprocessor */}
          <Rect
            x={-1.5}
            y={0}
            width={1.5}
            height={timelineHeight}
            visible={
              selectedPreprocessorId === preprocessor.id &&
              preprocessor.status !== "running" &&
              preprocessor.status !== "complete" &&
              clip?.type === "video"
            }
            cornerRadius={[cornerRadius, 0, 0, cornerRadius]}
            fill={"white"}
            stroke={"white"}
            onMouseOver={(e) => {
              e.target.getStage()!.container().style.cursor = "col-resize";
            }}
            onMouseDown={(e) => {
              if (!canResize) return;
              e.cancelBubble = true;
              e.evt.stopPropagation();
              setResizingPreprocessor({ id: preprocessor.id, side: "left" });
              e.target.getStage()!.container().style.cursor = "col-resize";
            }}
            onMouseLeave={(e) => {
              e.target.getStage()!.container().style.cursor = "default";
            }}
          />
          {/* Right drag handle for preprocessor */}
          <Rect
            x={preprocessorWidth}
            y={0}
            width={1.5}
            height={timelineHeight}
            visible={
              selectedPreprocessorId === preprocessor.id &&
              preprocessor.status !== "running" &&
              preprocessor.status !== "complete" &&
              clip?.type === "video"
            }
            cornerRadius={[0, cornerRadius, cornerRadius, 0]}
            fill={"white"}
            stroke={"white"}
            onMouseOver={(e) => {
              e.target.getStage()!.container().style.cursor = "col-resize";
            }}
            onMouseDown={(e) => {
              if (!canResize) return;
              e.cancelBubble = true;
              e.evt.stopPropagation();
              setResizingPreprocessor({ id: preprocessor.id, side: "right" });
              e.target.getStage()!.container().style.cursor = "col-resize";
            }}
            onMouseLeave={(e) => {
              e.target.getStage()!.container().style.cursor = "default";
            }}
          />
        </React.Fragment>
      </Group>

      {/* External control icons for completed preprocessors */}
      {preprocessor.status === "complete" &&
        preprocessorWidth >= 50 &&
        !isDraggingGlobal &&
        !assetMode &&
        !inputMode && (
          <Html>
            <div
              style={{
                position: "absolute",
                left: `${clipPosition.x + preprocessorX}px`,
                top: `${clipPosition.y}px`,
                width: `${preprocessorWidth}px`,
                height: `${timelineHeight}px`,
                pointerEvents: "none",
              }}
            >
              {/* Control icons - top right corner */}
              <div className="absolute top-1 right-1 flex items-center gap-1 pointer-events-auto">
                {/* Trash icon (only when selected) */}
                {selectedPreprocessorId === preprocessor.id && (
                  <div
                    onClick={(e) => {
                      e.stopPropagation();
                      removePreprocessorFromClip(clipId, preprocessor.id);
                    }}
                    className="flex items-center justify-center p-1 rounded cursor-pointer transition-all shadow-lg bg-brand-background border border-brand-light/10 hover:bg-brand-background hover:border-brand-light/40"
                  >
                    <LuTrash size={12} className="text-white" />
                  </div>
                )}
                {/* Select icon */}
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedPreprocessorId(
                      selectedPreprocessorId === preprocessor.id
                        ? null
                        : preprocessor.id,
                    );
                    if (assetMode) {
                      setSelectedAssetClipId(null);
                    } else {
                      ctrlSetSelectedClipIds([]);
                    }
                  }}
                  className={cn(
                    "flex items-center justify-center p-1 rounded cursor-pointer transition-all shadow-lg",
                    selectedPreprocessorId === preprocessor.id
                      ? "bg-brand-background border border-brand-light/10 hover:bg-brand-background"
                      : "bg-brand/80 border border-white/20 hover:bg-brand-background hover:border-white/40",
                  )}
                >
                  {selectedPreprocessorId === preprocessor.id ? (
                    <FaCheckSquare size={12} className="text-white" />
                  ) : (
                    <FaRegSquare size={12} className="text-white" />
                  )}
                </div>
              </div>
            </div>
          </Html>
        )}
    </>
  );
};
