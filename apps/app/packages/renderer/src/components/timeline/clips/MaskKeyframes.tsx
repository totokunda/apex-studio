import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Konva from "konva";
import { Group, Line } from "react-konva";
import { getGlobalFrame, getLocalFrame, useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { MaskClipProps, MaskData, VideoClipProps } from "@/lib/types";

interface DiamondProps {
  x: number;
  y: number;
  width: number;
  height: number;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  draggable?: boolean;
  clipId?: string;
  shadowColor?: string;
  shadowBlur?: number;
  shadowOpacity?: number;
}

export const Diamond: React.FC<DiamondProps> = ({
  x,
  y,
  width,
  height,
  fill = "#00C8B4",
  stroke = "black",
  strokeWidth = 2,
  draggable = false,
  clipId = "",
  shadowColor = "#000000",
  shadowBlur = 12,
  shadowOpacity = 0.3,
}) => {
  // Define diamond points relative to center
  const points = [
    0,
    -height / 2, // top
    width / 2,
    0, // right
    0,
    height / 2, // bottom
    -width / 2,
    0, // left
  ];
  const lineRef = useRef<Konva.Line>(null);
  const { selectedClipIds } = useControlsStore();
  const isDragging = useClipStore((s) => s.isDragging);

  useEffect(() => {
    if (selectedClipIds.includes(clipId) && lineRef.current) {
      lineRef.current?.moveToTop();
    }
  }, [clipId, selectedClipIds]);

  useEffect(() => {
    if (!isDragging && lineRef.current) {
      setTimeout(() => {
        lineRef.current?.show();
        lineRef.current?.moveToTop();
      }, 50);
    } else if (isDragging && lineRef.current) {
      lineRef.current?.hide();
    }
  }, [isDragging, lineRef]);

  return (
    <Line
      ref={lineRef}
      x={x}
      y={y}
      points={points}
      closed
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
      strokeScaleEnabled={false} // Prevent stroke distortion on scaling
      draggable={draggable}
      shadowColor={shadowColor}
      shadowBlur={shadowBlur}
      shadowOpacity={shadowOpacity}
      shadowOffsetX={0}
      shadowOffsetY={0}
      lineJoin="round"
      lineCap="round"
      // add bending
      tension={0.1}
    />
  );
};

type Position = {
  x: number;
  y: number;
};

type MaskKeyframesProps = {
  clip: VideoClipProps;
  clipPosition: Position;
  clipWidth: number;
  timelineHeight: number;
  isDragging: boolean;
  currentStartFrame: number;
  currentEndFrame: number;
};

type KeyframeEntry = {
  frame: number;
  data: MaskData;
};

const CLIP_DRAG_OFFSET = 1.5;
const KEYFRAME_RADIUS = 12;
// Avoid rendering thousands of keyframes at low zoom by enforcing a minimum pixel
// spacing between rendered diamonds. As zoom increases, `pxPerFrame` increases,
// which lowers the derived `minFrameDelta` and naturally renders more keyframes.
const MIN_KEYFRAME_PIXEL_SPACING = Math.max(4, Math.round(KEYFRAME_RADIUS * 0.25));

const cloneKeyframes = (keyframes: MaskClipProps["keyframes"]) => {
  if (keyframes instanceof Map) {
    return new Map(keyframes);
  }
  return { ...(keyframes as Record<number, MaskData>) };
};

const toSortedEntries = (
  keyframes: MaskClipProps["keyframes"],
): KeyframeEntry[] => {
  if (!keyframes) return [];
  if (keyframes instanceof Map) {
    return Array.from(keyframes.entries())
      .map(([frame, data]) => ({ frame, data }))
      .sort((a, b) => a.frame - b.frame);
  }
  return Object.entries(keyframes as Record<number, MaskData>)
    .map(([frame, data]) => ({ frame: Number(frame), data }))
    .sort((a, b) => a.frame - b.frame);
};

const replaceMask = (masks: MaskClipProps[], updatedMask: MaskClipProps) =>
  masks.map((mask) => (mask.id === updatedMask.id ? updatedMask : mask));

export const MaskKeyframes: React.FC<MaskKeyframesProps> = ({
  clip,
  clipPosition,
  clipWidth,
  timelineHeight,
  isDragging,
  currentStartFrame,
  currentEndFrame,
}) => {
  const updateClip = useClipStore((s) => s.updateClip);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const setFocusFrame = useControlsStore((s) => s.setFocusFrame);
  const selectedClipIds = useControlsStore((s) => s.selectedClipIds);

  const containerRef = useRef<Konva.Group>(null);

  const [selectedFrame, setSelectedFrame] = useState<number | null>(null);

  const mask = useMemo(() => {
    const masks = clip.masks || [];
    if (!masks.length) return null;
    if (selectedMaskId) {
      const selected = masks.find((m) => m.id === selectedMaskId);
      if (selected) return selected;
    }
    return masks[0] ?? null;
  }, [clip.masks, selectedMaskId]);

  const keyframes = useMemo(() => {
    if (!mask) return [];
    return toSortedEntries(mask.keyframes);
  }, [mask]);

  useEffect(() => {
    if (!mask) return;
    if (!selectedMaskId) {
      setSelectedMaskId(mask.id);
    }
  }, [mask, selectedMaskId, setSelectedMaskId]);

  // Ensure keyframes sit above preprocessors when the clip is selected
  useEffect(() => {
    const isSelected = selectedClipIds.includes(clip.clipId);
    if (!isSelected) return;
    const node = containerRef.current;
    if (!node) return;
    // Defer to end of tick to allow sibling mounts, then bring forward
    const t = window.setTimeout(() => {
      try {
        node.moveToTop();
        node.getLayer()?.batchDraw();
      } catch {}
    }, 0);
    return () => window.clearTimeout(t);
  }, [selectedClipIds, clip.clipId, keyframes]);

  const clipDuration = useMemo(() => {
    return Math.max(1, (currentEndFrame ?? 0) - (currentStartFrame ?? 0));
  }, [currentEndFrame, currentStartFrame]);

  const pxPerFrame = useMemo(() => {
    return clipWidth / clipDuration;
  }, [clipWidth, clipDuration]);

  const visualBaseX = isDragging
    ? clipPosition.x + CLIP_DRAG_OFFSET
    : clipPosition.x;
  const visualBaseY = isDragging
    ? clipPosition.y + CLIP_DRAG_OFFSET
    : clipPosition.y;
  const keyframeY = visualBaseY + timelineHeight / 2;

  const activeFrame = useMemo(() => {
    if (!keyframes.length) return null;
    const localFocus = Math.max(0, Math.round(getLocalFrame(focusFrame, clip)));

    let candidate: number | null = null;
    for (const { frame } of keyframes) {
      if (frame <= localFocus) {
        candidate = frame;
      } else {
        break;
      }
    }

    if (candidate === null) {
      return keyframes[0].frame;
    }
    return candidate;
  }, [clip, focusFrame, keyframes]);

  // Only one keyframe should be visually active at a time: prefer selected, else focus-derived
  const displayActiveFrame = useMemo(() => {
    return selectedFrame != null ? selectedFrame : activeFrame;
  }, [selectedFrame, activeFrame]);

  const visibleKeyframes = useMemo(() => {
    if (keyframes.length <= 1) return keyframes;

    // At high zoom, show everything.
    const safePxPerFrame = Math.max(1e-6, pxPerFrame);
    const minFrameDelta = Math.max(
      1,
      Math.ceil(MIN_KEYFRAME_PIXEL_SPACING / safePxPerFrame),
    );
    if (minFrameDelta <= 1) return keyframes;

    const keep = new Set<number>();
    keep.add(keyframes[0].frame);
    keep.add(keyframes[keyframes.length - 1].frame);
    if (displayActiveFrame != null) keep.add(displayActiveFrame);
    if (selectedFrame != null) keep.add(selectedFrame);

    let lastKeptFrame = keyframes[0].frame;
    for (let i = 1; i < keyframes.length - 1; i++) {
      const frame = keyframes[i].frame;
      // If a frame is explicitly kept (selected/active), accept it and move the cursor.
      if (keep.has(frame)) {
        lastKeptFrame = frame;
        continue;
      }
      if (frame - lastKeptFrame >= minFrameDelta) {
        keep.add(frame);
        lastKeptFrame = frame;
      }
    }

    return keyframes.filter(({ frame }) => keep.has(frame));
  }, [displayActiveFrame, keyframes, pxPerFrame, selectedFrame]);

  useEffect(() => {
    if (selectedFrame == null) return;
    if (!keyframes.some(({ frame }) => frame === selectedFrame)) {
      setSelectedFrame(null);
    }
  }, [keyframes, selectedFrame]);

  useEffect(() => {
    if (!selectedMaskId || !clip.masks?.length) {
      setSelectedFrame(null);
    }
  }, [clip.masks, selectedMaskId]);

  const commitMaskUpdate = useCallback(
    (updater: (mask: MaskClipProps) => MaskClipProps | null) => {
      if (!mask) return;
      const updated = updater(mask);
      if (!updated) return;
      updateClip(clip.clipId, {
        masks: replaceMask(clip.masks ?? [], updated),
      });
    },
    [clip.clipId, clip.masks, mask, updateClip],
  );

  const moveKeyframe = useCallback(
    (sourceFrame: number, targetFrame: number) => {
      if (!mask) return;
      if (sourceFrame === targetFrame) return;

      commitMaskUpdate((currentMask) => {
        const currentKeyframes = cloneKeyframes(currentMask.keyframes);
        let payload: MaskData | undefined;

        if (currentKeyframes instanceof Map) {
          payload = currentKeyframes.get(sourceFrame);
          if (!payload) return null;
          currentKeyframes.delete(sourceFrame);
          currentKeyframes.set(targetFrame, payload);
        } else {
          const record = currentKeyframes as Record<number, MaskData>;
          payload = record[sourceFrame];
          if (!payload) return null;
          delete record[sourceFrame];
          record[targetFrame] = payload;
        }

        setSelectedFrame(targetFrame);
        setFocusFrame(getGlobalFrame(targetFrame, clip));

        return {
          ...currentMask,
          keyframes: currentKeyframes,
          lastModified: Date.now(),
        };
      });
    },
    [clip, commitMaskUpdate, mask, setFocusFrame],
  );

  const removeKeyframe = useCallback(
    (frameToRemove: number) => {
      if (!mask) return;

      // If this is the final keyframe, remove the mask entirely
      const entriesNow = toSortedEntries(mask.keyframes);
      if (entriesNow.length <= 1) {
        const remainingMasks = (clip.masks ?? []).filter(
          (m) => m.id !== mask.id,
        );
        updateClip(clip.clipId, { masks: remainingMasks });
        const nextMaskId = remainingMasks[0]?.id ?? null;
        setSelectedMaskId(nextMaskId);
        setSelectedFrame(null);
        return;
      }

      // Otherwise, remove just this keyframe and keep the mask
      commitMaskUpdate((currentMask) => {
        const entries = toSortedEntries(currentMask.keyframes);
        const currentKeyframes = cloneKeyframes(currentMask.keyframes);
        if (currentKeyframes instanceof Map) {
          currentKeyframes.delete(frameToRemove);
        } else {
          delete (currentKeyframes as Record<number, MaskData>)[frameToRemove];
        }

        const remaining = entries.filter(
          ({ frame }) => frame !== frameToRemove,
        );
        const nearest = remaining.reduce<{
          frame: number;
          distance: number;
        } | null>((best, entry) => {
          const distance = Math.abs(entry.frame - frameToRemove);
          if (!best || distance < best.distance) {
            return { frame: entry.frame, distance };
          }
          return best;
        }, null);

        const fallbackFrame = nearest?.frame ?? remaining[0]?.frame ?? null;
        setSelectedFrame(fallbackFrame ?? null);
        if (fallbackFrame != null) {
          setFocusFrame(getGlobalFrame(fallbackFrame, clip));
        }

        return {
          ...currentMask,
          keyframes: currentKeyframes,
          lastModified: Date.now(),
        };
      });
    },
    [
      clip,
      commitMaskUpdate,
      mask,
      setFocusFrame,
      setSelectedMaskId,
      setSelectedFrame,
      updateClip,
    ],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      if (selectedFrame == null) return;
      const target = event.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName?.toLowerCase();
        const isEditable = (target as any).isContentEditable === true;
        if (tag === "input" || tag === "textarea" || isEditable) {
          return; // don't hijack typing/deleting in form fields
        }
      }
      event.preventDefault();
      try {
        (event as any).stopImmediatePropagation?.();
      } catch {}
      event.stopPropagation();
      removeKeyframe(selectedFrame);
    };

    // Capture phase ensures we preempt higher-level delete handlers
    window.addEventListener("keydown", handleKeyDown, true);
    return () => {
      window.removeEventListener("keydown", handleKeyDown, true as any);
    };
  }, [removeKeyframe, selectedFrame]);

  if (!mask || keyframes.length === 0) {
    return null;
  }

  const handleKeyframeClick = (frame: number) => {
    setSelectedMaskId(mask.id);
    setSelectedFrame(frame);
    setFocusFrame(getGlobalFrame(frame, clip));
  };

  const frameToX = (frame: number) => {
    const clamped = Math.max(0, Math.min(clipDuration, frame));
    return visualBaseX + clamped * pxPerFrame;
  };

  const dragBoundFunc = useMemo(() => {
    return (pos: Konva.Vector2d) => {
      const relativeX = pos.x - visualBaseX;
      const clampedRelative = Math.max(0, Math.min(clipWidth, relativeX));
      const nextFrame = Math.round(
        clampedRelative / Math.max(1e-6, pxPerFrame),
      );
      return {
        x: visualBaseX + nextFrame * pxPerFrame,
        y: keyframeY,
      };
    };
  }, [clipWidth, keyframeY, pxPerFrame, visualBaseX]);

  const handleDragEnd =
    (originalFrame: number) => (e: Konva.KonvaEventObject<DragEvent>) => {
      const absoluteX = e.target.x();
      const relativeX = absoluteX - visualBaseX;
      const nextFrame = Math.round(relativeX / Math.max(1e-6, pxPerFrame));
      const clamped = Math.max(0, Math.min(clipDuration, nextFrame));
      moveKeyframe(originalFrame, clamped);
    };

  const handleDragStart =
    (frame: number) => (e: Konva.KonvaEventObject<DragEvent>) => {
      e.cancelBubble = true;
      setSelectedMaskId(mask.id);
      setSelectedFrame(frame);
    };

  return (
    <Group ref={containerRef} listening>
      {visibleKeyframes.map(({ frame }) => {
        const isHighlighted = frame === displayActiveFrame;
        const fill = isHighlighted ? "#4C8DFF" : "#ffffff";
        const stroke = isHighlighted ? "#CFE2FF" : "rgba(180, 180, 180, 0.9)";
        const strokeWidth = isHighlighted ? 1.5 : 1.5;
        const glowColor = isHighlighted ? "#4C8DFF" : "#000000";
        const glowBlur = isHighlighted ? 18 : 10;
        const glowOpacity = isHighlighted ? 0.55 : 0.25;

        return (
          <Group
            key={`${mask.id}-${frame}`}
            x={frameToX(frame)}
            y={keyframeY}
            draggable
            dragBoundFunc={dragBoundFunc}
            onDragStart={handleDragStart(frame)}
            onDragEnd={handleDragEnd(frame)}
            onMouseDown={(e) => {
              e.cancelBubble = true;
            }}
            onClick={(e) => {
              e.cancelBubble = true;

              handleKeyframeClick(frame);
            }}
          >
            <Diamond
              x={0}
              y={0}
              width={KEYFRAME_RADIUS}
              height={KEYFRAME_RADIUS * 1.2}
              fill={fill}
              stroke={stroke}
              strokeWidth={strokeWidth}
              clipId={clip.clipId}
              shadowColor={glowColor}
              shadowBlur={glowBlur}
              shadowOpacity={glowOpacity}
            />
          </Group>
        );
      })}
    </Group>
  );
};

export default MaskKeyframes;
