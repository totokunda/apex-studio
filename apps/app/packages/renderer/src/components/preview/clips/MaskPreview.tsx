import React, { useEffect, useRef, useState } from "react";
import { Group } from "react-konva";
import Konva from "konva";
import { getLocalFrame, useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import {
  AnyClipProps,
  ClipTransform,
  MaskClipProps,
  MaskData,
  MaskShapeTool,
  PreprocessorClipType,
} from "@/lib/types";
import { useViewportStore } from "@/lib/viewport";
import { useMaskStore } from "@/lib/mask";
import { getCropOffset } from "@/components/preview/mask/touch";

import LassoMaskPreview from "./LassoMaskPreview";
import ShapeMaskPreview from "./ShapeMaskPreview";
import TouchMaskPreview from "./TouchMaskPreview";

interface MaskPreviewProps {
  clips: AnyClipProps[];
  sortClips: (clips: AnyClipProps[]) => AnyClipProps[];
  filterClips: (clips: AnyClipProps[], audio?: boolean) => AnyClipProps[];
  rectWidth: number;
  rectHeight: number;
}

interface MaskRenderData {
  mask: MaskClipProps;
  maskData: MaskData;
  activeKeyframe: number;
}

type GroupTransform = {
  x?: number;
  y?: number;
  rotation?: number;
};

interface MaskGroupTransforms {
  outer: GroupTransform;
  inner: GroupTransform;
}

const buildMaskGroupTransforms = (
  transform?: ClipTransform,
): MaskGroupTransforms => {
  if (!transform) {
    return { outer: {}, inner: {} };
  }

  const x = Number.isFinite(transform.x) ? transform.x : 0;
  const y = Number.isFinite(transform.y) ? transform.y : 0;
  const rotation = Number.isFinite(transform.rotation) ? transform.rotation : 0;

  // When a crop is applied to the clip, the visible media content is shifted
  // inside the clip's bounding box by an offset proportional to the crop
  // origin. The image preview accounts for this via the `crop` prop on the
  // Konva Image, but the saved mask geometry (lasso points, shapes, touch
  // points) remains in the original, uncropped coordinate space.
  //
  // To keep masks visually aligned with the cropped media, we apply the same
  // effective translation to the mask preview by shifting the inner group.
  // This preserves the existing rotation behaviour (masks rotate around the
  // clip origin) while re-centering them over the cropped content.
  const { offsetX: cropX, offsetY: cropY } = getCropOffset(transform);
  const innerOffsetX = -cropX;
  const innerOffsetY = -cropY;

  return {
    outer: { x, y, rotation },
    inner: { x: innerOffsetX, y: innerOffsetY },
  };
};

const MaskPreview: React.FC<MaskPreviewProps> = ({
  clips,
  sortClips,
  filterClips,
  rectWidth,
  rectHeight,
}) => {
  const { clipWithinFrame } = useClipStore();
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const [animationOffset, setAnimationOffset] = useState(0);
  const [masksToRender, setMasksToRender] = useState<MaskRenderData[]>([]);
  const [hasMasks, setHasMasks] = useState(false);
  const tool = useViewportStore((s) => s.tool);
  const setIsOverMask = useMaskStore((s) => s.setIsOverMask);
  const isOverMask = useMaskStore((s) => s.isOverMask);
  const updateClip = useClipStore((s) => s.updateClip);
  const getClipTransform = useClipStore((s) => s.getClipTransform);

  const ref = useRef<Konva.Group>(null);
  const prevMaskIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const masksData: MaskRenderData[] = [];
    let foundMasks = false;

    sortClips(filterClips(clips)).forEach((clip) => {
      if (clip.type !== "video" && clip.type !== "image") return;

      const clipIsActive = clipWithinFrame(clip, focusFrame);
      if (!clipIsActive) return;

      const masks = (clip as any).masks || [];

      masks.forEach((mask: MaskClipProps) => {
        const keyframes =
          mask.keyframes instanceof Map
            ? mask.keyframes
            : (mask.keyframes as Record<number, any>);

        const keyframeNumbers =
          keyframes instanceof Map
            ? Array.from(keyframes.keys())
                .map(Number)
                .sort((a, b) => a - b)
            : Object.keys(keyframes)
                .map(Number)
                .sort((a, b) => a - b);

        if (keyframeNumbers.length === 0) return;

        const startFrame = clip.startFrame ?? 0;
        const trimStart = isFinite(clip.trimStart ?? 0)
          ? (clip.trimStart ?? 0)
          : 0;
        const realStartFrame = startFrame + trimStart;
        const localFrame = focusFrame - realStartFrame;

        const nearestKeyframe = (frame: number) => {
          if (frame < keyframeNumbers[0]) return keyframeNumbers[0];
          const atOrBefore = keyframeNumbers.filter((k) => k <= frame).pop();
          return atOrBefore ?? keyframeNumbers[keyframeNumbers.length - 1];
        };

        const candidateLocal = nearestKeyframe(localFrame);
        const candidateGlobal = nearestKeyframe(focusFrame);
        const activeKeyframe =
          clip.type === "video" ? (candidateLocal ?? candidateGlobal) : 0;

        if (activeKeyframe !== undefined) {
          const maskData =
            keyframes instanceof Map
              ? keyframes.get(activeKeyframe)
              : keyframes[activeKeyframe];

          if (maskData) {
            masksData.push({
              mask,
              maskData,
              activeKeyframe,
            });
            foundMasks = true;
          }
        }
      });
    });

    setMasksToRender(masksData);
    setHasMasks(foundMasks);
  }, [clips, focusFrame, sortClips, filterClips, clipWithinFrame]);

  useEffect(() => {
    if (!hasMasks) {
      setAnimationOffset(0);
      return;
    }

    let animationFrameId: number;
    const animate = () => {
      setAnimationOffset((prev) => (prev + 0.5) % 20);
      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [hasMasks]);

  useEffect(() => {
    const currentMaskIds = new Set(masksToRender.map(({ mask }) => mask.id));

    if (isOverMask && currentMaskIds.size === 0) {
      setIsOverMask(false);
    }

    if (isOverMask && prevMaskIdsRef.current.size > 0) {
      const hadMaskRemoved = Array.from(prevMaskIdsRef.current).some(
        (prevId) => !currentMaskIds.has(prevId),
      );
      if (hadMaskRemoved && tool === "mask") {
        setIsOverMask(false);
      }
    }

    prevMaskIdsRef.current = currentMaskIds;
  }, [masksToRender, isOverMask, setIsOverMask, tool]);

  if (tool !== "mask") return null;

  return (
    <Group ref={ref}>
      {masksToRender.map(({ mask, maskData, activeKeyframe }) => {
        const clip = clips.find((c) => c.clipId === mask.clipId);
        if (!clip) return null;

        const clipTransform = getClipTransform(mask.clipId!) ?? clip.transform;
        const { outer, inner } = buildMaskGroupTransforms(clipTransform);
        const localFrame = getLocalFrame(focusFrame, clip);

        if (
          mask.tool === "lasso" &&
          maskData?.lassoPoints &&
          maskData.lassoPoints.length >= 6
        ) {
          const closedPoints = [
            ...maskData.lassoPoints,
            maskData.lassoPoints[0],
            maskData.lassoPoints[1],
          ];

          return (
            <Group key={`mask-${mask.id}-${activeKeyframe}`} {...outer}>
              <Group {...inner}>
                <LassoMaskPreview
                  mask={mask}
                  points={closedPoints}
                  animationOffset={animationOffset}
                  rectWidth={rectWidth}
                  rectHeight={rectHeight}
                />
              </Group>
            </Group>
          );
        }

        if (
          (mask.tool === "shape" || (mask.tool as any) === "rectangle") &&
          maskData?.shapeBounds
        ) {
          const bounds = maskData.shapeBounds;
          const shapeType: MaskShapeTool = bounds?.shapeType || "rectangle";

          return (
            <Group key={`mask-group-${mask.id}-${activeKeyframe}`} {...outer}>
              <Group {...inner}>
                <ShapeMaskPreview
                  key={`mask-${mask.id}-${activeKeyframe}`}
                  mask={mask}
                  x={bounds.x}
                  y={bounds.y}
                  width={bounds.width}
                  height={bounds.height}
                  rotation={bounds.rotation ?? 0}
                  shapeType={shapeType}
                  scaleX={bounds.scaleX ?? 1}
                  scaleY={bounds.scaleY ?? 1}
                  animationOffset={animationOffset}
                  rectWidth={rectWidth}
                  rectHeight={rectHeight}
                />
              </Group>
            </Group>
          );
        }

        if (mask.tool === "touch") {
          const activeDataKeyPoints =
            localFrame === activeKeyframe || clip.type === "image"
              ? maskData.touchPoints
              : [];

          const handleDeletePoints = (
            pointsToDelete: Array<{ x: number; y: number; label: 1 | 0 }>,
          ) => {
            const updatedTouchPoints = (maskData.touchPoints || []).filter(
              (point: { x: number; y: number; label: 1 | 0 }) => {
                return !pointsToDelete.some(
                  (p) =>
                    p.x === point.x &&
                    p.y === point.y &&
                    p.label === point.label,
                );
              },
            );

            const targetClip = clips.find((c) => c.clipId === mask.clipId);

            if (targetClip && mask.clipId) {
              const currentMasks =
                (targetClip as PreprocessorClipType).masks || [];

              if (updatedTouchPoints.length === 0) {
                const updatedMasks = currentMasks.filter(
                  (m) => m.id !== mask.id,
                );
                updateClip(mask.clipId, { masks: updatedMasks });

                const { setSelectedMaskId, selectedMaskId } =
                  useControlsStore.getState();
                if (selectedMaskId === mask.id) {
                  setSelectedMaskId(null);
                }
              } else {
                const keyframes =
                  mask.keyframes instanceof Map
                    ? mask.keyframes
                    : (mask.keyframes as Record<number, any>);
                const updatedKeyframes =
                  keyframes instanceof Map
                    ? new Map(keyframes)
                    : { ...keyframes };

                if (updatedKeyframes instanceof Map) {
                  updatedKeyframes.set(activeKeyframe, {
                    ...maskData,
                    touchPoints: updatedTouchPoints,
                  });
                } else {
                  updatedKeyframes[activeKeyframe] = {
                    ...maskData,
                    touchPoints: updatedTouchPoints,
                  };
                }

                const now = Date.now();
                const updatedMasks = currentMasks.map((m: any) =>
                  m.id === mask.id
                    ? { ...m, keyframes: updatedKeyframes, lastModified: now }
                    : m,
                );
                updateClip(mask.clipId, { masks: updatedMasks });
              }
            }
          };

          return (
            <Group key={`mask-${mask.id}-${activeKeyframe}`} {...outer}>
              <Group {...inner}>
                <TouchMaskPreview
                  clip={clip as PreprocessorClipType}
                  touchPoints={activeDataKeyPoints}
                  animationOffset={animationOffset}
                  rectWidth={rectWidth}
                  rectHeight={rectHeight}
                  onDeletePoints={handleDeletePoints}
                />
              </Group>
            </Group>
          );
        }
        return null;
      })}
    </Group>
  );
};

export default MaskPreview;
