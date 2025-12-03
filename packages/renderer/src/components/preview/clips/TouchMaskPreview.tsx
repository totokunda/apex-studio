import React, { useMemo, useEffect, useState, useRef } from "react";
import { Group, Circle, Line } from "react-konva";
import { useMask, useMaskStore } from "@/lib/mask";
import { ClipTransform, PreprocessorClipType } from "@/lib/types";
import { useControlsStore } from "@/lib/control";
import { useClipStore } from "@/lib/clip";
import { getMediaInfoCached } from "@/lib/media/utils";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
import _ from "lodash";
import { getLocalFrame } from "@/lib/clip";
import { getCropOffset } from "@/components/preview/mask/touch";

interface TouchMaskPreviewProps {
  clip: PreprocessorClipType;
  touchPoints?: Array<{ x: number; y: number; label: 1 | 0 }>;
  rectWidth: number;
  rectHeight: number;
  animationOffset: number;
  onDeletePoints?: (
    pointsToDelete: Array<{ x: number; y: number; label: 1 | 0 }>,
  ) => void;
}

const TouchMaskPreview: React.FC<TouchMaskPreviewProps> = ({
  clip,
  touchPoints,
  rectWidth,
  rectHeight,
  animationOffset,
  onDeletePoints,
}) => {
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const getClipTransform = useClipStore((s) => s.getClipTransform);
  const tool = useViewportStore((s) => s.tool);
  const groupRef = useRef<Konva.Group>(null);

  const touchDrawMode = useMaskStore((s) => s.touchDrawMode);
  const touchMaskRefetchToken = useMaskStore((s) => s.touchMaskRefetchToken);
  const setTouchMaskRefetchToken = useMaskStore(
    (s) => s.setTouchMaskRefetchToken,
  );
  const getAssetById = useClipStore((s) => s.getAssetById);
  const asset = useMemo(() => getAssetById(clip.assetId), [clip.assetId]);
  if (!asset) return null;
  const [renderedTouchPoints, setRenderedTouchPoints] = useState<
    Array<{ x: number; y: number; label: 1 | 0 }>
  >([]);
  const [contours, setContours] = useState<Array<Array<number>>>([]);
  const lastStoredDataRef = useRef<string>("");
  const lastDataRef = useRef<any>(null);
  const lastDataFrameRef = useRef<number | null>(null);
  const previousLoadingRef = useRef<boolean>(false);
  const [selectedPoints, setSelectedPoints] = useState<
    Array<{ x: number; y: number; label: 1 | 0 }>
  >([]);
  const { fps } = useControlsStore();
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const setIsMaskDragging = useMaskStore((s) => s.setIsMaskDragging);

  const currentFrame = useMemo(() => {
    return getLocalFrame(focusFrame, clip);
  }, [focusFrame, clip]);

  const currentMask = useMemo(
    () => (clip.masks || []).find((m) => m.tool === "touch"),
    [clip.masks],
  );

  const getFiniteNumber = (value: number | undefined | null): number =>
    Number.isFinite(value) ? (value as number) : 0;

  const normalizeMaskPointForRotation = (
    x: number,
    y: number,
    transform?: ClipTransform | null,
  ): { x: number; y: number } => {
    if (!transform) return { x, y };

    const { offsetX, offsetY } = getCropOffset(transform);
    const rotation = getFiniteNumber(transform.rotation);
    const originX = getFiniteNumber(transform.x);
    const originY = getFiniteNumber(transform.y);

    // No rotation: encode only the crop offset into mask space
    if (Math.abs(rotation) < 1e-6) {
      return {
        x: x - originX + offsetX,
        y: y - originY + offsetY,
      };
    }

    const angleRad = (rotation * Math.PI) / 180;
    const dx = x - originX;
    const dy = y - originY;
    const cos = Math.cos(angleRad);
    const sin = Math.sin(angleRad);
    const unrotatedX = cos * dx + sin * dy;
    const unrotatedY = -sin * dx + cos * dy;

    return {
      x: unrotatedX + offsetX,
      y: unrotatedY + offsetY,
    };
  };

  const denormalizeMaskPointForRotation = (
    x: number,
    y: number,
    transform?: ClipTransform | null,
  ): { x: number; y: number } => {
    if (!transform) return { x, y };

    // Inverse of the normalization used when storing mask points:
    // remove the crop offset in mask-space, then re-apply rotation
    // about the clip origin.
    const { offsetX, offsetY } = getCropOffset(transform);
    const baseX = x - offsetX;
    const baseY = y - offsetY;
    const originX = getFiniteNumber(transform.x);
    const originY = getFiniteNumber(transform.y);

    const rotation = getFiniteNumber(transform.rotation);
    if (Math.abs(rotation) < 1e-6) {
      return { x: baseX + originX, y: baseY + originY };
    }
    const angleRad = (rotation * Math.PI) / 180;
    const dx = baseX;
    const dy = baseY;
    const cos = Math.cos(angleRad);
    const sin = Math.sin(angleRad);
    const rotatedX = originX + cos * dx - sin * dy;
    const rotatedY = originY + sin * dx + cos * dy;
    return { x: rotatedX, y: rotatedY };
  };

  // Load existing stored contours and data on mount or when frame changes
  useEffect(() => {
    try {
      if (!currentMask || !currentMask.keyframes) {
        setContours([]);
        setRenderedTouchPoints([]);
        return;
      }

      const keyframes =
        currentMask.keyframes instanceof Map
          ? currentMask.keyframes
          : (currentMask.keyframes as Record<number, any>);

      const keyframeNumbers =
        keyframes instanceof Map
          ? Array.from(keyframes.keys()).sort((a, b) => a - b)
          : Object.keys(keyframes)
              .map(Number)
              .sort((a, b) => a - b);

      if (keyframeNumbers.length === 0) {
        setContours([]);
        setRenderedTouchPoints([]);
        return;
      }

      const activeKeyframe = keyframeNumbers
        .filter((k: number) => k <= currentFrame)
        .pop();
      if (activeKeyframe === undefined) {
        setContours([]);
        setRenderedTouchPoints([]);
        return;
      }

      const maskData =
        keyframes instanceof Map
          ? keyframes.get(activeKeyframe)
          : keyframes[activeKeyframe];

      if (!maskData) {
        setContours([]);
        setRenderedTouchPoints([]);
        return;
      }

      if (maskData.contours) {
        setContours(maskData.contours);
      } else {
        setContours([]);
      }

      if (maskData.touchPoints && currentFrame === activeKeyframe) {
        setRenderedTouchPoints(maskData.touchPoints);
      }
    } catch (error) {
      console.error("Error loading touch mask contours:", error);
    }
  }, [currentMask, currentFrame]);

  // Get media info and clip transform
  const mediaInfo = useMemo(
    () => getMediaInfoCached(asset.path) || null,
    [asset.path],
  );
  const clipTransform = useMemo(
    () => getClipTransform(clip.clipId || ""),
    [clip.clipId, getClipTransform],
  );

  const points = useMemo<Array<{ x: number; y: number }>>(() => {
    if (!touchPoints || touchPoints.length === 0) return [];

    // Stored touch points are in the same normalized clip space as other masks.
    // For the mask API we need display/canvas coordinates, so convert back to
    // world space using the current clip transform.
    if (!clipTransform) {
      return touchPoints.map((point) => ({ x: point.x, y: point.y }));
    }

    return touchPoints.map((point) => {
      const world = denormalizeMaskPointForRotation(
        point.x,
        point.y,
        clipTransform,
      );
      return { x: world.x, y: world.y };
    });
  }, [touchPoints, clipTransform]);

  const pointLabels = useMemo<Array<number>>(() => {
    if (!touchPoints) return [];
    return touchPoints.map((point) => point.label as number);
  }, [touchPoints]);

  const mapFrameToLocalFrame = useMemo(() => {
    if (clip.type === "image") return 0;
    const clipFps = mediaInfo?.stats.video?.averagePacketRate || 24;
    const startFrame = mediaInfo?.startFrame ?? 0;
    return (
      Math.round(currentFrame * (clipFps / fps)) +
      Math.round(startFrame * (clipFps / fps))
    );
  }, [currentFrame, fps, mediaInfo, clip.type]);

  const { data, loading } = useMask({
    id: currentMask?.id || "",
    inputPath: asset.path,
    tool: "touch",
    points: touchDrawMode === "point" && points.length > 0 ? points : undefined, // Only send points if we have touch points
    pointLabels:
      touchDrawMode === "point" && points.length > 0 ? pointLabels : undefined,
    frameNumber: mapFrameToLocalFrame,
    displayWidth: rectWidth,
    displayHeight: rectHeight,
    mediaInfo: mediaInfo || undefined,
    clipTransform: clipTransform || undefined,
    enabled:
      touchMaskRefetchToken !== null && touchPoints && touchPoints.length > 0,
  });

  // Once a mask request has actually run and completed (loading true → false), clear the global trigger
  useEffect(() => {
    if (touchMaskRefetchToken == null) {
      previousLoadingRef.current = loading;
      return;
    }

    const wasLoading = previousLoadingRef.current;
    const isLoading = loading;

    if (wasLoading && !isLoading) {
      setTouchMaskRefetchToken(null);
    }

    previousLoadingRef.current = isLoading;
  }, [loading, touchMaskRefetchToken, setTouchMaskRefetchToken]);

  useEffect(() => {
    if (!loading && data && data !== lastDataRef.current) {
      lastDataRef.current = data;
      lastDataFrameRef.current = currentFrame;
    }
  }, [data, loading, currentFrame]);

  // Update rendered touch points and contours when mask data is received and store in mask keyframes
  useEffect(() => {
    if (!currentMask) {
      return;
    }

    if (data?.contours && data.contours.length > 0 && !loading) {
      if (lastDataFrameRef.current !== currentFrame) {
        return;
      }

      // Contours from the mask API are in display/canvas coordinates that
      // already account for the clip transform. To keep all stored mask
      // geometry in the same clip‑relative space as lasso/shape, project the
      // contours back into mask space using the inverse of the clip transform.
      const normalizedContours =
        clipTransform && data.contours
          ? data.contours.map((contour) => {
              const out: number[] = [];
              for (let i = 0; i < contour.length; i += 2) {
                const x = contour[i];
                const y = contour[i + 1];
                if (!Number.isFinite(x) || !Number.isFinite(y)) {
                  out.push(x, y);
                  continue;
                }
                const p = normalizeMaskPointForRotation(
                  x,
                  y,
                  clipTransform,
                );
                out.push(p.x, p.y);
              }
              return out;
            })
          : data.contours;

      setContours(normalizedContours);
      setRenderedTouchPoints(touchPoints || []);

      try {
            // Store contours, touch points, and lasso strokes in mask keyframes
        const masks = clip.masks || [];
        const currentMask = masks.find((m) => m.tool === "touch");

        if (currentMask && currentMask.keyframes) {
          const keyframes =
            currentMask.keyframes instanceof Map
              ? currentMask.keyframes
              : (currentMask.keyframes as Record<number, any>);

          if (!keyframes) return;

          const keyframeNumbers =
            keyframes instanceof Map
              ? Array.from(keyframes.keys()).sort((a, b) => a - b)
              : Object.keys(keyframes)
                  .map(Number)
                  .sort((a, b) => a - b);

          const activeKeyframe = keyframeNumbers
            .filter((k: number) => k <= currentFrame)
            .pop();
          const hasIncomingTouchPoints = !!(
            touchPoints && touchPoints.length > 0
          );
          const hasNewInput = hasIncomingTouchPoints;

          if (activeKeyframe !== undefined || hasNewInput) {
            const getKeyframeData = (frame: number | undefined) => {
              if (frame === undefined) return undefined;
              return keyframes instanceof Map
                ? keyframes.get(frame)
                : keyframes[frame];
            };

            const hasCurrentKeyframe =
              keyframes instanceof Map
                ? keyframes.has(currentFrame)
                : Object.prototype.hasOwnProperty.call(keyframes, currentFrame);

            const shouldTargetCurrentFrame =
              hasCurrentKeyframe ||
              (activeKeyframe !== currentFrame && hasNewInput);
            let targetKeyframe = shouldTargetCurrentFrame
              ? currentFrame
              : activeKeyframe;

            if (clip.type === "image") {
              targetKeyframe = 0;
            }

            if (targetKeyframe === undefined) {
              return;
            }

            const maskDataForActive = getKeyframeData(activeKeyframe);
            const maskDataForCurrent = getKeyframeData(currentFrame);
            const maskDataForTarget =
              targetKeyframe === activeKeyframe
                ? maskDataForActive
                : (maskDataForCurrent ?? maskDataForActive);
            const baseMaskData = maskDataForTarget || {};

            // Only update if any data has actually changed
            const existingContours = maskDataForTarget?.contours;
            const existingTouchPoints = maskDataForTarget?.touchPoints;
            const effectiveTouchPoints =
              targetKeyframe === currentFrame && hasIncomingTouchPoints
                ? touchPoints
                : (existingTouchPoints ?? []);

            const dataKey = `${clip.clipId}-${targetKeyframe}-${JSON.stringify(data.contours)}-${JSON.stringify(effectiveTouchPoints)}`;
            if (lastStoredDataRef.current === dataKey) {
              return; // Already stored, skip
            }

            const contoursChanged = !_.isEqual(
              existingContours,
              normalizedContours,
            );
            const touchPointsChanged =
              targetKeyframe === currentFrame &&
              hasIncomingTouchPoints &&
              !_.isEqual(existingTouchPoints, effectiveTouchPoints);

            if (contoursChanged || touchPointsChanged) {
              const updatedKeyframes =
                keyframes instanceof Map
                  ? new Map(keyframes)
                  : { ...keyframes };

              if (updatedKeyframes instanceof Map) {
                updatedKeyframes.set(targetKeyframe, {
                  ...baseMaskData,
                  contours: normalizedContours,
                  touchPoints: effectiveTouchPoints,
                });
              } else {
                updatedKeyframes[targetKeyframe] = {
                  ...baseMaskData,
                  contours: normalizedContours,
                  touchPoints: effectiveTouchPoints,
                };
              }

              const { clips } = useClipStore.getState();
              const targetClip = clips.find((c) => c.clipId === clip.clipId);

              if (targetClip && clip.clipId) {
                const currentMasks =
                  (targetClip as PreprocessorClipType).masks || [];
                const updatedMasks = currentMasks.map((m) =>
                  m.id === currentMask.id
                    ? { ...m, keyframes: updatedKeyframes }
                    : m,
                );

                useClipStore
                  .getState()
                  .updateClip(clip.clipId, { masks: updatedMasks });

                // Mark this data as stored
                lastStoredDataRef.current = dataKey;
              }
            }
          }
        }
      } catch (error) {
        console.error("Error storing touch mask contours:", error);
      }
    }
  }, [data, loading, touchPoints, clip.clipId, currentFrame, currentMask]);

  // Calculate non-rendered touch points (points without contours yet)
  const nonRenderedTouchPoints = useMemo(() => {
    if (renderedTouchPoints.length === 0) return touchPoints || [];
    if (contours.length > 0) return [];
    return touchPoints || [];
  }, [touchPoints, renderedTouchPoints, contours]);

  // Handle keyboard events for deletion
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        (e.key === "Backspace" || e.key === "Delete") &&
        selectedPoints.length > 0 &&
        onDeletePoints
      ) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        onDeletePoints(selectedPoints);
        setSelectedPoints([]);
      }
    };

    window.addEventListener("keydown", handleKeyDown, { capture: true });
    return () =>
      window.removeEventListener("keydown", handleKeyDown, { capture: true });
  }, [selectedPoints, onDeletePoints]);

  // Toggle point selection
  const togglePointSelection = (point: {
    x: number;
    y: number;
    label: 1 | 0;
  }) => {
    try {
      // Deselect any mask transforms/dragging in other previews
      window.dispatchEvent(new CustomEvent("apex-mask-clear-transforms"));
      setIsMaskDragging(false);
      setSelectedMaskId(null);
    } catch {}
    setSelectedPoints((prev) => {
      const isSelected = prev.some(
        (p) => p.x === point.x && p.y === point.y && p.label === point.label,
      );
      if (isSelected) {
        return prev.filter(
          (p) =>
            !(p.x === point.x && p.y === point.y && p.label === point.label),
        );
      } else {
        return [...prev, point];
      }
    });
  };

  // Listen for mask selection elsewhere to clear touch point selection
  useEffect(() => {
    const handleClearTouch = () => {
      setSelectedPoints([]);
    };
    window.addEventListener(
      "apex-mask-clear-touch-selection",
      handleClearTouch as EventListener,
    );
    return () => {
      window.removeEventListener(
        "apex-mask-clear-touch-selection",
        handleClearTouch as EventListener,
      );
    };
  }, []);

  const displayContours = useMemo(() => {
    return contours;
  }, [contours, clipTransform]);

  return (
    // Do not apply an additional clipping rect here; the parent preview
    // group already clips to the viewport. Having a second clip rectangle
    // combined with the outer/inner transforms in MaskPreview causes the
    // rotated contours to be cropped when re-entering mask mode.
    <Group visible={tool === "mask"} ref={groupRef} clipX={0} clipY={0}>
      {/* Render contours with zebra stripes (generated from lasso strokes and touch points) */}
      {Array.isArray(displayContours) &&
        displayContours.length > 0 &&
        displayContours.map((contour, index) => (
          <Group key={`contour-${index}`}>
            {/* White stripe background */}
            <Line
              points={contour}
              stroke="#ffffff"
              strokeWidth={1}
              lineCap="round"
              lineJoin="round"
              closed={true}
              listening={false}
              fill="rgba(0, 127, 245, 0.4)"
            />

            {/* Black stripe foreground with animation */}
            <Line
              points={contour}
              stroke="#000000"
              strokeWidth={1}
              dash={[4.5, 4.5]}
              dashOffset={-animationOffset}
              lineCap="round"
              lineJoin="round"
              closed={true}
              listening={false}
            />
          </Group>
        ))}

      {/* Render rendered touch points (points that have contours) */}
      {Array.isArray(contours) &&
        contours.length > 0 &&
        touchDrawMode === "point" &&
        renderedTouchPoints.map((point, index) => {
          const isPositive = point.label === 1;
          const color = isPositive ? "#3b82f6" : "#ef4444"; // blue-500 : red-500
          const radius = 8;
          const iconSize = 6;
          const isSelected = selectedPoints.some(
            (p) =>
              p.x === point.x && p.y === point.y && p.label === point.label,
          );

          return (
            <Group
              key={`rendered-touch-point-${index}`}
              x={point.x}
              y={point.y}
              onMouseOver={() => {
                // make the cursor a pointer
                const stage = groupRef.current?.getStage();
                if (stage) {
                  const container = stage.container();
                  container.style.cursor = "pointer";
                }
              }}
              onMouseLeave={() => {
                // make the cursor a default
                const stage = groupRef.current?.getStage();
                if (stage) {
                  const container = stage.container();
                  container.style.cursor = "crosshair";
                }
              }}
            >
              {/* Outer glow */}
              <Circle
                radius={radius + 4}
                fill={color}
                opacity={isSelected ? 0.7 : 0.2}
                stroke={isSelected ? "white" : undefined}
                strokeWidth={isSelected ? 1 : 0}
                listening={true}
                onClick={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                  togglePointSelection(point);
                }}
                onTap={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                  togglePointSelection(point);
                }}
                onMouseDown={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                }}
                onTouchStart={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                }}
              />
              {/* Main circle */}
              <Circle
                radius={radius}
                fill={color}
                opacity={0.9}
                listening={true}
                onClick={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                  togglePointSelection(point);
                }}
                onTap={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                  togglePointSelection(point);
                }}
                onMouseDown={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                }}
                onTouchStart={(e) => {
                  e.cancelBubble = true;
                  e.evt?.preventDefault?.();
                  e.evt?.stopPropagation?.();
                }}
              />
              {/* Icon - Plus or X */}
              {isPositive ? (
                // Plus icon (horizontal and vertical lines)
                <>
                  <Line
                    points={[-iconSize / 2, 0, iconSize / 2, 0]}
                    stroke="#ffffff"
                    strokeWidth={2}
                    lineCap="round"
                    listening={false}
                  />
                  <Line
                    points={[0, -iconSize / 2, 0, iconSize / 2]}
                    stroke="#ffffff"
                    strokeWidth={2}
                    lineCap="round"
                    listening={false}
                  />
                </>
              ) : (
                // X icon (diagonal lines)
                <>
                  <Line
                    points={[
                      -iconSize / 2,
                      -iconSize / 2,
                      iconSize / 2,
                      iconSize / 2,
                    ]}
                    stroke="#ffffff"
                    strokeWidth={2}
                    lineCap="round"
                    listening={false}
                  />
                  <Line
                    points={[
                      iconSize / 2,
                      -iconSize / 2,
                      -iconSize / 2,
                      iconSize / 2,
                    ]}
                    stroke="#ffffff"
                    strokeWidth={2}
                    lineCap="round"
                    listening={false}
                  />
                </>
              )}
            </Group>
          );
        })}

      {/* Render non-rendered touch points (points without contours yet) */}
      {nonRenderedTouchPoints.map((point, index) => {
        const isPositive = point.label === 1;
        const color = isPositive ? "#3b82f6" : "#ef4444"; // blue-500 : red-500
        const radius = 8;
        const iconSize = 6;
        const isSelected = selectedPoints.some(
          (p) => p.x === point.x && p.y === point.y && p.label === point.label,
        );

        return (
          <Group
            key={`non-rendered-touch-point-${index}`}
            x={point.x}
            y={point.y}
            onMouseOver={() => {
              // make the cursor a pointer
              // get the stage container
              const stage = groupRef.current?.getStage();
              if (stage) {
                const container = stage.container();
                container.style.cursor = "pointer";
              }
            }}
            onMouseLeave={() => {
              // make the cursor a default
              const stage = groupRef.current?.getStage();
              if (stage) {
                const container = stage.container();
                container.style.cursor = "crosshair";
              }
            }}
          >
            {/* Outer glow */}
            <Circle
              radius={radius + 4}
              fill={color}
              opacity={isSelected ? 0.6 : 0.2}
              stroke={isSelected ? "white" : undefined}
              strokeWidth={isSelected ? 1 : 0}
              listening={true}
              onClick={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
                togglePointSelection(point);
              }}
              onTap={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
                togglePointSelection(point);
              }}
              onMouseDown={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
              }}
              onTouchStart={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
              }}
            />
            {/* Main circle */}
            <Circle
              radius={radius}
              fill={color}
              opacity={0.9}
              listening={true}
              onClick={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
                togglePointSelection(point);
              }}
              onTap={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
                togglePointSelection(point);
              }}
              onMouseDown={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
              }}
              onTouchStart={(e) => {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
              }}
            />
            {/* Icon - Plus or X */}
            {isPositive ? (
              // Plus icon (horizontal and vertical lines)
              <>
                <Line
                  points={[-iconSize / 2, 0, iconSize / 2, 0]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
                <Line
                  points={[0, -iconSize / 2, 0, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
              </>
            ) : (
              // X icon (diagonal lines)
              <>
                <Line
                  points={[
                    -iconSize / 2,
                    -iconSize / 2,
                    iconSize / 2,
                    iconSize / 2,
                  ]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
                <Line
                  points={[
                    iconSize / 2,
                    -iconSize / 2,
                    -iconSize / 2,
                    iconSize / 2,
                  ]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
              </>
            )}
          </Group>
        );
      })}
    </Group>
  );
};

export default TouchMaskPreview;
