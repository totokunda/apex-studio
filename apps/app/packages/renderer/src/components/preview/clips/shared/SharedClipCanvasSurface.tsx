import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { sanitizeCornerRadius } from "@/lib/konva/sanitizeCornerRadius";
import type { ClipTransform } from "@/lib/types";
import { useViewportStore } from "@/lib/viewport";
import Konva from "konva";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Group, Image, Line, Transformer } from "react-konva";

type Guides = {
  vCenter: boolean;
  hCenter: boolean;
  v25: boolean;
  v75: boolean;
  h25: boolean;
  h75: boolean;
  left: boolean;
  right: boolean;
  top: boolean;
  bottom: boolean;
};

type AspectFitSize = {
  displayWidth: number;
  displayHeight: number;
  offsetX: number;
  offsetY: number;
};

const EMPTY_GUIDES: Guides = {
  vCenter: false,
  hCenter: false,
  v25: false,
  v75: false,
  h25: false,
  h75: false,
  left: false,
  right: false,
  top: false,
  bottom: false,
};

const SNAP_THRESHOLD_PX = 4;

export const getAspectFitSize = (
  sourceWidth: number,
  sourceHeight: number,
  rectWidth: number,
  rectHeight: number,
): AspectFitSize => {
  if (!sourceWidth || !sourceHeight || !rectWidth || !rectHeight) {
    return { displayWidth: 0, displayHeight: 0, offsetX: 0, offsetY: 0 };
  }

  const aspectRatio = sourceWidth / sourceHeight;
  let displayWidth = rectWidth;
  let displayHeight = rectHeight;

  if (rectWidth / rectHeight > aspectRatio) {
    displayWidth = rectHeight * aspectRatio;
  } else {
    displayHeight = rectWidth / aspectRatio;
  }

  const offsetX = (rectWidth - displayWidth) / 2;
  const offsetY = (rectHeight - displayHeight) / 2;

  return { displayWidth, displayHeight, offsetX, offsetY };
};

type SharedClipCanvasSurfaceProps = {
  clipId: string;
  rectWidth: number;
  rectHeight: number;
  displayWidth: number;
  displayHeight: number;
  offsetX: number;
  offsetY: number;
  clipTransform?: ClipTransform;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageRef: React.RefObject<Konva.Image | null>;
  imageSource?: HTMLCanvasElement | null;
  overlap: boolean;
  inputMode?: boolean;
  isInFrame?: boolean;
  hidden?: boolean;
  overrideClip?: boolean;
};

const SharedClipCanvasSurface: React.FC<SharedClipCanvasSurfaceProps> = ({
  clipId,
  rectWidth,
  rectHeight,
  displayWidth,
  displayHeight,
  offsetX,
  offsetY,
  clipTransform,
  canvasRef,
  imageRef,
  imageSource,
  overlap,
  inputMode = false,
  isInFrame = true,
  hidden = false,
  overrideClip = false,
}) => {
  const tool = useViewportStore((s) => s.tool);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
  const clearSelection = useControlsStore((s) => s.clearSelection);
  const addClipSelection = useControlsStore((s) => s.addClipSelection);
  const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);

  const isSelected = useMemo(
    () => selectedClipIds.includes(clipId),
    [clipId, selectedClipIds],
  );

  const groupRef = useRef<Konva.Group>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef(0);
  const [guides, setGuides] = useState<Guides>(EMPTY_GUIDES);
  const [isInteracting, setIsInteracting] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [isTransforming, setIsTransforming] = useState(false);

  const clearGuides = useCallback(() => {
    setGuides(EMPTY_GUIDES);
  }, []);

  const updateGuidesAndMaybeSnap = useCallback(
    (opts: { snap: boolean }) => {
      if (isRotating) return;
      const node = imageRef.current;
      const group = groupRef.current;
      if (!node || !group) return;
      const thresholdLocal = SNAP_THRESHOLD_PX / Math.max(0.0001, scale);
      const client = node.getClientRect({
        skipShadow: true,
        skipStroke: true,
        relativeTo: group as any,
      });
      const centerX = client.x + client.width / 2;
      const centerY = client.y + client.height / 2;
      const dxToVCenter = rectWidth / 2 - centerX;
      const dyToHCenter = rectHeight / 2 - centerY;
      const dxToV25 = rectWidth * 0.25 - centerX;
      const dxToV75 = rectWidth * 0.75 - centerX;
      const dyToH25 = rectHeight * 0.25 - centerY;
      const dyToH75 = rectHeight * 0.75 - centerY;

      const nextGuides: Guides = {
        vCenter: Math.abs(dxToVCenter) <= thresholdLocal,
        hCenter: Math.abs(dyToHCenter) <= thresholdLocal,
        v25: Math.abs(dxToV25) <= thresholdLocal,
        v75: Math.abs(dxToV75) <= thresholdLocal,
        h25: Math.abs(dyToH25) <= thresholdLocal,
        h75: Math.abs(dyToH75) <= thresholdLocal,
        left: Math.abs(client.x - 0) <= thresholdLocal,
        right: Math.abs(client.x + client.width - rectWidth) <= thresholdLocal,
        top: Math.abs(client.y - 0) <= thresholdLocal,
        bottom: Math.abs(client.y + client.height - rectHeight) <= thresholdLocal,
      };
      setGuides(nextGuides);

      if (!opts.snap) return;

      let deltaX = 0;
      let deltaY = 0;
      if (nextGuides.vCenter) {
        deltaX += dxToVCenter;
      } else if (nextGuides.v25) {
        deltaX += dxToV25;
      } else if (nextGuides.v75) {
        deltaX += dxToV75;
      } else if (nextGuides.left) {
        deltaX += -client.x;
      } else if (nextGuides.right) {
        deltaX += rectWidth - (client.x + client.width);
      }
      if (nextGuides.hCenter) {
        deltaY += dyToHCenter;
      } else if (nextGuides.h25) {
        deltaY += dyToH25;
      } else if (nextGuides.h75) {
        deltaY += dyToH75;
      } else if (nextGuides.top) {
        deltaY += -client.y;
      } else if (nextGuides.bottom) {
        deltaY += rectHeight - (client.y + client.height);
      }
      if (deltaX !== 0 || deltaY !== 0) {
        node.x(node.x() + deltaX);
        node.y(node.y() + deltaY);
        setClipTransform(clipId, { x: node.x(), y: node.y() });
      }
    },
    [clipId, imageRef, isRotating, rectHeight, rectWidth, scale, setClipTransform],
  );

  const transformerBoundBoxFunc = useCallback(
    (_oldBox: any, newBox: any) => {
      if (isRotating) return newBox;
      const invScale = 1 / Math.max(0.0001, scale);
      const local = {
        x: (newBox.x - position.x) * invScale,
        y: (newBox.y - position.y) * invScale,
        width: newBox.width * invScale,
        height: newBox.height * invScale,
      };
      const thresholdLocal = SNAP_THRESHOLD_PX * invScale;

      const left = local.x;
      const right = local.x + local.width;
      const top = local.y;
      const bottom = local.y + local.height;
      const v25 = rectWidth * 0.25;
      const v75 = rectWidth * 0.75;
      const h25 = rectHeight * 0.25;
      const h75 = rectHeight * 0.75;

      if (Math.abs(left - 0) <= thresholdLocal) {
        local.x = 0;
        local.width = right - local.x;
      } else if (Math.abs(left - v25) <= thresholdLocal) {
        local.x = v25;
        local.width = right - local.x;
      } else if (Math.abs(left - v75) <= thresholdLocal) {
        local.x = v75;
        local.width = right - local.x;
      }
      if (Math.abs(rectWidth - right) <= thresholdLocal) {
        local.width = rectWidth - local.x;
      } else if (Math.abs(v75 - right) <= thresholdLocal) {
        local.width = v75 - local.x;
      } else if (Math.abs(v25 - right) <= thresholdLocal) {
        local.width = v25 - local.x;
      }
      if (Math.abs(top - 0) <= thresholdLocal) {
        local.y = 0;
        local.height = bottom - local.y;
      } else if (Math.abs(top - h25) <= thresholdLocal) {
        local.y = h25;
        local.height = bottom - local.y;
      } else if (Math.abs(top - h75) <= thresholdLocal) {
        local.y = h75;
        local.height = bottom - local.y;
      }
      if (Math.abs(rectHeight - bottom) <= thresholdLocal) {
        local.height = rectHeight - local.y;
      } else if (Math.abs(h75 - bottom) <= thresholdLocal) {
        local.height = h75 - local.y;
      } else if (Math.abs(h25 - bottom) <= thresholdLocal) {
        local.height = h25 - local.y;
      }

      const adjusted = {
        ...newBox,
        x: position.x + local.x * scale,
        y: position.y + local.y * scale,
        width: Math.max(local.width * scale, 1e-3),
        height: Math.max(local.height * scale, 1e-3),
      };
      return adjusted;
    },
    [isRotating, position.x, position.y, rectHeight, rectWidth, scale],
  );

  useEffect(() => {
    if (!isSelected) return;
    const tr = transformerRef.current;
    const img = imageRef.current;
    if (!tr || !img) return;
    const raf = requestAnimationFrame(() => {
      tr.nodes([img]);
      if (typeof (tr as any).forceUpdate === "function") {
        (tr as any).forceUpdate();
      }
      tr.getLayer()?.batchDraw?.();
    });
    return () => cancelAnimationFrame(raf);
  }, [imageRef, isSelected]);

  useEffect(() => {
    if (overrideClip || displayWidth <= 0 || displayHeight <= 0) return;
    const hasTransform = !!clipTransform;
    const width = clipTransform?.width ?? 0;
    const height = clipTransform?.height ?? 0;
    if (hasTransform && width > 0 && height > 0) return;

    setClipTransform(clipId, {
      x: offsetX,
      y: offsetY,
      width: displayWidth,
      height: displayHeight,
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
    });
  }, [
    clipId,
    clipTransform,
    displayHeight,
    displayWidth,
    offsetX,
    offsetY,
    overrideClip,
    setClipTransform,
  ]);

  useEffect(() => {
    if (!clipTransform || overrideClip) return;
    const currentWidth = clipTransform.width ?? 0;
    const currentHeight = clipTransform.height ?? 0;
    if (currentWidth > 0 && currentHeight > 0) return;

    const fallbackWidth =
      (displayWidth && displayWidth > 0 ? displayWidth : currentWidth) || 1;
    const fallbackHeight =
      (displayHeight && displayHeight > 0 ? displayHeight : currentHeight) || 1;

    setClipTransform(clipId, {
      ...clipTransform,
      x: offsetX,
      y: offsetY,
      width: Math.max(fallbackWidth, 1),
      height: Math.max(fallbackHeight, 1),
    });
  }, [
    clipId,
    clipTransform,
    displayHeight,
    displayWidth,
    offsetX,
    offsetY,
    overrideClip,
    setClipTransform,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !displayWidth || !displayHeight) return;
    const width = Math.floor(displayWidth);
    const height = Math.floor(displayHeight);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }, [canvasRef, displayHeight, displayWidth]);

  const handleDragMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      updateGuidesAndMaybeSnap({ snap: true });
      const node = imageRef.current;
      if (node) {
        setClipTransform(clipId, { x: node.x(), y: node.y() });
      } else {
        setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
      }
    },
    [clipId, imageRef, setClipTransform, updateGuidesAndMaybeSnap],
  );

  const handleDragStart = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.target.getStage()!.container().style.cursor = "grab";
      addClipSelection(clipId);
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
      setIsInteracting(true);
      updateGuidesAndMaybeSnap({ snap: true });
    },
    [addClipSelection, clipId, updateGuidesAndMaybeSnap],
  );

  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.target.getStage()!.container().style.cursor = "default";
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
      setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
      setIsInteracting(false);
      clearGuides();
    },
    [clearGuides, clipId, setClipTransform],
  );

  const handleClick = useCallback(() => {
    if (isFullscreen || hidden) return;
    clearSelection();
    addClipSelection(clipId);
  }, [addClipSelection, clearSelection, clipId, hidden, isFullscreen]);

  useEffect(() => {
    const transformer = transformerRef.current;
    if (!transformer) return;
    const bumpSuppress = () => {
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 300);
    };
    const persistTransform = () => {
      const node = imageRef.current;
      if (!node) return;
      const newWidth = node.width() * node.scaleX();
      const newHeight = node.height() * node.scaleY();
      setClipTransform(
        clipId,
        {
          x: node.x(),
          y: node.y(),
          width: newWidth,
          height: newHeight,
          scaleX: 1,
          scaleY: 1,
          rotation: node.rotation(),
        },
        true,
        true,
      );
      node.width(newWidth);
      node.height(newHeight);
      node.scaleX(1);
      node.scaleY(1);
    };
    const onTransformStart = () => {
      bumpSuppress();
      setIsTransforming(true);
      const active = (transformer as any)?.getActiveAnchor?.();
      const rotating = typeof active === "string" && active.includes("rotater");
      setIsRotating(!!rotating);
      setIsInteracting(true);
      if (!rotating) {
        updateGuidesAndMaybeSnap({ snap: false });
      } else {
        clearGuides();
      }
    };
    const onTransform = () => {
      bumpSuppress();
      if (!isRotating) {
        updateGuidesAndMaybeSnap({ snap: false });
      }
      persistTransform();
    };
    const onTransformEnd = () => {
      bumpSuppress();
      setIsTransforming(false);
      setIsInteracting(false);
      setIsRotating(false);
      clearGuides();
      persistTransform();
    };
    transformer.on("transformstart", onTransformStart);
    transformer.on("transform", onTransform);
    transformer.on("transformend", onTransformEnd);
    return () => {
      transformer.off("transformstart", onTransformStart);
      transformer.off("transform", onTransform);
      transformer.off("transformend", onTransformEnd);
    };
  }, [clearGuides, clipId, imageRef, isRotating, setClipTransform, updateGuidesAndMaybeSnap]);

  useEffect(() => {
    if (inputMode) return;
    const handleWindowClick = (e: MouseEvent) => {
      if (!isSelected) return;
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < suppressUntilRef.current) return;
      const stage = imageRef.current?.getStage();
      const container = stage?.container();
      const node = e.target;
      if (!container?.contains(node as Node)) return;
      if (!stage || !container || !imageRef.current) return;
      const containerRect = container.getBoundingClientRect();
      const pointerX = e.clientX - containerRect.left;
      const pointerY = e.clientY - containerRect.top;
      const imgRect = imageRef.current.getClientRect({
        skipShadow: true,
        skipStroke: true,
      });
      const insideImage =
        pointerX >= imgRect.x &&
        pointerX <= imgRect.x + imgRect.width &&
        pointerY >= imgRect.y &&
        pointerY <= imgRect.y + imgRect.height;

      if (!insideImage) {
        removeClipSelection(clipId);
      }
    };
    window.addEventListener("click", handleWindowClick);
    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, [clipId, imageRef, inputMode, isSelected, removeClipSelection]);

  const pixelCrop = useMemo(() => {
    const crop = clipTransform?.crop;
    if (!crop || !displayWidth || !displayHeight) return undefined;
    return {
      x: crop.x * displayWidth,
      y: crop.y * displayHeight,
      width: crop.width * displayWidth,
      height: crop.height * displayHeight,
    };
  }, [clipTransform?.crop, displayHeight, displayWidth]);

  const nodeWidth = useMemo(
    () =>
      clipTransform?.width && clipTransform.width > 0
        ? clipTransform.width
        : displayWidth || 1,
    [clipTransform?.width, displayWidth],
  );
  const nodeHeight = useMemo(
    () =>
      clipTransform?.height && clipTransform.height > 0
        ? clipTransform.height
        : displayHeight || 1,
    [clipTransform?.height, displayHeight],
  );
  const safeCornerRadius = useMemo(
    () =>
      sanitizeCornerRadius(clipTransform?.cornerRadius, nodeWidth, nodeHeight),
    [clipTransform?.cornerRadius, nodeHeight, nodeWidth],
  );

  if (hidden || !isInFrame) {
    return null;
  }

  return (
    <React.Fragment>
      <Group ref={groupRef} clipX={0} clipY={0} clipWidth={rectWidth} clipHeight={rectHeight}>
        <Image
          visible={!hidden}
          listening={!hidden}
          draggable={tool === "pointer" && !isTransforming && !inputMode && !hidden}
          ref={imageRef}
          cornerRadius={safeCornerRadius}
          opacity={(clipTransform?.opacity ?? 100) / 100}
          image={imageSource || canvasRef.current || undefined}
          x={clipTransform?.x ?? offsetX}
          y={clipTransform?.y ?? offsetY}
          width={nodeWidth}
          height={nodeHeight}
          scaleX={clipTransform?.scaleX ?? 1}
          scaleY={clipTransform?.scaleY ?? 1}
          rotation={clipTransform?.rotation ?? 0}
          crop={pixelCrop}
          onDragMove={handleDragMove}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          onClick={handleClick}
        />

        {tool === "pointer" && isSelected && isInteracting && !isRotating && !isFullscreen && (
          <React.Fragment>
            {guides.vCenter && (
              <Line
                listening={false}
                points={[rectWidth / 2, 0, rectWidth / 2, rectHeight]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.v25 && (
              <Line
                listening={false}
                points={[rectWidth * 0.25, 0, rectWidth * 0.25, rectHeight]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.v75 && (
              <Line
                listening={false}
                points={[rectWidth * 0.75, 0, rectWidth * 0.75, rectHeight]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.hCenter && (
              <Line
                listening={false}
                points={[0, rectHeight / 2, rectWidth, rectHeight / 2]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.h25 && (
              <Line
                listening={false}
                points={[0, rectHeight * 0.25, rectWidth, rectHeight * 0.25]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.h75 && (
              <Line
                listening={false}
                points={[0, rectHeight * 0.75, rectWidth, rectHeight * 0.75]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.left && (
              <Line
                listening={false}
                points={[0, 0, 0, rectHeight]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.right && (
              <Line
                listening={false}
                points={[rectWidth, 0, rectWidth, rectHeight]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.top && (
              <Line
                listening={false}
                points={[0, 0, rectWidth, 0]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
            {guides.bottom && (
              <Line
                listening={false}
                points={[0, rectHeight, rectWidth, rectHeight]}
                stroke={"#AE81CE"}
                strokeWidth={1}
                dash={[6, 4]}
              />
            )}
          </React.Fragment>
        )}
      </Group>
      <Transformer
        borderStroke="#AE81CE"
        anchorCornerRadius={8}
        anchorStroke="#E3E3E3"
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        visible={
          !hidden &&
          tool === "pointer" &&
          isSelected &&
          !isFullscreen &&
          overlap &&
          !inputMode
        }
        listening={!hidden}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
        boundBoxFunc={transformerBoundBoxFunc as any}
        ref={(node) => {
          transformerRef.current = node;
          if (node && imageRef.current) {
            node.nodes([imageRef.current]);
            if (typeof (node as any).forceUpdate === "function") {
              (node as any).forceUpdate();
            }
            node.getLayer()?.batchDraw?.();
          }
        }}
        enabledAnchors={["top-left", "bottom-right", "top-right", "bottom-left"]}
      />
    </React.Fragment>
  );
};

export default SharedClipCanvasSurface;
