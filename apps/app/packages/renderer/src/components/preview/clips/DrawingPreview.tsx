import React, { useCallback, useRef } from "react";
import { Group } from "react-konva";
import { DrawingClipProps, DrawingLine } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { useViewportStore } from "@/lib/viewport";
import { useControlsStore } from "@/lib/control";
import { useDrawingStore } from "@/lib/drawing";
import { KonvaEventObject } from "konva/lib/Node";
import Konva from "konva";
import DrawingLineComponent from "./custom/DrawingLine";
import { BaseClipApplicator } from "./apply/base";

interface DrawingPreviewProps extends DrawingClipProps {
  rectWidth: number;
  rectHeight: number;
  assetMode?: boolean;
  tempLinesOverride?: DrawingLine[];
  applicators: BaseClipApplicator[];
}

const DrawingPreview: React.FC<DrawingPreviewProps> = ({
  clipId,
  lines = [],
  rectWidth,
  rectHeight,
  tempLinesOverride,
  applicators,
}) => {
  const { updateClip } = useClipStore();
  const removeClip = useClipStore((s) => s.removeClip);
  const tool = useViewportStore((s) => s.tool);
  const isSelected = useControlsStore((s) =>
    s.selectedClipIds.includes(clipId),
  );
  const addClipSelection = useControlsStore((s) => s.addClipSelection);
  const clearSelection = useControlsStore((s) => s.clearSelection);
  const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
  const selectedLineId = useDrawingStore((s) => s.selectedLineId);
  const setSelectedLineId = useDrawingStore((s) => s.setSelectedLineId);
  const lineRefs = useRef<{ [key: string]: Konva.Line }>({});
  const groupRef = useRef<Konva.Group>(null);
  const suppressUntilRef = useRef(0);

  const handleLineClick = useCallback(
    (e: KonvaEventObject<any>, lineId: string) => {
      if (tool !== "pointer") return;
      e.cancelBubble = true;

      // Select the clip if not already selected
      if (!isSelected) {
        clearSelection();
        addClipSelection(clipId);
      }

      setSelectedLineId(lineId);
    },
    [
      tool,
      isSelected,
      clearSelection,
      addClipSelection,
      clipId,
      setSelectedLineId,
    ],
  );

  const handleTransformEnd = useCallback(
    (lineId: string) => {
      const line = lineRefs.current[lineId];
      if (!line) return;

      const clip = useClipStore
        .getState()
        .getClipById(clipId) as DrawingClipProps;
      if (!clip || !clip.lines) return;

      const updatedLines = clip.lines.map((l) => {
        if (l.lineId === lineId) {
          return {
            ...l,
            transform: {
              ...l.transform,
              x: line.x(),
              y: line.y(),
              scaleX: line.scaleX(),
              scaleY: line.scaleY(),
              rotation: line.rotation(),
            },
          };
        }
        return l;
      });

      updateClip(clipId, { lines: updatedLines });

      // Suppress clicks for 100ms after transform
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = now + 100;
    },
    [clipId, updateClip],
  );

  const handleDragEnd = useCallback(
    (lineId: string) => {
      const line = lineRefs.current[lineId];
      if (!line) return;

      const clip = useClipStore
        .getState()
        .getClipById(clipId) as DrawingClipProps;
      if (!clip || !clip.lines) return;

      const updatedLines = clip.lines.map((l) => {
        if (l.lineId === lineId) {
          return {
            ...l,
            transform: {
              ...l.transform,
              x: line.x(),
              y: line.y(),
            },
          };
        }
        return l;
      });

      updateClip(clipId, { lines: updatedLines });

      // Suppress clicks for 100ms after drag
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = now + 100;
    },
    [clipId, updateClip],
  );

  const setLineRef = useCallback((lineId: string, ref: Konva.Line | null) => {
    if (ref) {
      lineRefs.current[lineId] = ref;
    }
  }, []);

  const handleGroupClick = useCallback(
    (e: KonvaEventObject<any>) => {
      if (tool !== "pointer") return;
      // Check if the click target is the group itself (not a child element)
      if (e.target === e.currentTarget) {
        // Deselect the line
        setSelectedLineId(null);
      }
    },
    [tool, setSelectedLineId],
  );

  // Deselect line when clicking outside the drawing clip
  React.useEffect(() => {
    const handleWindowClick = (e: MouseEvent) => {
      if (!selectedLineId) return;
      // Only this clip (that owns the selected line) should handle deselection
      if (!(lines || []).some((l) => l.lineId === selectedLineId)) return;

      // Check suppression timestamp
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < suppressUntilRef.current) return;

      const stage = groupRef.current?.getStage();
      const container = stage?.container();
      const node = e.target;
      if (!container?.contains(node as Node)) return;
      if (!stage || !container || !groupRef.current) return;
      const containerRect = container.getBoundingClientRect();
      const pointerX = e.clientX - containerRect.left;
      const pointerY = e.clientY - containerRect.top;
      const groupRect = groupRef.current.getClientRect({
        skipShadow: true,
        skipStroke: true,
      });
      const insideGroup =
        pointerX >= groupRect.x &&
        pointerX <= groupRect.x + groupRect.width &&
        pointerY >= groupRect.y &&
        pointerY <= groupRect.y + groupRect.height;

      if (!insideGroup) {
        setSelectedLineId(null);
      }
    };
    window.addEventListener("click", handleWindowClick);
    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, [selectedLineId, setSelectedLineId, lines]);

  // Auto-delete empty drawing clips
  React.useEffect(() => {
    if (lines && lines.length === 0) {
      removeClip(clipId);
      removeClipSelection(clipId);
      setSelectedLineId(null);
    }
  }, [lines, clipId, removeClip, removeClipSelection, setSelectedLineId]);

  const renderLines = tempLinesOverride ?? lines;

  return (
    <Group
      ref={groupRef}
      clipX={0}
      clipY={0}
      clipWidth={rectWidth}
      clipHeight={rectHeight}
      onClick={handleGroupClick}
      onTap={handleGroupClick}
    >
      {renderLines.map((line) => {
        return (
          <DrawingLineComponent
            key={line.lineId}
            applicators={applicators}
            line={line}
            lineOpacity={line.opacity / 100}
            selectedLineId={selectedLineId ?? ""}
            handleLineClick={handleLineClick}
            handleDragEnd={handleDragEnd}
            handleTransformEnd={handleTransformEnd}
            isLineSelected={isSelected}
            setLineRef={setLineRef}
            rectWidth={rectWidth}
            rectHeight={rectHeight}
          />
        );
      })}
    </Group>
  );
};

export default DrawingPreview;
