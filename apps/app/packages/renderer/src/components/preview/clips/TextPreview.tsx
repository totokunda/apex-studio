import { TextClipProps } from "@/lib/types";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Text, Transformer, Group, Line, Rect } from "react-konva";
import { Html } from "react-konva-utils";
import { useControlsStore } from "@/lib/control";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
import { useClipStore } from "@/lib/clip";
import { BaseClipApplicator } from "./apply/base";
import ApplicatorFilter from "./custom/ApplicatorFilter";
// (duplicate removed)

//@ts-ignore
Konva.Filters.Applicator = ApplicatorFilter;

//@ts-ignore
Konva._fixTextRendering = true;

const applyTextTransform = (text: string, textTransform: string) => {
  if (textTransform === "uppercase") {
    return text.toUpperCase();
  }
  if (textTransform === "lowercase") {
    return text.toLowerCase();
  }
  if (textTransform === "capitalize") {
    // Capitalize each word
    return text
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }
  return text;
};

const TextPreview: React.FC<
  TextClipProps & {
    rectWidth: number;
    rectHeight: number;
    applicators: BaseClipApplicator[];
    assetMode?: boolean;
  }
> = ({ clipId, rectWidth, rectHeight, applicators }) => {
  const textRef = useRef<Konva.Text>(null);
  const backgroundRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef<number>(0);
  const tool = useViewportStore((s) => s.tool);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const clipTransform = useClipStore((s) => s.getClipTransform(clipId));
  const clip = useClipStore((s) => s.getClipById(clipId)) as TextClipProps;
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const isInFrame = useMemo(() => {
    const f = Number(focusFrame);
    if (!Number.isFinite(f)) return true;
    const start = Number((clip as any)?.startFrame ?? 0);
    const endRaw = (clip as any)?.endFrame;
    const end =
      typeof endRaw === "number" && Number.isFinite(endRaw) ? endRaw : Infinity;
    return f >= start && f <= end;
  }, [focusFrame, (clip as any)?.startFrame, (clip as any)?.endFrame]);
  const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
  const addClipSelection = useControlsStore((s) => s.addClipSelection);
  const clearSelection = useControlsStore((s) => s.clearSelection);
  const { selectedClipIds, isFullscreen } = useControlsStore();
  const isSelected = useMemo(
    () => selectedClipIds.includes(clipId),
    [clipId, selectedClipIds],
  );

  const [isEditing, setIsEditing] = useState(false);
  const [caretVisible, setCaretVisible] = useState(true);
  const [caretPosition, setCaretPosition] = useState<number>(0); // Character index for caret position
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const mouseDownPositionRef = useRef<{ x: number; y: number } | null>(null);
  const lastClickTimeRef = useRef<number>(0);
  const lastClickPositionRef = useRef<{ x: number; y: number } | null>(null);
  const clickCountRef = useRef<number>(0);
  const clickResetTimerRef = useRef<NodeJS.Timeout | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [temporaryText, setTemporaryText] = useState<string | null>(
    clip?.text ?? null,
  );

  // Editing model refs (avoid stale React state during rapid key repeats / fast typing)
  const editingTextRef = useRef<string>(clip?.text ?? "");
  const caretPositionRef = useRef<number>(0);
  const selectionStartRef = useRef<number | null>(null);
  const selectionEndRef = useRef<number | null>(null);

  // Batch clip-store writes to once per animation frame (keeps typing responsive)
  const pendingClipTextRef = useRef<string | null>(null);
  const clipTextFlushRafRef = useRef<number | null>(null);

  const groupRef = useRef<Konva.Group>(null);
  const SNAP_THRESHOLD_PX = 4;
  const [guides, setGuides] = useState({
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
  });

  const [isInteracting, setIsInteracting] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [isTransforming, setIsTransforming] = useState(false);

  const textTransform = clip?.textTransform ?? "none";

  const text = clip?.text ?? "Double-click to edit";
  const fontSize = clip?.fontSize ?? 48;
  const fontWeight = clip?.fontWeight ?? 400;
  const fontStyle = clip?.fontStyle ?? "normal";
  const fontFamily = clip?.fontFamily ?? "Arial";
  const color = clip?.color ?? "#000000";
  const textAlign = clip?.textAlign ?? "left";
  const verticalAlign = clip?.verticalAlign ?? "top";
  const textDecoration = clip?.textDecoration ?? "none";
  const colorOpacity = clip?.colorOpacity ?? 100;

  // Stroke properties
  const strokeEnabled = clip?.strokeEnabled ?? false;
  const stroke = clip?.stroke ?? "#000000";
  const strokeWidth = clip?.strokeWidth ?? 2;

  // Shadow properties
  const shadowEnabled = clip?.shadowEnabled ?? false;
  const shadowColor = clip?.shadowColor ?? "#000000";
  const shadowOpacity = clip?.shadowOpacity ?? 75;
  const shadowBlur = clip?.shadowBlur ?? 4;
  const shadowOffsetX = clip?.shadowOffsetX ?? 2;
  const shadowOffsetY = clip?.shadowOffsetY ?? 2;

  // Background properties
  const backgroundEnabled = clip?.backgroundEnabled ?? false;
  const backgroundColor = clip?.backgroundColor ?? "#000000";
  const backgroundOpacity = clip?.backgroundOpacity ?? 100;
  const backgroundCornerRadius = clip?.backgroundCornerRadius ?? 0;

  // Don't render if clip is not found
  if (!clip) {
    return null;
  }

  // If the playhead leaves this clip, exit editing mode to avoid stray keyboard/mouse handlers.
  useEffect(() => {
    if (!isInFrame && isEditing) {
      setIsEditing(false);
      setSelectionStart(null);
      setSelectionEnd(null);
    }
  }, [isInFrame, isEditing]);

  const updateGuidesAndMaybeSnap = useCallback(
    (opts: { snap: boolean }) => {
      if (isRotating) return;
      const node = textRef.current;
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
      const distVCenter = Math.abs(dxToVCenter);
      const distHCenter = Math.abs(dyToHCenter);
      const distV25 = Math.abs(dxToV25);
      const distV75 = Math.abs(dxToV75);
      const distH25 = Math.abs(dyToH25);
      const distH75 = Math.abs(dyToH75);
      const distLeft = Math.abs(client.x - 0);
      const distRight = Math.abs(client.x + client.width - rectWidth);
      const distTop = Math.abs(client.y - 0);
      const distBottom = Math.abs(client.y + client.height - rectHeight);

      const nextGuides = {
        vCenter: distVCenter <= thresholdLocal,
        hCenter: distHCenter <= thresholdLocal,
        v25: distV25 <= thresholdLocal,
        v75: distV75 <= thresholdLocal,
        h25: distH25 <= thresholdLocal,
        h75: distH75 <= thresholdLocal,
        left: distLeft <= thresholdLocal,
        right: distRight <= thresholdLocal,
        top: distTop <= thresholdLocal,
        bottom: distBottom <= thresholdLocal,
      };
      setGuides(nextGuides);

      if (opts.snap) {
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
      }
    },
    [rectWidth, rectHeight, scale, setClipTransform, clipId, isRotating],
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

      let adjusted = {
        ...newBox,
        x: position.x + local.x * scale,
        y: position.y + local.y * scale,
        width: local.width * scale,
        height: local.height * scale,
      };

      const MIN_SIZE_ABS = 1e-3;
      if (adjusted.width < MIN_SIZE_ABS) adjusted.width = MIN_SIZE_ABS;
      if (adjusted.height < MIN_SIZE_ABS) adjusted.height = MIN_SIZE_ABS;

      return adjusted;
    },
    [rectWidth, rectHeight, scale, position.x, position.y, isRotating],
  );

  useEffect(() => {
    if (!isSelected) return;
    const tr = transformerRef.current;
    const txt = textRef.current;
    if (!tr || !txt) return;
    const raf = requestAnimationFrame(() => {
      tr.nodes([txt]);
      if (typeof (tr as any).forceUpdate === "function") {
        (tr as any).forceUpdate();
      }
      tr.getLayer()?.batchDraw?.();
    });
    return () => cancelAnimationFrame(raf);
  }, [isSelected]);

  // Initialize default transform if missing
  const defaultWidth = 400;
  const defaultHeight = 100;
  const offsetX = (rectWidth - defaultWidth) / 2;
  const offsetY = (rectHeight - defaultHeight) / 2;

  useEffect(() => {
    if (!clipTransform) {
      setClipTransform(clipId, {
        x: offsetX,
        y: offsetY,
        width: defaultWidth,
        height: defaultHeight,
        scaleX: 1,
        scaleY: 1,
        rotation: 0,
      });
    }
  }, [clipTransform, offsetX, offsetY, clipId, setClipTransform]);

  const handleDragMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      updateGuidesAndMaybeSnap({ snap: true });
      const node = textRef.current;
      if (node) {
        // Immediately move background rect to avoid visual lag
        if (backgroundRef.current) {
          backgroundRef.current.x(node.x());
          backgroundRef.current.y(node.y());
          //@ts-ignore
          backgroundRef.current.getLayer()?.batchDraw?.();
        }
        setClipTransform(clipId, { x: node.x(), y: node.y() });
      } else {
        const nx = e.target.x();
        const ny = e.target.y();
        if (backgroundRef.current) {
          backgroundRef.current.x(nx);
          backgroundRef.current.y(ny);
          //@ts-ignore
          backgroundRef.current.getLayer()?.batchDraw?.();
        }
        setClipTransform(clipId, { x: nx, y: ny });
      }
    },
    [setClipTransform, clipId, updateGuidesAndMaybeSnap],
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
    [clipId, addClipSelection, updateGuidesAndMaybeSnap],
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
      setGuides({
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
      });
    },
    [setClipTransform, clipId],
  );

  // Helper function to get accurate character positions using DOM Range API
  const getDisplayedChars = useCallback(
    (
      text: string,
      fontStyleStr: string,
      maxWidth: number,
      alignment: string,
      textDec: string,
      fontStyleProp: string,
    ) => {
      // Use main document instead of iframe to access loaded fonts
      const container = document.createElement("div");
      container.style.font = fontStyleStr;
      container.style.width = `${maxWidth}px`;
      container.style.wordWrap = "break-word";
      container.style.whiteSpace = "pre-wrap";
      container.style.lineHeight = String(textRef.current?.lineHeight() || 1);
      container.style.textAlign = alignment;
      container.style.textDecoration = textDec;
      container.style.fontStyle = fontStyleProp;
      container.style.position = "absolute";
      container.style.left = "-9999px";
      container.style.top = "-9999px";
      container.style.visibility = "hidden";
      container.textContent = text;
      document.body.prepend(container);

      const range = document.createRange();

      // Get the full text BBox
      range.selectNode(container);
      const mainBBox = range.getBoundingClientRect();

      const chars: Array<{ char: string; rect: DOMRect }> = [];
      const textNode = container.firstChild as Text;

      if (textNode) {
        text.split("").forEach((char, index) => {
          range.setStart(textNode, index);
          range.setEnd(textNode, index + 1);
          const nodeBBox = range.getBoundingClientRect();

          // Get relative position
          const x = nodeBBox.left - mainBBox.left;
          const y = nodeBBox.top - mainBBox.top;
          const rect = new DOMRect(x, y, nodeBBox.width, nodeBBox.height);

          chars.push({ char, rect });
        });
      }

      container.remove();

      return chars.filter((char) => char.rect.height > 0);
    },
    [],
  );

  // Build a map of all character bounding boxes for debugging and interaction
  const characterBoundingBoxes = useMemo(() => {
    if (!isInFrame) return [];
    const node = textRef.current;
    if (!node) return [];

    const displayedText = applyTextTransform(
      temporaryText ?? text,
      textTransform,
    );
    if (!displayedText) return [];

    const nodeX = clipTransform?.x ?? offsetX;
    const nodeY = clipTransform?.y ?? offsetY;
    const nodeWidth = clipTransform?.width ?? defaultWidth;
    const nodeHeight = clipTransform?.height ?? defaultHeight;

    const fontStyleString = `${fontWeight >= 700 ? "bold" : "normal"} ${fontSize}px ${fontFamily}`;

    // Get accurate character positions using DOM
    const chars = getDisplayedChars(
      displayedText,
      fontStyleString,
      nodeWidth,
      textAlign,
      textDecoration,
      fontStyle,
    );

    if (chars.length === 0) return [];

    // Calculate text block position to match Konva's rendering
    // Konva applies vertical alignment, so we need to match it exactly
    const totalTextHeight = Math.max(...chars.map((c) => c.rect.bottom));

    let textBlockStartY = nodeY;
    if (verticalAlign === "middle") {
      textBlockStartY = nodeY + (nodeHeight - totalTextHeight) / 2;
    } else if (verticalAlign === "bottom") {
      textBlockStartY = nodeY + nodeHeight - totalTextHeight;
    }

    const boxes: Array<{
      index: number;
      x: number;
      y: number;
      width: number;
      height: number;
      char: string;
    }> = [];

    chars.forEach((charData, index) => {
      const { char, rect } = charData;

      // Add node offset and vertical alignment offset
      boxes.push({
        index: index,
        x: nodeX + rect.left,
        y: textBlockStartY + rect.top,
        width: rect.width,
        height: rect.height,
        char: char === "\n" ? "\\n" : char,
      });
    });

    return boxes;
  }, [
    isEditing,
    temporaryText,
    text,
    textTransform,
    fontSize,
    fontFamily,
    fontWeight,
    fontStyle,
    textDecoration,
    textAlign,
    verticalAlign,
    clipTransform,
    offsetX,
    offsetY,
    defaultWidth,
    defaultHeight,
    getDisplayedChars,
    isInFrame,
  ]);

  // Keep refs in sync when React state changes (also covers enter/exit edit mode)
  useEffect(() => {
    if (!isInFrame) return;
    if (!isEditing) return;
    editingTextRef.current = temporaryText ?? clip?.text ?? "";
    caretPositionRef.current = caretPosition;
    selectionStartRef.current = selectionStart;
    selectionEndRef.current = selectionEnd;
  }, [
    isInFrame,
    isEditing,
    temporaryText,
    clip?.text,
    caretPosition,
    selectionStart,
    selectionEnd,
  ]);

  const flushClipTextToStore = useCallback(
    (textToPersist?: string) => {
      const next = typeof textToPersist === "string"
        ? textToPersist
        : pendingClipTextRef.current;
      if (typeof next !== "string") return;
      pendingClipTextRef.current = null;
      const updateClipStore = useClipStore.getState().updateClip;
      updateClipStore(clipId, { text: next });
    },
    [clipId],
  );

  const scheduleClipTextToStore = useCallback(
    (next: string) => {
      pendingClipTextRef.current = next;
      if (clipTextFlushRafRef.current != null) return;
      clipTextFlushRafRef.current = requestAnimationFrame(() => {
        clipTextFlushRafRef.current = null;
        flushClipTextToStore();
      });
    },
    [flushClipTextToStore],
  );

  useEffect(() => {
    return () => {
      if (clipTextFlushRafRef.current != null) {
        cancelAnimationFrame(clipTextFlushRafRef.current);
        clipTextFlushRafRef.current = null;
      }
      // Best-effort final flush on unmount
      flushClipTextToStore();
    };
  }, [flushClipTextToStore]);

  // Handle keyboard input (ref-driven to ensure keystrokes apply in order)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      const getCurrent = () => {
        const currentText = editingTextRef.current ?? "";
        const caret = caretPositionRef.current ?? 0;
        const selStart = selectionStartRef.current;
        const selEnd = selectionEndRef.current;
        const hasSelection =
          selStart !== null && selEnd !== null && selStart !== selEnd;
        return { currentText, caret, selStart, selEnd, hasSelection };
      };

      const setSelection = (start: number | null, end: number | null) => {
        selectionStartRef.current = start;
        selectionEndRef.current = end;
        setSelectionStart(start);
        setSelectionEnd(end);
      };

      const setCaret = (pos: number) => {
        caretPositionRef.current = pos;
        setCaretPosition(pos);
      };

      const commitText = (nextText: string) => {
        editingTextRef.current = nextText;
        setTemporaryText(nextText);
        scheduleClipTextToStore(nextText);
      };

      const { currentText, caret, selStart, selEnd, hasSelection } = getCurrent();

      // Handle Select All (Ctrl/Cmd + A)
      if ((e.ctrlKey || e.metaKey) && e.key === "a") {
        e.preventDefault();
        setSelection(0, currentText.length);
        setCaret(currentText.length);
        return;
      }

      // Handle Copy (Ctrl/Cmd + C)
      if ((e.ctrlKey || e.metaKey) && e.key === "c") {
        e.preventDefault();
        if (hasSelection) {
          const start = Math.min(selStart!, selEnd!);
          const end = Math.max(selStart!, selEnd!);
          const selectedText = currentText.substring(start, end);
          navigator.clipboard.writeText(selectedText);
        }
        return;
      }

      // Handle Cut (Ctrl/Cmd + X)
      if ((e.ctrlKey || e.metaKey) && e.key === "x") {
        e.preventDefault();
        if (hasSelection) {
          const start = Math.min(selStart!, selEnd!);
          const end = Math.max(selStart!, selEnd!);
          const selectedText = currentText.substring(start, end);

          // Copy to clipboard
          navigator.clipboard.writeText(selectedText);

          // Delete selection
          const beforeSelection = currentText.substring(0, start);
          const afterSelection = currentText.substring(end);
          const updatedText = beforeSelection + afterSelection;

          commitText(updatedText);
          setCaret(start);
          setSelection(null, null);
        }
        return;
      }

      // Handle Paste (Ctrl/Cmd + V)
      if ((e.ctrlKey || e.metaKey) && e.key === "v") {
        e.preventDefault();
        navigator.clipboard.readText().then((clipboardText) => {
          if (!clipboardText) return;

          const ZERO_WIDTH_SPACE = "\u200B";
          const {
            currentText: textNow,
            caret: caretNow,
            selStart: selStartNow,
            selEnd: selEndNow,
            hasSelection: hasSelNow,
          } = getCurrent();
          let updatedText: string;
          let newCaretPosition: number;

          if (hasSelNow) {
            // Replace selection with pasted text
            const start = Math.min(selStartNow!, selEndNow!);
            const end = Math.max(selStartNow!, selEndNow!);
            const beforeSelection = textNow.substring(0, start);
            const afterSelection = textNow.substring(end);
            updatedText = beforeSelection + clipboardText + afterSelection;
            newCaretPosition = start + clipboardText.length;

            setSelection(null, null);
          } else {
            // Insert at caret
            const beforeCaret = textNow.substring(0, caretNow);
            let afterCaret = textNow.substring(caretNow);

            // Remove zero-width space if we're pasting right after it
            if (afterCaret.startsWith(ZERO_WIDTH_SPACE)) {
              afterCaret = afterCaret.substring(1);
            }

            updatedText = beforeCaret + clipboardText + afterCaret;
            newCaretPosition = caretNow + clipboardText.length;
          }

          commitText(updatedText);
          setCaret(newCaretPosition);
        });
        return;
      }

      // Handle Escape - Exit editing mode
      if (e.key === "Escape") {
        e.preventDefault();
        setIsEditing(false);
        setSelectionStart(null);
        setSelectionEnd(null);
        if (textRef.current) {
          textRef.current.getStage()!.container().style.cursor = "default";
        }
        return;
      }

      // Handle Home - Move to start of line
      if (e.key === "Home") {
        e.preventDefault();
        const beforeCaret = currentText.substring(0, caret);
        const lastNewlineIndex = beforeCaret.lastIndexOf("\n");
        const lineStart = lastNewlineIndex + 1;
        setCaret(lineStart);
        setSelection(null, null);
        return;
      }

      // Handle End - Move to end of line
      if (e.key === "End") {
        e.preventDefault();
        const afterCaret = currentText.substring(caret);
        const nextNewlineIndex = afterCaret.indexOf("\n");
        if (nextNewlineIndex >= 0) {
          setCaret(caret + nextNewlineIndex);
        } else {
          setCaret(currentText.length);
        }
        setSelection(null, null);
        return;
      }

      // Handle Arrow Keys
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (hasSelection) {
          // If there's a selection, move to start of selection
          const start = Math.min(selStart!, selEnd!);
          setCaret(start);
          setSelection(null, null);
        } else if (caret > 0) {
          setCaret(caret - 1);
        }
        return;
      }

      if (e.key === "ArrowRight") {
        e.preventDefault();
        if (hasSelection) {
          // If there's a selection, move to end of selection
          const end = Math.max(selStart!, selEnd!);
          setCaret(end);
          setSelection(null, null);
        } else if (caret < currentText.length) {
          setCaret(caret + 1);
        }
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();

        if (characterBoundingBoxes.length === 0) {
          setCaret(0);
          setSelection(null, null);
          return;
        }

        // Find current character box to get current X position
        const currentBox =
          characterBoundingBoxes[
            Math.min(caret, characterBoundingBoxes.length - 1)
          ];
        if (!currentBox) {
          setCaret(0);
          setSelection(null, null);
          return;
        }

        const currentX = currentBox.x;
        const currentY = currentBox.y;

        // Find all boxes on the previous line (lower Y value)
        const boxesAbove = characterBoundingBoxes.filter(
          (box) => box.y < currentY - 5,
        );

        if (boxesAbove.length === 0) {
          // No line above, move to start
          setCaret(0);
          setSelection(null, null);
          return;
        }

        // Find the closest line above
        const closestLineY = Math.max(...boxesAbove.map((box) => box.y));
        const boxesOnPreviousLine = boxesAbove.filter(
          (box) => Math.abs(box.y - closestLineY) < 5,
        );

        // Find the box on that line closest to current X position
        let closestBox = boxesOnPreviousLine[0];
        let minDistance = Math.abs(closestBox.x - currentX);

        for (const box of boxesOnPreviousLine) {
          const distance = Math.abs(box.x - currentX);
          if (distance < minDistance) {
            minDistance = distance;
            closestBox = box;
          }
        }

        setCaret(closestBox.index);
        setSelection(null, null);
        return;
      }

      if (e.key === "ArrowDown") {
        e.preventDefault();

        if (characterBoundingBoxes.length === 0) {
          setCaret(0);
          setSelection(null, null);
          return;
        }

        // Find current character box to get current X position
        const currentBox =
          characterBoundingBoxes[
            Math.min(caret, characterBoundingBoxes.length - 1)
          ];
        if (!currentBox) {
          setCaret(currentText.length);
          setSelection(null, null);
          return;
        }

        const currentX = currentBox.x;
        const currentY = currentBox.y;

        // Find all boxes on the next line (higher Y value)
        const boxesBelow = characterBoundingBoxes.filter(
          (box) => box.y > currentY + 5,
        );

        if (boxesBelow.length === 0) {
          // No line below, move to end
          setCaret(currentText.length);
          setSelection(null, null);
          return;
        }

        // Find the closest line below
        const closestLineY = Math.min(...boxesBelow.map((box) => box.y));
        const boxesOnNextLine = boxesBelow.filter(
          (box) => Math.abs(box.y - closestLineY) < 5,
        );

        // Find the box on that line closest to current X position
        let closestBox = boxesOnNextLine[0];
        let minDistance = Math.abs(closestBox.x - currentX);

        for (const box of boxesOnNextLine) {
          const distance = Math.abs(box.x - currentX);
          if (distance < minDistance) {
            minDistance = distance;
            closestBox = box;
          }
        }

        // Check if we should position after this character or before
        const charMidpoint = closestBox.x + closestBox.width / 2;
        if (currentX > charMidpoint && closestBox.index < currentText.length) {
          setCaret(Math.min(closestBox.index + 1, currentText.length));
        } else {
          setCaret(closestBox.index);
        }

        setSelection(null, null);
        return;
      }

      // Handle Delete key
      if (e.key === "Delete") {
        e.preventDefault();

        // Case 1: Text is selected - delete the selection
        if (hasSelection) {
          const start = Math.min(selStart!, selEnd!);
          const end = Math.max(selStart!, selEnd!);

          const beforeSelection = currentText.substring(0, start);
          const afterSelection = currentText.substring(end);
          const updatedText = beforeSelection + afterSelection;

          commitText(updatedText);
          setCaret(start);
          setSelection(null, null);
        }
        // Case 2: No selection - delete character after caret
        else if (caret < currentText.length) {
          const ZERO_WIDTH_SPACE = "\u200B";
          const beforeCaret = currentText.substring(0, caret);
          let afterCaret = currentText.substring(caret + 1);

          // Also remove zero-width space if it's after the deleted character
          if (afterCaret.startsWith(ZERO_WIDTH_SPACE)) {
            afterCaret = afterCaret.substring(1);
          }

          const updatedText = beforeCaret + afterCaret;

          commitText(updatedText);

          // Caret stays in same position
        }
        return;
      }

      // Handle Backspace
      if (e.key === "Backspace") {
        e.preventDefault();

        // Case 1: Text is selected - delete the selection
        if (hasSelection) {
          const start = Math.min(selStart!, selEnd!);
          const end = Math.max(selStart!, selEnd!);

          const beforeSelection = currentText.substring(0, start);
          const afterSelection = currentText.substring(end);
          const updatedText = beforeSelection + afterSelection;

          commitText(updatedText);
          setCaret(start);
          setSelection(null, null);
        }
        // Case 2: No selection - delete character before caret
        else if (caret > 0) {
          const ZERO_WIDTH_SPACE = "\u200B";
          let beforeCaret = currentText.substring(0, caret - 1);
          let afterCaret = currentText.substring(caret);

          // Also remove zero-width space after if it's there
          if (afterCaret.startsWith(ZERO_WIDTH_SPACE)) {
            afterCaret = afterCaret.substring(1);
          }

          const updatedText = beforeCaret + afterCaret;

          commitText(updatedText);
          setCaret(caret - 1);
        }
        return;
      }
      // Check if this is a printable character
      else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();

        const newChar = e.key;
        let updatedText: string;
        let newCaretPosition: number;
        const ZERO_WIDTH_SPACE = "\u200B";

        // If there's a selection, replace it with the new character
        if (
          selStart !== null &&
          selEnd !== null &&
          selStart !== selEnd
        ) {
          const start = Math.min(selStart, selEnd);
          const end = Math.max(selStart, selEnd);

          const beforeSelection = currentText.substring(0, start);
          const afterSelection = currentText.substring(end);
          updatedText = beforeSelection + newChar + afterSelection;
          newCaretPosition = start + 1;

          // Clear selection
          setSelection(null, null);
        } else {
          // No selection - insert at caret
          const beforeCaret = currentText.substring(0, caret);
          let afterCaret = currentText.substring(caret);

          // Remove zero-width space if we're typing right after it
          if (afterCaret.startsWith(ZERO_WIDTH_SPACE)) {
            afterCaret = afterCaret.substring(1);
          }

          updatedText = beforeCaret + newChar + afterCaret;
          newCaretPosition = caret + 1;
        }

        // Update the text
        commitText(updatedText);

        // Move caret forward
        setCaret(newCaretPosition);
      }
      // Handle Enter key for newlines
      else if (e.key === "Enter") {
        e.preventDefault();

        let updatedText: string;
        let newCaretPosition: number;
        const ZERO_WIDTH_SPACE = "\u200B"; // Invisible character for positioning

        // If there's a selection, replace it with newline
        if (hasSelection) {
          const start = Math.min(selStart!, selEnd!);
          const end = Math.max(selStart!, selEnd!);

          const beforeSelection = currentText.substring(0, start);
          const afterSelection = currentText.substring(end);

          // Add zero-width space after newline to help with positioning
          updatedText =
            beforeSelection + "\n" + ZERO_WIDTH_SPACE + afterSelection;
          newCaretPosition = start + 1;

          // Clear selection
          setSelection(null, null);
        } else {
          // No selection - insert at caret
          const beforeCaret = currentText.substring(0, caret);
          const afterCaret = currentText.substring(caret);

          // Add zero-width space after newline to help with positioning
          updatedText = beforeCaret + "\n" + ZERO_WIDTH_SPACE + afterCaret;
          newCaretPosition = caret + 1;
        }

        commitText(updatedText);
        setCaret(newCaretPosition);

        return;
      }
    },
    [clipId, characterBoundingBoxes, isEditing, isInFrame, scheduleClipTextToStore],
  );

  // Helper function to find the nearest character index using bounding boxes
  const getCharIndexAtPosition = useCallback(
    (mouseX: number, mouseY: number): number => {
      if (characterBoundingBoxes.length === 0) return 0;

      const displayedText = applyTextTransform(
        temporaryText ?? text,
        textTransform,
      );
      const maxIndex = displayedText.length;

      // First, check if we clicked directly inside any character box
      for (const box of characterBoundingBoxes) {
        if (
          mouseX >= box.x &&
          mouseX <= box.x + box.width &&
          mouseY >= box.y &&
          mouseY <= box.y + box.height
        ) {
          // Clicked inside this character - determine if closer to start or end
          const charMidpoint = box.x + box.width / 2;
          if (mouseX < charMidpoint) {
            // Closer to start of character
            return box.index;
          } else {
            // Closer to end of character
            return Math.min(box.index + 1, maxIndex);
          }
        }
      }

      // Not inside any character - find the line that was clicked
      // Group boxes by line (using Y coordinate with tolerance)
      const lines: Array<{
        y: number;
        height: number;
        boxes: typeof characterBoundingBoxes;
      }> = [];

      for (const box of characterBoundingBoxes) {
        // Find if this box belongs to an existing line (within 5px tolerance)
        let foundLine = lines.find((line) => Math.abs(line.y - box.y) < 5);

        if (foundLine) {
          foundLine.boxes.push(box);
        } else {
          // Create a new line
          lines.push({
            y: box.y,
            height: box.height,
            boxes: [box],
          });
        }
      }

      // Sort lines by Y position
      lines.sort((a, b) => a.y - b.y);

      // Sort boxes within each line by their index (not X position) to maintain correct order
      for (const line of lines) {
        line.boxes.sort((a, b) => a.index - b.index);
      }

      // Find the line that contains the click Y position
      let targetLine = lines.find((line) => {
        const lineTop = line.y;
        const lineBottom = line.y + line.height;
        return mouseY >= lineTop && mouseY <= lineBottom;
      });

      // If click is not within any line bounds, find the closest line
      if (!targetLine) {
        let minYDistance = Infinity;

        for (const line of lines) {
          const lineCenterY = line.y + line.height / 2;
          const distance = Math.abs(lineCenterY - mouseY);

          if (distance < minYDistance) {
            minYDistance = distance;
            targetLine = line;
          }
        }
      }

      // Get all boxes on the clicked line
      const boxesOnLine = targetLine?.boxes || [];
      if (boxesOnLine.length === 0) {
        return 0;
      }

      // Find the visual start and end of the line (leftmost and rightmost X positions)
      // But exclude newline characters from visual bounds
      const visualBoxes = boxesOnLine.filter((box) => box.char !== "\\n");
      const allBoxes = boxesOnLine;

      if (visualBoxes.length === 0) {
        // Only newline on this line
        return allBoxes[0].index;
      }

      // Find leftmost and rightmost visual characters
      let leftmostBox = visualBoxes[0];
      let rightmostBox = visualBoxes[0];

      for (const box of visualBoxes) {
        if (box.x < leftmostBox.x) {
          leftmostBox = box;
        }
        if (box.x + box.width > rightmostBox.x + rightmostBox.width) {
          rightmostBox = box;
        }
      }

      // If clicked before the visual start of the line, position at start
      if (mouseX < leftmostBox.x) {
        return leftmostBox.index;
      }

      // If clicked after the visual end of the line, position after the rightmost character
      // (which means before any newline if present)
      if (mouseX > rightmostBox.x + rightmostBox.width) {
        const endIndex = Math.min(rightmostBox.index + 1, maxIndex);
        return endIndex;
      }

      // Otherwise, find the closest character on this line (excluding newlines)
      let closestBox = visualBoxes[0];
      let minDistance = Math.abs(closestBox.x + closestBox.width / 2 - mouseX);

      for (const box of visualBoxes) {
        const centerX = box.x + box.width / 2;
        const distance = Math.abs(centerX - mouseX);

        if (distance < minDistance) {
          minDistance = distance;
          closestBox = box;
        }
      }

      // Determine if caret should be before or after this character
      const charMidpoint = closestBox.x + closestBox.width / 2;
      if (mouseX < charMidpoint) {
        return closestBox.index;
      } else {
        const endIndex = Math.min(closestBox.index + 1, maxIndex);
        return endIndex;
      }
    },
    [characterBoundingBoxes, temporaryText, text, textTransform],
  );

  const handleMouseDown = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (isFullscreen) {
        if (!isEditing) {
          clearSelection();
          addClipSelection(clipId);
        }
        return;
      }

      if (!isEditing) {
        clearSelection();
        addClipSelection(clipId);
        return;
      }

      const stage = e.target.getStage();
      if (stage) {
        const pointerPosition = stage.getPointerPosition();
        if (pointerPosition) {
          // Store mouse down position for drag detection
          mouseDownPositionRef.current = {
            x: pointerPosition.x,
            y: pointerPosition.y,
          };

          const group = groupRef.current;
          if (group) {
            const transform = group.getAbsoluteTransform().copy();
            transform.invert();
            const localPos = transform.point(pointerPosition);

            const charIndex = getCharIndexAtPosition(localPos.x, localPos.y);

            // Start potential drag
            setIsDragging(true);
            setSelectionStart(charIndex);
            setSelectionEnd(charIndex);
            setCaretPosition(charIndex);

            // Refocus input to ensure it captures keyboard events
            setTimeout(() => {
              inputRef.current?.focus();
            }, 0);
          }
        }
      }
    },
    [
      isFullscreen,
      isEditing,
      getCharIndexAtPosition,
      clearSelection,
      addClipSelection,
      clipId,
    ],
  );

  const handleMouseMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (!isDragging || !isEditing) return;

      const stage = e.target.getStage();
      if (stage) {
        const pointerPosition = stage.getPointerPosition();
        if (pointerPosition) {
          const group = groupRef.current;
          if (group) {
            const transform = group.getAbsoluteTransform().copy();
            transform.invert();
            const localPos = transform.point(pointerPosition);

            const charIndex = getCharIndexAtPosition(localPos.x, localPos.y);
            setSelectionEnd(charIndex);
            setCaretPosition(charIndex);
          }
        }
      }
    },
    [isDragging, isEditing, getCharIndexAtPosition],
  );

  const handleMouseUp = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (!isDragging) return;

      setIsDragging(false);

      // Check if we actually dragged or just clicked
      const stage = e.target.getStage();
      if (stage && mouseDownPositionRef.current) {
        const pointerPosition = stage.getPointerPosition();
        if (pointerPosition) {
          const dx = Math.abs(
            pointerPosition.x - mouseDownPositionRef.current.x,
          );
          const dy = Math.abs(
            pointerPosition.y - mouseDownPositionRef.current.y,
          );
          const didActuallyDrag = dx > 3 || dy > 3; // More than 3 pixels is considered a drag

          // If didn't drag (just clicked), clear selection
          if (!didActuallyDrag) {
            setSelectionStart(null);
            setSelectionEnd(null);
          }
          // Otherwise, if we dragged but start === end, also clear
          else if (selectionStart === selectionEnd) {
            setSelectionStart(null);
            setSelectionEnd(null);
          }
          // Otherwise keep the selection visible
        }
      }

      mouseDownPositionRef.current = null;
    },
    [isDragging, selectionStart, selectionEnd],
  );

  // Helper to find word boundaries at a given position
  const getWordBoundaries = useCallback(
    (position: number, textContent: string) => {
      if (!textContent) return { start: 0, end: 0 };

      let start = position;
      let end = position;

      // Find start of word (move back until we hit whitespace or start)
      while (start > 0 && !/\s/.test(textContent[start - 1])) {
        start--;
      }

      // Find end of word (move forward until we hit whitespace or end)
      while (end < textContent.length && !/\s/.test(textContent[end])) {
        end++;
      }

      return { start, end };
    },
    [],
  );

  // onClick is now handled by mousedown/mouseup
  // Keep this minimal handler for compatibility
  const handleClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    const currentTime = Date.now();
    const stage = e.target.getStage();
    if (stage) {
      const pointerPosition = stage.getPointerPosition();
      if (pointerPosition) {
        const timeSinceLastClick = currentTime - lastClickTimeRef.current;

        // Check if position is close
        let isPositionClose = false;
        if (lastClickPositionRef.current) {
          const dx = Math.abs(
            pointerPosition.x - lastClickPositionRef.current.x,
          );
          const dy = Math.abs(
            pointerPosition.y - lastClickPositionRef.current.y,
          );
          isPositionClose = dx < 10 && dy < 10;
        }

        // Update click count
        if (timeSinceLastClick < 500 && isPositionClose) {
          clickCountRef.current++;
        } else {
          clickCountRef.current = 1;
        }

        // Reset click count after delay
        if (clickResetTimerRef.current) {
          clearTimeout(clickResetTimerRef.current);
        }
        clickResetTimerRef.current = setTimeout(() => {
          clickCountRef.current = 0;
        }, 600);

        lastClickTimeRef.current = currentTime;
        lastClickPositionRef.current = {
          x: pointerPosition.x,
          y: pointerPosition.y,
        };
      }
    }
  }, []);

  const handleDblClick = useCallback(() => {
    if (tool !== "pointer" || isFullscreen) return;

    // If not already editing, enter editing mode and set caret to end
    if (!isEditing) {
      setIsEditing(true);
      const displayedText = applyTextTransform(
        temporaryText ?? text,
        textTransform,
      );
      setCaretPosition(displayedText.length);
      // Clear any selection
      setSelectionStart(null);
      setSelectionEnd(null);
    } else {
      // Already in editing mode - handle multi-click selection
      const currentText = temporaryText ?? text ?? "";

      if (clickCountRef.current === 2) {
        // Double-click: select word at caret
        const { start, end } = getWordBoundaries(caretPosition, currentText);
        setSelectionStart(start);
        setSelectionEnd(end);
        setCaretPosition(end);
      } else if (clickCountRef.current >= 3) {
        // Triple-click: select all
        setSelectionStart(0);
        setSelectionEnd(currentText.length);
        setCaretPosition(currentText.length);
      }
    }

    // set the cursor to text
    if (textRef.current) {
      textRef.current.getStage()!.container().style.cursor = "text";
    }
  }, [
    tool,
    isFullscreen,
    isEditing,
    temporaryText,
    text,
    textTransform,
    caretPosition,
    getWordBoundaries,
  ]);

  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
      if (clickResetTimerRef.current) {
        clearTimeout(clickResetTimerRef.current);
      }
    };
  }, []);

  // Stable filters array
  const filtersArray = useMemo(
    () => [
      //@ts-ignore
      Konva.Filters.Applicator,
    ],
    [],
  );

  // Timeline-aware applicator signature (type + clipId + start-end)
  const applicatorsSignature = useMemo(() => {
    if (!applicators || applicators.length === 0) return "none";
    try {
      return applicators
        .map((a) => {
          const type = a?.constructor?.name || "Unknown";
          const start = (a as any)?.getStartFrame?.() ?? "u";
          const end = (a as any)?.getEndFrame?.() ?? "u";
          const owner = (a as any)?.getClip?.()?.clipId ?? "u";
          return `${type}#${owner}@${start}-${end}`;
        })
        .join("|");
    } catch {
      return `len:${applicators.length}`;
    }
  }, [applicators]);

  // Store-driven active flag
  const focusFrameControls = useControlsStore((s) => s.focusFrame);
  const clipsState = useClipStore((s) => s.clips);
  const applicatorsActiveStore = useMemo(() => {
    const apps = applicators || [];
    if (!apps.length) return false;
    const getClipById = useClipStore.getState().getClipById;
    const frame =
      typeof focusFrameControls === "number" ? focusFrameControls : 0;
    return apps.some((a) => {
      const owned = (a as any)?.getClip?.();
      const id = owned?.clipId;
      if (!id) return false;
      const sc = getClipById(id) as any;
      if (!sc) return false;
      const start = sc.startFrame ?? 0;
      const end = sc.endFrame ?? 0;
      return frame >= start && frame <= end;
    });
  }, [clipsState, focusFrameControls, applicatorsSignature]);

  // Cache text node bitmap when visual attributes or active-range change and not interacting/editing
  useEffect(() => {
    const node = textRef.current;
    if (!node) return;
    if (isInteracting || isEditing) return;
    // Recompute cache only on material visual changes
    node.clearCache();
    node.cache({ pixelRatio: 2 });
    //@ts-ignore
    node.getLayer()?.batchDraw?.();
  }, [
    // Interaction guards
    isInteracting,
    isEditing,
    // Visual attributes affecting raster content
    applicatorsSignature,
    applicatorsActiveStore,
    text,
    temporaryText,
    textTransform,
    fontSize,
    fontFamily,
    fontStyle,
    fontWeight,
    textDecoration,
    color,
    colorOpacity,
    strokeEnabled,
    stroke,
    strokeWidth,
    shadowEnabled,
    shadowColor,
    shadowOpacity,
    shadowBlur,
    shadowOffsetX,
    shadowOffsetY,
    textAlign,
    verticalAlign,
    // Geometry
    clipTransform?.width,
    clipTransform?.height,
    clipTransform?.rotation,
  ]);

  // While editing: keep things snappy by rendering live (no filters/caching).
  // We also clear any existing cache once when entering edit mode to avoid
  // "stale bitmap until blur" behavior.
  useEffect(() => {
    if (!isInFrame) return;
    if (!isEditing) return;
    const node = textRef.current;
    if (!node) return;

    const bg = backgroundRef.current;
    const raf = requestAnimationFrame(() => {
      try {
        node.clearCache();
        bg?.clearCache();
        //@ts-ignore
        node.getLayer()?.batchDraw?.();
      } catch {
        // best-effort
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [isInFrame, isEditing]);

  // Cache background rect bitmap when its visual attributes or active-range change and not interacting/editing
  useEffect(() => {
    const bg = backgroundRef.current;
    if (!bg) return;
    if (isInteracting || isEditing) return;
    if (!backgroundEnabled) return;
    bg.clearCache();
    bg.cache({ pixelRatio: 2 });
    //@ts-ignore
    bg.getLayer()?.batchDraw?.();
  }, [
    isInteracting,
    isEditing,
    backgroundEnabled,
    backgroundColor,
    backgroundOpacity,
    backgroundCornerRadius,
    applicatorsSignature,
    applicatorsActiveStore,
    clipTransform?.width,
    clipTransform?.height,
    clipTransform?.rotation,
    clipTransform?.x,
    clipTransform?.y,
  ]);

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
        setGuides({
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
        });
      }
    };

    const onTransform = () => {
      bumpSuppress();
      const node = textRef.current;
      if (!node) return;

      // Calculate new dimensions from scale
      const newWidth = node.width() * node.scaleX();
      const newHeight = node.height() * node.scaleY();

      // Immediately reset scale and apply as width/height instead
      // This prevents visual scaling of the text content
      node.width(newWidth);
      node.height(newHeight);
      node.scaleX(1);
      node.scaleY(1);

      // Keep background rect perfectly in sync while transforming
      const bg = backgroundRef.current;
      if (bg) {
        bg.x(node.x());
        bg.y(node.y());
        bg.width(newWidth);
        bg.height(newHeight);
        bg.rotation(node.rotation());
        //@ts-ignore
        bg.getLayer()?.batchDraw?.();
      }

      setClipTransform(clipId, {
        x: node.x(),
        y: node.y(),
        width: newWidth,
        height: newHeight,
        scaleX: 1,
        scaleY: 1,
        rotation: node.rotation(),
      });

      if (!isRotating) {
        updateGuidesAndMaybeSnap({ snap: false });
      }
    };
    const onTransformEnd = () => {
      bumpSuppress();
      const node = textRef.current;
      if (node) {
        const newWidth = node.width() * node.scaleX();
        const newHeight = node.height() * node.scaleY();
        node.width(newWidth);
        node.height(newHeight);
        node.scaleX(1);
        node.scaleY(1);

        // Final sync for background rect at transform end
        const bg = backgroundRef.current;
        if (bg) {
          bg.x(node.x());
          bg.y(node.y());
          bg.width(newWidth);
          bg.height(newHeight);
          bg.rotation(node.rotation());
          //@ts-ignore
          bg.getLayer()?.batchDraw?.();
        }
        setClipTransform(clipId, {
          x: node.x(),
          y: node.y(),
          width: newWidth,
          height: newHeight,
          scaleX: 1,
          scaleY: 1,
          rotation: node.rotation(),
        });
      }
      setIsTransforming(false);
      setIsInteracting(false);
      setIsRotating(false);
      setGuides({
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
      });
    };
    transformer.on("transformstart", onTransformStart);
    transformer.on("transform", onTransform);
    transformer.on("transformend", onTransformEnd);
    return () => {
      transformer.off("transformstart", onTransformStart);
      transformer.off("transform", onTransform);
      transformer.off("transformend", onTransformEnd);
    };
  }, [
    transformerRef.current,
    updateGuidesAndMaybeSnap,
    setClipTransform,
    clipId,
    isRotating,
  ]);

  useEffect(() => {
    if (!isInFrame) return;
    const handleWindowClick = (e: MouseEvent) => {
      if (!isSelected) return;

      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < suppressUntilRef.current) return;
      const stage = textRef.current?.getStage();
      const container = stage?.container();
      const node = e.target;
      if (!container?.contains(node as Node)) return;
      if (!stage || !container || !textRef.current) return;
      const containerRect = container.getBoundingClientRect();
      const pointerX = e.clientX - containerRect.left;
      const pointerY = e.clientY - containerRect.top;
      const textRect = textRef.current.getClientRect({
        skipShadow: true,
        skipStroke: true,
      });
      const insideText =
        pointerX >= textRect.x &&
        pointerX <= textRect.x + textRect.width &&
        pointerY >= textRect.y &&
        pointerY <= textRect.y + textRect.height;
      if (!insideText) {
        removeClipSelection(clipId);
        setIsEditing(false);
        // set the cursor to default
        if (textRef.current) {
          textRef.current.getStage()!.container().style.cursor = "default";
        }
      }
    };
    window.addEventListener("click", handleWindowClick);
    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, [clipId, isSelected, removeClipSelection, isEditing, isInFrame]);

  useEffect(() => {
    if (!isInFrame) return;
    if (!isEditing) {
      // Clean up zero-width spaces when exiting edit mode
      const ZERO_WIDTH_SPACE = "\u200B";
      const currentText = temporaryText ?? clip?.text ?? "";
      if (currentText.includes(ZERO_WIDTH_SPACE)) {
        const cleanedText = currentText.replace(
          new RegExp(ZERO_WIDTH_SPACE, "g"),
          "",
        );
        setTemporaryText(cleanedText);
        const updateClipStore = useClipStore.getState().updateClip;
        updateClipStore(clipId, { text: cleanedText });
      } else {
        setTemporaryText(clip?.text ?? "");
      }
      setSelectionStart(null);
      setSelectionEnd(null);
    } else {
      // Focus the input when entering edit mode
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  }, [isEditing, clip?.text, temporaryText, clipId, isInFrame]);

  // Exit editing mode if node is deselected
  useEffect(() => {
    if (isEditing && !isSelected) {
      setIsEditing(false);
      setSelectionStart(null);
      setSelectionEnd(null);
    }
  }, [isEditing, isSelected]);

  // Ensure input stays focused after text updates
  useEffect(() => {
    if (
      isEditing &&
      inputRef.current &&
      document.activeElement !== inputRef.current
    ) {
      inputRef.current.focus();
    }
  }, [isEditing, temporaryText, caretPosition]);

  // Auto-expand node height when text exceeds bounds
  useEffect(() => {
    if (!isEditing) return;

    const node = textRef.current;
    if (!node || characterBoundingBoxes.length === 0) return;

    const nodeHeight = clipTransform?.height ?? defaultHeight;

    // Calculate total text height from character boxes
    const maxY = Math.max(
      ...characterBoundingBoxes.map((box) => box.y + box.height),
    );
    const minY = Math.min(...characterBoundingBoxes.map((box) => box.y));
    const totalTextHeight = maxY - minY;

    // Add some padding
    const requiredHeight = totalTextHeight + 20;

    // If text exceeds current height, expand the node
    if (requiredHeight > nodeHeight) {
      const newHeight = Math.ceil(requiredHeight);
      setClipTransform(clipId, { height: newHeight });
    }
  }, [
    isEditing,
    characterBoundingBoxes,
    clipTransform,
    defaultHeight,
    setClipTransform,
    clipId,
  ]);

  // Listen for global mouse up to end dragging
  useEffect(() => {
    if (!isInFrame) return;
    const handleGlobalMouseUp = () => {
      if (isDragging) {
        // Create a fake Konva event for the handler
        const stage = textRef.current?.getStage();
        if (stage) {
          const fakeEvent = {
            target: textRef.current,
          } as any;
          handleMouseUp(fakeEvent);
        }
      }
    };

    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => {
      window.removeEventListener("mouseup", handleGlobalMouseUp);
    };
  }, [isDragging, handleMouseUp, isInFrame]);

  // Blinking caret animation
  useEffect(() => {
    if (!isInFrame) return;
    if (!isEditing) {
      setCaretVisible(true);
      return;
    }

    // Reset to visible when caret position changes
    setCaretVisible(true);

    const interval = setInterval(() => {
      setCaretVisible((prev) => !prev);
    }, 530);

    return () => clearInterval(interval);
  }, [isEditing, caretPosition, isInFrame]);

  // Calculate merged selection rectangles (one per line to eliminate gaps)
  const selectedCharacterBoxes = useMemo(() => {
    if (
      selectionStart === null ||
      selectionEnd === null ||
      selectionStart === selectionEnd
    ) {
      return [];
    }

    const start = Math.min(selectionStart, selectionEnd);
    const end = Math.max(selectionStart, selectionEnd);

    const selectedBoxes = characterBoundingBoxes.filter(
      (box) => box.index >= start && box.index < end,
    );

    // Group boxes by line (Y coordinate)
    const lineGroups = new Map<number, typeof selectedBoxes>();
    for (const box of selectedBoxes) {
      const lineY = Math.round(box.y);
      if (!lineGroups.has(lineY)) {
        lineGroups.set(lineY, []);
      }
      lineGroups.get(lineY)!.push(box);
    }

    // Create merged rectangles for each line
    const mergedBoxes: Array<{
      index: number;
      x: number;
      y: number;
      width: number;
      height: number;
      char: string;
    }> = [];

    for (const boxes of lineGroups.values()) {
      if (boxes.length === 0) continue;

      // Find leftmost and rightmost boxes on this line
      const leftmost = boxes.reduce(
        (min, box) => (box.x < min.x ? box : min),
        boxes[0],
      );
      const rightmost = boxes.reduce(
        (max, box) => (box.x + box.width > max.x + max.width ? box : max),
        boxes[0],
      );

      // Create a merged rectangle spanning from leftmost to rightmost
      mergedBoxes.push({
        index: leftmost.index,
        x: leftmost.x,
        y: leftmost.y,
        width: rightmost.x + rightmost.width - leftmost.x,
        height: leftmost.height,
        char: "",
      });
    }

    return mergedBoxes;
  }, [selectionStart, selectionEnd, characterBoundingBoxes]);

  // Calculate visual caret position based on character index using bounding boxes
  const caretVisualPosition = useMemo(() => {
    if (!textRef.current || !isEditing) return null;

    const displayedText = applyTextTransform(
      temporaryText ?? text,
      textTransform,
    );
    const nodeX = clipTransform?.x ?? offsetX;
    const nodeY = clipTransform?.y ?? offsetY;
    const nodeWidth = clipTransform?.width ?? defaultWidth;
    const nodeHeight = clipTransform?.height ?? defaultHeight;
    const node = textRef.current;
    const lineHeight = fontSize * (node.lineHeight() || 1);

    // Helper to calculate vertically aligned Y position
    const getAlignedY = (contentHeight: number = lineHeight) => {
      if (verticalAlign === "middle") {
        return nodeY + (nodeHeight - contentHeight) / 2;
      } else if (verticalAlign === "bottom") {
        return nodeY + nodeHeight - contentHeight;
      }
      return nodeY;
    };

    if (!displayedText) {
      // Empty text - show caret at start position based on alignment
      let caretX = nodeX;
      if (textAlign === "center") {
        caretX = nodeX + nodeWidth / 2;
      } else if (textAlign === "right") {
        caretX = nodeX + nodeWidth;
      }
      return { x: caretX, y: getAlignedY(), height: fontSize };
    }

    // Clamp caret position to valid range
    const clampedCaretPosition = Math.max(
      0,
      Math.min(caretPosition, displayedText.length),
    );

    if (characterBoundingBoxes.length === 0) {
      // No boxes yet - show caret at start
      let caretX = nodeX;
      if (textAlign === "center") {
        caretX = nodeX + nodeWidth / 2;
      } else if (textAlign === "right") {
        caretX = nodeX + nodeWidth;
      }
      return { x: caretX, y: getAlignedY(), height: fontSize };
    }

    // If caret is at position 0, place it at the start of the first character (or at text start if no characters)
    if (clampedCaretPosition === 0) {
      if (characterBoundingBoxes.length > 0) {
        const firstBox = characterBoundingBoxes[0];
        return { x: firstBox.x, y: firstBox.y, height: firstBox.height };
      }
      let caretX = nodeX;
      if (textAlign === "center") {
        caretX = nodeX + nodeWidth / 2;
      } else if (textAlign === "right") {
        caretX = nodeX + nodeWidth;
      }
      return { x: caretX, y: getAlignedY(), height: fontSize };
    }

    // For any other position, use the character boxes directly since they have correct positions

    // If caret is at the end of text, place it after the last character
    if (clampedCaretPosition >= characterBoundingBoxes.length) {
      const lastBox = characterBoundingBoxes[characterBoundingBoxes.length - 1];
      return {
        x: lastBox.x + lastBox.width,
        y: lastBox.y,
        height: lastBox.height,
      };
    }

    // Find the character box at the caret position
    // The caret should appear before the character at caretPosition
    const charBox = characterBoundingBoxes[clampedCaretPosition];

    if (charBox) {
      // Place caret at the left edge of this character
      return { x: charBox.x, y: charBox.y, height: charBox.height };
    }

    // Fallback: place at the end of the previous character
    const prevBox = characterBoundingBoxes[clampedCaretPosition - 1];
    if (prevBox) {
      return {
        x: prevBox.x + prevBox.width,
        y: prevBox.y,
        height: prevBox.height,
      };
    }

    // Final fallback
    let caretX = nodeX;
    if (textAlign === "center") {
      caretX = nodeX + nodeWidth / 2;
    } else if (textAlign === "right") {
      caretX = nodeX + nodeWidth;
    }
    return { x: caretX, y: getAlignedY(), height: fontSize };
  }, [
    isEditing,
    caretPosition,
    characterBoundingBoxes,
    clipTransform,
    offsetX,
    offsetY,
    temporaryText,
    text,
    textTransform,
    fontSize,
    textAlign,
    defaultWidth,
    verticalAlign,
    defaultHeight,
  ]);

  const handleMouseOver = useCallback(() => {
    if (!textRef.current) return;
    if (isEditing) {
      textRef.current.getStage()!.container().style.cursor = "text";
    } else {
      textRef.current.getStage()!.container().style.cursor = "default";
    }
  }, [isEditing]);

  const handleMouseOut = useCallback(() => {
    if (!textRef.current) return;
    textRef.current.getStage()!.container().style.cursor = "default";
  }, []);

  if (!isInFrame) {
    return null;
  }

  return (
    <React.Fragment>
      <Group
        ref={groupRef}
        clipX={0}
        clipY={0}
        clipWidth={rectWidth}
        clipHeight={rectHeight}
      >
        {backgroundEnabled && (
          <Rect
            x={clipTransform?.x ?? offsetX}
            y={clipTransform?.y ?? offsetY}
            width={clipTransform?.width ?? defaultWidth}
            applicators={applicators}
            //@ts-ignore
            filters={isEditing ? undefined : filtersArray}
            height={clipTransform?.height ?? defaultHeight}
            ref={backgroundRef}
            fill={backgroundColor}
            opacity={
              ((backgroundOpacity ?? 100) / 100) *
              ((clipTransform?.opacity ?? 100) / 100)
            }
            cornerRadius={backgroundCornerRadius}
            rotation={clipTransform?.rotation ?? 0}
            listening={false}
          />
        )}

        <Text
          draggable={tool === "pointer" && !isTransforming && !isEditing}
          ref={textRef}
          text={
            isEditing
              ? applyTextTransform(temporaryText ?? text, textTransform)
              : applyTextTransform(text, textTransform)
          }
          fontSize={fontSize}
          fontFamily={fontFamily}
          fontStyle={`${fontStyle} ${fontWeight >= 700 ? "bold" : "normal"}`}
          textDecoration={textDecoration}
          fill={color}
          fillOpacity={(colorOpacity ?? 100) / 100}
          stroke={strokeEnabled ? stroke : undefined}
          strokeWidth={strokeEnabled ? strokeWidth : undefined}
          strokeEnabled={strokeEnabled}
          shadowColor={shadowEnabled ? shadowColor : undefined}
          shadowBlur={shadowEnabled ? shadowBlur : undefined}
          shadowOpacity={
            shadowEnabled ? (shadowOpacity ?? 100) / 100 : undefined
          }
          shadowOffsetX={shadowEnabled ? shadowOffsetX : undefined}
          shadowOffsetY={shadowEnabled ? shadowOffsetY : undefined}
          shadowEnabled={shadowEnabled}
          align={textAlign}
          verticalAlign={verticalAlign}
          opacity={(clipTransform?.opacity ?? 100) / 100}
          backgroundEnabled={backgroundEnabled}
          backgroundColor={backgroundColor}
          backgroundOpacity={backgroundOpacity}
          backgroundCornerRadius={backgroundCornerRadius}
          x={clipTransform?.x ?? offsetX}
          y={clipTransform?.y ?? offsetY}
          width={clipTransform?.width ?? defaultWidth}
          height={clipTransform?.height ?? defaultHeight}
          scaleX={1}
          scaleY={1}
          rotation={clipTransform?.rotation ?? 0}
          onDragMove={handleDragMove}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          onClick={handleClick}
          onDblClick={handleDblClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          applicators={applicators}
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          //@ts-ignore
          filters={isEditing ? undefined : filtersArray}
          onTransform={(e) => {
            const node = e.target as Konva.Text;
            const newWidth = node.width() * node.scaleX();
            const newHeight = node.height() * node.scaleY();
            node.setAttrs({
              width: newWidth,
              height: newHeight,
              scaleX: 1,
              scaleY: 1,
            });
            // Also keep background rect in sync during resize
            if (backgroundRef.current) {
              backgroundRef.current.x(node.x());
              backgroundRef.current.y(node.y());
              backgroundRef.current.width(newWidth);
              backgroundRef.current.height(newHeight);
              //@ts-ignore
              backgroundRef.current.getLayer()?.batchDraw?.();
            }
          }}
        />

        {tool === "pointer" &&
          isSelected &&
          isInteracting &&
          !isRotating &&
          !isEditing &&
          !isFullscreen && (
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

      {/* Editing UI - wrapped in Group with same rotation as text */}
      <Group
        x={clipTransform?.x ?? offsetX}
        y={clipTransform?.y ?? offsetY}
        rotation={clipTransform?.rotation ?? 0}
        listening={false}
      >
        {/* Text selection highlight - with rotation */}
        {isEditing &&
          selectedCharacterBoxes.map((box) => {
            const textX = clipTransform?.x ?? offsetX;
            const textY = clipTransform?.y ?? offsetY;
            return (
              <Rect
                key={`highlight-${box.index}`}
                x={box.x - textX}
                y={box.y - textY}
                width={box.width}
                height={box.height}
                fill="#ADD8E6"
                opacity={0.5}
                listening={false}
              />
            );
          })}

        {/* Caret - with rotation */}
        {(() => {
          const shouldShow =
            isEditing &&
            caretVisible &&
            caretVisualPosition &&
            selectionStart === null &&
            selectionEnd === null;
          if (!shouldShow) return null;

          const textX = clipTransform?.x ?? offsetX;
          const textY = clipTransform?.y ?? offsetY;

          return (
            <Line
              x={0}
              y={0}
              points={[
                caretVisualPosition.x - textX,
                caretVisualPosition.y - textY,
                caretVisualPosition.x - textX,
                caretVisualPosition.y - textY + caretVisualPosition.height,
              ]}
              stroke={color}
              strokeWidth={2}
              listening={false}
              opacity={(colorOpacity ?? 100) / 100}
            />
          );
        })()}
      </Group>

      {/* Hidden input for keyboard capture */}
      {isEditing && (
        <Html>
          <input
            ref={inputRef}
            type="text"
            onKeyDown={handleKeyDown}
            style={{
              position: "absolute",
              top: "-9999px",
              left: "-9999px",
              width: "1px",
              height: "1px",
              opacity: 0,
              border: "none",
              outline: "none",
              padding: 0,
              margin: 0,
            }}
            autoFocus
          />
        </Html>
      )}

      <Transformer
        borderStroke="#AE81CE"
        anchorCornerRadius={8}
        anchorStroke="#E3E3E3"
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
        rotateEnabled={!isEditing}
        boundBoxFunc={transformerBoundBoxFunc as any}
        visible={tool === "pointer" && isSelected && !isFullscreen}
        keepRatio={false}
        ref={(node) => {
          transformerRef.current = node;
          if (node && textRef.current) {
            node.nodes([textRef.current]);
            if (typeof (node as any).forceUpdate === "function") {
              (node as any).forceUpdate();
            }
            node.getLayer()?.batchDraw?.();
          }
        }}
        enabledAnchors={[
          "top-left",
          "bottom-right",
          "top-right",
          "bottom-left",
          "middle-left",
          "middle-right",
          "top-center",
          "bottom-center",
        ]}
      />
    </React.Fragment>
  );
};

export default TextPreview;
