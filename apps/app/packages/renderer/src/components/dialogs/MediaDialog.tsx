import React, {
  useState,
  useEffect,
  useMemo,
  useRef,
  useCallback,
} from "react";
import { Stage, Layer, Group, Rect, Transformer, Path } from "react-konva";
import Konva from "konva";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { LuPause, LuPlay } from "react-icons/lu";
import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { useViewportStore } from "@/lib/viewport";
import VideoPreview from "@/components/preview/clips/VideoPreview";
import ImagePreview from "@/components/preview/clips/ImagePreview";
import { getApplicatorsForClip } from "@/lib/applicator-utils";
import { useWebGLHaldClut } from "@/components/preview/webgl-filters";
import {
  AnyClipProps,
  AudioClipProps,
  ClipTransform,
  ImageClipProps,
  MaskClipProps,
  VideoClipProps,
} from "@/lib/types";
import TimelineSelector from "@/components/properties/model/inputs/timeline/TimelineSelector";
import { BASE_LONG_SIDE } from "@/lib/settings";
import AudioPreview from "../preview/clips/AudioPreview";
import { CircularAudioVisualizer } from "../properties/model/inputs/CircularAudioVisualizer";
import ShapePreview from "../preview/clips/ShapePreview";
import TextPreview from "../preview/clips/TextPreview";
import DrawingPreview from "../preview/clips/DrawingPreview";
import { remapMaskWithClipTransformProportional } from "@/lib/mask/clipTransformUtils";
import { BaseClipApplicator } from "@/components/preview/clips/apply/base";

interface PartialTimelineSelectorProps {
  mode: "frame" | "range";
  inputId: string;
  /**
   * When true, the dialog should render an audio-only preview UI (e.g. for AudioInput
   * selecting a video/group/model just for its audio track). VideoInput should NOT set this.
   */
  audioOnly?: boolean;
}

interface MediaDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (data: {
    rotation: number;
    aspectRatio: string;
    crop?: { x: number; y: number; width: number; height: number };
    transformWidth?: number;
    transformHeight?: number;
    transformX?: number;
    transformY?: number;
    originalTransform?: ClipTransform;
  }) => void;
  clipOverride?: AnyClipProps | null;
  timelineSelectorProps?: PartialTimelineSelectorProps;
  focusFrame: number;
  setFocusFrame: (frame: number) => void;
  canCrop?: boolean;
  selectionRange?: [number, number];
  // Optional external playback controls (e.g. per-input range playback)
  isPlayingExternal?: boolean;
  onPlay?: () => void;
  onPause?: () => void;
  // Optional max duration (in frames) for input range selection
  maxDuration?: number;
}

const ASPECT_RATIOS = [
  { label: "Original", value: "original" },
  { label: "1:1", value: "1:1" },
  { label: "16:9", value: "16:9" },
  { label: "9:16", value: "9:16" },
  { label: "4:3", value: "4:3" },
  { label: "3:4", value: "3:4" },
  { label: "Custom", value: "custom" },
];

// 1. EXTRACTED COMPONENT TO HANDLE STABLE OVERRIDES IN MEDIADIALOG
const MediaDialogClipItem = React.memo(({
  clip,
  previewWidth,
  previewHeight,
  focusFrame,
  inputId,
  getApplicators,
  clipWithinFrame,
  getClipById,
  groupCentering,
}: {
  clip: AnyClipProps;
  previewWidth: number;
  previewHeight: number;
  focusFrame: number;
  inputId?: string;
  getApplicators: (id: string, frame: number) => BaseClipApplicator[];
  clipWithinFrame: (clip: AnyClipProps, frame: number, overlap: boolean, padding: number) => boolean;
  getClipById: (id: string) => AnyClipProps | undefined;
  groupCentering: { groupId: string; dx: number; dy: number; scale: number } | null;
}) => {
  const startFrame = clip.startFrame || 0;
  const groupStart = clip.groupId
    ? getClipById(clip.groupId)?.startFrame || 0
    : 0;
  const relativeStart = startFrame - groupStart;
  const hasOverlap =
    (clip.type === "video" || clip.type === "image") &&
    (clip.groupId ? relativeStart : startFrame) > 0
      ? true
      : false;
  // Use global frame for bounds/effects, but keep local focus for child playback math
  const effectiveGlobalFrame = clip.groupId
    ? focusFrame + groupStart
    : focusFrame;
  const clipAtFrame = clipWithinFrame(
    clip,
    effectiveGlobalFrame,
    hasOverlap,
    0,
  );
  if (!clipAtFrame && clip.groupId) return null;

  const applicators = getApplicators(
    clip.clipId,
    effectiveGlobalFrame,
  );

  // 2. MEMOIZE THE OVERRIDE CLIP
  const overrideToUse = useMemo(() => {
    const override = { ...clip };

    if (
        override &&
        override.transform &&
        (override.type === "image" ||
        override.type === "video") &&
        !override.clipId?.startsWith("media:") &&
        !override.groupId
    ) {
        const t = { ...(override.transform ?? {}) };
        t.rotation =
        override.originalTransform?.rotation ?? 0;

        // In the media dialog, always show the full underlying item for
        // non-group clips, even if a crop was previously applied.
        // Reset crop and base width/height to the native media dimensions.
        // @ts-ignore - mediaWidth/mediaHeight exist on image/video clips
        const nativeW =
        override.mediaWidth ||
        0;
        // @ts-ignore
        const nativeH =
        override.mediaHeight ||
        0;

        if (nativeW > 0 && nativeH > 0) {
        t.width = nativeW;
        t.height = nativeH;
        }

        if ((t as any).crop) {
        delete (t as any).crop;
        }

        // Use the logical preview rect as the "canvas" the input should fit inside
        const canvasW = previewWidth;
        const canvasH = previewHeight;

        if (canvasW > 0 && canvasH > 0) {
        const rawW = t.width ?? canvasW;
        const rawH = t.height ?? canvasH;
        const baseSx =
            typeof t.scaleX === "number" ? t.scaleX : 1;
        const baseSy =
            typeof t.scaleY === "number" ? t.scaleY : 1;
        const deg =
            typeof t.rotation === "number" ? t.rotation : 0;
        const rad = (deg * Math.PI) / 180;
        const c = Math.cos(rad);
        const s = Math.sin(rad);

        // First, compute the axis-aligned bounding box of the current transform
        const w0 = rawW * baseSx;
        const h0 = rawH * baseSy;
        const x1_0 = w0 * c;
        const y1_0 = w0 * s;
        const x2_0 = -h0 * s;
        const y2_0 = h0 * c;
        const x3_0 = w0 * c - h0 * s;
        const y3_0 = w0 * s + h0 * c;
        const minX0 = Math.min(0, x1_0, x2_0, x3_0);
        const maxX0 = Math.max(0, x1_0, x2_0, x3_0);
        const minY0 = Math.min(0, y1_0, y2_0, y3_0);
        const maxY0 = Math.max(0, y1_0, y2_0, y3_0);
        const aabbW0 = maxX0 - minX0;
        const aabbH0 = maxY0 - minY0;

        // If the bounding box is larger than the canvas, scale it down uniformly
        let fitScale = 1;
        if (
            aabbW0 > 0 &&
            aabbH0 > 0 &&
            (aabbW0 > canvasW || aabbH0 > canvasH)
        ) {
            const scaleXFit = canvasW / aabbW0;
            const scaleYFit = canvasH / aabbH0;
            fitScale = Math.min(scaleXFit, scaleYFit, 1);
        }

        const sx = baseSx * fitScale;
        const sy = baseSy * fitScale;
        const w = rawW * sx;
        const h = rawH * sy;

        const x1 = w * c;
        const y1 = w * s;
        const x2 = -h * s;
        const y2 = h * c;
        const x3 = w * c - h * s;
        const y3 = w * s + h * c;
        const minX = Math.min(0, x1, x2, x3);
        const maxX = Math.max(0, x1, x2, x3);
        const minY = Math.min(0, y1, y2, y3);
        const maxY = Math.max(0, y1, y2, y3);
        const aabbW = maxX - minX;
        const aabbH = maxY - minY;

        t.scaleX = sx;
        t.scaleY = sy;
        t.x = (canvasW - aabbW) / 2 - minX;
        t.y = (canvasH - aabbH) / 2 - minY;

        override.transform = t;
        }
    }

    if (
        groupCentering &&
        override?.groupId === groupCentering.groupId &&
        override.transform &&
        (override.type === "image" ||
        override.type === "video" ||
        override.type === "shape" ||
        override.type === "text" ||
        override.type === "draw")
    ) {
        const t = { ...(override.transform ?? {}) };
        const baseX = t.x ?? 0;
        const baseY = t.y ?? 0;
        const existingScaleX =
        typeof t.scaleX === "number" ? t.scaleX : 1;
        const existingScaleY =
        typeof t.scaleY === "number" ? t.scaleY : 1;
        const scale =
        typeof (groupCentering as any).scale === "number"
            ? (groupCentering as any).scale
            : 1;

        t.x = baseX * scale + groupCentering.dx;
        t.y = baseY * scale + groupCentering.dy;
        t.scaleX = existingScaleX * scale;
        t.scaleY = existingScaleY * scale;

        override.transform = t;
    }

    if ((override as any).masks?.length > 0) {
        const masks = [...(override as any).masks] as MaskClipProps[];
        
        (override as any).masks = masks.map((mask: MaskClipProps) => {
        // Calculate native dimensions for intermediate transform
        // @ts-ignore
        const nW = clip.mediaWidth || clip.width || 0;
        // @ts-ignore
        const nH = clip.mediaHeight || clip.height || 0;

        // Native transform at origin
        const nativeTransform: ClipTransform = {
            x: 0,
            y: 0,
            width: nW,
            height: nH,
            scaleX: 1,
            scaleY: 1,
            rotation: clip.originalTransform?.rotation ?? 0,
            opacity: 1,
            cornerRadius: 0,
        };

        // Zeroed current transform (preserve scale/crop but move to origin)
        const currentTransform =
            clip.transform ?? nativeTransform;
        const zeroCurrentTransform: ClipTransform = {
            ...currentTransform,
            x: 0,
            y: 0,
        };

        // Zeroed override transform
        const overrideTransform = override.transform!;

        let xOffset = 0;
        let yOffset = 0;

        const realCrop = clip.transform?.crop && (clip.transform.crop.width != 1 || clip.transform.crop.height != 1 || clip.transform.crop.x != 0 || clip.transform.crop.y != 0);

        if (realCrop && clip.transform?.crop) {
            // determine how much to offsetX 
            const fullWidth = clip.transform.width / clip.transform.crop.width;
            const fullHeight = clip.transform.height / clip.transform.crop.height;
            const offsetX = fullWidth * clip.transform.crop.x;
            const offsetY = fullHeight * clip.transform.crop.y;
            xOffset = -offsetX;
            yOffset = -offsetY;
        }

        const zeroOverrideTransform: ClipTransform = {
            ...overrideTransform,
            x: xOffset ,
            y: yOffset,
        };

        // Map: Current(0,0) -> Native(0,0)
        // We explicitly use the calculated zeroCurrentTransform as the source of truth
        // for the current mask coordinate space, ignoring any potentially stale transform
        // on the mask object itself.
        const maskForRemap = { ...mask, transform: undefined };
        
        const toOriginal = remapMaskWithClipTransformProportional(
            maskForRemap,
            zeroCurrentTransform,
            nativeTransform,
        );

        // Map: Native(0,0) -> Override(0,0)
        const toOverride = remapMaskWithClipTransformProportional(
            toOriginal,
            nativeTransform,
            zeroOverrideTransform,
        );
        
        // Restore actual transform position
        toOverride.transform = { ...overrideTransform };
        toOverride.transform

        return toOverride;
        });
    }

    if (override?.clipId?.startsWith("media:")) {
        override.transform = undefined;
    }

    return override;
  }, [clip, previewWidth, previewHeight, groupCentering]);


  switch (clip.type) {
    case "video":
      return (
        <VideoPreview
          key={clip.clipId}
          {...{...(clip as any), hidden: false}}
          overrideClip={overrideToUse}
          rectWidth={previewWidth}
          rectHeight={previewHeight}
          applicators={applicators}
          overlap={true}
          inputMode={true}
          focusFrameOverride={focusFrame}
          inputId={inputId}
          decoderKey={`media-dialog::${clip.clipId}`}
        />
      );
    case "image":
      return (
        <ImagePreview
          key={clip.clipId}
          {...{...(clip as any), hidden: false}}
          overrideClip={overrideToUse}
          rectWidth={previewWidth}
          rectHeight={previewHeight}
          applicators={applicators}
          overlap={true}
          inputMode={true}
          inputId={inputId}
          focusFrameOverride={focusFrame}
        />
      );
    case "shape":
      return (
        <ShapePreview
          key={clip.clipId}
          {...{...(clip as any), hidden: false}}
          rectWidth={previewWidth}
          rectHeight={previewHeight}
          applicators={applicators}
          assetMode={true}
          inputMode={true}
          inputId={inputId}
          focusFrameOverride={focusFrame}
        />
      );
    case "text":
      return (
        <TextPreview
          key={clip.clipId}
          {...{...(clip as any), hidden: false}}
          rectWidth={previewWidth}
          rectHeight={previewHeight}
          applicators={applicators}
          assetMode={true}
          inputMode={true}
          inputId={inputId}
          focusFrameOverride={focusFrame}
        />
      );
    case "draw":
      return (
        <DrawingPreview
          key={clip.clipId}
          {...{...(clip as any), hidden: false}}
          rectWidth={previewWidth}
          rectHeight={previewHeight}
          assetMode={true}
          applicators={applicators}
          inputMode={true}
          inputId={inputId}
          focusFrameOverride={focusFrame}
        />
      );
    default:
      return null;
  }
});

export const MediaDialog: React.FC<MediaDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  clipOverride,
  timelineSelectorProps,
  focusFrame,
  setFocusFrame,
  canCrop = true,
  isPlayingExternal,
  onPlay,
  onPause,
  selectionRange,
  maxDuration,
}) => {
  const [rotation, setRotation] = useState(0);
  const [aspectRatio, setAspectRatio] = useState("original");
  const [size, setSize] = useState({ width: 0, height: 0 });

  const [containerNode, setContainerNode] = useState<HTMLDivElement | null>(
    null,
  );
  const divTimelineSelectorRef = useRef<HTMLDivElement>(null);
  const { getClipTransform, clips, timelines, clipWithinFrame } =
    useClipStore();
  const editorAspectRatio = useViewportStore((s) => s.aspectRatio);

  const selectedClipIds = useControlsStore((state) => state.selectedClipIds);
  const getClipById = useClipStore((state) => state.getClipById);
  const haldClutInstance = useWebGLHaldClut();
  const isPlaying =
    typeof isPlayingExternal === "boolean"
      ? isPlayingExternal
      : useControlsStore((state) => state.isPlaying);
  const pause = useControlsStore((state) => state.pause);

  const clipId = clipOverride?.clipId ?? selectedClipIds[0];

  // Get the single selected clip
  const selectedClip = clipOverride ?? getClipById(clipId);

  // Scrubber logic
  const [isDragging, setIsDragging] = useState(false);
  const [dragProgress, setDragProgress] = useState<number | null>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  const handleScrubberMove = useCallback(
    (clientX: number) => {
      if (
        !progressBarRef.current ||
        !selectedClip ||
        selectedClip.type !== "video"
      )
        return;

      const videoClip = selectedClip as VideoClipProps;
      const startFrame = videoClip.startFrame || 0;
      const endFrame = videoClip.endFrame || 0;
      const duration = Math.max(1, endFrame - startFrame);

      const rect = progressBarRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const newProgress = Math.max(0, Math.min(x / rect.width, 1));

      requestAnimationFrame(() => {
        setDragProgress(newProgress);
      });

      const newFrame = Math.round(startFrame + newProgress * duration);
      setFocusFrame(newFrame);
    },
    [selectedClip, setFocusFrame],
  );

  const handleScrubberMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      handleScrubberMove(e.clientX);
    },
    [isDragging, handleScrubberMove],
  );

  const handleScrubberMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragProgress(null);
  }, []);

  const handleProgressBarMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      if (isPlaying) pause();
      handleScrubberMove(e.clientX);
    },
    [handleScrubberMove, isPlaying, pause],
  );

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleScrubberMouseMove);
      document.addEventListener("mouseup", handleScrubberMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleScrubberMouseMove);
        document.removeEventListener("mouseup", handleScrubberMouseUp);
      };
    }
  }, [isDragging, handleScrubberMouseMove, handleScrubberMouseUp]);

  const progress = useMemo(() => {
    if (isDragging && dragProgress !== null) return dragProgress;
    if (!selectedClip || selectedClip.type !== "video") return 0;

    const videoClip = selectedClip as VideoClipProps;
    const startFrame = videoClip.startFrame || 0;
    const endFrame = videoClip.endFrame || 0;
    const duration = Math.max(1, endFrame - startFrame);

    // Calculate relative position in clip
    const relativeFrame = focusFrame - startFrame;
    return Math.max(0, Math.min(relativeFrame / duration, 1));
  }, [isDragging, dragProgress, selectedClip, focusFrame]);

  const [rangeStart, rangeEnd] = selectionRange ?? [0, 1];
  const rangeSpan = Math.max(1, rangeEnd - rangeStart);
  const selectionProgress = useMemo(() => {
    if (!selectedClip || timelineSelectorProps?.mode !== "range") return 0;
    const clampedFrame = Math.max(rangeStart, Math.min(rangeEnd, focusFrame));
    const localFrame = clampedFrame - rangeStart;
    return Math.max(0, Math.min(localFrame / rangeSpan, 1));
  }, [selectedClip, focusFrame, rangeStart, rangeEnd, rangeSpan]);

  const isSelectionPlaying =
    typeof isPlayingExternal === "boolean" ? isPlayingExternal : isPlaying;

  const getApplicators = useCallback(
    (id: string, frameOverride?: number) => {
      return getApplicatorsForClip(id, {
        haldClutInstance,
        focusFrameOverride: frameOverride,
      });
    },
    [haldClutInstance],
  );

  // Resize observer for the container
  useEffect(() => {
    if (!containerNode) return;

    const updateSize = () => {
      const width = containerNode.offsetWidth;
      const height = containerNode.offsetHeight;
      // Provide some padding so the transformer handles are not cut off
      const padding = 128;
      const availableWidth = width - padding;
      const availableHeight = height - padding;


      setSize({
        width: availableWidth,
        height: availableHeight,
      });
    };

    // Initial measure
    updateSize();

    // Small delay to ensure layout is stable after dialog animation
    const timer = setTimeout(updateSize, 100);

    const observer = new ResizeObserver(() => {
      updateSize();
    });
    observer.observe(containerNode);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, [containerNode, isOpen]); // Re-run when node changes or dialog opens/closes

  const handleReset = () => {
    setRotation(0);
    setAspectRatio("original");
    // Reset the interactive crop rect back to full media display rect
    if (cropRectRef.current && mediaDisplayRect) {
      const { width: mW, height: mH, x: mX, y: mY } = mediaDisplayRect;

      // Clear any previous manual scaling
      cropRectRef.current.scaleX(1);
      cropRectRef.current.scaleY(1);

      cropRectRef.current.width(mW);
      cropRectRef.current.height(mH);
      cropRectRef.current.x(mX);
      cropRectRef.current.y(mY);

      if (transformerRef.current && cropRectRef.current) {
        transformerRef.current.nodes([cropRectRef.current]);
      }
      setOverlayPath(getOverlayPath());
    }
  };

  const handleConfirm = () => {
    if (!cropRectRef.current || !mediaDisplayRect || !selectedClip) {
      onConfirm({ rotation, aspectRatio });
      onClose();
      return;
    }

    // Calculate crop in NORMALIZED coordinates (0-1) relative to the display rect
    const {
      width: dispW,
      height: dispH,
      x: dispX,
      y: dispY,
    } = mediaDisplayRect;
    const cropNode = cropRectRef.current;
    const cropX = cropNode.x();
    const cropY = cropNode.y();
    const cropW = cropNode.width() * cropNode.scaleX();
    const cropH = cropNode.height() * cropNode.scaleY();

    const clip = selectedClip as VideoClipProps | ImageClipProps;
    // @ts-ignore
    const nativeW = clip.mediaWidth || clip.width || 0;
    // @ts-ignore
    const nativeH = clip.mediaHeight || clip.height || 0;

    if (dispW > 0 && dispH > 0) {
      const normalizedCrop = {
        x: (cropX - dispX) / dispW,
        y: (cropY - dispY) / dispH,
        width: cropW / dispW,
        height: cropH / dispH,
      };

      // Clamp to 0-1 to avoid precision issues causing invalid crops
      normalizedCrop.x = Math.max(0, Math.min(1, normalizedCrop.x));
      normalizedCrop.y = Math.max(0, Math.min(1, normalizedCrop.y));
      normalizedCrop.width = Math.max(0, Math.min(1, normalizedCrop.width));
      normalizedCrop.height = Math.max(0, Math.min(1, normalizedCrop.height));

      // Calculate new clip transform dimensions to match crop aspect ratio
      // Logic: Maintain the current scale (zoom) of the content relative to the native resolution.
      // This prevents shrinking when switching aspect ratios and keeps the visual size consistent.

      let currentTransform = clipOverride
        ? clipOverride.transform
        : (getClipTransform(selectedClip.clipId) as ClipTransform);
      let originalTransform = clipOverride
        ? clipOverride.originalTransform
        : (getClipTransform(selectedClip.clipId) as ClipTransform);
      if (!currentTransform) {
        const contentRatio = nativeW / nativeH;
        const displayW = BASE_LONG_SIDE * contentRatio;
        const displayH = BASE_LONG_SIDE;

        currentTransform = {
          x: 0,
          y: 0,
          width: displayW,
          height: displayH,
          scaleX: 1,
          scaleY: 1,
          rotation: 0,
        } as ClipTransform;
      }

      if (!originalTransform) {
        originalTransform = { ...currentTransform };
      } else {
        originalTransform = undefined;
      }

      // Calculate current scale based on the currently visible portion of the native width
      const currentCropW = currentTransform.crop
        ? currentTransform.crop.width
        : 1.0;
      const currentCropH = currentTransform.crop
        ? currentTransform.crop.height
        : 1.0;
      const effectiveNativeW = nativeW * currentCropW;
      const effectiveNativeH = nativeH * currentCropH;
      const currentScaleWidth =
        effectiveNativeW > 0 ? currentTransform.width / effectiveNativeW : 1;
      const currentScaleHeight =
        effectiveNativeH > 0 ? currentTransform.height / effectiveNativeH : 1;
      const currentScale = Math.max(currentScaleWidth, currentScaleHeight);

      const newWidth = nativeW * normalizedCrop.width * currentScale;
      const newHeight = nativeH * normalizedCrop.height * currentScale;

      // Compute new transform position so the remaining visible area stays in place
      const currentCropX = currentTransform.crop ? currentTransform.crop.x : 0;
      const currentCropY = currentTransform.crop ? currentTransform.crop.y : 0;
      const deltaCropX = normalizedCrop.x - currentCropX;
      const deltaCropY = normalizedCrop.y - currentCropY;

      const deltaNativeX = nativeW * deltaCropX;
      const deltaNativeY = nativeH * deltaCropY;

      const baseX =
        typeof currentTransform.x === "number" ? currentTransform.x : 0;
      const baseY =
        typeof currentTransform.y === "number" ? currentTransform.y : 0;

      // When we crop away pixels on the left/top, we shift the transform in the
      // opposite direction so the visible content appears to stay put.
      const newX = baseX + deltaNativeX * currentScale;
      const newY = baseY + deltaNativeY * currentScale;

      onConfirm({
        rotation,
        aspectRatio,
        crop: normalizedCrop,
        transformWidth: newWidth,
        transformHeight: newHeight,
        transformX: newX,
        transformY: newY,
        originalTransform: originalTransform,
      });
    } else {
      onConfirm({ rotation, aspectRatio });
    }
    onClose();
  };

  const handleOpenChange = (open: boolean) => {
    if (!open) onClose();
  };

  // Ensure we only render valid clips

  const cropRectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);

  const sortClips = useCallback(
    (clipsToSort: AnyClipProps[]) => {
      // Treat each group as a single sortable unit; then expand children in defined order
      type GroupUnit = {
        kind: "group";
        id: string;
        y: number;
        start: number;
        children: AnyClipProps[];
      };
      type SingleUnit = {
        kind: "single";
        y: number;
        start: number;
        clip: AnyClipProps;
      };

      const groups = clipsToSort.filter(
        (c) => c.type === "group",
      ) as AnyClipProps[];
      const childrenSet = new Set<string>(
        groups.flatMap((g) => {
          const nested = ((g as any).children as string[][] | undefined) ?? [];
          return nested.flat();
        }),
      );

      // Build group units
      const groupUnits: GroupUnit[] = groups.map((g) => {
        const y =
          timelines.find((t) => t.timelineId === g.timelineId)?.timelineY ?? 0;
        const start = g.startFrame ?? 0;
        const nested = ((g as any).children as string[][] | undefined) ?? [];
        const childIdsFlat = nested.flat();
        const children = childIdsFlat
          .map((id) => clips.find((c) => c.clipId === id))
          .filter(Boolean) as AnyClipProps[];
        return { kind: "group", id: g.clipId, y, start, children };
      });

      // Build single units for non-group, non-child clips
      const singleUnits: SingleUnit[] = clipsToSort
        .filter((c) => c.type !== "group" && !childrenSet.has(c.clipId))
        .map((c) => {
          const y =
            timelines.find((t) => t.timelineId === c.timelineId)?.timelineY ??
            0;
          const start = c.startFrame ?? 0;
          return { kind: "single", y, start, clip: c };
        });

      // Sort units: lower on screen first (higher y), then earlier start
      const units = [...groupUnits, ...singleUnits].sort((a, b) => {
        if (a.y !== b.y) return b.y - a.y;
        return a.start - b.start;
      });

      // Flatten units back to clip list; for groups, expand children in their defined order
      const result: AnyClipProps[] = [];
      for (const u of units) {
        if (u.kind === "single") {
          result.push(u.clip);
        } else {
          // Within a group, render lower timelines first (higher y), then earlier starts
          // Ensure children are ordered as in group's children list (reversed like main Preview)
          result.push(...u.children.reverse());
        }
      }
      return result;
    },
    [timelines, clips],
  );

  const toRender = useMemo(() => {
    if (clipOverride) {
      if (clipOverride.type === "group") {
        // Let sorter handle group flattening so child order matches main Preview
        return sortClips([clipOverride]);
      }
      return [clipOverride];
    }
    if (!clipId) return [] as AnyClipProps[];
    const c = getClipById(clipId);
    if (!c) return [] as AnyClipProps[];
    if (c.type === "group") {
      return sortClips([c]);
    }
    return [c];
  }, [clipOverride, clipId, getClipById, clips, sortClips]);

  const isValidClip = toRender.length > 0;

  // Determine the logical canvas (world) size for the dialog preview.
  // For non-image/video clips (or groups containing only non-image/video),
  // we base this on the real editor aspect ratio so layout matches the main editor.
  const previewRect = useMemo(() => {
    
    if (!isValidClip) {
      return { width: 0, height: 0 };
    }

    const hasMediaClip = toRender.some(
      (clip) => clip.type === "image" || clip.type === "video",
    );

    // Check if this is a group render (any clip has a groupId)
    // Groups must use editor aspect ratio to match the coordinate system
    // their children's transforms were defined in
    const isGroupRender = toRender.some((clip) => clip.groupId);

    // Groups and non-media-only clips use the editor aspect ratio
    if (!hasMediaClip || isGroupRender) {
      const ratio =
        editorAspectRatio.height === 0
          ? 0
          : editorAspectRatio.width / editorAspectRatio.height;
      const baseShortSide = BASE_LONG_SIDE;


      if (!Number.isFinite(ratio) || ratio <= 0) {
        return { width: size.width, height: size.height };
      }


      return {
        width: baseShortSide * ratio,
        height: baseShortSide,
      };
    }


    // Single image / video clips still use the container-driven size
    return { width: size.width, height: size.height };
  }, [
    isValidClip,
    toRender,
    editorAspectRatio.width,
    editorAspectRatio.height,
    size.width,
    size.height,
  ]);


  // Map logical canvas to the actual dialog container size
  const view = useMemo(() => {
    const { width: worldW, height: worldH } = previewRect;

    if (!worldW || !worldH || !size.width || !size.height) {
      return {
        scale: 1,
        x: size.width / 2 + 16,
        y: size.height / 2 + 32,
        offsetX: worldW / 2,
        offsetY: worldH / 2,
      };
    }

    const scaleX = size.width / worldW;
    const scaleY = size.height / worldH;
    const scale = Math.min(scaleX, scaleY);

    return {
      scale,
      x: size.width / 2 + 16,
      y: size.height / 2 + 32,
      offsetX: worldW / 2,
      offsetY: worldH / 2,
    };
  }, [previewRect.width, previewRect.height, size.width, size.height]);

  const groupCentering = useMemo(() => {
    const canvasW = previewRect.width;
    const canvasH = previewRect.height;
    if (!canvasW || !canvasH) return null;

    const groupParent: AnyClipProps | null =
      clipOverride && clipOverride.type === "group"
        ? clipOverride
        : selectedClip && (selectedClip as AnyClipProps).type === "group"
          ? (selectedClip as AnyClipProps)
          : null;

    if (!groupParent) return null;

    const gid = groupParent.clipId;
    const children = toRender.filter(
      (c) =>
        c.groupId === gid &&
        (c.type === "image" ||
          c.type === "video" ||
          c.type === "shape" ||
          c.type === "text" ||
          c.type === "draw"),
    );

    if (!children.length) return null;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const child of children) {
      const t = child.transform;
      if (!t) continue;

      const rawW = t.width ?? canvasW;
      const rawH = t.height ?? canvasH;
      const sx = typeof t.scaleX === "number" ? t.scaleX : 1;
      const sy = typeof t.scaleY === "number" ? t.scaleY : 1;
      const w = rawW * sx;
      const h = rawH * sy;
      const deg = typeof t.rotation === "number" ? t.rotation : 0;
      const rad = (deg * Math.PI) / 180;
      const c = Math.cos(rad);
      const s = Math.sin(rad);

      const baseX = t.x ?? 0;
      const baseY = t.y ?? 0;

      const x0 = baseX;
      const y0 = baseY;
      const x1 = baseX + w * c;
      const y1 = baseY + w * s;
      const x2 = baseX - h * s;
      const y2 = baseY + h * c;
      const x3 = baseX + w * c - h * s;
      const y3 = baseY + w * s + h * c;

      minX = Math.min(minX, x0, x1, x2, x3);
      minY = Math.min(minY, y0, y1, y2, y3);
      maxX = Math.max(maxX, x0, x1, x2, x3);
      maxY = Math.max(maxY, y0, y1, y2, y3);
    }

    if (
      !isFinite(minX) ||
      !isFinite(minY) ||
      !isFinite(maxX) ||
      !isFinite(maxY)
    ) {
      return null;
    }

    const groupWidth = maxX - minX;
    const groupHeight = maxY - minY;

    if (groupWidth <= 0 || groupHeight <= 0) {
      return null;
    }

    // Uniform scale so the entire group fits inside the canvas
    const scaleW = canvasW / groupWidth;
    const scaleH = canvasH / groupHeight;
    const scale = Math.min(1, scaleW, scaleH);

    const scaledWidth = groupWidth * scale;
    const scaledHeight = groupHeight * scale;

    // After scaling around origin (0,0), bbox becomes [minX*scale, maxX*scale], etc.
    // Choose dx,dy so the scaled bbox is centered in the canvas.
    const dx = (canvasW - scaledWidth) / 2 - minX * scale;
    const dy = (canvasH - scaledHeight) / 2 - minY * scale;

    return { groupId: gid, dx, dy, scale };
  }, [
    previewRect.width,
    previewRect.height,
    toRender,
    clipOverride,
    selectedClip,
  ]);

  // Helper to create the overlay path string
  // It draws a large outer rectangle and subtracts the inner crop rectangle
  const getOverlayPath = useCallback(() => {
    if (!cropRectRef.current || !size.width) return "";

    const crop = cropRectRef.current;
    const cx = crop.x();
    const cy = crop.y();
    const cw = crop.width() * crop.scaleX();
    const ch = crop.height() * crop.scaleY();

    // Outer rectangle (canvas size)
    const outer = `M 0 0 L ${size.width} 0 L ${size.width} ${size.height} L 0 ${size.height} Z`;

    // Inner rectangle (crop area) - drawn counter-clockwise to create a hole
    const inner = `M ${cx} ${cy} L ${cx} ${cy + ch} L ${cx + cw} ${cy + ch} L ${cx + cw} ${cy} Z`;

    return outer + " " + inner;
  }, [size]);

  const [overlayPath, setOverlayPath] = useState("");

  // Calculate the displayed size and position of the clip
  const mediaDisplayRect = useMemo(() => {
    if (!isValidClip || !size.width || !size.height) return null;

    const clip = selectedClip as VideoClipProps | ImageClipProps;
    // @ts-ignore
    const clipW = clip.mediaWidth || clip.width;
    // @ts-ignore
    const clipH = clip.mediaHeight || clip.height;

    if (!clipW || !clipH) {
      return {
        x: 0,
        y: 0,
        width: size.width,
        height: size.height,
      };
    }

    // Calculate the displayed size of the clip within the stage (which uses contain/fit logic)
    const stageRatio = size.width / size.height;
    const clipRatio = clipW / clipH;

    let displayW, displayH;

    if (clipRatio > stageRatio) {
      // Limited by width
      displayW = size.width;
      displayH = size.width / clipRatio;
    } else {
      // Limited by height
      displayH = size.height;
      displayW = size.height * clipRatio;
    }

    const x = (size.width - displayW) / 2;
    const y = (size.height - displayH) / 2;

    return { width: displayW, height: displayH, x, y };
  }, [isValidClip, size, selectedClip]);

  // Initialize crop rect from existing clip transform if available, keyed by clipId to handle clip changes
  useEffect(() => {
    if (!mediaDisplayRect || !cropRectRef.current || !selectedClip) return;

    const { width: mW, height: mH, x: mX, y: mY } = mediaDisplayRect;

    // Always reset scale so programmatic changes are not affected by previous manual resizes
    cropRectRef.current.scaleX(1);
    cropRectRef.current.scaleY(1);

    // Check if this clip already has a crop defined
    if (selectedClip.transform?.crop) {
      const {
        x: nX,
        y: nY,
        width: nW,
        height: nH,
      } = selectedClip.transform.crop;
      cropRectRef.current.width(nW * mW);
      cropRectRef.current.height(nH * mH);
      cropRectRef.current.x(mX + nX * mW);
      cropRectRef.current.y(mY + nY * mH);
      if (transformerRef.current && cropRectRef.current) {
        transformerRef.current.nodes([cropRectRef.current]);
      }

      setOverlayPath(getOverlayPath());
    } else {
      // Reset to full size if no crop exists for this clip
      cropRectRef.current.width(mW);
      cropRectRef.current.height(mH);
      cropRectRef.current.x(mX);
      cropRectRef.current.y(mY);
      if (transformerRef.current && cropRectRef.current) {
        transformerRef.current.nodes([cropRectRef.current]);
      }
      setOverlayPath(getOverlayPath());
    }
  }, [
    mediaDisplayRect,
    selectedClip?.transform,
    selectedClip?.clipId,
    getOverlayPath,
  ]); // Re-run explicitly when clipId changes

  const applyAspectRatioCrop = useCallback(
    (ratio: string, options?: { force?: boolean }) => {
      if (transformerRef.current && cropRectRef.current) {
        transformerRef.current.nodes([cropRectRef.current]);
      }

      const currentCrop = selectedClip?.transform?.crop;

      if (!options?.force && currentCrop) {
        return;
      }

      if (!cropRectRef.current || !mediaDisplayRect) {
        return;
      }

      const { width: mW, height: mH, x: mX, y: mY } = mediaDisplayRect;

      let newW = mW;
      let newH = mH;

      if (ratio !== "original") {
        const [wStr, hStr] = ratio.split(":");
        const rW = parseFloat(wStr);
        const rH = parseFloat(hStr);

        if (!isNaN(rW) && !isNaN(rH) && rH !== 0) {
          const targetRatio = rW / rH;
          const mediaRatio = mW / mH;

          if (mediaRatio > targetRatio) {
            // Media is wider relative to target -> constrain width based on height
            newH = mH;
            newW = mH * targetRatio;
          } else {
            // Media is taller (or equal) -> constrain height based on width
            newW = mW;
            newH = mW / targetRatio;
          }
        }
      }

      const newX = mX + (mW - newW) / 2;
      const newY = mY + (mH - newH) / 2;

      // Clear any previous manual scaling before applying programmatic size
      cropRectRef.current.scaleX(1);
      cropRectRef.current.scaleY(1);

      cropRectRef.current.width(newW);
      cropRectRef.current.height(newH);
      cropRectRef.current.x(newX);
      cropRectRef.current.y(newY);

      setOverlayPath(getOverlayPath());
    },
    [mediaDisplayRect, getOverlayPath, selectedClip],
  );

  // Update overlay and aspect ratio when transformer changes (user resize/drag)
  const handleTransform = useCallback(() => {
    setOverlayPath(getOverlayPath());

    if (!cropRectRef.current) return;

    const w = cropRectRef.current.width() * cropRectRef.current.scaleX();
    const h = cropRectRef.current.height() * cropRectRef.current.scaleY();

    if (!h) return;

    const currentRatio = w / h;
    const EPS = 0.01;

    const matched = ASPECT_RATIOS.find((ratio) => {
      if (ratio.value === "original" || ratio.value === "custom") return false;
      const [wStr, hStr] = ratio.value.split(":");
      const rW = parseFloat(wStr);
      const rH = parseFloat(hStr);
      if (isNaN(rW) || isNaN(rH) || rH === 0) return false;
      const target = rW / rH;
      return Math.abs(currentRatio - target) < EPS;
    });

    if (matched) {
      if (aspectRatio !== matched.value) {
        setAspectRatio(matched.value);
      }
    } else {
      if (aspectRatio !== "custom") {
        setAspectRatio("custom");
      }
    }
  }, [getOverlayPath, aspectRatio]);

  // Initial overlay path
  useEffect(() => {
    if (size.width > 0) {
      setOverlayPath(getOverlayPath());
    }
  }, [size, getOverlayPath]);

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-4xl w-full h-[80vh] flex flex-col bg-background text-foreground p-0 gap-0 overflow-hidden dark font-poppins border-brand-light/10">
        <DialogHeader className="px-6 py-6 border-b border-brand-light/10 shrink-0 bg-brand-background">
          <DialogTitle className="text-start text-[13px] font-medium">
            {" "}
          </DialogTitle>
        </DialogHeader>
        <div
          ref={setContainerNode}
          className="flex-1 bg-black/50 relative flex items-center justify-center overflow-hidden"
        >
          {!isValidClip ? (
            <div className="text-muted-foreground opacity-50">
              No valid media selected
            </div>
          ) : toRender.every((clip) => clip.type === "audio") &&
            timelineSelectorProps ? (
            <>
              <AudioPreview
                key={toRender[0]?.clipId}
                {...(toRender[0] as AudioClipProps)}
                overrideClip={toRender[0] as AnyClipProps as AudioClipProps}
                inputMode={true}
                inputId={timelineSelectorProps.inputId}
                overlap={true}
                rectWidth={size.width}
                rectHeight={size.height}
              />
              <CircularAudioVisualizer
                inputId={timelineSelectorProps.inputId}
                width={size.width}
                height={size.height}
                active={isSelectionPlaying}
              />
            </>
          ) : timelineSelectorProps && timelineSelectorProps.mode === "range" && 
            timelineSelectorProps.audioOnly &&
            toRender.some((clip) => clip.type === "video" || clip.type === "group" || clip.type === "model") ? (
            /* For video/group/model clips from AudioInput, show ONLY audio visualizer (no video) */
            <>
              {/* Render AudioPreview for video, model and audio clips */}
              {toRender.map((clip) => {
                if (clip.type !== "video" && clip.type !== "model" && clip.type !== "audio") return null;
                
                // For clips that are children of a group, compute timing relative to group start
                const clipStartFrame = clip.startFrame || 0;
                const clipEndFrame = clip.endFrame || 0;
                
                // Get the ORIGINAL group's startFrame from the clips store
                const originalGroup = clip.groupId ? clips.find((c) => c.clipId === clip.groupId) : null;
                const originalGroupStart = originalGroup?.startFrame || 0;
                
                // Convert absolute timeline frames to group-relative frames
                const relativeStart = clip.groupId ? clipStartFrame - originalGroupStart : clipStartFrame;
                const relativeEnd = clip.groupId ? clipEndFrame - originalGroupStart : clipEndFrame;
                const hasOverlap = relativeStart > 0;
                
                return (
                  <AudioPreview
                    key={`audio-${clip.clipId}`}
                    {...{...(clip as any), hidden: false, startFrame: relativeStart, endFrame: relativeEnd}}
                    overlap={hasOverlap}
                    rectWidth={size.width}
                    rectHeight={size.height}
                    inputMode={true}
                    inputId={timelineSelectorProps?.inputId}
                    preserveInputTiming={true}
                  />
                );
              })}
              {/* Render AudioPreview for group clips */}
              {toRender.map((clip) => {
                if (clip.type !== "group") return null;
                const nested = ((clip as any).children as string[][] | undefined) ?? [];
                const childIdsFlat = nested.flat();
                const childClips = childIdsFlat
                  .map((id) => clips.find((c) => c.clipId === id))
                  .filter((c): c is AnyClipProps => 
                    !!c && (c.type === "video" || c.type === "audio" || c.type === "model")
                  );
                
                // Get the ORIGINAL group's startFrame from clips store
                const originalGroup = clips.find((c) => c.clipId === clip.clipId);
                const originalGroupStart = originalGroup?.startFrame || 0;
                
                return childClips.map((childClip) => {
                  // Child clips have absolute timeline frames, convert to group-relative
                  const childStart = (childClip.startFrame || 0) - originalGroupStart;
                  const childEnd = (childClip.endFrame || 0) - originalGroupStart;
                  const hasOverlap = childStart > 0;
                  
                  return (
                    <AudioPreview
                      key={`group-audio-${childClip.clipId}`}
                      {...{...(childClip as any), hidden: false, startFrame: childStart, endFrame: childEnd}}
                      overlap={hasOverlap}
                      rectWidth={size.width}
                      rectHeight={size.height}
                      inputMode={true}
                      inputId={timelineSelectorProps?.inputId}
                      preserveInputTiming={true}
                    />
                  );
                });
              })}
              <CircularAudioVisualizer
                inputId={timelineSelectorProps.inputId}
                width={size.width}
                height={size.height}
                active={isSelectionPlaying}
              />
            </>
          ) : (
            <div className="relative w-full h-full flex flex-col">
              <div className="flex-1 relative flex items-center justify-center overflow-hidden">
                <Stage
                  width={size.width + 32}
                  height={size.height + 64}
                  className="bg-transparent"
                >
                  <Layer>
                    <Group
                      x={view.x}
                      y={view.y}
                      offsetX={view.offsetX}
                      offsetY={view.offsetY}
                      scaleX={view.scale}
                      scaleY={view.scale}
                      rotation={rotation}
                    >
                      <Rect
                        width={previewRect.width || size.width}
                        height={previewRect.height || size.height}
                        fill="transparent"
                      />
                      {toRender.map((clip) => {
                        if (clip.type === "group") return null;
                        // 3. USE THE NEW MEMOIZED ITEM COMPONENT HERE
                        return (
                          <MediaDialogClipItem
                            key={clip.clipId}
                            clip={clip}
                            previewWidth={previewRect.width || size.width}
                            previewHeight={previewRect.height || size.height}
                            focusFrame={focusFrame}
                            inputId={timelineSelectorProps?.inputId}
                            getApplicators={getApplicators}
                            clipWithinFrame={clipWithinFrame}
                            getClipById={getClipById}
                            groupCentering={groupCentering}
                          />
                        );
                      })}
                      {/* Render AudioPreview for video and model clips that have audio */}
                      {toRender.map((clip) => {
                        if (clip.type !== "video" && clip.type !== "model") return null;
                        const startFrame = clip.startFrame || 0;
                        const groupStart = clip.groupId
                          ? getClipById(clip.groupId)?.startFrame || 0
                          : 0;
                        const relativeStart = startFrame - groupStart;
                        const hasOverlap =
                          (clip.groupId ? relativeStart : startFrame) > 0;
                        
                        return (
                          <AudioPreview
                            key={`audio-${clip.clipId}`}
                            {...{...(clip as any), hidden: false}}
                            overlap={hasOverlap}
                            rectWidth={previewRect.width || size.width}
                            rectHeight={previewRect.height || size.height}
                            inputMode={true}
                            inputId={timelineSelectorProps?.inputId}
                          />
                        );
                      })}
                      {/* Render AudioPreview for group clips by processing their child clips */}
                      {toRender.map((clip) => {
                        if (clip.type !== "group") return null;
                        const nested = ((clip as any).children as string[][] | undefined) ?? [];
                        const childIdsFlat = nested.flat();
                        const childClips = childIdsFlat
                          .map((id) => clips.find((c) => c.clipId === id))
                          .filter((c): c is AnyClipProps => 
                            !!c && (c.type === "video" || c.type === "audio" || c.type === "model")
                          );
                        
                        // Get the ORIGINAL group's startFrame from clips store, not from the normalized input clip
                        const originalGroup = clips.find((c) => c.clipId === clip.clipId);
                        const originalGroupStart = originalGroup?.startFrame || 0;
                        
                        return childClips.map((childClip) => {
                          // Child clips have absolute timeline frames, convert to group-relative
                          const childStart = (childClip.startFrame || 0) - originalGroupStart;
                          const childEnd = (childClip.endFrame || 0) - originalGroupStart;
                          const hasOverlap = childStart > 0;
                          
                          return (
                            <AudioPreview
                              key={`group-audio-${childClip.clipId}`}
                              {...{...(childClip as any), hidden: false, startFrame: childStart, endFrame: childEnd}}
                              overlap={hasOverlap}
                              rectWidth={previewRect.width || size.width}
                              rectHeight={previewRect.height || size.height}
                              inputMode={true}
                              inputId={timelineSelectorProps?.inputId}
                              preserveInputTiming={true}
                            />
                          );
                        });
                      })}
                    </Group>
                    <Group y={32} x={16} visible={canCrop}>
                      {/* Dark overlay outside crop area */}
                      <Path
                        data={overlayPath}
                        fill="rgba(0,0,0,0.6)"
                        fillRule="evenodd"
                        listening={false}
                      />

                      {/* Crop Transformer Rect */}
                      <Rect
                        ref={cropRectRef}
                        stroke="transparent"
                        strokeWidth={1}
                        fill="transparent"
                        draggable
                        dragBoundFunc={(pos) => {
                          if (!mediaDisplayRect || !cropRectRef.current)
                            return pos;

                          const groupOffsetX = 16;
                          const groupOffsetY = 32;

                          const {
                            x: localMinX,
                            y: localMinY,
                            width: boundaryW,
                            height: boundaryH,
                          } = mediaDisplayRect;
                          const minX = localMinX + groupOffsetX;
                          const minY = localMinY + groupOffsetY;
                          const maxX = minX + boundaryW;
                          const maxY = minY + boundaryH;

                          const node = cropRectRef.current;
                          const w = node.width() * node.scaleX();
                          const h = node.height() * node.scaleY();

                          let x = pos.x;
                          let y = pos.y;

                          if (x < minX) x = minX;
                          if (y < minY) y = minY;
                          if (x + w > maxX) x = maxX - w;
                          if (y + h > maxY) y = maxY - h;

                          return { x, y };
                        }}
                        onTransform={handleTransform}
                        onDragMove={handleTransform}
                      />

                      <Transformer
                        ref={transformerRef}
                        flipEnabled={false}
                        boundBoxFunc={(oldBox, newBox) => {
                          // limit resize
                          if (newBox.width < 5 || newBox.height < 5) {
                            return oldBox;
                          }

                          if (mediaDisplayRect) {
                            const groupOffsetX = 16;
                            const groupOffsetY = 32;

                            const {
                              x: localMinX,
                              y: localMinY,
                              width: boundaryW,
                              height: boundaryH,
                            } = mediaDisplayRect;
                            const minX = localMinX + groupOffsetX;
                            const minY = localMinY + groupOffsetY;
                            const maxX = minX + boundaryW;
                            const maxY = minY + boundaryH;

                            let { x, y, width, height } = newBox;

                            if (x < minX) {
                              width -= minX - x;
                              x = minX;
                            }
                            if (y < minY) {
                              height -= minY - y;
                              y = minY;
                            }
                            if (x + width > maxX) {
                              width = maxX - x;
                            }
                            if (y + height > maxY) {
                              height = maxY - y;
                            }

                            return {
                              ...newBox,
                              x,
                              y,
                              width,
                              height,
                            };
                          }

                          return newBox;
                        }}
                        anchorCornerRadius={100}
                        anchorFill="white"
                        anchorStroke="white"
                        anchorSize={14}
                        borderStroke="white"
                        borderStrokeWidth={3}
                        keepRatio={false}
                        rotateEnabled={false}
                        enabledAnchors={[
                          "top-left",
                          "top-center",
                          "top-right",
                          "middle-right",
                          "middle-left",
                          "bottom-left",
                          "bottom-center",
                          "bottom-right",
                        ]}
                        onTransform={handleTransform}
                      />
                    </Group>
                  </Layer>
                </Stage>
              </div>
              {isValidClip &&
                toRender.some((clip) => clip.type === "video") &&
                !timelineSelectorProps && (
                  <div className=" pb-4 flex justify-center px-4 w-full ">
                    <div className="bg-black/40 backdrop-blur-md rounded-md px-6 py-3 w-full flex items-center border border-white/10 shadow-sm">
                      <div
                        ref={progressBarRef}
                        className="relative w-full h-1 bg-white/20 rounded-full cursor-pointer group"
                        onMouseDown={handleProgressBarMouseDown}
                      >
                        <div
                          className={`absolute h-full bg-brand-primary rounded-full pointer-events-none `}
                          style={{ width: `${progress * 100}%` }}
                        />
                        <div
                          className={`absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg ${
                            isDragging ? "cursor-grabbing" : "cursor-grab"
                          }`}
                          style={{ left: `calc(${progress * 100}% - 6px)` }}
                        />
                      </div>
                    </div>
                  </div>
                )}
            </div>
          )}
        </div>
        {timelineSelectorProps && timelineSelectorProps.mode === "range" && (
          <div className="pb-4 flex justify-center px-4 w-full bg-black/50">
            <div className=" rounded-md px-4 py-3 pt-0 w-full flex items-center gap-x-4  shadow-sm">
              <button
                type="button"
                onClick={() => {
                  if (!onPlay || !onPause) return;
                  if (isSelectionPlaying) {
                    onPause();
                  } else {
                    onPlay();
                  }
                }}
                className="flex items-center justify-center w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
              >
                {isSelectionPlaying ? (
                  <LuPause className="w-4 h-4" />
                ) : (
                  <LuPlay className="w-4 h-4" />
                )}
              </button>
              <div className="flex-1 flex flex-col gap-y-1">
                <div className="relative w-full h-1.5 bg-white/20 rounded-full overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-brand-light rounded-full"
                    style={{ width: `${selectionProgress * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
        {timelineSelectorProps && (
          <div
            ref={divTimelineSelectorRef}
            className="px-6 py-4 border-t border-brand-light/10 bg-brand-background shrink-0 flex items-center justify-between"
          >
            <TimelineSelector
              clip={clipOverride ?? (selectedClip as AnyClipProps)}
              height={64}
              width={(divTimelineSelectorRef.current?.clientWidth ?? 0) - 60}
              mode={timelineSelectorProps.mode}
              inputId={timelineSelectorProps.inputId}
              maxDuration={maxDuration}
            />
          </div>
        )}
        <div className="px-6 py-4 border-t border-brand-light/10 bg-brand-background shrink-0 flex items-center justify-between">
          <div className="flex items-center gap-6 flex-1">
            {/* Aspect Ratio Control */}
            {canCrop && (
              <div className="flex items-center gap-3">
                <span className="text-[12px] font-medium">Aspect Ratio</span>
                <Select
                  value={aspectRatio}
                  onValueChange={(value) => {
                    setAspectRatio(value);
                    if (value !== "custom") {
                      applyAspectRatioCrop(value, { force: true });
                    }
                  }}
                  
                >
                  <SelectTrigger className="w-[100px] h-7.5! bg-secondary/50 border-none text-[11px] font-medium rounded-[6px]">
                    <SelectValue placeholder="Select ratio" />
                  </SelectTrigger>
                  <SelectContent className="bg-background text-foreground font-poppins dark z-101!">
                    {ASPECT_RATIOS.map((ratio) => (
                      <SelectItem
                        key={ratio.value}
                        value={ratio.value}
                        className="text-brand-light text-[11px] font-medium"
                      >
                        {ratio.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            {canCrop && (
              <Button
                variant="ghost"
                onClick={handleReset}
                className="h-7 px-5 hover:bg-secondary/80 text-[11.5px]! font-medium bg-brand-light/10 rounded-[6px]"
              >
                Reset
              </Button>
            )}
            <Button
              onClick={canCrop ? handleConfirm : () => onClose()}
              className="h-7 px-5 bg-brand-primary hover:bg-brand-primary/90 text-white text-[11.5px]! font-medium bg-brand-accent hover:bg-brand-accent-two-shade rounded-[6px]"
            >
              {canCrop ? "Confirm" : "Done"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
