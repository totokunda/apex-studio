import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { Stage, Layer, Group, Rect } from "react-konva";
import type Konva from "konva";
import { useClipStore } from "@/lib/clip";
import { useViewportStore } from "@/lib/viewport";
import { BASE_LONG_SIDE } from "@/lib/settings";
import { AnyClipProps } from "@/lib/types";
import VideoPreview from "@/components/preview/clips/VideoPreview";
import ImagePreview from "@/components/preview/clips/ImagePreview";
import ShapePreview from "@/components/preview/clips/ShapePreview";
import TextPreview from "@/components/preview/clips/TextPreview";
import DrawingPreview from "@/components/preview/clips/DrawingPreview";
import AudioPreview from "@/components/preview/clips/AudioPreview";
import { getApplicatorsForClip } from "@/lib/applicator-utils";
import { useWebGLHaldClut } from "@/components/preview/webgl-filters";
import { useInputControlsStore } from "@/lib/inputControl";

export type OffscreenPosterStageHandle = {
  toCanvas: (
    pixelRatio?: number,
    quality?: number,
  ) => Promise<HTMLCanvasElement | null>;
  toDataURL: (opts?: {
    mimeType?: string;
    quality?: number;
    pixelRatio?: number;
  }) => Promise<string | null>;
};

type Props = {
  width: number;
  height: number;
  frame?: number; // single frame mode
  // range playback (inclusive)
  startFrame?: number;
  endFrame?: number;
  step?: number;
  pixelRatio?: number;
  onFrame?: (frame: number, canvas: HTMLCanvasElement) => void | Promise<void>;
  autoStart?: boolean;
  inputId?: string;
  clipId?: string;
  clip?: AnyClipProps;
  baseLongSide?: number;
  ratioOverride?: number;
  // Offscreen fast mode: skip delays and allow downstream fast decoding
  offscreenFast?: boolean;
  // If true, do not wait for rAF between frames in range mode
  noDelay?: boolean;
  // Called when range playback completes
  onEnd?: () => void;
};

const OffscreenPosterStage = forwardRef<OffscreenPosterStageHandle, Props>(
  (
    {
      width,
      height,
      frame,
      startFrame,
      endFrame,
      step = 1,
      pixelRatio = 1,
      onFrame,
      autoStart = true,
      inputId,
      clipId,
      clip: clipOverride,
      baseLongSide = BASE_LONG_SIDE,
      ratioOverride,
      offscreenFast = false,
      noDelay = false,
      onEnd,
    },
    ref,
  ) => {
    const stageRef = useRef<Konva.Stage>(null);
    const aspectRatio = useViewportStore((s) => s.aspectRatio);
    const getClipById = useClipStore((s) => s.getClipById);
    const clips = useClipStore((s) => s.clips);
    const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
    const timelines = useClipStore((s) => s.timelines);
    const haldClutInstance = useWebGLHaldClut();
    const setInputFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
    const [currentFrame, setCurrentFrame] = useState<number>(
      frame ?? startFrame ?? 0,
    );
    const cancelRef = useRef<boolean>(false);

    // Sync currentFrame when explicit single-frame prop changes
    useEffect(() => {
      if (typeof frame === "number" && Number.isFinite(frame)) {
        setCurrentFrame(Math.trunc(frame));
      }
    }, [frame]);

    useImperativeHandle(
      ref,
      () => ({
        async toCanvas(_pixelRatio = 1, quality = 1.0) {
          const stage = stageRef.current;
          if (!stage) return null;
          await new Promise(requestAnimationFrame);
          if (typeof stage.toCanvas === "function") {
            return stage.toCanvas({ pixelRatio: 1, quality });
          }
          const dataURL = stage.toDataURL({ pixelRatio: 1, quality });
          const img = new Image();
          img.src = dataURL;
          await img.decode();
          const canvas = document.createElement("canvas");
          canvas.width = Math.round(width);
          canvas.height = Math.round(height);
          const ctx = canvas.getContext("2d");
          if (!ctx) return null;
          ctx.drawImage(img, 0, 0);
          return canvas;
        },
        async toDataURL(opts) {
          const stage = stageRef.current;
          if (!stage) return null;
          await new Promise(requestAnimationFrame);
          const { mimeType, quality } = opts || {};
          return stage.toDataURL({ mimeType, quality, pixelRatio: 1 });
        },
      }),
      [width, height],
    );

    const sortClips = useCallback(
      (clipsToSort: AnyClipProps[]) => {
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
            const nested =
              ((g as any).children as string[][] | undefined) ?? [];
            return nested.flat();
          }),
        );

        const groupUnits: GroupUnit[] = groups.map((g) => {
          const y =
            timelines.find((t) => t.timelineId === g.timelineId)?.timelineY ??
            0;
          const start = g.startFrame ?? 0;
          const nested = ((g as any).children as string[][] | undefined) ?? [];
          const childIdsFlat = nested.flat();
          const children = childIdsFlat
            .map((id) => clips.find((c) => c.clipId === id))
            .filter(Boolean) as AnyClipProps[];
          return { kind: "group", id: g.clipId, y, start, children };
        });

        const singleUnits: SingleUnit[] = clipsToSort
          .filter((c) => c.type !== "group" && !childrenSet.has(c.clipId))
          .map((c) => {
            const y =
              timelines.find((t) => t.timelineId === c.timelineId)?.timelineY ??
              0;
            const start = c.startFrame ?? 0;
            return { kind: "single", y, start, clip: c };
          });

        const units = [...groupUnits, ...singleUnits].sort((a, b) => {
          if (a.y !== b.y) return b.y - a.y;
          return a.start - b.start;
        });

        const result: AnyClipProps[] = [];
        for (const u of units) {
          if (u.kind === "single") {
            result.push(u.clip);
          } else {
            result.push(...u.children.reverse());
          }
        }
        return result;
      },
      [timelines, clips],
    );

    const rectDims = useMemo(() => {
      const editorRatio = aspectRatio.width / aspectRatio.height;
      const ratio =
        typeof ratioOverride === "number" &&
        isFinite(ratioOverride) &&
        ratioOverride > 0
          ? ratioOverride
          : editorRatio;
      if (!Number.isFinite(ratio) || ratio <= 0) {
        return { rectWidth: 0, rectHeight: 0 };
      }
      return { rectWidth: baseLongSide * ratio, rectHeight: baseLongSide };
    }, [aspectRatio.width, aspectRatio.height, baseLongSide, ratioOverride]);

    const view = useMemo(() => {
      const { rectWidth, rectHeight } = rectDims;
      if (!rectWidth || !rectHeight || !width || !height)
        return { scale: 1, x: 0, y: 0 };
      const scaleX = width / rectWidth;
      const scaleY = height / rectHeight;
      // Use cover scaling so posters fill their containers
      const scale = Math.max(scaleX, scaleY);
      const x = (width - rectWidth * scale) / 2;
      const y = (height - rectHeight * scale) / 2;
      return { scale, x, y };
    }, [rectDims.rectWidth, rectDims.rectHeight, width, height]);

    const toRender = useMemo(() => {
      if (clipOverride) {
        if (clipOverride.type === "group") {
          return sortClips([clipOverride]);
        }
        return [clipOverride];
      }
      if (!clipId) return [] as AnyClipProps[];
      const c = getClipById(clipId);
      if (!c) return [] as AnyClipProps[];
      if (c.type === "group") return sortClips([c]);
      return [c];
    }, [clipOverride, clipId, getClipById, clips, sortClips]);

    const getApplicators = useCallback(
      (id: string, frameOverride?: number) => {
        return getApplicatorsForClip(id, {
          haldClutInstance,
          focusFrameOverride: frameOverride,
        });
      },
      [haldClutInstance],
    );

    const { rectWidth, rectHeight } = rectDims;

    // Drive input focus for video/image components which use inputMode to pick frames
    useEffect(() => {
      setInputFocusFrame(currentFrame, inputId);
    }, [currentFrame, inputId, setInputFocusFrame]);

    // Internal playback loop (range mode)
    useEffect(() => {
      cancelRef.current = false;
      const hasRange =
        Number.isFinite(startFrame as number) &&
        Number.isFinite(endFrame as number);
      if (!hasRange || !autoStart) return;

      const run = async () => {
        const s = Math.max(0, Math.trunc(startFrame as number));
        const e = Math.max(s, Math.trunc(endFrame as number));
        const inc = Math.max(1, Math.trunc(step as number));
        for (let f = s; f <= e; f += inc) {
          if (cancelRef.current) break;
          setCurrentFrame(f);
          if (!noDelay) {
            await new Promise(requestAnimationFrame);
            await new Promise(requestAnimationFrame);
          }
          if (cancelRef.current) break;
          try {
            const canvas = await stageRef.current?.toCanvas({ pixelRatio });
            if (canvas && onFrame) {
              await onFrame(f, canvas);
            }
          } catch {}
        }
        try {
          onEnd?.();
        } catch {}
      };
      run();
      return () => {
        cancelRef.current = true;
      };
    }, [
      startFrame,
      endFrame,
      step,
      pixelRatio,
      onFrame,
      autoStart,
      noDelay,
      onEnd,
    ]);

    return (
      <div
        aria-hidden
        style={{
          position: "absolute",
          left: -10000,
          top: -10000,
          width,
          height,
          opacity: 0,
          pointerEvents: "none",
        }}
      >
        <Stage ref={stageRef} width={width} height={height} listening={false}>
          <Layer listening={false}>
            <Group
              x={view.x}
              y={view.y}
              scaleX={view.scale}
              scaleY={view.scale}
              listening={false}
            >
              <Rect
                x={0}
                y={0}
                width={rectWidth}
                height={rectHeight}
                fill={"#000000"}
                listening={false}
              />
              {toRender.map((clip) => {
                if (clip.type === "group") return null;
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
                // Use global frame for bounds/effects; keep local currentFrame for child playback
                const effectiveGlobalFrame = clip.groupId
                  ? currentFrame + groupStart
                  : currentFrame;
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
                const overrideToUse = clipOverride ? clip : undefined;
                if (
                  (!overrideToUse?.groupId &&
                    overrideToUse?.type === "image") ||
                  overrideToUse?.type === "video"
                ) {
                  overrideToUse.originalTransform = undefined;
                  overrideToUse.transform = undefined;
                  overrideToUse.height = undefined;
                  overrideToUse.width = undefined;
                }

                switch (clip.type) {
                  case "video":
                    return (
                      <VideoPreview
                        key={clip.clipId}
                        {...(clip as any)}
                        overrideClip={overrideToUse}
                        rectWidth={rectWidth}
                        rectHeight={rectHeight}
                        applicators={applicators}
                        overlap={true}
                        inputMode={true}
                        inputId={inputId}
                        focusFrameOverride={currentFrame}
                        currentLocalFrameOverride={currentFrame}
                        offscreenFast={offscreenFast}
                      />
                    );
                  case "image":
                    return (
                      <ImagePreview
                        key={clip.clipId}
                        {...(clip as any)}
                        overrideClip={overrideToUse}
                        rectWidth={rectWidth}
                        rectHeight={rectHeight}
                        applicators={applicators}
                        overlap={true}
                        inputMode={true}
                        inputId={inputId}
                        focusFrameOverride={currentFrame}
                        currentLocalFrameOverride={currentFrame}
                      />
                    );
                  case "shape":
                    return (
                      <ShapePreview
                        key={clip.clipId}
                        {...(clip as any)}
                        rectWidth={rectWidth}
                        rectHeight={rectHeight}
                        applicators={applicators}
                        assetMode={true}
                      />
                    );
                  case "text":
                    return (
                      <TextPreview
                        key={clip.clipId}
                        {...(clip as any)}
                        rectWidth={rectWidth}
                        rectHeight={rectHeight}
                        applicators={applicators}
                        assetMode={true}
                      />
                    );
                  case "draw":
                    return (
                      <DrawingPreview
                        key={clip.clipId}
                        {...(clip as any)}
                        rectWidth={rectWidth}
                        rectHeight={rectHeight}
                        assetMode={true}
                      />
                    );
                  default:
                    return null;
                }
              })}
              {toRender.map((clip) => {
                if (clip.type !== "video") return null;
                const startFrame = clip.startFrame || 0;
                const groupStart = clip.groupId
                  ? getClipById(clip.groupId)?.startFrame || 0
                  : 0;
                const relativeStart = startFrame - groupStart;
                const hasOverlap =
                  (clip.groupId ? relativeStart : startFrame) > 0
                    ? true
                    : false;
                const effectiveGlobalFrame = clip.groupId
                  ? currentFrame + groupStart
                  : currentFrame;
                const clipAtFrame = clipWithinFrame(
                  clip,
                  effectiveGlobalFrame,
                  hasOverlap,
                  0,
                );
                if (!clipAtFrame && clip.groupId) return null;
                const overrideToUse = clipOverride ? clip : undefined;
                return (
                  <AudioPreview
                    key={clip.clipId}
                    {...(clip as any)}
                    overrideClip={overrideToUse}
                    overlap={hasOverlap}
                    rectWidth={rectWidth}
                    rectHeight={rectHeight}
                    inputMode={true}
                    inputId={inputId}
                  />
                );
              })}
            </Group>
          </Layer>
        </Stage>
      </div>
    );
  },
);

export default OffscreenPosterStage;
