import React, {
  useCallback,
  useMemo,
  useEffect,
  useRef,
  useState,
} from "react";
import { Stage, Layer, Group, Rect } from "react-konva";
import { useClipStore } from "@/lib/clip";
import { useViewportStore } from "@/lib/viewport";
import { BASE_LONG_SIDE } from "@/lib/settings";
import { AnyClipProps, ImageClipProps, VideoClipProps } from "@/lib/types";
import VideoPreview from "@/components/preview/clips/VideoPreview";
import ImagePreview from "@/components/preview/clips/ImagePreview";
import ShapePreview from "@/components/preview/clips/ShapePreview";
import TextPreview from "@/components/preview/clips/TextPreview";
import DrawingPreview from "@/components/preview/clips/DrawingPreview";
import { getApplicatorsForClip } from "@/lib/applicator-utils";
import { useWebGLHaldClut } from "@/components/preview/webgl-filters";
import { useInputControlsStore } from "@/lib/inputControl";
import AudioPreview from "@/components/preview/clips/AudioPreview";
import { getMediaInfo } from "@/lib/media/utils";

// Kept for parity with other components if needed later; not used in poster sorting
// const getGroupChildren = (groupClip: AnyClipProps, allClips: AnyClipProps[]) => {
//   const nested = ((groupClip as any).children as string[][] | undefined) ?? [];
//   const childIdsFlat = nested.flat();
//   const children = childIdsFlat
//     .map(id => allClips.find(c => c.clipId === id))
//     .filter(Boolean) as AnyClipProps[];
//   return children;
// }

import { remapMaskForMediaDialog } from "@/lib/mask/transformUtils";
import { ClipTransform, MaskClipProps } from "@/lib/types";
import { BaseClipApplicator } from "@/components/preview/clips/apply/base";
import DynamicModelPreview from "@/components/preview/clips/DynamicModelPreview";

// 1. EXTRACTED COMPONENT TO HANDLE STABLE OVERRIDES
const PosterClipItem = React.memo(({
  clip,
  rectWidth,
  rectHeight,
  focusFrame,
  inputId,
  getApplicators,
  clipWithinFrame,
  getClipById,
  isDialogOpen,
}: {
  clip: AnyClipProps;
  rectWidth: number;
  rectHeight: number;
  focusFrame: number;
  inputId?: string;
  getApplicators: (id: string, frame: number) => BaseClipApplicator[];
  clipWithinFrame: (clip: AnyClipProps, frame: number, overlap: boolean, padding: number) => boolean;
  getClipById: (id: string) => AnyClipProps | undefined;
  isDialogOpen?: boolean;
}) => {

  const groupStart = clip.groupId
    ? getClipById(clip.groupId)?.startFrame || 0
    : 0;

  const effectiveGlobalFrame = clip.groupId
    ? focusFrame + groupStart
    : focusFrame;
    
  const applicators = getApplicators(
    clip.clipId,
    effectiveGlobalFrame,
  );

  // 2. MEMOIZE THE OVERRIDE CLIP TO PREVENT VIDEO PREVIEW RESETS
  const overrideToUse = useMemo(() => {
    const override = { ...clip };
    if (
        override?.transform &&
        !override.groupId &&
        override.originalTransform
    ) {
        if (override.transform.crop) {
            override.transform = {
                ...override.originalTransform,
                width: rectWidth,
                height: rectHeight,
                x: 0,
                y: 0,
                scaleX: 1,
                scaleY: 1,
                rotation: 0,
                crop: override.transform.crop,
                opacity: 100,
            };
        } else {
            override.transform = {
                ...override.originalTransform,
            };
        }

        // Remap masks
        if ((override as any).masks?.length > 0) {
            const masks = [
                ...(override as any).masks,
            ] as MaskClipProps[];

            (override as any).masks = masks.map(
                (mask: MaskClipProps) => {
                    // @ts-ignore
                    const nW = clip.mediaWidth || clip.width || 0;
                    // @ts-ignore
                    const nH = clip.mediaHeight || clip.height || 0;

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

                    const currentTransform = clip.transform ?? nativeTransform;
                    const zeroCurrentTransform: ClipTransform = {
                        ...currentTransform,
                        x: 0,
                        y: 0,
                    };

                    const overrideTransform = override.transform!;
                    const zeroOverrideTransform: ClipTransform = {
                        ...overrideTransform,
                        x: 0,
                        y: 0,
                    };

                    const maskForRemap = { ...mask, transform: undefined };

                    const toOriginal = remapMaskForMediaDialog(
                        maskForRemap,
                        zeroCurrentTransform,
                        nativeTransform,
                    );

                    const toOverride = remapMaskForMediaDialog(
                        toOriginal,
                        nativeTransform,
                        zeroOverrideTransform,
                    );

                    toOverride.transform = { ...overrideTransform };
                    return toOverride;
                },
            );
        }
    }
    return override;
  }, [clip, rectWidth, rectHeight, isDialogOpen]);
  

  switch (clip.type) {
    case "video":
      return (
        <VideoPreview
          key={`${clip.clipId}-${isDialogOpen}`}
          {...{...(clip as any), hidden: false}}
          overrideClip={overrideToUse}
          rectWidth={rectWidth}
          rectHeight={rectHeight}
          applicators={applicators}
          overlap={true}
          inputMode={true}
          focusFrameOverride={focusFrame}
          inputId={inputId}
          decoderKey={`poster::${clip.clipId}`}
        />
      );
    case "image":
      return (
        <ImagePreview
          key={clip.clipId}
          {...{...(clip as any), hidden: false}}
          overrideClip={overrideToUse}
          rectWidth={rectWidth}
          rectHeight={rectHeight}
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
          {...{...(clip as any), hidden: false}}
          rectWidth={rectWidth}
          rectHeight={rectHeight}
          assetMode={true}
          applicators={applicators}
        />
      );
    case "model":
      return (
        <DynamicModelPreview
          key={clip.clipId}
          clip={clip}
          applicators={applicators}
          overlap={true}
          rectWidth={rectWidth}
          rectHeight={rectHeight}
        />
      );
    default:
      return null;
  }
});

const TimelineClipPosterPreview: React.FC<{
  clipId?: string;
  clip?: AnyClipProps;
  width: number;
  height: number;
  inputId?: string;
  ratioOverride?: number;
  audioOnly?: boolean;
  needsStage?: boolean;
  isDialogOpen?: boolean;
}> = ({
  clipId,
  clip: clipOverride,
  width,
  height,
  inputId,
  ratioOverride,
  audioOnly = false,
  needsStage = false,
  isDialogOpen = false,
}) => {
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const getClipById = useClipStore((s) => s.getClipById);
  const clips = useClipStore((s) => s.clips);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
  const haldClutInstance = useWebGLHaldClut();
  const focusFrame = useInputControlsStore(
    (s) => s.getFocusFrame(inputId ?? "") ?? 0,
  );

  const timelines = useClipStore((s) => s.timelines);
  const isOnTimeline = useMemo(() => {
    return getClipById(clipId ?? "") !== undefined && clipId !== undefined;
  }, [clipId, getClipById]);

  const ratioCacheByClipIdRef = useRef<Record<string, number>>({});
  const [contentRatio, setContentRatio] = useState<number | null>(null);
  const getAssetById = useClipStore((s) => s.getAssetById);

  const rootClip = useMemo<AnyClipProps | null>(() => {
    if (clipOverride) return clipOverride as AnyClipProps;
    if (!clipId) return null;
    return getClipById(clipId) as AnyClipProps | null;
  }, [clipOverride, clipId, getClipById]);

  useEffect(() => {
    const clip = rootClip;
    if (!clip) {
      setContentRatio(null);
      return;
    }
    if (clip.type === "group") {
      setContentRatio(null);
      return;
    }
    const cached = ratioCacheByClipIdRef.current[clip.clipId];
    if (typeof cached === "number" && cached > 0) {
      setContentRatio(cached);
      return;
    }
    const asset = getAssetById((clip as VideoClipProps | ImageClipProps).assetId);
    const src = asset?.path;
    if (!src) {
      setContentRatio(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo(src);
        const w =
          (info as any)?.stats?.video?.width ??
          (info as any)?.stats?.image?.width ??
          (info as any)?.width ??
          (info as any)?.streams?.[0]?.width;
        const h =
          (info as any)?.stats?.video?.height ??
          (info as any)?.stats?.image?.height ??
          (info as any)?.height ??
          (info as any)?.streams?.[0]?.height;
        let r = Number(w) / Number(h);
        const transform = (clip as any)?.transform as ClipTransform | undefined;
        const crop = transform?.crop;
        if (
          crop &&
          crop.width > 0 &&
          crop.height > 0 &&
          Number.isFinite(Number(w)) &&
          Number.isFinite(Number(h)) &&
          Number(w) > 0 &&
          Number(h) > 0
        ) {
          r = (Number(w) * crop.width) / (Number(h) * crop.height);
        }

        if (!cancelled && Number.isFinite(r) && r > 0) {
          ratioCacheByClipIdRef.current[clip.clipId] = r;
          setContentRatio(r);
        }
      } catch {
        if (!cancelled) setContentRatio(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [rootClip]);

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

  const rectDims = useMemo(() => {
    const editorRatio = aspectRatio.width / aspectRatio.height;
    const ratio =
      typeof ratioOverride === "number" && ratioOverride > 0
        ? ratioOverride
        : !isOnTimeline && typeof contentRatio === "number" && contentRatio > 0
          ? contentRatio
          : editorRatio;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return { rectWidth: 0, rectHeight: 0 };
    }
    return { rectWidth: BASE_LONG_SIDE * ratio, rectHeight: BASE_LONG_SIDE };
  }, [
    aspectRatio.width,
    aspectRatio.height,
    rootClip,
    contentRatio,
    ratioOverride,
  ]);

  const view = useMemo(() => {
    const { rectWidth, rectHeight } = rectDims;
    if (!rectWidth || !rectHeight || !width || !height)
      return { scale: 1, x: 0, y: 0 };
    const scaleX = width / rectWidth;
    const scaleY = height / rectHeight;
    // Use cover scaling so posters fill the container
    const scale = Math.max(scaleX, scaleY);
    const x = (width - rectWidth * scale) / 2;
    const y = (height - rectHeight * scale) / 2;
    return { scale, x, y };
  }, [rectDims.rectWidth, rectDims.rectHeight, width, height]);

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

  const getApplicators = React.useCallback(
    (id: string, frameOverride?: number) => {
      return getApplicatorsForClip(id, {
        haldClutInstance,
        focusFrameOverride: frameOverride,
      });
    },
    [haldClutInstance],
  );

  const { rectWidth, rectHeight } = rectDims;

  const StageComponent = needsStage ? Stage : React.Fragment;
  const StageProps = needsStage
    ? { width: audioOnly ? 1 : width, height: audioOnly ? 1 : height }
    : {};


  return (
    <div className="w-full h-auto flex flex-col items-center justify-start bg-black">
      <StageComponent
        key={`${clipOverride?.clipId || clipId || "none"}${audioOnly ? ":audio" : ""}`}
        {...StageProps}
      >
        <Layer>
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
            />
            {audioOnly
              ? null
              : toRender.map((clip) => {
                  // 3. USE THE NEW MEMOIZED ITEM COMPONENT HERE
                  return (
                    <PosterClipItem
                        key={`${clip.clipId}`}
                        clip={clip}
                        rectWidth={rectWidth}
                        rectHeight={rectHeight}
                        focusFrame={focusFrame}
                        inputId={inputId}
                        getApplicators={getApplicators}
                        clipWithinFrame={clipWithinFrame}
                        getClipById={getClipById}
                        isDialogOpen={isDialogOpen}
                    />
                  );
                })}
            {toRender.map((clip) => {
              if (clip.type !== "video" && clip.type !== "audio") return null;
              const startFrame = clip.startFrame || 0;
              const groupStart = clip.groupId
                ? getClipById(clip.groupId)?.startFrame || 0
                : 0;
              const relativeStart = startFrame - groupStart;
              const hasOverlap =
                (clip.groupId ? relativeStart : startFrame) > 0 ? true : false;

         
              const overrideToUse = clipOverride ? clip : undefined;
              return (
                <AudioPreview
                  key={clip.clipId}
                  {...{...(clip as any), hidden: false}}
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
      </StageComponent>
    </div>
  );
};

export default TimelineClipPosterPreview;
