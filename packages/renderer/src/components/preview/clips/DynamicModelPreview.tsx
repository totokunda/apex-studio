import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnyClipProps } from "@/lib/types";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import VideoPreview from "./VideoPreview";
import ImagePreview from "./ImagePreview";
import { BaseClipApplicator } from "./apply/base";
import { useClipStore } from "@/lib/clip";

interface DynamicModelPreviewProps {
  clip: AnyClipProps;
  rectWidth: number;
  rectHeight: number;
  applicators: BaseClipApplicator[];
  overlap: boolean;
}

const DynamicModelPreview: React.FC<DynamicModelPreviewProps> = ({
  clip,
  rectWidth,
  rectHeight,
  applicators,
  overlap,
}) => {
  const src = (clip as any)?.src || "";
  const [tick, setTick] = useState(0);
  const [activeSrc, setActiveSrc] = useState(src);
  const lastAspectJobIdRef = useRef<string | undefined>(undefined);

  // Resolve info for the active source only; keep showing previous src until new info is ready
  const info = useMemo(() => getMediaInfoCached(activeSrc), [activeSrc, tick]);

  // Fallback type guess by file extension while media info is being resolved
  const typeGuess = useMemo(() => {
    const normalized = (activeSrc || "")
      .split("?")[0]
      .split("#")[0]
      .toLowerCase();
    if (/\.(mp4|mov|webm|m4v|avi|mkv)$/.test(normalized)) return "video";
    if (/\.(png|jpg|jpeg|webp|bmp|gif)$/.test(normalized)) return "image";
    return null;
  }, [activeSrc]);

  // When src changes, prefetch media info for the new src and switch only when ready
  useEffect(() => {
    let cancelled = false;
    if (!src || src === activeSrc)
      return () => {
        cancelled = true;
      };
    const cached = getMediaInfoCached(src);
    if (cached) {
      if (!cancelled) {
        setActiveSrc(src);
        setTick((v) => v + 1);
      }
      return () => {
        cancelled = true;
      };
    }
    (async () => {
      try {
        await getMediaInfo(src, { sourceDir: "apex-cache" });
      } finally {
        if (!cancelled) {
          setActiveSrc(src);
          setTick((v) => v + 1);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [src]);

  // When a new job becomes active, update aspect ratio and reset transforms once
  useEffect(() => {
    const jobId = (clip as any)?.activeJobId as string | undefined;
    if (!jobId) return;
    if (lastAspectJobIdRef.current === jobId) return;

    let cancelled = false;
    (async () => {
      try {
        const targetSrc = (clip as any)?.src || activeSrc;
        let info = getMediaInfoCached(targetSrc);
        if (!info) {
          try {
            info = await getMediaInfo(targetSrc, { sourceDir: "apex-cache" });
          } catch {}
        }
        if (cancelled) return;
        const dims = (() => {
          const v = (info as any)?.video;
          const im = (info as any)?.image;
          if (
            v &&
            typeof (v as any).displayWidth === "number" &&
            typeof (v as any).displayHeight === "number"
          ) {
            return {
              w: Math.max(0, (v as any).displayWidth || 0),
              h: Math.max(0, (v as any).displayHeight || 0),
            };
          }
          if (
            im &&
            typeof (im as any).width === "number" &&
            typeof (im as any).height === "number"
          ) {
            return {
              w: Math.max(0, (im as any).width || 0),
              h: Math.max(0, (im as any).height || 0),
            };
          }
          return { w: 0, h: 0 };
        })();
        if (dims.w > 0 && dims.h > 0) {
          const ar = dims.w / Math.max(1, dims.h);
          try {
            const updateClip = useClipStore.getState().updateClip;
            updateClip(clip.clipId, {
              transform: undefined,
              originalTransform: undefined,
              mediaWidth: dims.w as any,
              mediaHeight: dims.h as any,
              mediaAspectRatio: ar as any,
            } as any);
            lastAspectJobIdRef.current = jobId;
          } catch {}
        }
      } catch {}
    })();
    return () => {
      cancelled = true;
    };
  }, [(clip as any)?.activeJobId, clip.clipId, activeSrc]);

  if (info?.video || (!info && typeGuess === "video" && activeSrc)) {
    return (
      <VideoPreview
        {...({ ...(clip as any), src: activeSrc } as any)}
        rectWidth={rectWidth}
        rectHeight={rectHeight}
        applicators={applicators}
        overlap={overlap}
      />
    );
  }
  if (info?.image || (!info && typeGuess === "image" && activeSrc)) {
    return (
      <ImagePreview
        {...({ ...(clip as any), src: activeSrc } as any)}
        rectWidth={rectWidth}
        rectHeight={rectHeight}
        applicators={applicators}
        overlap={overlap}
      />
    );
  }
  // While probing type, render nothing; effect above will trigger rerender when info is ready
  return null;
};

export default DynamicModelPreview;
