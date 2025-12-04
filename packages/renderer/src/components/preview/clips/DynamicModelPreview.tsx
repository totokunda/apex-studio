import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnyClipProps, ImageClipProps, ModelClipProps, VideoClipProps } from "@/lib/types";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import VideoPreview from "./VideoPreview";
import ImagePreview from "./ImagePreview";
import { BaseClipApplicator } from "./apply/base";
import { useClipStore } from "@/lib/clip";
import { updateClip } from "@app/preload/src/projects";
import _ from "lodash";

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

  const getAssetById = useClipStore((s) => s.getAssetById);
  const src = useMemo(() => getAssetById((clip as ModelClipProps)?.assetId ?? "")?.path || "", [(clip as ModelClipProps)?.assetId, getAssetById]);
  const lastAspectJobIdRef = useRef<string | undefined>(undefined);
  const {updateClip} = useClipStore();
  // Resolve info for the active source only; keep showing previous src until new info is ready
  const info = useMemo(() => getMediaInfoCached(src), [src]);


  // Fallback type guess by file extension while media info is being resolved
  const typeGuess = useMemo(() => {
    const normalized = (src || "")
      .split("?")[0]
      .split("#")[0]
      .toLowerCase();
    if (/\.(mp4|mov|webm|m4v|avi|mkv)$/.test(normalized)) return "video";
    if (/\.(png|jpg|jpeg|webp|bmp|gif)$/.test(normalized)) return "image";
    return null;
  }, [src]);


  // When a new job becomes active, update aspect ratio and reset transforms once
  useEffect(() => {
    const jobId = (clip as ModelClipProps)?.activeJobId as string | undefined;
    if (!jobId) return;
    if (lastAspectJobIdRef.current === jobId) return;

    try {
       if (!info) return;

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
    
  }, [(clip as ModelClipProps)?.activeJobId, clip.clipId, src]);

  useEffect(() => {
    
    if (info && info.video) {
      let patches: Partial<ModelClipProps> = {};
      if (!isFinite(clip.trimStart ?? 0)) {
        patches.trimStart = 0;
      } 
      if (!isFinite(clip.trimEnd ?? 0)) {
        patches.trimEnd = 0;
      }
      console.log("patches", patches);
      if (!_.isEmpty(patches)) {
        console.log("updating clip", patches);
        updateClip(clip.clipId, patches);
      }
    }
  }, [info, clip.clipId, updateClip]);




  if (info?.video || (!info && typeGuess === "video" && src)) {
    return (
      <VideoPreview
        {...({
          ...(clip as VideoClipProps),
        })}
        rectWidth={rectWidth}
        rectHeight={rectHeight}
        applicators={applicators}
        overlap={overlap}
      />
    );
  }

  if (info?.image || (!info && typeGuess === "image" && src)) {
    return (
      <ImagePreview
        {...({
          ...(clip as ImageClipProps),
        })}
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
