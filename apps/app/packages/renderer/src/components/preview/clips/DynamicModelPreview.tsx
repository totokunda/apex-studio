import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnyClipProps, ImageClipProps, ModelClipProps, VideoClipProps } from "@/lib/types";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import VideoPreview from "./VideoPreview";
import ImagePreview from "./ImagePreview";
import { BaseClipApplicator } from "./apply/base";
import { useClipStore } from "@/lib/clip";

import _ from "lodash";

interface DynamicModelPreviewProps {
  clip: ModelClipProps;
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


  useEffect(() => {
    
    if (info && info.video) {
      let patches: Partial<ModelClipProps> = {};
      if (!isFinite(clip.trimStart ?? 0)) {
        patches.trimStart = 0;
      } 
      if (!isFinite(clip.trimEnd ?? 0)) {
        patches.trimEnd = 0;
      }
      if (!_.isEmpty(patches)) {
        updateClip(clip.clipId, patches);
      }
    }
  }, [info, clip.clipId, updateClip]);



  if (info?.video || (!info && typeGuess === "video" && src)) {
    return (
      <VideoPreview
        {...({
          ...(clip as unknown as VideoClipProps),
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
          ...(clip as unknown as ImageClipProps),
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
