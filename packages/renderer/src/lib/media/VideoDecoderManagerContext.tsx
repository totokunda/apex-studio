import React, { createContext, useContext, useRef, useEffect } from "react";
import { VideoDecoderManager } from "./video-decoder";
import { Asset } from "../types";
import { useClipStore } from "../clip";
import { ClipProps } from "../types";
import { getMediaInfoCached } from "./utils";
import { useProjectsStore } from "../projects";

const VideoDecoderManagerContext = createContext<VideoDecoderManager | null>(null);

type ProviderProps = {
    children: React.ReactNode;
};

const assetClipSignature = (assets: Record<string, Asset>, clips: ClipProps[]) => {
  return Object.values(assets).map(a => a.id).join(",") + clips.map(c => c.clipId).join(",");
}

export const VideoDecoderManagerProvider: React.FC<ProviderProps> = ({ children }) => {
    const managerRef = useRef<VideoDecoderManager | null>(null);
    const { assets, clips } = useClipStore();
    const assetClipSig = assetClipSignature(assets, clips);
    const activePoject = useProjectsStore((s) => s.getActiveProject());

    useEffect(() => {
      for (const clip of clips) {
        if (clip.type === "video") {
          let mediaInfo = getMediaInfoCached(clip.assetId);

          const decoderId = `${clip.assetId}::${clip.clipId}`;
          if (mediaInfo && mediaInfo.video && !managerRef.current?.hasAsset(decoderId)) {
            const decoderConfig = mediaInfo.videoDecoderConfig;
            if (decoderConfig) {
              managerRef.current?.addAsset(assets[clip.assetId], {
                mediaInfo,
                videoDecoderConfig: decoderConfig,
                logicalId: decoderId,
                folderUuid: activePoject?.folderUuid,
              });
            }
          }
        }
      }
    }, [assetClipSig]);
    

    if (!managerRef.current) {
        managerRef.current = new VideoDecoderManager();
    }

    return (
        <VideoDecoderManagerContext.Provider value={managerRef.current}>
            {children}
        </VideoDecoderManagerContext.Provider>
    );
};

export function useVideoDecoderManager(): VideoDecoderManager {
    const ctx = useContext(VideoDecoderManagerContext);
    if (!ctx) {
        throw new Error("useVideoDecoderManager must be used within a VideoDecoderManagerProvider");
    }
    return ctx;
}


