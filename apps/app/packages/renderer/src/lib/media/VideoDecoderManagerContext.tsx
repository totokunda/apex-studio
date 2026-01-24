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
    const activeProject = useProjectsStore((s) => s.getActiveProject());
    const activeFolderUuid = activeProject?.folderUuid;

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
                folderUuid: activeFolderUuid,
              });
            }
          }
        }
      }
    }, [assetClipSig, activeFolderUuid]);
    

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
    if (ctx) return ctx;

    // Hardening: in case a consumer mounts under a different React root (or a future
    // entrypoint forgets to mount the provider), avoid crashing the app.
    // This fallback should be rare; we warn once to keep it visible in dev logs.
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    return getFallbackVideoDecoderManager();
}

let FALLBACK_MANAGER: VideoDecoderManager | null = null;
let DID_WARN_FALLBACK = false;
function getFallbackVideoDecoderManager(): VideoDecoderManager {
  if (!FALLBACK_MANAGER) {
    FALLBACK_MANAGER = new VideoDecoderManager();
  }
  if (!DID_WARN_FALLBACK) {
    DID_WARN_FALLBACK = true;
    // eslint-disable-next-line no-console
    console.warn(
      "[VideoDecoderManager] useVideoDecoderManager() was called without a VideoDecoderManagerProvider. Falling back to a singleton manager.",
    );
  }
  return FALLBACK_MANAGER;
}


