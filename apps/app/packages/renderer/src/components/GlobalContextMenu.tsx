import React, { useEffect, useRef, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { useContextMenuStore } from "@/lib/context-menu";
import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { cn } from "@/lib/utils";
import { useViewportStore } from "@/lib/viewport";
import type { AnyClipProps, ImageClipProps, VideoClipProps } from "@/lib/types";
import { prepareExportClipsForValue } from "@/lib/prepareExportClips";
import { getMediaInfoCached } from "@/lib/media/utils";
import { exportSequence, exportClip } from "@app/export-renderer";
import { saveImageToPath } from "@app/preload";
import ClipExportModal, {
  ClipExportSettings,
} from "@/components/dialogs/ClipExportModal";

const Key: React.FC<{ text: string }> = ({ text }) => (
  <span className=" text-[10px] text-brand-light/60">{text}</span>
);

const GlobalContextMenu: React.FC = () => {
  const { open, position, items, groups, closeMenu, setPosition, target } =
    useContextMenuStore();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clipsStore = useClipStore();
  const timelines = useClipStore((s) => s.timelines);
  const getClipsForGroup = useClipStore((s) => s.getClipsForGroup);
  const getClipsByType = useClipStore((s) => s.getClipsByType);
  const getClipPositionScore = useClipStore((s) => s.getClipPositionScore);
  const fps = useControlsStore((s) => s.fps);
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [exportKind, setExportKind] = useState<
    "audio" | "image" | "video" | null
  >(null);
  const [exportClipId, setExportClipId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<number | null>(null);

  const getDefaultNameForClip = (
    clip: AnyClipProps | null | undefined,
  ): string => {
    if (!clip) return "export";
    try {
      const asset = getAssetById((clip as any)?.assetId);
      if (!asset) return "export";
      const parts = asset.path.split(/[\\/]/);
      const last = parts[parts.length - 1] || "";
      const stem = last.replace(/\.[^.]+$/, "");
      return stem || "export";
    } catch {
      return "export";
    }
  };

  const shouldShowMenu = open && !exportModalOpen;

  useEffect(() => {
    if (!shouldShowMenu) return;
    const onAny = (e: Event) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      )
        closeMenu();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeMenu();
    };
    document.addEventListener("pointerdown", onAny, true);
    document.addEventListener("mousedown", onAny, true);
    document.addEventListener("click", onAny, true);
    document.addEventListener("contextmenu", onAny, true);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onAny, true);
      document.removeEventListener("mousedown", onAny, true);
      document.removeEventListener("click", onAny, true);
      document.removeEventListener("contextmenu", onAny, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [shouldShowMenu, closeMenu]);

  // Clamp to viewport bounds after mount
  useEffect(() => {
    if (!shouldShowMenu) return;
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    let nx = position.x,
      ny = position.y;
    const BOTTOM_MARGIN = 10; // ensure some space from bottom edge
    if (rect.right > window.innerWidth) nx -= rect.right - window.innerWidth;
    if (rect.bottom > window.innerHeight - BOTTOM_MARGIN)
      ny -= rect.bottom - (window.innerHeight - BOTTOM_MARGIN);
    if (rect.left < 0) nx += -rect.left;
    if (rect.top < 0) ny += -rect.top;
    if (nx !== position.x || ny !== position.y) setPosition({ x: nx, y: ny });
  }, [shouldShowMenu, position.x, position.y, setPosition]);

  const handleExport = useCallback(
    async (settings: ClipExportSettings) => {
      if (!exportClipId || !exportKind) return;
      let clip = {...(clipsStore.getClipById(exportClipId) as
        | AnyClipProps
        | undefined)} as AnyClipProps;
      if (!clip) return;

      let likeImage = clip.type === "image"
      let likeVideo = clip.type === "video"

      if (clip.type === "model") {
        let asset = getAssetById(clip.assetId as string);
        if (asset) {
          likeImage = asset.type === "image";
          likeVideo = asset.type === "video";
        }
        if (clip.originalTransform) {
          clip.originalTransform.x = 0;
          clip.originalTransform.y = 0;
        }
        if (clip.transform) {
          clip.transform.x = 0;
          clip.transform.y = 0;
        }
      }


      const basePath = settings.path.trim();
      const baseName = settings.name.trim();
      if (!basePath || !baseName) return;

      let extension = "";
      if (settings.kind === "audio" && settings.audioFormat) {
        extension = settings.audioFormat;
      } else if (settings.kind === "image" && settings.imageFormat) {
        extension =
          settings.imageFormat === "jpeg" ? "jpg" : settings.imageFormat;
      } else if (settings.kind === "video" && settings.videoFormat) {
        extension = settings.videoFormat;
      }
      const outpath = `${basePath}/${baseName}${extension ? `.${extension}` : ""}`;

      try {
        setExporting(true);
        setExportProgress(0);

        if (settings.kind === "audio") {
          const audioClip = { ...(clip as any) };
          await exportSequence({
            mode: "audio",
            clips: [audioClip as any],
            fps,
            filename: outpath,
            audioOptions: {
              format: settings.audioFormat ?? "mp3",
            },
            onProgress: ({ ratio }) => {
              setExportProgress(typeof ratio === "number" ? ratio : 0);
            },
          });
        } else if (settings.kind === "video") {
          const baseH = settings.resolution ?? 1080;
          // Derive aspect ratio from the clip's native media dimensions when possible.
          let nativeW = 0;
          let nativeH = 0;
          if (likeImage) {
            const info = getMediaInfoCached(
              (clip as ImageClipProps).assetId,
            );
            nativeW = info?.image?.width ?? 0;
            nativeH = info?.image?.height ?? 0;
          } else if (likeVideo) {
            const info = getMediaInfoCached(
              (clip as VideoClipProps).assetId,
            );
            nativeW = info?.video?.displayWidth ?? 0;
            nativeH = info?.video?.displayHeight ?? 0;
          }

          const fallbackAspect =
            aspectRatio && aspectRatio.height > 0
              ? aspectRatio.width / aspectRatio.height
              : 16 / 9;
          const aspect =
            nativeW > 0 && nativeH > 0 ? nativeW / nativeH : fallbackAspect;

          const targetH = baseH;
          const targetW = Math.max(1, Math.round(targetH * aspect));

          const prepared = prepareExportClipsForValue(
            clip as AnyClipProps,
            {
              aspectRatio,
              getAssetById,
              getClipsForGroup,
              getClipsByType,
              getClipPositionScore,
              timelines,
            },
            {
              clearMasks: false,
              applyCentering: true,
              useOriginalTransform: true,
              dimensionsFrom: "clip",
              target: { width: targetW, height: targetH },
            },
          );

          const { exportClips } = prepared;


          const videoEncoderOptions: any = {
            format: settings.videoFormat ?? "mp4",
            codec: (settings.videoFormat === "webm" ? "vp9" : "h264") as any,
            bitrate: "8000k",
            resolution: { width: targetW, height: targetH },
          };

          if (exportClips.length === 1) {
            await exportClip({
              mode: "video",
              clip: exportClips[0],
              fps,
              width: targetW,
              height: targetH,
              encoderOptions: videoEncoderOptions,
              backgroundColor: "#000000",
              filename: outpath,
              onProgress: ({ ratio }) => {
                setExportProgress(typeof ratio === "number" ? ratio : 0);
              },
            });
          } else {
            await exportSequence({
              mode: "video",
              clips: exportClips,
              fps,
              width: targetW,
              height: targetH,
              encoderOptions: videoEncoderOptions,
              backgroundColor: "#000000",
              filename: outpath,
              onProgress: ({ ratio }) => {
                setExportProgress(typeof ratio === "number" ? ratio : 0);
              },
            });
          }
        } else if (settings.kind === "image") {
          const baseH = settings.resolution ?? 1080;
          // Derive aspect ratio from the clip's native media dimensions when possible.
          let nativeW = 0;
          let nativeH = 0;
          if (likeImage) {
            const info = getMediaInfoCached(
              (clip as ImageClipProps).assetId,
            );
            nativeW = info?.image?.width ?? 0;
            nativeH = info?.image?.height ?? 0;
          } else if (likeVideo) {
            const info = getMediaInfoCached(
              (clip as VideoClipProps).assetId,
            );
            nativeW = info?.video?.displayWidth ?? 0;
            nativeH = info?.video?.displayHeight ?? 0;
          }

          const fallbackAspect =
            aspectRatio && aspectRatio.height > 0
              ? aspectRatio.width / aspectRatio.height
              : 16 / 9;
          const aspect =
            nativeW > 0 && nativeH > 0 ? nativeW / nativeH : fallbackAspect;

          const targetH = baseH;
          const targetW = Math.max(1, Math.round(targetH * aspect));

 
          const prepared = prepareExportClipsForValue(
            clip as AnyClipProps,
            {
              aspectRatio,
              getAssetById,
              getClipsForGroup,
              getClipsByType,
              getClipPositionScore,
              timelines,
            },
            {
              clearMasks: false,
              applyCentering: false,
              useOriginalTransform: true,
              dimensionsFrom: "clip",
              target: { width: targetW, height: targetH },
            },
          );

          const { exportClips } = prepared;
          const frame =
            likeVideo || clip.type === "group"
              ? ((clip as any).selectedFrame ?? 0)
              : 0;

       
          if (exportClips.length === 1) {
            const result = await exportClip({
              mode: "image",
              width: targetW,
              height: targetH,
              imageFrame: frame,
              clip: exportClips[0],
              backgroundColor: "#000000",
              encoderOptions: {
                resolution: { width: targetW, height: targetH },
              },
              fps,
              onProgress: ({ ratio }) => {
                setExportProgress(typeof ratio === "number" ? ratio : 0);
              },
            });


            if (result instanceof Blob) {
              const buf = new Uint8Array(await result.arrayBuffer());
              await saveImageToPath(buf, outpath);
            }
          } else {
            const result = await exportSequence({
              mode: "image",
              width: targetW,
              height: targetH,
              imageFrame: frame,
              clips: exportClips,
              fps,
              onProgress: ({ ratio }) => {
                setExportProgress(typeof ratio === "number" ? ratio : 0);
              },
            });

            if (result instanceof Blob) {
              const buf = new Uint8Array(await result.arrayBuffer());
              await saveImageToPath(buf, outpath);
            }
          }
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      } finally {
        setExporting(false);
        setExportProgress(null);
        setExportModalOpen(false);
      }
    },
    [
      exportClipId,
      exportKind,
      clipsStore,
      fps,
      aspectRatio,
      getClipsForGroup,
      getClipsByType,
      getClipPositionScore,
      timelines,
      setExportProgress,
    ],
  );

  const onSelect = (action: string) => {
    if (target?.type === "textSelection") {
      if (action === "copy") {
        try {
          // Prefer the browser's native selection copy.
          const ok = document.execCommand?.("copy");
          if (ok) {
            closeMenu();
            return;
          }
        } catch {
          // fall through
        }
        try {
          const sel = window.getSelection?.();
          const text = sel?.toString?.() ?? "";
          if (text.trim()) navigator.clipboard?.writeText?.(text);
        } catch {
          // swallow
        }
      }
      closeMenu();
      return;
    }

    if (target?.type === "clip") {
      const ids = target.clipIds;
      if (action === "copy") clipsStore.copyClips(ids);
      else if (action === "cut") {
        clipsStore.cutClips(ids);
        useControlsStore.getState().clearSelection();
      } else if (action === "paste")
        clipsStore.pasteClips(useControlsStore.getState().focusFrame);
      else if (action === "delete") {
        ids.forEach((id) => clipsStore.removeClip(id));
        useControlsStore.getState().clearSelection();
      } else if (action === "split")
        clipsStore.splitClip(
          useControlsStore.getState().focusFrame,
          target.primaryClipId,
        );
      else if (action === "separateAudio" && target.isVideo)
        clipsStore.separateClip(target.primaryClipId);
      else if (action === "export") {
        try {
          const primaryId = target.primaryClipId;
          const clip = clipsStore.getClipById(primaryId) as
            | AnyClipProps
            | undefined;
          if (!clip) return;
          let kind: "audio" | "image" | "video" | null = null;
          if (clip.type === "audio") kind = "audio";
          else if (clip.type === "image") kind = "image";
          else if (clip.type === "video") kind = "video";
          else if (clip.type === "model") {
            let asset = getAssetById(clip.assetId as string);
            if (asset) {
              kind = asset.type === "video" ? "video" : "image";
            }
          }
          if (!kind) return;
          setExportClipId(primaryId);
          setExportKind(kind);
          setExportModalOpen(true);
        } catch {
          // swallow
        }
      } else if (action === "group") {
        clipsStore.groupClips(ids);
      } else if (action === "ungroup") {
        clipsStore.ungroupClips(target.primaryClipId);
      } else if (action === "convertToMedia") {
        clipsStore.convertToMedia(target.primaryClipId);
      }
    } else if (target?.type === "timeline") {
      if (action === "paste") {
        const frame = useControlsStore.getState().focusFrame;
        clipsStore.pasteClips(frame, target.timelineId);
        // Attempt to fix any overlaps by resolving again (pasteClips already resolves overlaps globally)
        // If needed, we could snap pasted clips to nearest valid gaps on target timeline here.
      }
    }
    closeMenu();
  };

  const content =
    groups && groups.length > 0 ? (
      <div className="p-1">
        {groups.map(
          (group, gi) =>
            group.items.length > 0 && (
              <div key={group.id}>
                {group.label && (
                  <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-brand-light/50">
                    {group.label}
                  </div>
                )}
                {group.items.map((item) => (
                  <button
                    key={item.id}
                    disabled={item.disabled}
                    onClick={() => onSelect(item.action)}
                    className={cn(
                      "w-full px-2.5 py-1.5 text-left text-[11.5px] flex items-center rounded justify-between hover:bg-brand-light/10 text-brand-light",
                      item.disabled &&
                        "opacity-50 cursor-default! hover:bg-brand",
                    )}
                  >
                    <span>{item.label}</span>
                    {item.shortcut && <Key text={item.shortcut} />}
                  </button>
                ))}
                {gi < groups.length - 1 && (
                  <div className="my-1 h-px bg-brand-light/10 -mx-1" />
                )}
              </div>
            ),
        )}
      </div>
    ) : (
      <div className="py-1">
        {items.map((item) => (
          <button
            key={item.id}
            disabled={item.disabled}
            onClick={() => onSelect(item.action)}
            className="w-full px-2.5 py-1.5 text-left text-[11px] flex items-center justify-between hover:bg-brand-light/10 disabled:opacity-50 text-brand-light"
          >
            <span>{item.label}</span>
            {item.shortcut && <Key text={item.shortcut} />}
          </button>
        ))}
      </div>
    );

  return (
    <>
      {shouldShowMenu &&
        createPortal(
          <div
            ref={containerRef}
            style={{
              position: "fixed",
              left: position.x,
              top: position.y,
              zIndex: 100,
            }}
            className="w-52 font-poppins select-none rounded-md border border-brand-light/10 bg-brand-background-light shadow-lg"
          >
            {content}
          </div>,
          document.body,
        )}
      {exportKind && (
        <ClipExportModal
          open={exportModalOpen}
          kind={exportKind}
          defaultName={getDefaultNameForClip(
            exportClipId
              ? (clipsStore.getClipById(exportClipId) as
                  | AnyClipProps
                  | undefined)
              : undefined,
          )}
          onOpenChange={setExportModalOpen}
          onExport={handleExport}
          isExporting={exporting}
          exportProgress={exportProgress}
        />
      )}
    </>
  );
};

export default GlobalContextMenu;
