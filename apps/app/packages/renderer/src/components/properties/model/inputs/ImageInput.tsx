import React, { useEffect, useMemo, useRef, useState } from "react";
import { useDndMonitor } from "@dnd-kit/core";
import Droppable from "@/components/dnd/Droppable";
import { IoImageOutline } from "react-icons/io5";
import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@radix-ui/react-tabs";
import { LuSearch, LuUpload } from "react-icons/lu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { MediaItem, MediaThumb } from "@/components/media/Item";
import { getMediaInfo } from "@/lib/media/utils";
import { listConvertedMedia } from "@app/preload";
import { ScrollArea } from "@/components/ui/scroll-area";
import TimelineSearch from "./timeline/TimelineSearch";
import { useClipStore } from "@/lib/clip";
import { useAssetControlsStore } from "@/lib/assetControl";
import { useProjectsStore } from "@/lib/projects";
import TimelineClipPosterPreview from "./TimelineClipPosterPreview";
import {
  AnyClipProps,
  clipSignature,
  ClipTransform,
  ImageClipProps,
  VideoClipProps,
} from "@/lib/types";
import { VIDEO_EXTS, IMAGE_EXTS } from "@/lib/settings";
import {
  getLowercaseExtension,
  pickMediaPaths,
  importMediaPaths,
  getPathForFile,
} from "@app/preload";
import { useViewportStore } from "@/lib/viewport";
import {
  useMediaLibraryVersion,
  bumpMediaLibraryVersion,
} from "@/lib/media/library";
import { useInputControlsStore } from "@/lib/inputControl";
import { usePreprocessorsListQuery } from "@/lib/preprocessor/queries";
import { useControlsStore } from "@/lib/control";
import { TbEdit, TbPhoto } from "react-icons/tb";
import { MediaDialog } from "@/components/dialogs/MediaDialog";

export type ImageSelection = AnyClipProps | null;

interface ImageInputProps {
  label?: string;
  description?: string;
  height?: number;
  inputId: string;
  value: ImageSelection;
  onChange: (value: ImageSelection) => void;
  clipId: string;
  panelSize: number;
  preprocessorRef?: string;
  preprocessorName?: string;
  applyPreprocessorInitial?: boolean;
  onChangeComposite?: (value: {
    selection: ImageSelection;
    preprocessor_ref?: string;
    preprocessor_name?: string;
    apply_preprocessor?: boolean;
  }) => void;
}

interface PopoverImageProps {
  value: ImageSelection;
  onChange: (value: ImageSelection) => void;
  clipId: string | null;
  onSelectComplete?: () => void;
}

const PopoverImage: React.FC<PopoverImageProps> = ({
  value,
  onChange,
  clipId,
  onSelectComplete,
}) => {
  const isUserInteractingRef = useRef(false);
  const [selectedTab, setSelectedTab] = useState<"timeline" | "library">(
    "library",
  );
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredMediaItems, setFilteredMediaItems] = useState<MediaItem[]>([]);
  const { fps } = useControlsStore();
  const addAsset = useClipStore((s) => s.addAsset);
  // media cache not needed for list sync here; we subscribe to library version instead
  const { clips } = useClipStore();
  const getClipById = useClipStore((s) => s.getClipById);
  const getActiveProject = useProjectsStore((s) => s.getActiveProject);
  const clearSelectedAsset = useAssetControlsStore((s) => s.clearSelectedAsset);
  const setSelectedAssetChangeHandler = useAssetControlsStore(
    (s) => s.setSelectedAssetChangeHandler,
  );
  useEffect(() => {
    setFilteredMediaItems(
      mediaItems.filter((media) =>
        media.name.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
    );
  }, [searchQuery, mediaItems]);

  const mediaLibraryVersion = useMediaLibraryVersion();

  useEffect(() => {
    (async () => {
      const list = await listConvertedMedia(getActiveProject()?.folderUuid);
      const infoPromises = list.map((it) => getMediaInfo(it.assetUrl));
      const infos = await Promise.all(infoPromises);
      let results: MediaItem[] = list.map((it, idx) => ({
        name: it.name,
        type: it.type,
        absPath: it.absPath,
        assetUrl: it.assetUrl,
        dateAddedMs: it.dateAddedMs,
        mediaInfo: infos[idx],
        hasProxy: it.hasProxy,
      }));
      results = results
        .sort((a, b) =>
          a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
        )
        .filter((media) => media.type === "image" || media.type === "video");
      setMediaItems(results);
    })();
  }, [mediaLibraryVersion]);

  const handleUpload = async () => {
    try {
      const filters = [
        {
          name: "Image/Video Files",
          extensions: VIDEO_EXTS.concat(IMAGE_EXTS),
        },
        { name: "Video Files", extensions: VIDEO_EXTS },
        { name: "Image Files", extensions: IMAGE_EXTS },
      ];
      const picked = await pickMediaPaths({
        directory: false,
        filters,
        title: "Choose image/video file(s) to import",
      });
      const paths = (picked ?? []).filter((p) => {
        const ext = getLowercaseExtension(p);
        return VIDEO_EXTS.includes(ext) || IMAGE_EXTS.includes(ext);
      });
      if (paths.length === 0) return;
      const existingNames = new Set(mediaItems.map((it) => it.name));
      await importMediaPaths(paths, undefined, getActiveProject()?.folderUuid);
      const list = await listConvertedMedia(getActiveProject()?.folderUuid);
      const infoPromises = list.map((it) => getMediaInfo(it.assetUrl));
      const infos = await Promise.all(infoPromises);
      let results: MediaItem[] = list.map((it, idx) => ({
        name: it.name,
        type: it.type,
        absPath: it.absPath,
        assetUrl: it.assetUrl,
        dateAddedMs: it.dateAddedMs,
        mediaInfo: infos[idx],
        hasProxy: it.hasProxy,
      }));

      results = results
        .sort((a, b) =>
          a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
        )
        .filter((media) => media.type === "image" || media.type === "video");
      setMediaItems(results);
      const newlyAdded = results.filter((it) => !existingNames.has(it.name));
      if (newlyAdded.length > 0) {
        const first = newlyAdded[0];
        clearSelectedAsset();
        const ext = getLowercaseExtension(first.assetUrl);
        const isVideo = VIDEO_EXTS.includes(ext);
        if (isVideo) {
          const mw =
            first.mediaInfo?.video?.displayWidth ??
            (first.mediaInfo as any)?.video?.width ??
            0;
          const mh =
            first.mediaInfo?.video?.displayHeight ??
            (first.mediaInfo as any)?.video?.height ??
            0;
          const mar = mw && mh ? mw / mh : undefined;
          const duration = Math.max(
            1,
            Math.floor((first.mediaInfo?.duration || 0) * fps),
          );
          const asset = addAsset({ path: first.assetUrl, modelInputAsset: true });
          const clip: VideoClipProps = {
            type: "video",
            clipId: `media:${first.assetUrl}`,
            assetId: asset.id,
            startFrame: 0,
            endFrame: duration,
            mediaWidth: mw || undefined,
            mediaHeight: mh || undefined,
            mediaAspectRatio:
              typeof mar === "number" && isFinite(mar) && mar > 0
                ? mar
                : undefined,
            preprocessors: [],
            masks: [],
          } as any;
          onChange(clip as AnyClipProps);
          onSelectComplete?.();
        } else {
          const iw = first.mediaInfo?.image?.width ?? 0;
          const ih = first.mediaInfo?.image?.height ?? 0;
          const iar = iw && ih ? iw / ih : undefined;
          const asset = addAsset({
            path: first.assetUrl,
            modelInputAsset: true,
          });
          const clip: ImageClipProps = {
            type: "image",
            clipId: `media:${first.assetUrl}`,
            assetId: asset.id,
            startFrame: 0,
            endFrame: 1,
            mediaWidth: iw || undefined,
            mediaHeight: ih || undefined,
            mediaAspectRatio:
              typeof iar === "number" && isFinite(iar) && iar > 0
                ? iar
                : undefined,
            preprocessors: [],
            masks: [],
          } as any;
          onChange(clip as AnyClipProps);
          onSelectComplete?.();
        }
      }
      bumpMediaLibraryVersion();
    } catch (e) {
      // swallow errors here; MediaMenu handles toasts, but we keep silent in this compact picker
    }
  };

  const numEligibleTimelineAssets = useMemo(() => {
    return clips.filter(
      (clip) =>
        clip.type !== "filter" &&
        clip.type !== "audio" &&
        clip.clipId !== clipId,
    ).length;
  }, [clips]);

  // Direct change handler so timeline selection/deselection can update this input
  const assetSelectionHandler = React.useCallback(
    (clipId: string | null) => {
      if (!clipId) {
        onChange(null);
        onSelectComplete?.();
        return;
      }
      const selectedClip = getClipById(clipId) as AnyClipProps | undefined;
      if (!selectedClip || selectedClip.type === "audio") return;
      const duration = Math.max(
        1,
        (selectedClip.endFrame ?? 0) - (selectedClip.startFrame ?? 0),
      );
      const projected = {
        ...selectedClip,
        startFrame: 0,
        endFrame: duration,
      } as AnyClipProps;
      onChange(projected);
      onSelectComplete?.();
    },
    [getClipById, onChange, onSelectComplete],
  );

  return (
    <PopoverContent
      side="left"
      align="start"
      sideOffset={20}
      className={cn(
        "p-2 z-90 dark h-full flex flex-col gap-y-3 border border-brand-light/10 rounded-[7px] font-poppins transition-all duration-150",
        selectedTab === "timeline" ? "w-[600px]" : "w-96",
      )}
      onOpenAutoFocus={() => {
        isUserInteractingRef.current = true;
        // TimelineSearch selection highlighting also uses a global asset-controls store
        // (see TimelineClip), so we must clear it when opening a new picker to avoid
        // leaking "Selected" state across different ImageInput slots.
        clearSelectedAsset();
        setSelectedAssetChangeHandler(assetSelectionHandler);
      }}
      onCloseAutoFocus={() => {
        isUserInteractingRef.current = false;
        setSelectedAssetChangeHandler(null);
      }}
    >
      <Tabs
        className=""
        value={selectedTab}
        onValueChange={(value) =>
          setSelectedTab(value as "timeline" | "library")
        }
      >
        <div
          className={cn(
            "w-full flex flex-row items-center gap-x-2 justify-between",
          )}
        >
          <TabsList
            className={cn(
              "w-full  text-brand-light text-[10.5px] rounded font-medium text-start flex flex-row shadow overflow-hidden",
              numEligibleTimelineAssets === 0
                ? "justify-start"
                : "justify-between  cursor-pointer  bg-brand-background-light",
            )}
          >
            <TabsTrigger
              value="library"
              className={cn(
                " w-full py-1.5 flex items-center",
                selectedTab === "library" && numEligibleTimelineAssets > 0
                  ? "bg-brand-accent-shade"
                  : "",
                numEligibleTimelineAssets === 0
                  ? "cursor-default justify-start px-2.5"
                  : "cursor-pointer   justify-center px-4",
              )}
            >
              Media Library
            </TabsTrigger>
            <TabsTrigger
              hidden={numEligibleTimelineAssets === 0}
              value="timeline"
              className={cn(
                "px-4 w-full py-1.5 cursor-pointer flex items-center justify-center",
                selectedTab === "timeline" ? "bg-brand-accent-shade" : "",
              )}
            >
              Timeline Assets
            </TabsTrigger>
          </TabsList>
          <button
            onClick={handleUpload}
            className={cn(
              "w-fit h-full mr-2 flex flex-row items-center justify-center gap-x-1.5 bg-brand-background-light hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded py-1.5",
              numEligibleTimelineAssets === 0 ? "px-5" : "px-3",
            )}
          >
            <LuUpload className="w-3.5 h-3.5 text-brand-light" />
            <span className="text-brand-light text-[10.5px] font-medium">
              Upload
            </span>
          </button>
        </div>
        <TabsContent
          value="library"
          className="w-full h-full flex flex-col py-2 gap-y-2 outline-none"
        >
          <div className="w-full flex flex-row items-center justify-between gap-x-2">
            <span className="relative w-full">
              <LuSearch className="w-3.5 h-3.5 text-brand-light/50 absolute left-2 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search for media"
                className="w-full h-full pl-8 text-brand-light text-[10.5px] font-normal bg-brand rounded-[7px] border border-brand-light/10 p-2 outline-none"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </span>
          </div>
          <ScrollArea className="w-full h-96">
            <div className="w-full h-full grid grid-cols-2 gap-3">
              {filteredMediaItems.map((media) => (
                <div
                  key={media.name}
                  onClick={() => {
                    // Selecting a library item should clear timeline selection
                    clearSelectedAsset();
                    const targetClipId = `media:${media.assetUrl}`;
                    const isSame = value && value.clipId === targetClipId;
                    if (isSame) {
                      onChange(null);
                      onSelectComplete?.();
                      return;
                    }
                    const ext = getLowercaseExtension(media.assetUrl);
                    const isVideo = VIDEO_EXTS.includes(ext);
                    if (isVideo) {
                      const duration = Math.max(
                        1,
                        Math.floor((media.mediaInfo?.duration || 0) * fps),
                      );
                      const mw =
                        media.mediaInfo?.video?.displayWidth ??
                        (media.mediaInfo as any)?.video?.width ??
                        0;
                      const mh =
                        media.mediaInfo?.video?.displayHeight ??
                        (media.mediaInfo as any)?.video?.height ??
                        0;
                      const mar = mw && mh ? mw / mh : undefined;
                      const asset = addAsset({
                        path: media.assetUrl,
                        modelInputAsset: true,
                      });
                      const clip: VideoClipProps = {
                        type: "video",
                        clipId: targetClipId,
                        assetId: asset.id,
                        startFrame: 0,
                        endFrame: duration,
                        mediaWidth: mw || undefined,
                        mediaHeight: mh || undefined,
                        mediaAspectRatio:
                          typeof mar === "number" && isFinite(mar) && mar > 0
                            ? mar
                            : undefined,
                        preprocessors: [],
                        masks: [],
                      } as any;
                      onChange(clip as AnyClipProps);
                      onSelectComplete?.();
                    } else {
                      const iw = media.mediaInfo?.image?.width ?? 0;
                      const ih = media.mediaInfo?.image?.height ?? 0;
                      const iar = iw && ih ? iw / ih : undefined;
                      const asset = addAsset({
                        path: media.assetUrl,
                        modelInputAsset: true,
                      });
                      const clip: ImageClipProps = {
                        type: "image",
                        clipId: targetClipId,
                        assetId: asset.id,
                        startFrame: 0,
                        endFrame: 1,
                        mediaWidth: iw || undefined,
                        mediaHeight: ih || undefined,
                        mediaAspectRatio:
                          typeof iar === "number" && isFinite(iar) && iar > 0
                            ? iar
                            : undefined,
                        preprocessors: [],
                        masks: [],
                      } as any;
                      onChange(clip as AnyClipProps);
                      onSelectComplete?.();
                    }
                  }}
                  className={cn(
                    "w-full flex flex-col items-center justify-center gap-y-1.5 cursor-pointer group relative",
                  )}
                >
                  <div className="relative">
                    <div
                      className={cn(
                        "absolute top-0 left-0 w-full h-full bg-brand-background-light/50 backdrop-blur-sm rounded-md z-20 group-hover:opacity-100 transition-all duration-200 flex items-center justify-center",
                        value && value.clipId === `media:${media.assetUrl}`
                          ? "opacity-100"
                          : "opacity-0",
                      )}
                    >
                      <div
                        className={cn(
                          "rounded-full py-1 px-3  bg-brand-light/10 flex items-center justify-center font-medium text-[10.5px] w-fit",
                          value && value.clipId === `media:${media.assetUrl}`
                            ? "bg-brand-light/20"
                            : "",
                        )}
                      >
                        {value && value.clipId === `media:${media.assetUrl}`
                          ? "Selected"
                          : "Use as Input"}
                      </div>
                    </div>
                    <MediaThumb key={media.name} item={media} />
                  </div>
                  <div className="text-brand-light/90 text-[9.5px] text-start truncate w-full text-ellipsis overflow-hidden group-hover:text-brand-light transition-all duration-200">
                    {media.name}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="timeline" className="outline-none">
          <TimelineSearch
            isAssetSelected={(clipId) => {
              if (!value) return false;
              return clipId === value.clipId;
            }}
            types={["image", "video", "group", "text", "shape", "draw"]}
            excludeClipId={clipId}
          />
        </TabsContent>
      </Tabs>
    </PopoverContent>
  );
};

const ImageInput: React.FC<ImageInputProps> = ({
  label,
  description,
  height,
  inputId,
  value,
  onChange,
  clipId,
  panelSize,
  preprocessorRef,
  preprocessorName,
  applyPreprocessorInitial,
  onChangeComposite,
}) => {
  // Each ImageInput instance must have a unique droppable id; otherwise DnD-kit
  // can't distinguish which slot is being hovered/dropped on in ImageInputList.
  const droppableId = useMemo(() => `image-input:${inputId}`, [inputId]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageContainerRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState<{ w: number; h: number }>({
    w: 0,
    h: 0,
  });
  const [mediaClip, setMediaClip] = useState<AnyClipProps | null>(null);
  const [isOverDropZone, setIsOverDropZone] = useState(false);
  const externalDragCounterRef = useRef(0);
  const getClipById = useClipStore((s) => s.getClipById);
  const getPreprocessorsForClip = useClipStore(
    (s) => s.getPreprocessorsForClip,
  );
  const updateModelInput = useClipStore((s) => s.updateModelInput);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const { data: preprocessors = [] } = usePreprocessorsListQuery({
    enabled: !!preprocessorRef,
  });
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const { focusFrameByInputId } = useInputControlsStore();
  const focusFrameForInput = focusFrameByInputId[inputId] ?? 0;
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const addAsset = useClipStore((s) => s.addAsset);

  const isClipOnTimeline = useMemo(() => {
    if (!value) return false;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return false;
    return true;
  }, [value, getClipById]);

  const resolvedPreprocessorName = useMemo(() => {
    if (!preprocessorRef) return preprocessorName;
    const found = (preprocessors || []).find((p) => p.id === preprocessorRef);
    return found?.name || preprocessorName || preprocessorRef;
  }, [preprocessors, preprocessorRef, preprocessorName]);
  // Preprocessor toggle state and defaulting
  const [applyPreprocessor, setApplyPreprocessor] = useState<boolean>(true);
  const { fps } = useControlsStore();

  useEffect(() => {
    if (typeof applyPreprocessorInitial === "boolean") {
      setApplyPreprocessor(applyPreprocessorInitial);
      return;
    }
    if (!preprocessorRef) {
      setApplyPreprocessor(true);
      return;
    }
    const sel = value as AnyClipProps | null;
    if (sel) {
      const clip = sel;
      if (clip && (clip.type === "video" || clip.type === "image")) {
        const exists = (getPreprocessorsForClip(clip.clipId) || []).some(
          (p) => (p.preprocessor?.id || "") === preprocessorRef,
        );
        setApplyPreprocessor(!exists);
        return;
      }
    }
    setApplyPreprocessor(true);
  }, [
    value,
    preprocessorRef,
    applyPreprocessorInitial,
    getClipById,
    getPreprocessorsForClip,
  ]);

  const emitSelection = React.useCallback(
    (next: ImageSelection) => {
      // If this input is being cleared, also clear asset-mode selection so an empty
      // input doesn't keep showing a stale "Selected" state in TimelineSearch.
      if (!next) {
        try {
          useAssetControlsStore.getState().clearSelectedAsset();
        } catch {}
      }
      if (onChangeComposite) {
        onChangeComposite({
          selection: next,
          preprocessor_ref: preprocessorRef,
          preprocessor_name: resolvedPreprocessorName,
          apply_preprocessor: applyPreprocessor,
        });
      } else {
        onChange(next);
      }
      // Apply mapped height/width if configured on this input
      (async () => {
        try {
          if (!next) return;
          const clip: any = getClipById(clipId) as any;
          const manifest = clip?.manifest;
          const ui = manifest?.spec?.ui || manifest?.ui;
          const inputsArr: any[] = Array.isArray(ui?.inputs) ? ui.inputs : [];
          if (!inputsArr.length) return;
          const self =
            inputsArr.find((inp: any) => String(inp?.id) === String(inputId)) ||
            {};
          const mapHId: string | undefined = self?.map_h;
          const mapWId: string | undefined = self?.map_w;
          if (!mapHId && !mapWId) return;
          const scaleById: string | undefined = self?.scale_by;
          let targetShortSide: number | undefined;
          if (scaleById) {
            const scaleInp = inputsArr.find(
              (inp: any) => String(inp?.id) === String(scaleById),
            );
            if (scaleInp) {
              const type = String(scaleInp?.type || "");
              const valStr = String(scaleInp?.value ?? scaleInp?.default ?? "");
              if (type === "number" || type === "number+slider") {
                const n = Number(valStr);
                if (Number.isFinite(n) && n > 0)
                  targetShortSide = Math.floor(n);
              } else if (type === "select") {
                const opts: any[] = Array.isArray(scaleInp?.options)
                  ? scaleInp.options
                  : [];
                const selected = opts.find(
                  (o: any) => String(o?.value) === valStr,
                );
                const resCandidate = Number(selected?.value ?? NaN);
                if (Number.isFinite(resCandidate) && resCandidate > 0)
                  targetShortSide = Math.floor(resCandidate);
              } else {
                const n = Number(valStr);
                if (Number.isFinite(n) && n > 0)
                  targetShortSide = Math.floor(n);
              }
            }
          }
          // Determine intrinsic media dimensions (fallback to mediaInfo if missing)
          let src: string | undefined = getAssetById((next as ImageClipProps | VideoClipProps).assetId)?.path;
          let mw = Number((next as ImageClipProps | VideoClipProps)?.mediaWidth ?? 0);
          let mh = Number((next as ImageClipProps | VideoClipProps)?.mediaHeight ?? 0);
          if ((!mw || !mh) && src) {
            try {
              const info = await getMediaInfo(src);
              const wCandidate =
                info?.video?.displayWidth ??
                info?.image?.width ??
                (Number.isFinite(mw) ? mw : undefined);
              const hCandidate =
                info?.video?.displayHeight ??
                info?.image?.height ??
                (Number.isFinite(mh) ? mh : undefined);
              mw = Number(wCandidate ?? 0);
              mh = Number(hCandidate ?? 0);
            } catch {}
          }
          if (!mw || !mh) return;

          let outW = mw;
          let outH = mh;

          if (
            Number.isFinite(targetShortSide as number) &&
            (targetShortSide as number) > 0
          ) {
            const res = targetShortSide as number;
            // Treat the scale-by value (usually resolution) as the target
            // for the *shorter* side, and scale the longer side to match AR.
            if (mw >= mh) {
              // Landscape: height is shorter
              const scale = res / mh;
              outH = res;
              outW = Math.floor(Math.max(1, mw * scale));
            } else {
              // Portrait: width is shorter
              const scale = res / mw;
              outW = res;
              outH = Math.floor(Math.max(1, mh * scale));
            }
          }

          // Prevent ModelInputsPanel resolution/aspect sync from overriding mapped values
          if (mapHId === "height" || mapWId === "width") {
            updateModelInput(clipId, "resolution", { value: "custom" } as any);
            updateModelInput(clipId, "aspect_ratio", {
              value: "custom",
            } as any);
          }
          if (mapHId)
            updateModelInput(clipId, mapHId, { value: String(outH) } as any);
          if (mapWId)
            updateModelInput(clipId, mapWId, { value: String(outW) } as any);
        } catch {}
      })();
    },
    [
      onChangeComposite,
      onChange,
      preprocessorRef,
      resolvedPreprocessorName,
      applyPreprocessor,
      getClipById,
      clipId,
      inputId,
      updateModelInput,
    ],
  );

  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const getActiveProject = useProjectsStore((s) => s.getActiveProject);
  const { getTimelineDuration, getFocusFrame } = useInputControlsStore();

  const setInputTimelineDuration = useInputControlsStore(
    (s) => s.setTimelineDuration,
  );
  const setInputFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
  const selectionKey = useMemo(() => {
    if (!value) return "null";
    return value.clipId;
  }, [value]);
  const lastSelectionKeyRef = useRef<string | null>(null);
  const viewportRatio = useMemo(() => {
    const r = aspectRatio.width / aspectRatio.height;
    return Number.isFinite(r) && r > 0 ? r : 1;
  }, [aspectRatio.width, aspectRatio.height]);
  const ratioCacheByClipIdRef = useRef<Record<string, number>>({});
  const [contentRatio, setContentRatio] = useState<number | null>(null);
  const selectedClipForRatio = useMemo<AnyClipProps | null>(() => {
    if (!value) return null;
    return (mediaClip ?? (value as AnyClipProps)) || null;
  }, [value, mediaClip]);

  const selectedClipRatioSignature = clipSignature(
    selectedClipForRatio as AnyClipProps,
  );
  const divRef = useRef<HTMLDivElement>(null);

  const liveTimelineClip = useClipStore((s) => {
    if (!value) return null;
    if ((value as any)?.disableTimelineSync) return null;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return null;
    return (s.getClipById(cid) as AnyClipProps | undefined) ?? null;
  });

  const handleConfirm = (data: {
    rotation: number;
    aspectRatio: string;
    crop?: { x: number; y: number; width: number; height: number };
    transformWidth?: number;
    transformHeight?: number;
    transformX?: number;
    transformY?: number;
    originalTransform?: ClipTransform;
  }) => {
    if (!mediaClip) return;

    const newTransform: ClipTransform = {
      ...(mediaClip.transform ?? {}),
      ...(data.rotation ? { rotation: data.rotation } : {}),
      ...(data.crop ? { crop: data.crop } : { crop: undefined }),
      ...(data.transformWidth ? { width: data.transformWidth } : {}),
      ...(data.transformHeight ? { height: data.transformHeight } : {}),
      ...(typeof data.transformX === "number" ? { x: data.transformX } : {}),
      ...(typeof data.transformY === "number" ? { y: data.transformY } : {}),
    } as ClipTransform;
    const clipToUse = { ...mediaClip, transform: newTransform } as AnyClipProps;
    if (data.originalTransform) {
      clipToUse.originalTransform = { ...data.originalTransform };
    }
    setMediaClip(clipToUse);
    emitSelection(clipToUse);

    if (liveTimelineClip) {
      setClipTransform(liveTimelineClip.clipId, newTransform);
    }
  };

  const displayRatio = useMemo(() => {
    const clip = selectedClipForRatio;
    if (!clip) return viewportRatio;
    if (clip.type === "group") return viewportRatio;
    if (
      typeof contentRatio === "number" &&
      Number.isFinite(contentRatio) &&
      contentRatio > 0
    )
      return contentRatio;
    return viewportRatio;
  }, [selectedClipRatioSignature, contentRatio, viewportRatio]);

  useEffect(() => {
    const clip = selectedClipForRatio;
    if (!clip) {
      setContentRatio(null);
      return;
    }
    if (clip.type !== "video" && clip.type !== "image") {
      setContentRatio(null);
      return;
    }

    const transform = (clip as any)?.transform as ClipTransform | undefined;
    const crop = transform?.crop;
    const cropKey = crop
      ? `${crop.x.toFixed(4)},${crop.y.toFixed(4)},${crop.width.toFixed(4)},${crop.height.toFixed(4)}`
      : "";
    const cacheKey = cropKey ? `${clip.clipId}|${cropKey}` : clip.clipId;

    const cached = ratioCacheByClipIdRef.current[cacheKey];
    if (typeof cached === "number" && cached > 0) {
      setContentRatio(cached);
      return;
    }

    const asset = getAssetById((clip as ImageClipProps | VideoClipProps).assetId);
    const src = asset?.path;
    if (!src) {
      setContentRatio(null);
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo(src);
        let w = info?.video?.displayWidth ?? info?.image?.width ?? 0;
        let h = info?.video?.displayHeight ?? info?.image?.height ?? 0;

        if (
          crop &&
          crop.width > 0 &&
          crop.height > 0 &&
          Number.isFinite(w) &&
          Number.isFinite(h) &&
          w > 0 &&
          h > 0
        ) {
          w = w * crop.width;
          h = h * crop.height;
        }

        const r = w / h;
        if (!cancelled && Number.isFinite(r) && r > 0) {
          ratioCacheByClipIdRef.current[cacheKey] = r;
          setContentRatio(r);
        }
      } catch {
        if (!cancelled) setContentRatio(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedClipRatioSignature]);
  // Determine stage rendering dynamically in render branch

  // Keep stage size scaled, max height 512px
  useEffect(() => {
    const el = stageContainerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      let w = Math.max(1, panelSize);
      let h = Math.max(1, w / displayRatio);
      if (h > (height ?? 400)) {
        h = height ?? 400;
        w = Math.max(1, h * displayRatio);
      }
      setStageSize({ w, h });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [panelSize, displayRatio, height]);

  // Ensure width updates immediately when panelSize changes, even if height is unchanged
  useEffect(() => {
    let w = Math.max(1, panelSize);
    let h = Math.max(1, w / displayRatio);
    if (h > (height ?? 400)) {
      h = height ?? 400;
      w = Math.max(1, h * displayRatio);
    }
    setStageSize({ w, h });
  }, [panelSize, displayRatio, height]);

  // DnD monitor: track hover-over state and handle drop selection
  useDndMonitor({
    onDragStart: () => {
      setIsOverDropZone(false);
    },
    onDragMove: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id as string | undefined;
      const isValid =
        !!data && (data.type === "image" || data.type === "video");
      setIsOverDropZone(isValid && overId === droppableId);
    },
    onDragCancel: () => {
      setIsOverDropZone(false);
    },
    onDragEnd: (event) => {
      const overId = event.over?.id as string | undefined;
      const data = event.active?.data?.current as MediaItem | undefined;
      const isValid =
        !!data && (data.type === "image" || data.type === "video");
      if (isValid && overId === droppableId) {
        const assetUrl = (data as MediaItem).assetUrl;
        const targetClipId = `media:${assetUrl}`;
        const isSame = value && value.clipId === targetClipId;
        if (isSame) {
          emitSelection(null);
        } else {
          const ext = getLowercaseExtension(assetUrl);
          const isVideo = VIDEO_EXTS.includes(ext);
          if (isVideo) {
            const duration = Math.max(
              1,
              Math.floor((data.mediaInfo?.duration || 0) * fps),
            );
            const mw =
              data.mediaInfo?.video?.displayWidth ??
              (data.mediaInfo as any)?.video?.width ??
              0;
            const mh =
              data.mediaInfo?.video?.displayHeight ??
              (data.mediaInfo as any)?.video?.height ??
              0;
            const mar = mw && mh ? mw / mh : undefined;
            const asset = addAsset({
              path: data.assetUrl,
              modelInputAsset: true,
            });
            const clip: VideoClipProps = {
              type: "video",
              clipId: targetClipId,
              assetId: asset.id,
              startFrame: 0,
              endFrame: duration,
              mediaWidth: mw || undefined,
              mediaHeight: mh || undefined,
              mediaAspectRatio:
                typeof mar === "number" && isFinite(mar) && mar > 0
                  ? mar
                  : undefined,
              preprocessors: [],
              masks: [],
            } as any;
            emitSelection(clip as AnyClipProps);
          } else {
            const iw = data.mediaInfo?.image?.width ?? 0;
            const ih = data.mediaInfo?.image?.height ?? 0;
            const iar = iw && ih ? iw / ih : undefined; 
            const asset = addAsset({
              path: data.assetUrl,
              modelInputAsset: true,
            });
            const clip: ImageClipProps = {
              type: "image",
              clipId: targetClipId,
              assetId: asset.id,
              startFrame: 0,
              endFrame: 1,
              mediaWidth: iw || undefined,
              mediaHeight: ih || undefined,
              mediaAspectRatio:
                typeof iar === "number" && isFinite(iar) && iar > 0
                  ? iar
                  : undefined,
              preprocessors: [],
              masks: [],
            } as any;
            emitSelection(clip as AnyClipProps);
          }
        }
      }
      setIsOverDropZone(false);
    },
  });

  // Use provided clip selection directly for preview and timeline syncing
  useEffect(() => {
    const canvasEl = canvasRef.current;
    if (!value) {
      if (canvasEl) {
        const ctx = canvasEl.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      }
      setStageSize({ w: 0, h: 0 });
      setMediaClip(null);
    
      const currentDuration = getTimelineDuration(inputId);
      const currentFocus = getFocusFrame(inputId);
      if (currentDuration[0] !== 0 || currentDuration[1] !== 1) {
        setInputTimelineDuration(0, 1, inputId);
      }
      if ((currentFocus ?? 0) !== 0) {
        setInputFocusFrame(0, inputId);
      }
      return;
    }
    setMediaClip(value as AnyClipProps);
    const start = Math.max(0, Math.round(value.startFrame ?? 0));
    const end = Math.max(start + 1, Math.round(value.endFrame ?? start + 1));
    const span = Math.max(1, end - start);
    const currentDuration = getTimelineDuration(inputId);
    const currentFocus = getFocusFrame(inputId);
    if (currentDuration[0] !== 0 || currentDuration[1] !== span) {
      setInputTimelineDuration(0, span, inputId);
    }
    const clampedFocus = Math.max(0, Math.min(span - 1, currentFocus ?? 0));
    if (currentFocus !== clampedFocus) {
      setInputFocusFrame(clampedFocus, inputId);
    }
  }, [value, inputId, setInputFocusFrame, setInputTimelineDuration]);

  useEffect(() => {
    const prevKey = lastSelectionKeyRef.current;
    lastSelectionKeyRef.current = selectionKey;
    if (!value) return;
    if (prevKey !== selectionKey) {
      return;
    }
    // selection is a clip; focus syncing is managed via store
  }, [selectionKey, value]);

  const handleToggleApply = React.useCallback(() => {
    const next = !applyPreprocessor;
    setApplyPreprocessor(next);
    if (onChangeComposite) {
      onChangeComposite({
        selection: value ?? null,
        preprocessor_ref: preprocessorRef,
        preprocessor_name: resolvedPreprocessorName,
        apply_preprocessor: next,
      });
    }
  }, [
    applyPreprocessor,
    onChangeComposite,
    value,
    preprocessorRef,
    resolvedPreprocessorName,
  ]);

  const handleExternalDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    externalDragCounterRef.current++;
    setIsOverDropZone(true);
  };

  const handleExternalDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleExternalDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    externalDragCounterRef.current--;
    if (externalDragCounterRef.current <= 0) {
      externalDragCounterRef.current = 0;
      setIsOverDropZone(false);
    }
  };

  const handleExternalDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOverDropZone(false);
    externalDragCounterRef.current = 0;
    const files = Array.from(e.dataTransfer.files);
    if (files.length !== 1) return;
    let path: string | undefined;
    try {
      path = getPathForFile(files[0]);
    } catch {}
    if (!path) return;
    const ext = getLowercaseExtension(path);
    const isAllowed = VIDEO_EXTS.includes(ext) || IMAGE_EXTS.includes(ext);
    if (!isAllowed) return;
    try {
      const before = await listConvertedMedia(getActiveProject()?.folderUuid);
      const existingNames = new Set(before.map((it) => it.name));
      await importMediaPaths([path], undefined, getActiveProject()?.folderUuid);
      const after = await listConvertedMedia(getActiveProject()?.folderUuid);
      const newlyAdded = after
        .filter((it) => !existingNames.has(it.name))
        .filter((it) => it.type === "image" || it.type === "video");
      const first = newlyAdded[0] ?? null;
      if (first) {
        if (VIDEO_EXTS.includes(getLowercaseExtension(first.assetUrl))) {
          try {
            const info = await getMediaInfo(first.assetUrl);
            const duration = Math.max(
              1,
              Math.floor((info?.duration || 0) * fps),
            );
            const mw =
              info?.video?.displayWidth ?? (info as any)?.video?.width ?? 0;
            const mh =
              info?.video?.displayHeight ?? (info as any)?.video?.height ?? 0;
            const mar = mw && mh ? mw / mh : undefined;
            const asset = addAsset({
              path: first.assetUrl,
              modelInputAsset: true,
            });
            const clip: VideoClipProps = {
              type: "video",
              clipId: `media:${first.assetUrl}`,
              assetId: asset.id,
              startFrame: 0,
              endFrame: duration,
              mediaWidth: mw || undefined,
              mediaHeight: mh || undefined,
              mediaAspectRatio:
                typeof mar === "number" && isFinite(mar) && mar > 0
                  ? mar
                  : undefined,
              preprocessors: [],
              masks: [],
            } as any;
            emitSelection(clip as AnyClipProps);
          } catch {}
        } else {
          const iw = (await getMediaInfo(first.assetUrl))?.image?.width ?? 0; // best effort
          const ih = (await getMediaInfo(first.assetUrl))?.image?.height ?? 0;
          const iar = iw && ih ? iw / ih : undefined;
          const asset = addAsset({
            path: first.assetUrl,
            modelInputAsset: true,
          });
          const clip: ImageClipProps = {
            type: "image",
            clipId: `media:${first.assetUrl}`,
            assetId: asset.id,
            startFrame: 0,
            endFrame: 1,
            mediaWidth: iw || undefined,
            mediaHeight: ih || undefined,
            mediaAspectRatio:
              typeof iar === "number" && isFinite(iar) && iar > 0
                ? iar
                : undefined,
            preprocessors: [],
            masks: [],
          } as any;
          emitSelection(clip as AnyClipProps);
        }
      }
      bumpMediaLibraryVersion();
    } catch {}
  };

  // Keep preview/input in sync with timeline clip updates/deletions when selection references timeline

  useEffect(() => {
    if (!value) return;
    if ((value as any)?.disableTimelineSync) return;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return; // media assets managed via media info
    if (!liveTimelineClip) {
      // clip removed from timeline → clear input
      emitSelection(null);
      return;
    }
    // Ignore audio-only timeline clips for image input
    if ((liveTimelineClip as AnyClipProps).type === "audio") {
      emitSelection(null);
      return;
    }
    if ((liveTimelineClip as AnyClipProps).hidden) {
      emitSelection(null);
      return;
    }
    const sigCurrent = clipSignature(value as AnyClipProps);
    const sigLive = clipSignature(liveTimelineClip as AnyClipProps);
    if (sigCurrent !== sigLive) {
      emitSelection({ ...(liveTimelineClip as AnyClipProps) });
    }
  }, [value, liveTimelineClip, emitSelection]);

  return (
    <Droppable className="w-full" id={droppableId} accepts={["media"]}>
      <div
        ref={divRef}
        className="flex flex-col items-start w-full gap-y-1 min-w-0   h-full"
        style={{ height: height ? `${height}px` : undefined }}
      >
        <div
          className={cn(
            "w-full flex flex-col items-start justify-start  h-full",
            {
              "p-0 rounded-md": !!height,
              "p-3 border border-brand-light/5 rounded-[7px]": !height,
            },
          )}
        >
          {(label || description) && (
            <div className="w-full flex flex-col items-start justify-start mb-3">
              <div className="w-full flex flex-col">
                <div className="flex flex-col items-start justify-start">
                  {label && (
                    <label className="text-brand-light text-[10.5px] w-full font-medium text-start">
                      {label}
                    </label>
                  )}
                  {description && (
                    <span className="text-brand-light/80 text-[9.5px] w-full text-start">
                      {description}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
          <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
            <PopoverTrigger
              asChild
              onClick={(e) => {
                if (!isPopoverOpen && value) {
                  e.preventDefault();
                  e.stopPropagation();
                }
              }}
            >
              <div
                ref={stageContainerRef}
                onDragEnter={handleExternalDragEnter}
                onDragOver={handleExternalDragOver}
                onDragLeave={handleExternalDragLeave}
                onDrop={handleExternalDrop}
                style={{ height: value ? stageSize.h : undefined }}
                className={cn(
                  "w-full flex flex-col items-center justify-center gap-y-3 h-full shadow-accent group  cursor-pointer relative overflow-hidden",
                  value
                    ? ""
                    : "border-dashed hover:opacity-70 p-4 border-brand-light/10 border bg-brand-background-light/50 ",
                  height ? "h-full rounded-md" : "h-48 rounded",
                )}
              >
                {isOverDropZone && !height && (
                  <div className="absolute inset-0 z-30 bg-brand-background-light/40 backdrop-blur-sm pointer-events-none transition-opacity duration-150" />
                )}
                {value ? (
                  stageSize.w > 0 && stageSize.h > 0 ? (
                    (() => {
                      if (!mediaClip) return null;
                      const clipToRender = mediaClip;
                      return (
                        <TimelineClipPosterPreview
                          key={clipToRender.clipId}
                          clip={clipToRender}
                          width={stageSize.w}
                          height={stageSize.h}
                          ratioOverride={displayRatio}
                          inputId={inputId}
                          needsStage={true}
                          isDialogOpen={isDialogOpen}
                        />
                      );
                    })()
                  ) : (
                    <canvas
                      ref={canvasRef}
                      className="w-full h-auto rounded-[6px]"
                    />
                  )
                ) : (
                  <>
                    <IoImageOutline
                      className={cn(" text-brand-light", {
                        "w-6 h-6": height,
                        "w-1/4 h-1/4": !height,
                      })}
                    />
                    <span
                      className={cn(
                        "text-brand-light text-[10.5px] w-full text-center font-medium",
                        {
                          "text-[11.5px]": height,
                          "text-[10.5px]": !height,
                        },
                      )}
                    >
                      {height
                        ? "Click to select an image."
                        : "Click or drag and drop an image here."}
                    </span>
                  </>
                )}
                {value && (
                  <div
                    className={cn(
                      "absolute bottom-0 h-full left-0 right-0 z-50 bg-brand/60 hover:backdrop-blur-sm gap-x-4 transition-opacity duration-150 opacity-0 hover:opacity-100 flex items-center justify-center",
                      {
                        "rounded-md": !!height,
                        rounded: !height,
                      },
                    )}
                  >
                    <div className="flex flex-col justify-center items-center gap-y-2 p-4  rounded-md w-full">
                      <button
                        onClick={() => setIsPopoverOpen(true)}
                        className={cn(
                          "z-30 duration-150 flex items-center px-6 py-2  bg-brand-light hover:bg-brand-light/90 shadow-md justify-center rounded text-[10.5px] font-poppins font-medium text-brand-accent-two-shade transition-colors",
                          {
                            "w-28 gap-x-1": !!height,
                            "w-40 gap-x-2": !height,
                          },
                        )}
                      >
                        <TbPhoto className="w-4 h-4" />
                        <span className="">
                          {height ? "Select" : "Select Media"}
                        </span>
                      </button>
                      {value &&
                        mediaClip &&
                        (mediaClip?.type === "video" ||
                          mediaClip.type === "group" ||
                          (mediaClip.type === "image" &&
                            !isClipOnTimeline)) && (
                          <button
                            onClick={() => setIsDialogOpen(true)}
                            className={cn(
                              "z-30 duration-150 justify-center shadow-md flex items-center  px-6 py-2 font-poppins font-medium bg-brand-accent-two-shade  hover:bg-brand-accent-two-shade/90 rounded text-[10.5px] text-brand-light transition-colors",
                              {
                                "w-28 gap-x-1": !!height,
                                "w-40 gap-x-2": !height,
                              },
                            )}
                          >
                            <TbEdit className="w-4 h-4" />
                            {height ? "Edit" : "Edit Media"}
                          </button>
                        )}
                    </div>
                  </div>
                )}
              </div>
            </PopoverTrigger>
            <PopoverImage
              value={value}
              onChange={emitSelection}
              clipId={clipId}
          onSelectComplete={() => setIsPopoverOpen(false)}
            />
          </Popover>
          <div
            className={cn("w-full flex flex-col items-start justify-start ", {
              "mt-0": !!height,
              "mt-3": !height,
            })}
          >
            {preprocessorRef && (
              <div className=" flex flex-row-reverse items-center gap-x-3 justify-between">
                <span className="text-brand-light text-[10px] font-medium">
                  Apply {resolvedPreprocessorName}
                </span>
                <button
                  type="button"
                  aria-pressed={applyPreprocessor}
                  aria-label={applyPreprocessor ? "Disable" : "Enable"}
                  onClick={handleToggleApply}
                  className={cn(
                    "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none",
                    applyPreprocessor
                      ? "bg-blue-600"
                      : "bg-brand-background border border-brand-light/10",
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-4 w-4 transform rounded-full bg-brand-light shadow transition-transform",
                      applyPreprocessor ? "translate-x-4.5" : "translate-x-0.5",
                    )}
                  />
                </button>
              </div>
            )}

            <MediaDialog
              isOpen={isDialogOpen}
              onClose={() => setIsDialogOpen(false)}
              onConfirm={handleConfirm}
              clipOverride={mediaClip}

              timelineSelectorProps={
                mediaClip && mediaClip.type !== "image"
                  ? { mode: "frame", inputId: inputId }
                  : undefined
              }
              focusFrame={focusFrameForInput}
              canCrop={!isClipOnTimeline}
              setFocusFrame={(frame) => setInputFocusFrame(frame, inputId)}
            />
          </div>
        </div>
      </div>
    </Droppable>
  );
};

export default ImageInput;
