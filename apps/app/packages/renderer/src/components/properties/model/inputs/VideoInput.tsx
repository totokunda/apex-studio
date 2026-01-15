import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDndMonitor } from "@dnd-kit/core";
import Droppable from "@/components/dnd/Droppable";
import { MdMovie } from "react-icons/md";
import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@radix-ui/react-tabs";
import { LuPause, LuPlay, LuSearch, LuUpload } from "react-icons/lu";
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
  VideoClipProps,
  clipSignature,
  ClipTransform,
} from "@/lib/types";
import { DEFAULT_FPS, VIDEO_EXTS } from "@/lib/settings";
import {
  getLowercaseExtension,
  importMediaPaths,
  pickMediaPaths,
  getPathForFile,
} from "@app/preload";
import { useViewportStore } from "@/lib/viewport";
import {
  useMediaLibraryVersion,
  bumpMediaLibraryVersion,
} from "@/lib/media/library";

import { useInputControlsStore } from "@/lib/inputControl";
import { useControlsStore } from "@/lib/control";
import { usePreprocessorsListQuery } from "@/lib/preprocessor/queries";
import { TbEdit, TbVideo } from "react-icons/tb";
import { MediaDialog } from "@/components/dialogs/MediaDialog";

export type VideoSelection = AnyClipProps | null;

interface VideoInputProps {
  label?: string;
  description?: string;
  inputId: string;
  maxDuration?: number;
  value: VideoSelection;
  onChange: (value: VideoSelection) => void;
  clipId: string;
  panelSize: number;
  preprocessorRef?: string;
  preprocessorName?: string;
  applyPreprocessorInitial?: boolean;
  onChangeComposite?: (value: {
    selection: VideoSelection;
    preprocessor_ref?: string;
    preprocessor_name?: string;
    apply_preprocessor?: boolean;
  }) => void;
}

interface PopoverVideoProps {
  value: VideoSelection;
  onChange: (value: VideoSelection) => void;
  clipId: string | null;
}

const PopoverVideo: React.FC<PopoverVideoProps> = ({
  value,
  onChange,
  clipId,
}) => {
  const isUserInteractingRef = useRef(false);
  const [selectedTab, setSelectedTab] = useState<"timeline" | "library">(
    "library",
  );
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredMediaItems, setFilteredMediaItems] = useState<MediaItem[]>([]);
  const { clips } = useClipStore();
  const getClipById = useClipStore((s) => s.getClipById);
  const clearSelectedAsset = useAssetControlsStore((s) => s.clearSelectedAsset);
  const setSelectedAssetChangeHandler = useAssetControlsStore(
    (s) => s.setSelectedAssetChangeHandler,
  );
  const mediaLibraryVersion = useMediaLibraryVersion();
  const { fps } = useControlsStore();
  const addAsset = useClipStore((s) => s.addAsset);
  const getActiveProject = useProjectsStore((s) => s.getActiveProject);
  useEffect(() => {
    const next = mediaItems.filter((media) =>
      media.name.toLowerCase().includes(searchQuery.toLowerCase()),
    );
    setFilteredMediaItems(next);
  }, [searchQuery, mediaItems]);

  useEffect(() => {
    (async () => {
      try {
        const list = await listConvertedMedia(getActiveProject()?.folderUuid);
        const infoPromises = list.map((it) => getMediaInfo(it.assetUrl));
        const infos = await Promise.all(infoPromises);
        const results: MediaItem[] = list
          .map((it, idx) => ({
            name: it.name,
            type: it.type,
            absPath: it.absPath,
            assetUrl: it.assetUrl,
            dateAddedMs: it.dateAddedMs,
            mediaInfo: infos[idx],
            hasProxy: it.hasProxy,
          }))
          .filter((media) => media.type === "video")
          .sort((a, b) =>
            a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
          );
        setMediaItems(results);
      } catch {
        // Swallow errors; UI handles toasts elsewhere.
      }
    })();
  }, [mediaLibraryVersion]);

  const numEligibleTimelineAssets = useMemo(() => {
    return clips.filter(
      (clip) =>
        clip.type !== "filter" &&
        clip.type !== "audio" &&
        clip.clipId !== clipId,
    ).length;
  }, [clips, clipId]);

  const handleUpload = useCallback(async () => {
    try {
      const filters = [{ name: "Video Files", extensions: VIDEO_EXTS }];
      const picked = await pickMediaPaths({
        directory: false,
        filters,
        title: "Choose video file(s) to import",
      });
      const paths = (picked ?? []).filter((p) =>
        VIDEO_EXTS.includes(getLowercaseExtension(p)),
      );
      if (paths.length === 0) return;
      const before = await listConvertedMedia(getActiveProject()?.folderUuid);
      const existingNames = new Set(before.map((it) => it.name));
      await importMediaPaths(paths, undefined, getActiveProject()?.folderUuid);
      const after = await listConvertedMedia(getActiveProject()?.folderUuid);
      const infoPromises = after.map((it) => getMediaInfo(it.assetUrl));
      const infos = await Promise.all(infoPromises);
      const results: MediaItem[] = after
        .map((it, idx) => ({
          name: it.name,
          type: it.type,
          absPath: it.absPath,
          assetUrl: it.assetUrl,
          dateAddedMs: it.dateAddedMs,
          mediaInfo: infos[idx],
          hasProxy: it.hasProxy,
        }))
        .filter((media) => media.type === "video")
        .sort((a, b) =>
          a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
        );
      setMediaItems(results);

      const newlyAdded = results.filter((it) => !existingNames.has(it.name));
      if (newlyAdded.length > 0) {
        const first = newlyAdded[0];
        const durationFrames = Math.max(
          1,
          Math.floor((first.mediaInfo?.duration || 0) * fps),
        );
        clearSelectedAsset();
        const mw =
          first.mediaInfo?.video?.displayWidth ??
          (first.mediaInfo as any)?.video?.width ??
          0;
        const mh =
          first.mediaInfo?.video?.displayHeight ??
          (first.mediaInfo as any)?.video?.height ??
          0;
        const mar = mw && mh ? mw / mh : undefined;
        const asset = addAsset({ path: first.assetUrl, modelInputAsset: true });
        const clip: VideoClipProps = {
          type: "video",
          clipId: `media:${first.assetUrl}`,
          assetId: asset.id,
          startFrame: 0,
          endFrame: Math.max(1, durationFrames),
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
      }
      bumpMediaLibraryVersion();
    } catch {
      // ignore upload errors here
    }
  }, [clearSelectedAsset, onChange, fps]);

  const assetSelectionHandler = React.useCallback(
    (selectedClipId: string | null) => {
      if (!selectedClipId) {
        onChange(null);
        return;
      }
      const clip = getClipById(selectedClipId) as AnyClipProps | undefined;
      if (!clip || clip.type === "audio") return;
      const clipDuration = Math.max(
        1,
        (clip.endFrame ?? 0) - (clip.startFrame ?? 0),
      );
      // create a new asset for the clip
      const newClip = {
        ...clip,
        startFrame: 0,
        endFrame: clipDuration,
      } as AnyClipProps;
      if (clip.type === "video") {
        const asset = addAsset({
          path: clip.assetId,
          modelInputAsset: true,
        });
        (newClip as VideoClipProps).assetId = asset.id;
      }
      onChange(newClip);
    },
    [getClipById, onChange],
  );

  const renderMediaLibrary = () => (
    <div className="w-full h-full flex flex-col py-2 gap-y-2 outline-none">
      <div className="w-full flex flex-row items-center justify-between gap-x-2">
        <span className="relative w-full">
          <LuSearch className="w-3.5 h-3.5 text-brand-light/50 absolute left-2 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search for video"
            className="w-full h-full pl-8 text-brand-light text-[10.5px] font-normal bg-brand rounded-[7px] border border-brand-light/10 p-2 outline-none"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </span>
      </div>
      <ScrollArea className="w-full h-96">
        <div className="w-full h-full grid grid-cols-2 gap-3">
          {filteredMediaItems.map((media) => {
            const isSelected = value?.clipId === `media:${media.assetUrl}`;
            const durationFrames = Math.max(
              1,
              Math.floor((media.mediaInfo?.duration || 0) * fps),
            );
            return (
              <div
                key={media.name}
                onClick={() => {
                  clearSelectedAsset();
                  if (isSelected) {
                    onChange(null);
                  } else {
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
                      clipId: `media:${media.assetUrl}`,
                      assetId: asset.id,
                      assetIdHistory: [asset.id],
                      startFrame: 0,
                      endFrame: durationFrames,
                      mediaWidth: mw || undefined,
                      mediaHeight: mh || undefined,
                      mediaAspectRatio:
                        typeof mar === "number" && isFinite(mar) && mar > 0
                          ? mar
                          : undefined,
                      preprocessors: [],
                      masks: [],
                    };
                    onChange(clip as AnyClipProps);
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
                      isSelected ? "opacity-100" : "opacity-0",
                    )}
                  >
                    <div
                      className={cn(
                        "rounded-full py-1 px-3 bg-brand-light/10 flex items-center justify-center font-medium text-[10.5px] w-fit",
                        isSelected ? "bg-brand-light/20" : "",
                      )}
                    >
                      {isSelected ? "Selected" : "Use as Input"}
                    </div>
                  </div>
                  <MediaThumb key={media.name} item={media} />
                </div>
                <div className="text-brand-light/90 text-[9.5px] text-start truncate w-full text-ellipsis overflow-hidden group-hover:text-brand-light transition-all duration-200">
                  {media.name}
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
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
        clearSelectedAsset();
        setSelectedAssetChangeHandler(assetSelectionHandler);
      }}
      onCloseAutoFocus={() => {
        isUserInteractingRef.current = false;
        setSelectedAssetChangeHandler(null);
      }}
    >
      <Tabs
        value={selectedTab}
        onValueChange={(val) => setSelectedTab(val as "timeline" | "library")}
      >
        <div
          className={cn(
            "w-full flex flex-row items-center gap-x-2 justify-between",
          )}
        >
          <TabsList
            className={cn(
              "w-full text-brand-light text-[10.5px] rounded font-medium text-start flex flex-row shadow overflow-hidden",
              numEligibleTimelineAssets === 0
                ? "justify-start"
                : "justify-between cursor-pointer bg-brand-background-light",
            )}
          >
            <TabsTrigger
              value="library"
              className={cn(
                "w-full py-1.5 flex items-center",
                selectedTab === "library" && numEligibleTimelineAssets > 0
                  ? "bg-brand-accent-shade"
                  : "",
                numEligibleTimelineAssets === 0
                  ? "cursor-default justify-start px-2.5"
                  : "cursor-pointer justify-center px-4",
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
        <TabsContent value="library">{renderMediaLibrary()}</TabsContent>
        <TabsContent value="timeline" className="outline-none">
          <TimelineSearch
            isAssetSelected={(clipId) => {
              if (!value) return false;
              return clipId === value.clipId;
            }}
            types={["image", "video", "group", "text", "shape", "draw"]}
            excludeClipId={clipId || undefined}
          />
        </TabsContent>
      </Tabs>
    </PopoverContent>
  );
};

const VideoInput: React.FC<VideoInputProps> = ({
  label,
  description,
  inputId,
  value,
  onChange,
  clipId,
  panelSize,
  preprocessorRef,
  preprocessorName,
  applyPreprocessorInitial,
  onChangeComposite,
  maxDuration,
}) => {
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
  const resolvedPreprocessorName = useMemo(() => {
    if (!preprocessorRef) return preprocessorName;
    const found = (preprocessors || []).find((p) => p.id === preprocessorRef);
    return found?.name || preprocessorName || preprocessorRef;
  }, [preprocessors, preprocessorRef, preprocessorName]);

  const { clearSelectedAsset } = useAssetControlsStore();

  const aspectRatio = useViewportStore((s) => s.aspectRatio);

  const setInputFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
  const setInputFps = useInputControlsStore((s) => s.setFps);
  const {
    fpsByInputId,
    focusFrameByInputId,
    selectedRangeByInputId,
    setFocusFrame,
    setSelectedRange,
    clearSelectedRange,
    clearFocusFrame,
    play,
    pause,
  } = useInputControlsStore();
  const isPlaying = useInputControlsStore(
    (s) => !!s.isPlayingByInputId[inputId],
  );

  const fpsForInput = fpsByInputId[inputId] ?? DEFAULT_FPS;

  const selectedRangeTuple = selectedRangeByInputId[inputId] ?? [0, 0];
  const focusFrameForInput = focusFrameByInputId[inputId] ?? 0;
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);

  // If selection is cleared externally (not via emitSelection), also clear
  // any persisted per-input selection state.
  useEffect(() => {
    if (value) return;
    pause(inputId);
    clearSelectedRange(inputId);
    clearFocusFrame(inputId);
  }, [value, inputId, pause, clearSelectedRange, clearFocusFrame]);

  const { fps } = useControlsStore();
  const playbackStartTimestampRef = useRef<number | null>(null);
  const playbackLastFrameRef = useRef<number>(0);
  const ratioCacheByClipIdRef = useRef<Record<string, number>>({});
  const [contentRatio, setContentRatio] = useState<number | null>(null);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const scrubberTrackRef = useRef<HTMLDivElement | null>(null);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const addAsset = useClipStore((s) => s.addAsset);

  // Initialize input fps synchronously before first render to avoid slow playback.
  // We intentionally do not depend on the unstable setter reference here to avoid
  // triggering an infinite render loop; instead we keep the latest setter in a ref.
  const setInputFpsRef = React.useRef(setInputFps);
  React.useEffect(() => {
    setInputFpsRef.current = setInputFps;
  }, [setInputFps]);
  React.useLayoutEffect(() => {
    setInputFpsRef.current(fps, inputId);
  }, [fps, inputId]);

  // Simplified: requestedStart/end unused

  const selectionKey = useMemo(() => {
    if (!value) return "null";
    return value.clipId;
  }, [value]);

  const getActiveProject = useProjectsStore((s) => s.getActiveProject);

  const viewportRatio = useMemo(() => {
    const r = aspectRatio.width / aspectRatio.height;
    return Number.isFinite(r) && r > 0 ? r : 1;
  }, [aspectRatio.width, aspectRatio.height]);
  const selectedClipForRatio = useMemo<AnyClipProps | null>(() => {
    if (!value) return null;
    return (mediaClip ?? (value as AnyClipProps)) || null;
  }, [value, mediaClip]);
  const selectedClipRatioSignature = clipSignature(
    selectedClipForRatio as AnyClipProps,
  );
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
    const asset = getAssetById(clip.assetId);
    if (!asset) return;
    const src = asset.path;
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

  // Preprocessor toggle state and default
  const [applyPreprocessor, setApplyPreprocessor] = useState<boolean>(true);
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

  const emitSelection = useCallback(
    (next: VideoSelection) => {
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

      // When unselecting, clear any persisted per-input selection state.
      if (!next) {
        pause(inputId);
        clearSelectedRange(inputId);
        clearFocusFrame(inputId);
        return;
      }

      // Reset selected range to [0, clip_length] only when a NEW clip is selected
      const currentClipId = value?.clipId ?? null;
      const nextClipId = next?.clipId ?? null;
      const isNewClip = next && nextClipId !== currentClipId;
      if (isNewClip) {
        const clipStart = Math.max(0, Math.round(next.startFrame ?? 0));
        const clipEnd = Math.max(
          clipStart + 1,
          Math.round(next.endFrame ?? clipStart + 1),
        );
        let span = clipEnd - clipStart;
        if (typeof maxDuration === "number" && maxDuration > 0) {
          span = Math.min(span, Math.max(1, Math.floor(maxDuration)));
        }
        setSelectedRange(0, span, inputId);
        setFocusFrame(0, inputId);
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
          let src: string | undefined = getAssetById((next as VideoClipProps).assetId)?.path;
          let mw = Number((next as VideoClipProps)?.mediaWidth ?? 0);
          let mh = Number((next as VideoClipProps)?.mediaHeight ?? 0);
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
      maxDuration,
      setSelectedRange,
      setFocusFrame,
      clearSelectedRange,
      clearFocusFrame,
      pause,
      value,
    ],
  );

  const handleToggleApply = useCallback(() => {
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

  useEffect(() => {
    const el = stageContainerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      let w = Math.max(1, panelSize);
      let h = Math.max(1, w / displayRatio);
      if (h > 400) {
        h = 400;
        w = Math.max(1, h * displayRatio);
      }
      setStageSize({ w, h });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [panelSize, displayRatio]);

  useEffect(() => {
    let w = Math.max(1, panelSize);
    let h = Math.max(1, w / displayRatio);
    if (h > 400) {
      h = 400;
      w = Math.max(1, h * displayRatio);
    }
    setStageSize({ w, h });
  }, [panelSize, displayRatio]);

  const [rangeStartForInput, rangeEndForInput] = useMemo<
    [number, number]
  >(() => {
    const start = Math.max(0, Math.round(selectedRangeTuple?.[0] ?? 0));
    const endRaw = Math.max(
      start + 1,
      Math.round(selectedRangeTuple?.[1] ?? start + 1),
    );
    return [start, endRaw];
  }, [selectedRangeTuple?.[0], selectedRangeTuple?.[1]]);
  const scrubberProgress = useMemo(() => {
    const clampedFocus = Math.max(
      rangeStartForInput,
      Math.min(
        Math.max(rangeStartForInput, rangeEndForInput - 1),
        Math.round(focusFrameForInput ?? rangeStartForInput),
      ),
    );
    const span = Math.max(1, rangeEndForInput - rangeStartForInput);
    if (span <= 1) return 0;
    return (clampedFocus - rangeStartForInput) / (span - 1);
  }, [focusFrameForInput, rangeStartForInput, rangeEndForInput]);

  const scrubToClientX = useCallback(
    (clientX: number) => {
      if (!scrubberTrackRef.current) return;
      const rect = scrubberTrackRef.current.getBoundingClientRect();
      if (!rect || rect.width <= 0) return;
      const rawRatio = (clientX - rect.left) / rect.width;
      const ratio = Math.max(0, Math.min(1, rawRatio));
      const span = Math.max(1, rangeEndForInput - rangeStartForInput);
      const nextFrame =
        span <= 1
          ? rangeStartForInput
          : rangeStartForInput + Math.round(ratio * (span - 1));
      setInputFocusFrame(nextFrame, inputId);
    },
    [inputId, rangeStartForInput, rangeEndForInput, setInputFocusFrame],
  );

  useEffect(() => {
    if (!isScrubbing) return;
    const handleMove = (e: MouseEvent) => {
      scrubToClientX(e.clientX);
    };
    const handleUp = () => {
      setIsScrubbing(false);
    };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [isScrubbing, scrubToClientX]);

  const rangeSummary = useMemo(() => {
    const start = rangeStartForInput;
    const endDisplay = Math.max(start, rangeEndForInput);
    const spanFrames = Math.max(1, rangeEndForInput - start);
    const threshold = Math.max(1, Math.floor(fpsForInput * 5));
    if (spanFrames < threshold) {
      return `${start}f – ${endDisplay}f`;
    }
    const formatTime = (secondsRaw: number) => {
      const totalSeconds = Math.floor(secondsRaw);
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = Math.floor(totalSeconds % 60);
      return hours > 0
        ? `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
        : `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
    };
    const startTime = start / Math.max(1, fpsForInput);
    const endTime = endDisplay / Math.max(1, fpsForInput);
    return `${formatTime(startTime)} – ${formatTime(endTime)}`;
  }, [rangeStartForInput, rangeEndForInput, fpsForInput]);

  useEffect(() => {
    playbackLastFrameRef.current = rangeStartForInput;
    playbackStartTimestampRef.current = null;
  }, [rangeStartForInput, selectionKey]);

  useDndMonitor({
    onDragStart: () => {
      setIsOverDropZone(false);
    },
    onDragMove: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id as string | undefined;
      const isValid = !!data && data.type === "video";
      setIsOverDropZone(isValid && overId === "video-input");
    },
    onDragCancel: () => {
      setIsOverDropZone(false);
    },
    onDragEnd: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id as string | undefined;
      const isValid = !!data && data.type === "video";
      if (isValid && overId === "video-input") {
        clearSelectedAsset();
        void (async () => {
          try {
            const info = data.mediaInfo ?? (await getMediaInfo(data.assetUrl));
            const durationFrames = Math.max(
              1,
              Math.floor((info?.duration || 0) * fps),
            );
            const mw =
              info?.video?.displayWidth ?? (info as any)?.video?.width ?? 0;
            const mh =
              info?.video?.displayHeight ?? (info as any)?.video?.height ?? 0;
            const mar = mw && mh ? mw / mh : undefined;
            const asset = addAsset({
              path: data.assetUrl,
              modelInputAsset: true,
            });
            const clip: VideoClipProps = {
              type: "video",
              clipId: `media:${data.assetUrl}`,
              assetId: asset.id,
              startFrame: 0,
              endFrame: durationFrames,
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
          } catch {
            const asset = addAsset({
              path: data?.assetUrl || "",
              modelInputAsset: true,
            });
            const clip: VideoClipProps = {
              type: "video",
              clipId: `media:${data?.assetUrl || ""}`,
              assetId: asset.id,
              startFrame: 0,
              endFrame: 1,
              preprocessors: [],
              masks: [],
            } as any;
            emitSelection(clip as AnyClipProps);
          }
        })();
      }
      setIsOverDropZone(false);
    },
  });

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
    if (!VIDEO_EXTS.includes(ext)) return;
    try {
      const before = await listConvertedMedia(getActiveProject()?.folderUuid);
      const existingNames = new Set(before.map((it) => it.name));
      await importMediaPaths([path], undefined, getActiveProject()?.folderUuid);
      const after = await listConvertedMedia(getActiveProject()?.folderUuid);
      const infoPromises = after.map((it) => getMediaInfo(it.assetUrl));
      const infos = await Promise.all(infoPromises);
      const results: MediaItem[] = after
        .map((it, idx) => ({
          name: it.name,
          type: it.type,
          absPath: it.absPath,
          assetUrl: it.assetUrl,
          dateAddedMs: it.dateAddedMs,
          mediaInfo: infos[idx],
          hasProxy: it.hasProxy,
        }))
        .filter((media) => media.type === "video")
        .sort((a, b) =>
          a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
        );
      const newlyAdded = results.filter((it) => !existingNames.has(it.name));
      if (newlyAdded.length > 0) {
        const first = newlyAdded[0];
        const durationFrames = Math.max(
          1,
          Math.floor((first.mediaInfo?.duration || 0) * fps),
        );
        clearSelectedAsset();
        const mw =
          first.mediaInfo?.video?.displayWidth ??
          (first.mediaInfo as any)?.video?.width ??
          0;
        const mh =
          first.mediaInfo?.video?.displayHeight ??
          (first.mediaInfo as any)?.video?.height ??
          0;
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
          endFrame: durationFrames,
          mediaWidth: mw || undefined,
          mediaHeight: mh || undefined,
          mediaAspectRatio:
            typeof mar === "number" && isFinite(mar) && mar > 0
              ? mar
              : undefined,
          preprocessors: [],
          masks: [],
        } as any;
        emitSelection(clip);
      }
      bumpMediaLibraryVersion();
    } catch {
      // ignore errors
    }
  };

  // Simplified preview: set mediaClip from store for timeline clips, or compute media asset duration once
  useEffect(() => {
    if (!value) {
      setMediaClip(null);
      return;
    }
    const clip = value as AnyClipProps;
    const cid = String(clip.clipId || "");
    if (!cid.startsWith("media:")) {
      const live = getClipById(cid) as AnyClipProps | undefined;
      // check if model with assetId is video
      let newType = live?.type;
      if (live?.type === "model") {
        const asset = getAssetById(live?.assetId ?? "");
        newType = asset?.type;
      }

      setMediaClip({
        ...live,
        type: newType,
        masks: (value as VideoClipProps).masks ?? [],
      } as AnyClipProps);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo((clip as VideoClipProps).assetId);
        const durationFrames = Math.max(
          1,
          Math.floor((info?.duration || 0) * fps),
        );
        if (!cancelled)
          setMediaClip({
            ...clip,
            startFrame: 0,
            endFrame: durationFrames,
          } as AnyClipProps);
      } catch {
        if (!cancelled) setMediaClip(clip);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [value, getClipById, fps]);

  // Simplified: no emission of start/end; latest clip is fetched from store when needed

  useEffect(() => {
    const [rangeStart, rangeEnd] = selectedRangeTuple;
    const clampedFocus = Math.max(
      rangeStart,
      Math.min(rangeEnd, Math.round(focusFrameForInput ?? rangeStart)),
    );
    if (clampedFocus !== focusFrameForInput) {
      setInputFocusFrame(clampedFocus, inputId);
    }
  }, [selectedRangeTuple, focusFrameForInput, inputId, setInputFocusFrame]);

  // Local RAF playback removed; store-managed playback handles focus updates

  const previewClip = useMemo<AnyClipProps | null>(() => {
    if (!value) return null;

    return mediaClip ?? (value as AnyClipProps);
  }, [value, mediaClip]);

  // Keep preview/input in sync with timeline clip updates/deletions when selection references timeline
  const liveTimelineClip = useClipStore((s) => {
    if (!value) return null;
    if ((value as any)?.disableTimelineSync) return null;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return null;
    return (s.getClipById(cid) as AnyClipProps | undefined) ?? null;
  });

  const handleConfirm = useCallback(
    (data: {
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
      const updatedClip: AnyClipProps = {
        ...mediaClip,
        transform: newTransform,
      };
      if (data.originalTransform) {
        (updatedClip as any).originalTransform = { ...data.originalTransform };
      }
      setMediaClip(updatedClip);
      emitSelection(updatedClip as AnyClipProps);
      if (liveTimelineClip) {
        setClipTransform(liveTimelineClip.clipId, newTransform);
      }
    },
    [mediaClip, emitSelection, liveTimelineClip, setClipTransform],
  );

  useEffect(() => {
    if (!value) return;
    if ((value as any)?.disableTimelineSync) return;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return; // media assets managed separately
    if (!liveTimelineClip) {
      emitSelection(null);
      return;
    }
    if ((liveTimelineClip as AnyClipProps).hidden) {
      emitSelection(null);
      return;
    }
    if ((liveTimelineClip as AnyClipProps).type === "audio") {
      emitSelection(null);
      return;
    }
    const sigCurrent = clipSignature(value as AnyClipProps);
    const sigLive = clipSignature(liveTimelineClip as AnyClipProps);
    if (sigCurrent !== sigLive) {
      emitSelection({ ...(liveTimelineClip as AnyClipProps) });
    }
  }, [value, liveTimelineClip, emitSelection]);

  const isClipOnTimeline = useMemo(() => {
    if (!value) return false;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return false;
    return true;
  }, [value, getClipById]);

  

  // Reset selected range only when the selected clipId actually changes AND range is uninitialized
  const lastClipSignatureRef = useRef<string | null>(null);
  useEffect(() => {
    if (!previewClip || !previewClip.type) return;
    const clipStart = Math.max(0, Math.round(previewClip.startFrame ?? 0));
    const clipEnd = Math.max(
      clipStart + 1,
      Math.round(previewClip.endFrame ?? clipStart + 1),
    );
    let span = clipEnd - clipStart;
    if (span <= 1) {
      return;
    }
    if (typeof maxDuration === "number" && maxDuration > 0) {
      span = Math.min(span, Math.max(1, Math.floor(maxDuration)));
    }
    const currentClipId = String(previewClip.clipId || "");

    // Persist previous clipId per input across unmounts so navigation does not force a reset
    let sameClipAsBefore = false;
    try {
      const store: any = window as any;
      if (!store.__apexPrevClipIdByInput) {
        store.__apexPrevClipIdByInput = new Map<string, string>();
      }
      const map = store.__apexPrevClipIdByInput as Map<string, string>;
      const prevId = map.get(inputId);
      sameClipAsBefore = prevId === currentClipId;
      if (!sameClipAsBefore) map.set(inputId, currentClipId);
    } catch {
      // Fallback for non-window environments: use local ref within the session
      sameClipAsBefore = lastClipSignatureRef.current === currentClipId;
      if (!sameClipAsBefore) lastClipSignatureRef.current = currentClipId;
    }

    if (sameClipAsBefore) return;

    const [curStart, curEnd] = selectedRangeTuple;
    
    // Check if current range is valid and persisted (not uninitialized [0, 0])
    // If range is valid and within bounds, preserve it (likely restored from persistence)
    const isUninitialized = curStart === 0 && curEnd === 0;
    const isValidRange = curEnd > curStart && curStart >= 0 && curEnd <= span;
    
    // Only reset to full span if range is uninitialized or invalid for this clip
    if (isUninitialized || !isValidRange) {
      setSelectedRange(0, span, inputId);
    }
  }, [
    previewClip?.clipId,
    previewClip?.startFrame,
    previewClip?.endFrame,
    inputId,
    setSelectedRange,
    selectedRangeTuple[0],
    selectedRangeTuple[1],
  ]);

  // Ensure selectedRange is always valid (min 1 frame, within [clip.start, clip.end])
  useEffect(() => {
    if (!previewClip || !previewClip.type) return;

    const clipStart = Math.max(0, Math.round(previewClip.startFrame ?? 0));
    const clipEnd = Math.max(
      clipStart + 1,
      Math.round(previewClip.endFrame ?? clipStart + 1),
    );
    const span = clipEnd - clipStart;
    if (span < 1) {
      return;
    }

    // if selectedRangeTuple is [0, 0], set it to [0, span]
    if (selectedRangeTuple[0] === 0 && selectedRangeTuple[1] === 0) {
      setSelectedRange(0, span, inputId);
      return;
    }
    let curStart = Math.round(selectedRangeTuple?.[0] ?? 0);
    let curEnd = Math.round(selectedRangeTuple?.[1] ?? curStart + 1);
    let desiredStart = Math.max(0, Math.min(span - 1, curStart));
    let desiredEnd = Math.max(desiredStart + 1, Math.min(span, curEnd));

    if (typeof maxDuration === "number" && maxDuration > 0) {
      const maxSpan = Math.max(1, Math.floor(maxDuration));
      const currentSpan = desiredEnd - desiredStart;
      if (currentSpan > maxSpan) {
        desiredEnd = desiredStart + maxSpan;
      }
    }

    if (desiredStart !== curStart || desiredEnd !== curEnd) {
      setSelectedRange(desiredStart, desiredEnd, inputId);
    }
  }, [
    previewClip,
    selectedRangeTuple?.[0],
    selectedRangeTuple?.[1],
    setSelectedRange,
    inputId,
    maxDuration,
  ]);


  // Simplified: do not mirror timeline changes by emitting; preview pulls latest from store via mediaClip

  const handleTogglePlayback = useCallback(() => {
    if (!previewClip) return;
    if (rangeEndForInput <= rangeStartForInput) {
      setFocusFrame(rangeStartForInput, inputId);
      return;
    }
    if (isPlaying) {
      pause(inputId);
    } else {
      play(inputId);
    }
  }, [
    previewClip,
    isPlaying,
    rangeStartForInput,
    rangeEndForInput,
    inputId,
    setFocusFrame,
    play,
    pause,
  ]);

  const showTimeline = Boolean(
    previewClip &&
    (previewClip.type === "video" || previewClip.type === "group"),
  );

  const playDisabled = !previewClip || rangeEndForInput <= rangeStartForInput;

  return (
    <Droppable className="w-full" id="video-input" accepts={["media"]}>
      <div className="flex flex-col items-start w-full gap-y-1 min-w-0 bg-brand rounded-[7px] border border-brand-light/5 h-full">
        <div className="w-full h-full flex flex-col items-start justify-start p-3 ">
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
                  "w-full flex flex-col items-center justify-center gap-y-3 shadow-accent cursor-pointer relative overflow-hidden group",
                  value
                    ? ""
                    : "border-dashed h-48 p-4 border-brand-light/10 border bg-brand-background-light/50 rounded",
                )}
              >
                {isOverDropZone && (
                  <div className="absolute inset-0 z-30 bg-brand-background-light/40 backdrop-blur-sm pointer-events-none transition-opacity duration-150" />
                )}
                {value ? (
                  stageSize.w > 0 && stageSize.h > 0 ? (
                    previewClip ? (
                      <TimelineClipPosterPreview
                        key={previewClip.clipId}
                        needsStage={true}
                        clip={previewClip}
                        width={stageSize.w}
                        height={stageSize.h}
                        inputId={inputId}
                        ratioOverride={displayRatio}
                        isDialogOpen={isDialogOpen}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-brand-light/70 text-[12px]">
                        Unable to preview clip.
                      </div>
                    )
                  ) : (
                    <div className="w-full h-32 flex items-center justify-center text-brand-light/70 text-[10px]">
                      Preparing preview...
                    </div>
                  )
                ) : (
                  <>
                    <MdMovie className="w-1/4 h-1/4 text-brand-light" />
                    <span className="text-brand-light text-[11px] w-full text-center font-medium">
                      Click or drag and drop a video here.
                    </span>
                  </>
                )}
                {value && (
                  <div className="absolute bottom-0 h-full left-0 right-0 z-50 bg-brand/60 hover:backdrop-blur-sm gap-x-4 transition-opacity duration-150 opacity-0 hover:opacity-100 flex items-center justify-center">
                    <div className="flex flex-col justify-center items-center gap-y-2 p-4 rounded-md w-full">
                      <button
                        onClick={() => setIsPopoverOpen(true)}
                        className="z-30 duration-150 flex items-center gap-x-2 px-6 py-2 w-40 bg-brand-light hover:bg-brand-light/90 shadow-md justify-center rounded text-[10.5px] font-poppins font-medium text-brand-accent-two-shade transition-colors"
                      >
                        <TbVideo className="w-4 h-4" />
                        <span>Select Media</span>
                      </button>
                      <button
                        onClick={() => setIsDialogOpen(true)}
                        className="z-30 duration-150 w-40 justify-center shadow-md flex items-center gap-x-2 px-6 py-2 font-poppins font-medium bg-brand-accent-two-shade  hover:bg-brand-accent-two-shade/90 rounded text-[10.5px] text-brand-light transition-colors"
                      >
                        <TbEdit className="w-4 h-4" />
                        Edit Video
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </PopoverTrigger>
            <PopoverVideo
              value={value}
              onChange={emitSelection}
              clipId={clipId}
            />
          </Popover>
          {showTimeline && value && (
            <div className="w-full flex flex-col gap-y-3 mt-3">
              <div className="w-full flex flex-col gap-y-1">
                <div
                  ref={scrubberTrackRef}
                  className={cn(
                    "relative w-full h-1.5 rounded-full bg-brand-background-light/70",
                    playDisabled
                      ? "opacity-60 cursor-default"
                      : "cursor-pointer",
                  )}
                  onMouseDown={(e) => {
                    if (playDisabled) return;
                    e.preventDefault();
                    setIsScrubbing(true);
                    scrubToClientX(e.clientX);
                  }}
                >
                  <div
                    className="absolute inset-y-0 left-0 bg-brand-light/80 rounded-full pointer-events-none"
                    style={{ width: `${scrubberProgress * 100}%` }}
                  />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-white shadow-sm pointer-events-none"
                    style={{ left: `${scrubberProgress * 100}%` }}
                  />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={handleTogglePlayback}
                  disabled={playDisabled}
                  className={cn(
                    "flex items-center gap-x-1.5 px-3 py-1.5 rounded bg-brand-background-light text-[10.5px] font-medium text-brand-light transition-colors",
                    playDisabled
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-brand-light/10 cursor-pointer",
                  )}
                >
                  {isPlaying ? (
                    <LuPause className="w-3.5 h-3.5" />
                  ) : (
                    <LuPlay className="w-3.5 h-3.5" />
                  )}
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <span className="text-brand-light/70 text-[10px] font-medium">
                  {rangeSummary}
                </span>
              </div>
            </div>
          )}
          {preprocessorRef && (
            <div className=" flex flex-row-reverse items-center gap-x-3 justify-between mt-3">
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
            timelineSelectorProps={{ mode: "range", inputId }}
            focusFrame={focusFrameForInput}
            setFocusFrame={(frame) => setInputFocusFrame(frame, inputId)}
            maxDuration={
              typeof maxDuration === "number"
                ? maxDuration
                : undefined
            }
            isPlayingExternal={isPlaying}
            onPlay={() => play(inputId)}
            onPause={() => pause(inputId)}
            canCrop={
              !isClipOnTimeline
            }
            selectionRange={selectedRangeTuple}
          />
        </div>
      </div>
    </Droppable>
  );
};

export default VideoInput;
