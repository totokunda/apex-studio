import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDndMonitor } from "@dnd-kit/core";
import Droppable from "@/components/dnd/Droppable";
import { MdAudiotrack } from "react-icons/md";
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
import { getPathForFile, listConvertedMedia } from "@app/preload";
import { ScrollArea } from "@/components/ui/scroll-area";
import TimelineSearch from "./timeline/TimelineSearch";
import { useClipStore } from "@/lib/clip";
import { useAssetControlsStore } from "@/lib/assetControl";
import { useProjectsStore } from "@/lib/projects";
import { AnyClipProps, AudioClipProps } from "@/lib/types";
import { AUDIO_EXTS, DEFAULT_FPS } from "@/lib/settings";
import {
  getLowercaseExtension,
  importMediaPaths,
  pickMediaPaths,
} from "@app/preload";
import { useViewportStore } from "@/lib/viewport";
import {
  useMediaLibraryVersion,
  bumpMediaLibraryVersion,
} from "@/lib/media/library";
import { useInputControlsStore } from "@/lib/inputControl";
import { useControlsStore } from "@/lib/control";
import AudioPreview from "@/components/preview/clips/AudioPreview";
import { VIDEO_EXTS } from "@/lib/settings";
import { TbEdit } from "react-icons/tb";
import { MediaDialog } from "@/components/dialogs/MediaDialog";
import { CircularAudioVisualizer } from "./CircularAudioVisualizer";

const isVideo = (path: string) => {
  const ext = getLowercaseExtension(path);
  return VIDEO_EXTS.includes(ext);
};

export type AudioSelection = AudioClipProps | null;

interface AudioInputProps {
  label?: string;
  description?: string;
  inputId: string;
  value: AudioSelection;
  onChange: (value: AudioSelection) => void;
  clipId: string;
  panelSize: number;
}

interface PopoverAudioProps {
  value: AudioSelection;
  onChange: (value: AudioSelection) => void;
  clipId: string | null;
}

const PopoverAudio: React.FC<PopoverAudioProps> = ({
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
  const addAsset = useClipStore((s) => s.addAsset);
  const setSelectedAssetChangeHandler = useAssetControlsStore(
    (s) => s.setSelectedAssetChangeHandler,
  );
  const mediaLibraryVersion = useMediaLibraryVersion();
  const { fps } = useControlsStore();
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
          .filter((media) => {
            if (media.type === "audio") return true;
            const mediaInfo = media.mediaInfo;
            // if has audio track, return true
            if (mediaInfo?.audio) {
              void getMediaInfo(media.assetUrl + "#audio");
              return true;
            }
            return false;
          })
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
      (clip) => clip.type === "audio" && clip.clipId !== clipId && !clip.hidden,
    ).length;
  }, [clips, clipId]);

  const handleUpload = useCallback(async () => {
    try {
      const filters = [{ name: "Audio Files", extensions: AUDIO_EXTS }];
      const picked = await pickMediaPaths({
        directory: false,
        filters,
        title: "Choose audio file(s) to import",
      });
      const paths = (picked ?? []).filter((p) =>
        AUDIO_EXTS.includes(getLowercaseExtension(p)),
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
        .filter((media) => media.type === "audio")
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
        let url = first.assetUrl + (isVideo(first.assetUrl) ? "#audio" : "");
        const asset = addAsset({ path: url, modelInputAsset: true });
        const clip: AudioClipProps = {
          type: "audio",
          clipId: `media:${first.assetUrl}`,
          assetId: asset.id,
          startFrame: 0,
          endFrame: Math.max(1, durationFrames),
        };
        onChange(clip);
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
      if (!clip) return;
      const clipDuration = Math.max(
        1,
        (clip.endFrame ?? 0) - (clip.startFrame ?? 0),
      );
      if (clip.type === "audio") {
        onChange({
          ...(clip as AnyClipProps),
          startFrame: 0,
          endFrame: clipDuration,
          type: "audio",
        } as AudioClipProps);
        return;
      }
      onChange({
        ...clip,
        startFrame: 0,
        endFrame: clipDuration,
        type: "audio",
      } as AudioClipProps);
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
            placeholder="Search for audio"
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
                    const url =
                      media.assetUrl +
                      (isVideo(media.assetUrl) ? "#audio" : "");
                    const asset = addAsset({
                      path: url,
                      modelInputAsset: true,
                    });
                    const clip: AudioClipProps = {
                      type: "audio",
                      clipId: `media:${media.assetUrl}`,
                      assetId: asset.id,
                      startFrame: 0,
                      endFrame: durationFrames,
                    } as any;
                    onChange(clip);
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
        "p-2 z-[90] dark h-full flex flex-col gap-y-3 border border-brand-light/10 rounded-[7px] font-poppins transition-all duration-150",
        selectedTab === "timeline" ? "w-[600px]" : "w-96",
      )}
      onOpenAutoFocus={() => {
        isUserInteractingRef.current = true;
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
            types={["audio"]}
            excludeClipId={clipId || undefined}
          />
        </TabsContent>
      </Tabs>
    </PopoverContent>
  );
};

const AudioInput: React.FC<AudioInputProps> = ({
  label,
  description,
  inputId,
  value,
  onChange,
  clipId,
  panelSize,
}) => {
  const stageContainerRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState<{ w: number; h: number }>({
    w: 0,
    h: 0,
  });
  const [mediaClip, setMediaClip] = useState<AnyClipProps | null>(null);
  const [isOverDropZone, setIsOverDropZone] = useState(false);
  const externalDragCounterRef = useRef(0);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const { clearSelectedAsset } = useAssetControlsStore();

  const aspectRatio = useViewportStore((s) => s.aspectRatio);

  const setInputFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
  // Simplified: no programmatic range writes; selector manages its own range
  const getClipById = useClipStore((s) => s.getClipById);
  const setInputFps = useInputControlsStore((s) => s.setFps);
  const {
    fpsByInputId,
    focusFrameByInputId,
    selectedRangeByInputId,
    setFocusFrame,
    setSelectedRange,
    play,
    pause,
  } = useInputControlsStore();
  const isPlaying = useInputControlsStore(
    (s) => !!s.isPlayingByInputId[inputId],
  );

  const fpsForInput = fpsByInputId[inputId] ?? DEFAULT_FPS;
  const selectedRangeTuple = selectedRangeByInputId[inputId] ?? [0, 1];
  const focusFrameForInput = focusFrameByInputId[inputId] ?? 0;

  const { fps } = useControlsStore();
  const playbackStartTimestampRef = useRef<number | null>(null);
  const playbackLastFrameRef = useRef<number>(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const scrubberTrackRef = useRef<HTMLDivElement | null>(null);
  const addAsset = useClipStore((s) => s.addAsset);
  const getActiveProject = useProjectsStore((s) => s.getActiveProject);
  // Initialize input fps synchronously before first render to avoid slow playback.
  // Use a ref to avoid depending on the unstable setter reference.
  const setInputFpsRef = React.useRef(setInputFps);
  React.useEffect(() => {
    setInputFpsRef.current = setInputFps;
  }, [setInputFps]);
  React.useLayoutEffect(() => {
    setInputFpsRef.current(fps, inputId);
  }, [fps, inputId]);

  const selectionKey = useMemo(() => {
    if (!value) return "null";
    return value.clipId;
  }, [value]);

  // Simplified: no non-range signature; preview reads latest from store as needed

  const viewportRatio = useMemo(() => {
    const r = 16 / 9;
    return Number.isFinite(r) && r > 0 ? r : 1;
  }, [aspectRatio.width, aspectRatio.height]);

  const emitSelection = useCallback(
    (next: AudioSelection) => {
      onChange(next);
    },
    [onChange],
  );
  useEffect(() => {
    const el = stageContainerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      const width = Math.max(1, panelSize);
      const height = Math.max(1, width / viewportRatio);
      setStageSize({ w: width, h: height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [panelSize, viewportRatio]);

  useEffect(() => {
    setStageSize((prev) => ({
      w: Math.max(1, panelSize),
      h: Math.max(1, panelSize / viewportRatio || prev.h),
    }));
  }, [panelSize, viewportRatio]);

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

  const rangeSummary = useMemo(() => {
    const start = rangeStartForInput;
    const endDisplay = Math.max(start, rangeEndForInput - 1);
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
      const isValid =
        !!data &&
        (data.type === "audio" ||
          (data.type === "video" &&
            data.mediaInfo?.audio !== undefined &&
            data.mediaInfo?.audio !== null));
      setIsOverDropZone(isValid && overId === "audio-input");
    },
    onDragCancel: () => {
      setIsOverDropZone(false);
    },
    onDragEnd: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id as string | undefined;
      const isValid =
        !!data &&
        (data.type === "audio" ||
          (data.type === "video" &&
            data.mediaInfo?.audio !== undefined &&
            data.mediaInfo?.audio !== null));
      if (!isValid) return;
      if (isValid && overId === "audio-input") {
        clearSelectedAsset();
        void (async () => {
          try {
            const info = data.mediaInfo ?? (await getMediaInfo(data.assetUrl));
            const durationFrames = Math.max(
              1,
              Math.floor((info?.duration || 0) * fps),
            );
            let url = data.assetUrl + (isVideo(data.assetUrl) ? "#audio" : "");
            const asset = addAsset({ path: url, modelInputAsset: true });
            const clip: AudioClipProps = {
              type: "audio",
              clipId: `media:${data.assetUrl}`,
              assetId: asset.id,
              startFrame: 0,
              endFrame: durationFrames,
            };
            emitSelection(clip);
          } catch {
            let url =
              data?.assetUrl + (isVideo(data?.assetUrl || "") ? "#audio" : "");
            const asset = addAsset({ path: url, modelInputAsset: true });
            const clip: AudioClipProps = {
              type: "audio",
              clipId: `media:${data?.assetUrl || ""}`,
              assetId: asset.id,
              startFrame: 0,
              endFrame: 1,
            };
            emitSelection(clip);
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
    if (!AUDIO_EXTS.includes(ext)) return;
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
        .filter((media) => media.type === "audio")
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
        let url = first.assetUrl + (isVideo(first.assetUrl) ? "#audio" : "");
        const asset = addAsset({ path: url, modelInputAsset: true });
        const clip: AudioClipProps = {
          type: "audio",
          clipId: `media:${first.assetUrl}`,
          assetId: asset.id,
          startFrame: 0,
          endFrame: durationFrames,
        };
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
      setMediaClip(live ?? null);
      return;
    }
    // media asset: compute duration once for preview
    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo((clip as AudioClipProps).assetId as string); 
        const durationFrames = Math.max(
          1,
          Math.floor((info?.duration || 0) * fps),
        );
        if (!cancelled)
          setMediaClip({
            ...clip,
            endFrame: durationFrames,
            startFrame: 0,
          } as AnyClipProps);
      } catch {
        if (!cancelled) setMediaClip(clip);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [value, getClipById, fps]);

  // No emission of start/end from range changes; we fetch latest clip from store when needed

  useEffect(() => {
    const [rangeStart, rangeEnd] = selectedRangeTuple;
    const clampedFocus = Math.max(
      rangeStart,
      Math.min(rangeEnd, Math.round(focusFrameForInput ?? rangeStart)),
    );
    if (clampedFocus !== focusFrameForInput && !isPlaying) {
      setInputFocusFrame(clampedFocus, inputId);
    }
  }, [
    selectedRangeTuple?.[0],
    selectedRangeTuple?.[1],
    focusFrameForInput,
    inputId,
    setInputFocusFrame,
    isPlaying,
    mediaClip?.clipId,
  ]);

  const previewClip = useMemo<AnyClipProps | null>(() => {
    if (!value) return null;
    return mediaClip ?? (value as AnyClipProps);
  }, [value, mediaClip]);

  // Clear selection if referenced timeline clip is deleted
  const liveTimelineClip = useClipStore((s) => {
    if (!value) return null;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return null;
    return (s.getClipById(cid) as AnyClipProps | undefined) ?? null;
  });
  useEffect(() => {
    if (!value) return;
    const cid = String(value.clipId || "");
    if (cid.startsWith("media:")) return;
    if (!liveTimelineClip) {
      emitSelection(null);
      return;
    }
    if ((liveTimelineClip as AnyClipProps).hidden) {
      emitSelection(null);
      return;
    }
  }, [value, liveTimelineClip, emitSelection]);

  // Reset selected range only when the selected clipId actually changes
  const lastClipSignatureRef = useRef<string | null>(null);
  useEffect(() => {
    if (!previewClip) return;
    const clipStart = Math.max(0, Math.round(previewClip.startFrame ?? 0));
    const clipEnd = Math.max(
      clipStart + 1,
      Math.round(previewClip.endFrame ?? clipStart + 1),
    );
    const span = Math.max(1, clipEnd - clipStart);
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
    const isAlreadyFull = curStart === 0 && curEnd === span;
    if (!isAlreadyFull) {
      setSelectedRange(0, span, inputId);
    }
  }, [
    previewClip?.clipId,
    previewClip?.startFrame,
    previewClip?.endFrame,
    inputId,
    setSelectedRange,
    selectedRangeTuple?.[0],
    selectedRangeTuple?.[1],
  ]);

  // Ensure selectedRange is always valid (min 1 frame, within [clip.start, clip.end])
  useEffect(() => {
    if (!previewClip) return;
    const clipStart = Math.max(0, Math.round(previewClip.startFrame ?? 0));
    const clipEnd = Math.max(
      clipStart + 1,
      Math.round(previewClip.endFrame ?? clipStart + 1),
    );
    const span = Math.max(1, clipEnd - clipStart);
    let curStart = Math.round(selectedRangeTuple?.[0] ?? 0);
    let curEnd = Math.round(selectedRangeTuple?.[1] ?? curStart + 1);
    const desiredStart = Math.max(0, Math.min(span - 1, curStart));
    const desiredEnd = Math.max(desiredStart + 1, Math.min(span, curEnd));
    if (desiredStart !== curStart || desiredEnd !== curEnd) {
      setSelectedRange(desiredStart, desiredEnd, inputId);
    }
  }, [
    previewClip,
    selectedRangeTuple?.[0],
    selectedRangeTuple?.[1],
    setSelectedRange,
    inputId,
  ]);

  // Default selectedRange to the full duration on first load (store default [0,1])
  useEffect(() => {
    if (!previewClip) return;
    if (!Array.isArray(selectedRangeTuple)) return;
    const isDefault =
      selectedRangeTuple[0] === 0 && selectedRangeTuple[1] === 1;
    if (!isDefault) return;
    const clipStart = Math.max(0, Math.round(previewClip.startFrame ?? 0));
    const clipEnd = Math.max(
      clipStart + 1,
      Math.round(previewClip.endFrame ?? clipStart + 1),
    );
    const span = Math.max(1, clipEnd - clipStart);
    setSelectedRange(0, span, inputId);
  }, [
    previewClip,
    selectedRangeTuple?.[0],
    selectedRangeTuple?.[1],
    setSelectedRange,
    inputId,
  ]);

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
    (previewClip.type === "audio" || previewClip.type === "group"),
  );

  const playDisabled = !previewClip || rangeEndForInput <= rangeStartForInput;

  const handleDialogConfirm = useCallback(
    (_data: {
      rotation: number;
      aspectRatio: string;
      crop?: { x: number; y: number; width: number; height: number };
      transformWidth?: number;
      transformHeight?: number;
      transformX?: number;
      transformY?: number;
    }) => {
      // Audio does not currently apply spatial transforms; selection range is edited via TimelineSelector inside MediaDialog.
    },
    [],
  );

  return (
    <Droppable className="w-full h-full" id="audio-input" accepts={["media"]}>
      <div className="flex flex-col items-start w-full gap-y-1 min-w-0 bg-brand rounded-[7px] border border-brand-light/5 h-auto">
        <div className="w-full h-full flex flex-col items-start justify-start p-3">
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
                style={{ height: value ? Math.max(1, stageSize.h) : undefined }}
                className={cn(
                  "w-full flex flex-col items-center justify-center gap-y-3 shadow-accent cursor-pointer relative overflow-hidden group",
                  value ? "" : "border-dashed",
                  value
                    ? ""
                    : "p-4 border-brand-light/10 border bg-brand-background-light/50 rounded",
                )}
              >
                {isOverDropZone && (
                  <div className="absolute inset-0 z-30 bg-brand-background-light/40 backdrop-blur-sm pointer-events-none transition-opacity duration-150" />
                )}
                {value ? (
                  stageSize.w > 0 && stageSize.h > 0 ? (
                    previewClip ? (
                      <>
                        <AudioPreview
                          key={previewClip.clipId}
                          {...(previewClip as AudioClipProps)}
                          overrideClip={
                            previewClip as AnyClipProps as AudioClipProps
                          }
                          inputMode={true}
                          inputId={inputId}
                          overlap={true}
                          rectWidth={stageSize.w}
                          rectHeight={stageSize.h}
                          disabled={isDialogOpen}
                        />
                        <CircularAudioVisualizer
                          inputId={inputId}
                          width={stageSize.w}
                          height={stageSize.h}
                          active={isPlaying}
                        />
                      </>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-brand-light/70 text-[12px]">
                        Unable to preview audio file.
                      </div>
                    )
                  ) : (
                    <div className="w-full h-32 flex items-center justify-center text-brand-light/70 text-[10px]">
                      Preparing preview...
                    </div>
                  )
                ) : (
                  <div
                    style={{ height: Math.max(1, stageSize.h - 40) }}
                    className="flex flex-col items-center justify-center gap-y-2"
                  >
                    <MdAudiotrack className="w-1/4 h-1/4 text-brand-light " />
                    <span className="text-brand-light text-[11px] w-full text-center font-medium">
                      Click or drag and drop a audio file here.
                    </span>
                  </div>
                )}
                {value && (
                  <div className="absolute bottom-0 h-full left-0 right-0 z-50 bg-brand/60 hover:backdrop-blur-sm gap-x-4 transition-opacity duration-150 opacity-0 hover:opacity-100 flex items-center justify-center">
                    <div className="flex flex-col justify-center items-center gap-y-2 p-4 rounded-md w-full">
                      <button
                        onClick={() => setIsPopoverOpen(true)}
                        className="z-30 duration-150 flex items-center gap-x-2 px-6 py-2 w-40 bg-brand-light hover:bg-brand-light/90 shadow-md justify-center rounded text-[10.5px] font-poppins font-medium text-brand-accent-two-shade transition-colors"
                      >
                        <MdAudiotrack className="w-4 h-4" />
                        <span>Select Media</span>
                      </button>
                      <button
                        onClick={() => setIsDialogOpen(true)}
                        className="z-30 duration-150 w-40 justify-center shadow-md flex items-center gap-x-2 px-6 py-2 font-poppins font-medium bg-brand-accent-two-shade hover:bg-brand-accent-two-shade/90 rounded text-[10.5px] text-brand-light transition-colors"
                      >
                        <TbEdit className="w-4 h-4" />
                        Edit Audio
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </PopoverTrigger>
            <PopoverAudio
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
          <MediaDialog
            isOpen={isDialogOpen}
            onClose={() => setIsDialogOpen(false)}
            onConfirm={handleDialogConfirm}
            clipOverride={mediaClip}
            timelineSelectorProps={{ mode: "range", inputId }}
            focusFrame={focusFrameForInput}
            setFocusFrame={(frame) => setInputFocusFrame(frame, inputId)}
            canCrop={false}
            isPlayingExternal={isPlaying}
            onPlay={() => play(inputId)}
            onPause={() => pause(inputId)}
            selectionRange={selectedRangeTuple as [number, number]}
          />
        </div>
      </div>
    </Droppable>
  );
};

export default AudioInput;
