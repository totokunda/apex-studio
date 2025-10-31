import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDndMonitor } from '@dnd-kit/core';
import Droppable from '@/components/dnd/Droppable';
import { MdAudiotrack } from 'react-icons/md';
import { cn } from '@/lib/utils';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@radix-ui/react-tabs';
import { LuPause, LuPlay, LuSearch, LuUpload } from 'react-icons/lu';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { MediaItem, MediaThumb } from '@/components/media/Item';
import { getMediaInfo } from '@/lib/media/utils';
import { getPathForFile, listConvertedMedia } from '@app/preload';
import { ScrollArea } from '@/components/ui/scroll-area';
import TimelineSearch from './timeline/TimelineSearch';
import { useClipStore } from '@/lib/clip';
import { useAssetControlsStore } from '@/lib/assetControl';
import { AnyClipProps, AudioClipProps } from '@/lib/types';
import { AUDIO_EXTS, DEFAULT_FPS } from '@/lib/settings';
import { getLowercaseExtension, importMediaPaths, pickMediaPaths } from '@app/preload';
import { useViewportStore } from '@/lib/viewport';
import { useMediaLibraryVersion, bumpMediaLibraryVersion } from '@/lib/media/library';
import TimelineSelector from './timeline/TimelineSelector';
import { useInputControlsStore } from '@/lib/inputControl';
import { useControlsStore } from '@/lib/control';
import AudioPreview from '@/components/preview/clips/AudioPreview';

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

const PopoverAudio: React.FC<PopoverAudioProps> = ({ value, onChange, clipId }) => {
  const isUserInteractingRef = useRef(false);
  const [selectedTab, setSelectedTab] = useState<'timeline' | 'library'>('library');
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredMediaItems, setFilteredMediaItems] = useState<MediaItem[]>([]);
  const { clips } = useClipStore();
  const getClipById = useClipStore((s) => s.getClipById);
  const clearSelectedAsset = useAssetControlsStore((s) => s.clearSelectedAsset);
  const setSelectedAssetChangeHandler = useAssetControlsStore((s) => s.setSelectedAssetChangeHandler);
  const mediaLibraryVersion = useMediaLibraryVersion();
  const {fps} = useControlsStore();
  useEffect(() => {
    const next = mediaItems.filter((media) =>
      media.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    setFilteredMediaItems(next);
  }, [searchQuery, mediaItems]);

  useEffect(() => {
    (async () => {
      try {
        const list = await listConvertedMedia();
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
          .filter((media) => media.type === 'audio')
          .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
        setMediaItems(results);
      } catch {
        // Swallow errors; UI handles toasts elsewhere.
      }
    })();
  }, [mediaLibraryVersion]);

  const numEligibleTimelineAssets = useMemo(() => {
    return clips.filter((clip) => clip.type !== 'filter' && clip.type !== 'audio' && clip.clipId !== clipId).length;
  }, [clips, clipId]);

  const handleUpload = useCallback(async () => {
    try {
      const filters = [
        { name: 'Audio Files', extensions: AUDIO_EXTS },
      ];
      const picked = await pickMediaPaths({ directory: false, filters, title: 'Choose audio file(s) to import' });
      const paths = (picked ?? []).filter((p) => AUDIO_EXTS.includes(getLowercaseExtension(p)));
      if (paths.length === 0) return;
      const before = await listConvertedMedia();
      const existingNames = new Set(before.map((it) => it.name));
      await importMediaPaths(paths);
      const after = await listConvertedMedia();
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
        .filter((media) => media.type === 'audio')
        .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
      setMediaItems(results);

      const newlyAdded = results.filter((it) => !existingNames.has(it.name));
      if (newlyAdded.length > 0) {
        const first = newlyAdded[0];
        const durationFrames = Math.max(1, Math.floor((first.mediaInfo?.duration || 0) * fps));
        clearSelectedAsset();
        const clip: AudioClipProps = {
          type: 'audio',
          clipId: `media:${first.assetUrl}`,
          src: first.assetUrl,
          startFrame: 0,
          endFrame: Math.max(1, durationFrames),
        }
        onChange(clip);
      }
      bumpMediaLibraryVersion();
    } catch {
      // ignore upload errors here
    }
  }, [clearSelectedAsset, onChange, fps]);

  const assetSelectionHandler = React.useCallback((selectedClipId: string | null) => {
    if (!selectedClipId) {
      onChange(null);
      return;
    }
    const clip = getClipById(selectedClipId) as AnyClipProps | undefined;
    if (!clip || clip.type === 'audio') return;
    const clipDuration = Math.max(1, (clip.endFrame ?? 0) - (clip.startFrame ?? 0));
    onChange({ ...clip, startFrame: 0, endFrame: clipDuration, type:'audio' } as AudioClipProps);
  }, [getClipById, onChange]);

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
            const durationFrames = Math.max(1, Math.floor((media.mediaInfo?.duration || 0) * fps));
            return (
              <div
                key={media.name}
                onClick={() => {
                  clearSelectedAsset();
                  if (isSelected) {
                    onChange(null);
                  } else {
                    const clip: AudioClipProps = {
                      type: 'audio',
                      clipId: `media:${media.assetUrl}`,
                      src: media.assetUrl,
                      startFrame: 0,
                      endFrame: durationFrames,
                    } as any;
                    onChange(clip);
                  }
                }}
                className={cn(
                  'w-full flex flex-col items-center justify-center gap-y-1.5 cursor-pointer group relative'
                )}
              >
                <div className="relative">
                  <div
                    className={cn(
                      'absolute top-0 left-0 w-full h-full bg-brand-background-light/50 backdrop-blur-sm rounded-md z-20 group-hover:opacity-100 transition-all duration-200 flex items-center justify-center',
                      isSelected ? 'opacity-100' : 'opacity-0'
                    )}
                  >
                    <div
                      className={cn(
                        'rounded-full py-1 px-3 bg-brand-light/10 flex items-center justify-center font-medium text-[10.5px] w-fit',
                        isSelected ? 'bg-brand-light/20' : ''
                      )}
                    >
                      {isSelected ? 'Selected' : 'Use as Input'}
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
        'p-2 z-[90] dark h-full flex flex-col gap-y-3 border border-brand-light/10 rounded-[7px] font-poppins transition-all duration-150',
        selectedTab === 'timeline' ? 'w-[600px]' : 'w-96'
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
      <Tabs value={selectedTab} onValueChange={(val) => setSelectedTab(val as 'timeline' | 'library')}>
        <div className={cn('w-full flex flex-row items-center gap-x-2 justify-between')}>
          <TabsList
            className={cn(
              'w-full text-brand-light text-[10.5px] rounded font-medium text-start flex flex-row shadow overflow-hidden',
              numEligibleTimelineAssets === 0 ? 'justify-start' : 'justify-between cursor-pointer bg-brand-background-light'
            )}
          >
            <TabsTrigger
              value="library"
              className={cn(
                'w-full py-1.5 flex items-center',
                selectedTab === 'library' && numEligibleTimelineAssets > 0 ? 'bg-brand-accent-shade' : '',
                numEligibleTimelineAssets === 0 ? 'cursor-default justify-start px-2.5' : 'cursor-pointer justify-center px-4'
              )}
            >
              Media Library
            </TabsTrigger>
            <TabsTrigger
              hidden={numEligibleTimelineAssets === 0}
              value="timeline"
              className={cn(
                'px-4 w-full py-1.5 cursor-pointer flex items-center justify-center',
                selectedTab === 'timeline' ? 'bg-brand-accent-shade' : ''
              )}
            >
              Timeline Assets
            </TabsTrigger>
          </TabsList>
          <button
            onClick={handleUpload}
            className={cn(
              'w-fit h-full mr-2 flex flex-row items-center justify-center gap-x-1.5 bg-brand-background-light hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded py-1.5',
              numEligibleTimelineAssets === 0 ? 'px-5' : 'px-3'
            )}
          >
            <LuUpload className="w-3.5 h-3.5 text-brand-light" />
            <span className="text-brand-light text-[10.5px] font-medium">Upload</span>
          </button>
        </div>
        <TabsContent value="library">{renderMediaLibrary()}</TabsContent>
        <TabsContent value="timeline" className="outline-none">
          <TimelineSearch types={['audio']} excludeClipId={clipId || undefined} />
        </TabsContent>
      </Tabs>
    </PopoverContent>
  );
};

const AudioInput: React.FC<AudioInputProps> = ({ label, description, inputId, value, onChange, clipId, panelSize }) => {

  const stageContainerRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const [mediaClip, setMediaClip] = useState<AnyClipProps | null>(null);
  const [isOverDropZone, setIsOverDropZone] = useState(false);
  const externalDragCounterRef = useRef(0);
  
  const {
    clearSelectedAsset,
  } = useAssetControlsStore();


  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  
  
  const setInputFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
  const getSelectedRange = useInputControlsStore((s) => s.getSelectedRange);
  const setInputSelectedRange = useInputControlsStore((s) => s.setSelectedRange);
  const setInputFps = useInputControlsStore((s) => s.setFps);
  const { fpsByInputId, focusFrameByInputId, selectedRangeByInputId, setFocusFrame} = useInputControlsStore();
  const isPlaying = useInputControlsStore((s) => !!s.isPlayingByInputId[inputId]);

  const fpsForInput = fpsByInputId[inputId] ?? DEFAULT_FPS;
  const selectedRangeTuple = selectedRangeByInputId[inputId] ?? [0, 1];
  const focusFrameForInput = focusFrameByInputId[inputId] ?? 0;

  const {fps} = useControlsStore();
  const playbackStartTimestampRef = useRef<number | null>(null);
  const playbackLastFrameRef = useRef<number>(0);
  const lastSelectionKeyRef = useRef<string | null>(null);
  const durationCacheByClipIdRef = useRef<Record<string, number>>({});
  const fpsCacheByClipIdRef = useRef<Record<string, number>>({});

  const valueClipId = value?.clipId ?? null;
  const requestedStartFrame = useMemo(() => Math.max(0, Math.round(value?.startFrame ?? 0)), [value?.startFrame]);
  const requestedEndFrame = useMemo(
    () => Math.max(requestedStartFrame + 1, Math.round(value?.endFrame ?? requestedStartFrame + 1)),
    [requestedStartFrame, value?.endFrame]
  );

  const selectionKey = useMemo(() => {
    if (!value) return 'null';
    return value.clipId;
  }, [value]);


  const viewportRatio = useMemo(() => {
    const r = aspectRatio.width / aspectRatio.height;
    return Number.isFinite(r) && r > 0 ? r : 1;
  }, [aspectRatio.width, aspectRatio.height]);

  const emitSelection = useCallback((next: AudioSelection) => {
    onChange(next);
  }, [onChange]);


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

  const [rangeStartForInput, rangeEndForInput] = useMemo<[number, number]>(() => {
    const start = Math.max(0, Math.round(selectedRangeTuple?.[0] ?? 0));
    const endRaw = Math.max(start + 1, Math.round(selectedRangeTuple?.[1] ?? start + 1));
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
        ? `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
        : `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
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
      const isValid = !!data && data.type === 'audio';
      setIsOverDropZone(isValid && overId === 'audio-input');
    },
    onDragCancel: () => {
      setIsOverDropZone(false);
    },
    onDragEnd: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id as string | undefined;
      const isValid = !!data && data.type === 'audio';
      if (isValid && overId === 'audio-input') {
        clearSelectedAsset();
        void (async () => {
          try {
            const info = data.mediaInfo ?? await getMediaInfo(data.assetUrl);
            const durationFrames = Math.max(1, Math.floor((info?.duration || 0) * fps));
            const clip: AudioClipProps = {
              type: 'audio',
              clipId: `media:${data.assetUrl}`,
              src: data.assetUrl,
              startFrame: 0,
              endFrame: durationFrames,
            }
            emitSelection(clip);
          } catch {
            const clip: AudioClipProps = {
              type: 'audio',
              clipId: `media:${data?.assetUrl || ''}`,
              src: data?.assetUrl || '',
              startFrame: 0,
              endFrame: 1,
            }
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
      const before = await listConvertedMedia();
      const existingNames = new Set(before.map((it) => it.name));
      await importMediaPaths([path]);
      const after = await listConvertedMedia();
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
        .filter((media) => media.type === 'audio')
        .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
      const newlyAdded = results.filter((it) => !existingNames.has(it.name));
      if (newlyAdded.length > 0) {
        const first = newlyAdded[0];
        
        const durationFrames = Math.max(1, Math.floor((first.mediaInfo?.duration || 0) * fps));

        clearSelectedAsset();
        const clip: AudioClipProps = {
          type: 'audio',
          clipId: `media:${first.assetUrl}`,
          src: first.assetUrl,
          startFrame: 0,
          endFrame: durationFrames,
        }
        emitSelection(clip);
      }
      bumpMediaLibraryVersion();
    } catch {
      // ignore errors
    }
  };


  const lastSyncedSignatureRef = useRef<string | null>(null);
  const lastEmittedRangeRef = useRef<string | null>(null);

  const applyRangeAndTimeline = useCallback(
    (desiredStart: number, desiredEnd: number, opts?: { duration?: number; fps?: number; focus?: number }) => {
      const store = useInputControlsStore.getState();
      const [currentRangeStart, currentRangeEnd] = getSelectedRange(inputId);
      if (currentRangeStart !== desiredStart || currentRangeEnd !== desiredEnd) {
        setInputSelectedRange(desiredStart, desiredEnd, inputId);
      }

      if (typeof opts?.fps === 'number') {
        const currentFps = store.getFps(inputId);
        if (currentFps !== opts.fps) {
          setInputFps(opts.fps, inputId);
        }
      }
      if (typeof opts?.focus === 'number') {
        const currentFocus = store.getFocusFrame(inputId);
        if (currentFocus !== opts.focus) {
          setInputFocusFrame(opts.focus, inputId);
        }
      }
    },
    [inputId, setInputFocusFrame, setInputFps, setInputSelectedRange]
  );

  useEffect(() => {
    const signature = !value ? 'null' : `${selectionKey}:${requestedStartFrame}:${requestedEndFrame}`;
    if (!value) {
      lastSyncedSignatureRef.current = null;
      lastEmittedRangeRef.current = 'null';
      setMediaClip((prev) => (prev ? null : prev));
      applyRangeAndTimeline(0, 1, { duration: 1, fps: DEFAULT_FPS, focus: 0 });
      return;
    }
    if (lastSyncedSignatureRef.current === signature) {
      return;
    }
    lastSyncedSignatureRef.current = signature;
    lastEmittedRangeRef.current = `${requestedStartFrame}-${requestedEndFrame}:${selectionKey}`;

    const clip = value as AnyClipProps;
    const isMediaAudio = clip.type === 'audio' && typeof (clip as any)?.src === 'string' && String(clip.clipId || '').startsWith('media:');

    const applyWithDuration = (durationFrames: number, fpsToUse?: number) => {
      const desiredStart = Math.max(0, Math.min(durationFrames - 1, requestedStartFrame));
      const desiredEnd = Math.max(desiredStart + 1, Math.min(durationFrames, requestedEndFrame));
      const preview: AnyClipProps = { ...clip, startFrame: 0, endFrame: durationFrames } as AnyClipProps;
      setMediaClip((prev) => {
        if (
          prev &&
          prev.clipId === preview.clipId &&
          (prev.startFrame ?? 0) === (preview.startFrame ?? 0) &&
          (prev.endFrame ?? 0) === (preview.endFrame ?? 0)
        ) {
          return prev;
        }
        return preview;
      });
      applyRangeAndTimeline(desiredStart, desiredEnd, {
        duration: durationFrames,
        fps: typeof fpsToUse === 'number' ? fpsToUse : fps,
        focus: desiredStart,
      });
    };

    if (isMediaAudio) {
      const cachedDur = durationCacheByClipIdRef.current[clip.clipId];
      const cachedFps = fpsCacheByClipIdRef.current[clip.clipId];
      if (typeof cachedDur === 'number' && cachedDur > 0 && typeof cachedFps === 'number' && cachedFps > 0) {
        applyWithDuration(cachedDur, cachedFps);
        return;
      }
      void (async () => {
        try {
          const info = await getMediaInfo((clip as any).src as string);
          const durationFrames = Math.max(1, Math.floor((info?.duration || 0) * fps));
          durationCacheByClipIdRef.current[clip.clipId] = durationFrames;
          fpsCacheByClipIdRef.current[clip.clipId] = fps;
          applyWithDuration(durationFrames, fps);
        } catch {
          const fallback = Math.max(1, Math.round((clip.endFrame ?? 0) - (clip.startFrame ?? 0)));
          applyWithDuration(fallback, fps);
        }
      })();
      return;
    }

    const durationFrames = Math.max(1, Math.round((clip.endFrame ?? 0) - (clip.startFrame ?? 0)));
    applyWithDuration(durationFrames, fps);
  }, [value, selectionKey, requestedStartFrame, requestedEndFrame, applyRangeAndTimeline, fps]);


  useEffect(() => {
    const prevKey = lastSelectionKeyRef.current;
    lastSelectionKeyRef.current = selectionKey;
    if (!value) return;
    if (prevKey !== selectionKey) return;
    const normalizedStart = rangeStartForInput;
    const normalizedEnd = rangeEndForInput;
    const rangeKey = `${normalizedStart}-${normalizedEnd}:${selectionKey}`;
    if (rangeKey === lastEmittedRangeRef.current) return;
    if (normalizedStart === requestedStartFrame && normalizedEnd === requestedEndFrame) return;
    lastEmittedRangeRef.current = rangeKey;
    emitSelection({
      ...(value as AnyClipProps),
      startFrame: normalizedStart,
      endFrame: normalizedEnd,
    } as AudioClipProps);
  }, [
    rangeStartForInput,
    rangeEndForInput,
    selectionKey,
    valueClipId,
    emitSelection,
    requestedStartFrame,
    requestedEndFrame,
  ]);


  useEffect(() => {
    const [rangeStart, rangeEnd] = selectedRangeTuple;
    const clampedFocus = Math.max(rangeStart, Math.min(rangeEnd - 1, Math.round(focusFrameForInput ?? rangeStart)));
    if (clampedFocus !== focusFrameForInput) {
      setInputFocusFrame(clampedFocus, inputId);
    }
  }, [selectedRangeTuple, focusFrameForInput, inputId, setInputFocusFrame]);

  const previewClip = useMemo<AnyClipProps | null>(() => {
    if (!value) return null;
    return mediaClip ?? (value as AnyClipProps);
  }, [value, mediaClip]);

  const handleTogglePlayback = useCallback(() => {
    if (!previewClip) return;
    if (rangeEndForInput <= rangeStartForInput) {
      setFocusFrame(rangeStartForInput, inputId);
      return;
    }
    const store = useInputControlsStore.getState();
    if (isPlaying) {
      store.pause(inputId);
      return;
    }
    playbackStartTimestampRef.current = null;
    playbackLastFrameRef.current = rangeStartForInput;
    setFocusFrame(rangeStartForInput, inputId);
    store.play(inputId);
  }, [previewClip, isPlaying, rangeStartForInput, rangeEndForInput, inputId, setFocusFrame]);

  const showTimeline = Boolean(previewClip && (previewClip.type === 'audio' || previewClip.type === 'group'));

  const playDisabled = !previewClip || rangeEndForInput <= rangeStartForInput;

  return (
    <Droppable className="w-full h-full" id="audio-input" accepts={['media']}>
      <div className="flex flex-col items-start w-full gap-y-1 min-w-0 bg-brand rounded-[7px] border border-brand-light/5 h-auto">
        <div className="w-full h-full flex flex-col items-start justify-start p-3">
          <div className="w-full flex flex-col items-start justify-start mb-3">
            <div className="w-full flex flex-col">
              <div className="flex flex-col items-start justify-start">
                {label && <label className="text-brand-light text-[10.5px] w-full font-medium text-start">{label}</label>}
                {description && <span className="text-brand-light/80 text-[9.5px] w-full text-start">{description}</span>}
              </div>

            </div>
          </div>
          <Popover>
            <PopoverTrigger className="w-full">
              <div
                ref={stageContainerRef}
                onDragEnter={handleExternalDragEnter}
                onDragOver={handleExternalDragOver}
                onDragLeave={handleExternalDragLeave}
                onDrop={handleExternalDrop}
                
                className={cn(
                  'w-full flex flex-col items-center justify-center gap-y-3 shadow-accent hover:opacity-70 cursor-pointer relative overflow-hidden',
                  value ? '' : 'border-dashed',
                  value ? '' : 'p-4 border-brand-light/10 border bg-brand-background-light/50'
                )}
              >
                {isOverDropZone && (
                  <div className="absolute inset-0 z-30 bg-brand-background-light/40 backdrop-blur-sm pointer-events-none transition-opacity duration-150" />
                )}
                {value ? (
                  stageSize.w > 0 && stageSize.h > 0 ? (
                    previewClip ? (
                     <AudioPreview key={previewClip.clipId} {...(previewClip as AudioClipProps)} overrideClip={previewClip as AudioClipProps} inputMode={true} inputId={inputId} overlap={true} rectWidth={stageSize.w} rectHeight={stageSize.h} />
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
                  <>
                    <MdAudiotrack className="w-10 h-10 text-brand-light" />
                    <span className="text-brand-light text-[11px] w-full text-center font-medium">
                      Click or drag and drop a audio file here.
                    </span>
                  </>
                )}
              </div>
            </PopoverTrigger>
            <PopoverAudio value={value} onChange={emitSelection} clipId={clipId} />
          </Popover>
          {showTimeline && value && (
            <div className="w-full flex flex-col gap-y-2 mt-3">
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={handleTogglePlayback}
                  disabled={playDisabled}
                  className={cn(
                    'flex items-center gap-x-1.5 px-3 py-1.5 rounded bg-brand-background-light text-[10.5px] font-medium text-brand-light transition-colors',
                    playDisabled
                      ? 'opacity-50 cursor-not-allowed'
                      : 'hover:bg-brand-light/10 cursor-pointer'
                  )}
                >
                  {isPlaying ? <LuPause className="w-3.5 h-3.5" /> : <LuPlay className="w-3.5 h-3.5" />}
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <span className="text-brand-light/70 text-[10px] font-medium">{rangeSummary}</span>
              </div>
              <TimelineSelector
                inputId={inputId}
                clip={previewClip as AnyClipProps}
                width={Math.max(1, stageSize.w)}
                height={44}
                mode="range"
              />
            </div>
          )}
        </div>
      </div>
    </Droppable>
  );
};

export default AudioInput;
