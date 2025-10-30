import React, { useEffect, useMemo, useRef, useState     } from 'react'
import { useDndMonitor } from '@dnd-kit/core'
import Droppable from '@/components/dnd/Droppable';
import { IoImageOutline } from "react-icons/io5";
import { cn } from '@/lib/utils';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@radix-ui/react-tabs';
import { LuSearch, LuUpload } from 'react-icons/lu';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { MediaItem, MediaThumb } from '@/components/media/Item';
import { getMediaInfo } from '@/lib/media/utils';
import { listConvertedMedia } from '@app/preload';
import { ScrollArea } from '@/components/ui/scroll-area';
import TimelineSearch from './timeline/TimelineSearch';
import { useClipStore } from '@/lib/clip';
import { useAssetControlsStore } from '@/lib/assetControl';
import TimelineClipPosterPreview from './TimelineClipPosterPreview';
import { AnyClipProps, ImageClipProps, VideoClipProps } from '@/lib/types';
import { VIDEO_EXTS, IMAGE_EXTS, DEFAULT_FPS } from '@/lib/settings';
import { getLowercaseExtension, pickMediaPaths, importMediaPaths, getPathForFile } from '@app/preload';
import { useViewportStore } from '@/lib/viewport';
import { useMediaLibraryVersion, bumpMediaLibraryVersion } from '@/lib/media/library';
import TimelineSelector from './timeline/TimelineSelector';
import { useInputControlsStore } from '@/lib/inputControl';
import { usePreprocessorsListStore } from '@/lib/preprocessor/list-store';

type ImageSelectionMedia = { kind: 'media'; assetUrl: string; frame?: number };
type ImageSelectionClip = { kind: 'clip'; clipId: string; frame?: number };
export type ImageSelection = ImageSelectionMedia | ImageSelectionClip | null;

interface ImageInputProps {
  label?: string;
  description?: string;
  inputId: string;
  value: ImageSelection;
  onChange: (value: ImageSelection) => void;
  clipId: string;
  panelSize: number;
  preprocessorRef?: string;
  preprocessorName?: string;
  applyPreprocessorInitial?: boolean;
  onChangeComposite?: (value: { selection: ImageSelection; preprocessor_ref?: string; preprocessor_name?: string; apply_preprocessor?: boolean }) => void;
}

interface PopoverImageProps {
    value: ImageSelection;
    onChange: (value: ImageSelection) => void;
    clipId: string | null;
}

const PopoverImage: React.FC<PopoverImageProps> = ({ value, onChange, clipId }) => {
    const isUserInteractingRef = useRef(false);
    const [selectedTab, setSelectedTab] = useState<'timeline' | 'library'>('library');
    const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [filteredMediaItems, setFilteredMediaItems] = useState<MediaItem[]>([]);
    // media cache not needed for list sync here; we subscribe to library version instead
    const {clips} = useClipStore();
    const getClipById = useClipStore((s) => s.getClipById);
    const clearSelectedAsset = useAssetControlsStore((s) => s.clearSelectedAsset);
  const setSelectedAssetChangeHandler = useAssetControlsStore((s) => s.setSelectedAssetChangeHandler);
    useEffect(() => {
        setFilteredMediaItems(mediaItems.filter((media) => media.name.toLowerCase().includes(searchQuery.toLowerCase())));
    }, [searchQuery, mediaItems]);

    const mediaLibraryVersion = useMediaLibraryVersion();

    useEffect(() => {
        (async () => {
        const list = await listConvertedMedia();
        const infoPromises = list.map(it => getMediaInfo(it.assetUrl));
        const infos = await Promise.all(infoPromises);
        let results: MediaItem[] = list.map((it, idx) => ({
          name: it.name,
          type: it.type,
          absPath: it.absPath,
          assetUrl: it.assetUrl,
          dateAddedMs: it.dateAddedMs,
          mediaInfo: infos[idx],
          hasProxy: it.hasProxy
        }));
        results = results.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase())).filter((media) => (media.type === 'image' || media.type === 'video'));
        setMediaItems(results);
        })();
    }, [mediaLibraryVersion]);

    const handleUpload = async () => {
        try {
            const filters = [
                { name: 'Image/Video Files', extensions: VIDEO_EXTS.concat(IMAGE_EXTS) },
                { name: 'Video Files', extensions: VIDEO_EXTS },
                { name: 'Image Files', extensions: IMAGE_EXTS },
            ];
            const picked = await pickMediaPaths({ directory: false, filters, title: 'Choose image/video file(s) to import' });
            const paths = (picked ?? []).filter((p) => {
                const ext = getLowercaseExtension(p);
                return VIDEO_EXTS.includes(ext) || IMAGE_EXTS.includes(ext);
            });
            if (paths.length === 0) return;
            const existingNames = new Set(mediaItems.map(it => it.name));
            await importMediaPaths(paths);
            const list = await listConvertedMedia();
            const infoPromises = list.map(it => getMediaInfo(it.assetUrl));
            const infos = await Promise.all(infoPromises);
            let results: MediaItem[] = list.map((it, idx) => ({
              name: it.name,
              type: it.type,
              absPath: it.absPath,
              assetUrl: it.assetUrl,
              dateAddedMs: it.dateAddedMs,
              mediaInfo: infos[idx],
              hasProxy: it.hasProxy
            }));
            results = results.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase())).filter((media) => (media.type === 'image' || media.type === 'video'));
            setMediaItems(results);
            const newlyAdded = results.filter(it => !existingNames.has(it.name));
            if (newlyAdded.length > 0) {
                const first = newlyAdded[0];
                clearSelectedAsset();
                onChange({ kind: 'media', assetUrl: first.assetUrl, frame: 0 });
            }
            bumpMediaLibraryVersion();
        } catch (e) {
            // swallow errors here; MediaMenu handles toasts, but we keep silent in this compact picker
        }
    };

    const numEligibleTimelineAssets = useMemo(() => {
        return clips.filter((clip) => (clip.type !== 'filter' && clip.type !== 'audio' && clip.clipId !== clipId)).length;
    }, [clips])

    // Direct change handler so timeline selection/deselection can update this input
    const assetSelectionHandler = React.useCallback((clipId: string | null) => {
        if (!clipId) {
            onChange(null);
            return;
        }
        const selectedClip = getClipById(clipId);
        if (!selectedClip) return;
        onChange({ kind: 'clip', clipId: selectedClip.clipId, frame: 0 });
    }, [getClipById, onChange]);

    return (
        <PopoverContent 
            side='left' 
            align='start'
            sideOffset={20} 
            className={cn('p-2 z-[90] dark h-full flex flex-col gap-y-3 border border-brand-light/10 rounded-[7px] font-poppins transition-all duration-150', selectedTab === 'timeline' ? 'w-[600px]' : 'w-96')}
            onOpenAutoFocus={() => { isUserInteractingRef.current = true; setSelectedAssetChangeHandler(assetSelectionHandler); }} onCloseAutoFocus={() => { isUserInteractingRef.current = false; setSelectedAssetChangeHandler(null); }}>
                <Tabs className='' value={selectedTab} onValueChange={(value) => setSelectedTab(value as 'timeline' | 'library')}>
                    <div className={cn('w-full flex flex-row items-center gap-x-2 justify-between')}>
                <TabsList className={cn('w-full  text-brand-light text-[10.5px] rounded font-medium text-start flex flex-row shadow overflow-hidden',
                    numEligibleTimelineAssets === 0 ? 'justify-start' : 'justify-between  cursor-pointer  bg-brand-background-light'
                )}>
                    <TabsTrigger  value="library" className={cn(' w-full py-1.5 flex items-center', selectedTab === 'library' && numEligibleTimelineAssets > 0 ? 'bg-brand-accent-shade' : '',
                    numEligibleTimelineAssets === 0 ? 'cursor-default justify-start px-2.5' : 'cursor-pointer   justify-center px-4' ,

                    )}>
                        Media Library
                    </TabsTrigger>
                    <TabsTrigger hidden={numEligibleTimelineAssets === 0}  value="timeline" className={cn('px-4 w-full py-1.5 cursor-pointer flex items-center justify-center', selectedTab === 'timeline' ? 'bg-brand-accent-shade' : '')}>
                        Timeline Assets
                    </TabsTrigger>
                </TabsList>
                <button onClick={handleUpload} className={cn('w-fit h-full mr-2 flex flex-row items-center justify-center gap-x-1.5 bg-brand-background-light hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded py-1.5', 
                    numEligibleTimelineAssets === 0 ? 'px-5' : 'px-3'

                )}>
                        <LuUpload className='w-3.5 h-3.5 text-brand-light' />
                        <span className='text-brand-light text-[10.5px] font-medium'>Upload</span>
                    </button>
                </div>
                <TabsContent value="library" className='w-full h-full flex flex-col py-2 gap-y-2 outline-none'>
                    <div className='w-full flex flex-row items-center justify-between gap-x-2'>
                    <span className='relative w-full'>
                    <LuSearch className='w-3.5 h-3.5 text-brand-light/50 absolute left-2 top-1/2 -translate-y-1/2' />
                    <input type="text" placeholder='Search for media' className="w-full h-full pl-8 text-brand-light text-[10.5px] font-normal bg-brand rounded-[7px] border border-brand-light/10 p-2 outline-none" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
                    </span>
                    </div>
                    <ScrollArea  className='w-full h-96'>
                        <div className='w-full h-full grid grid-cols-2 gap-3'>
                            {filteredMediaItems.map((media) => (
                                <div key={media.name} onClick={() => {
                                    // Selecting a library item should clear timeline selection
                                    clearSelectedAsset();
                                    const next = { kind: 'media', assetUrl: media.assetUrl } as const;
                                    const isSame = value && value.kind === 'media' && value.assetUrl === media.assetUrl;
                                    onChange(isSame ? null : { ...next, frame: 0 });
                                }} className={cn('w-full flex flex-col items-center justify-center gap-y-1.5 cursor-pointer group relative')}>
                                    <div className='relative'>
                                    <div className={cn('absolute top-0 left-0 w-full h-full bg-brand-background-light/50 backdrop-blur-sm rounded-md z-20 group-hover:opacity-100 transition-all duration-200 flex items-center justify-center', (value && value.kind === 'media' && value.assetUrl === media.assetUrl) ? 'opacity-100' : 'opacity-0')}>
                                      <div className={cn('rounded-full py-1 px-3  bg-brand-light/10 flex items-center justify-center font-medium text-[10.5px] w-fit', (value && value.kind === 'media' && value.assetUrl === media.assetUrl) ? 'bg-brand-light/20' : '')}>
                                        {(value && value.kind === 'media' && value.assetUrl === media.assetUrl) ? 'Selected' : 'Use as Input'}
                                      </div>
                                    </div>
                                     <MediaThumb key={media.name} item={media} />
                                </div>
                                <div className='text-brand-light/90 text-[9.5px] text-start truncate w-full text-ellipsis overflow-hidden group-hover:text-brand-light transition-all duration-200'>{media.name}</div>
                                </div>
                            ))}
                        </div>
                    </ScrollArea>
                </TabsContent>
                <TabsContent value="timeline" className='outline-none'>
                    <TimelineSearch types={['image', 'video', 'group', 'text', 'shape', 'draw']} excludeClipId={clipId} />
                </TabsContent>
             </Tabs>
        </PopoverContent>
    )
}


const ImageInput: React.FC<ImageInputProps> = ({ label, description, inputId, value, onChange, clipId, panelSize, preprocessorRef, preprocessorName, applyPreprocessorInitial, onChangeComposite }) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const stageContainerRef = useRef<HTMLDivElement | null>(null);
    const [stageSize, setStageSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
    const [mediaClip, setMediaClip] = useState<AnyClipProps | null>(null);
    const [isOverDropZone, setIsOverDropZone] = useState(false);
    const externalDragCounterRef = useRef(0);
    const getClipById = useClipStore((s) => s.getClipById);
    const getPreprocessorsForClip = useClipStore((s) => s.getPreprocessorsForClip);
    const { preprocessors, load } = usePreprocessorsListStore();
    useEffect(() => {
        if (preprocessorRef) {
            void load();
        }
    }, [preprocessorRef, load]);
    const resolvedPreprocessorName = useMemo(() => {
        if (!preprocessorRef) return preprocessorName;
        const found = (preprocessors || []).find(p => p.id === preprocessorRef);
        return found?.name || preprocessorName || preprocessorRef;
    }, [preprocessors, preprocessorRef, preprocessorName]);
    // Preprocessor toggle state and defaulting
    const [applyPreprocessor, setApplyPreprocessor] = useState<boolean>(true);

    useEffect(() => {
        if (typeof applyPreprocessorInitial === 'boolean') {
            setApplyPreprocessor(applyPreprocessorInitial);
            return;
        }
        if (!preprocessorRef) {
            setApplyPreprocessor(true);
            return;
        }
        const sel = value;
        if (sel && sel.kind === 'clip') {
            const clip = getClipById(sel.clipId);
            if (clip && (clip.type === 'video' || clip.type === 'image')) {
                const exists = (getPreprocessorsForClip(clip.clipId) || []).some(p => (p.preprocessor?.id || '') === preprocessorRef);
                setApplyPreprocessor(!exists);
                return;
            }
        }
        setApplyPreprocessor(true);
    }, [value, preprocessorRef, applyPreprocessorInitial, getClipById, getPreprocessorsForClip]);

    const emitSelection = React.useCallback((next: ImageSelection) => {
        if (onChangeComposite) {
            onChangeComposite({ selection: next, preprocessor_ref: preprocessorRef, preprocessor_name: resolvedPreprocessorName, apply_preprocessor: applyPreprocessor });
        } else {
            onChange(next);
        }
    }, [onChangeComposite, onChange, preprocessorRef, resolvedPreprocessorName, applyPreprocessor]);

    const clearSelectedAsset = useAssetControlsStore((s) => s.clearSelectedAsset);
    const aspectRatio = useViewportStore((s) => s.aspectRatio);

    const setInputTimelineDuration = useInputControlsStore((s) => s.setTimelineDuration);
    const setInputFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
    const {focusFrameByInputId} = useInputControlsStore();
    const focusFrameForInput = focusFrameByInputId[inputId] ?? 0;
    const selectionKey = useMemo(() => {
        if (!value) return 'null';
        if (value.kind === 'media') return `media:${value.assetUrl}`;
        return `clip:${value.clipId}`;
    }, [value]);
    const lastSelectionKeyRef = useRef<string | null>(null);
    const viewportRatio = useMemo(() => {
        const r = aspectRatio.width / aspectRatio.height;
        return Number.isFinite(r) && r > 0 ? r : 1;
    }, [aspectRatio.width, aspectRatio.height]);
    // Determine stage rendering dynamically in render branch

    // Keep stage size exactly equal to the trigger box area
    useEffect(() => {
        const el = stageContainerRef.current;
        if (!el) return;
        const obs = new ResizeObserver((entries) => {
            const rect = entries[0]?.contentRect;
            if (!rect) return;
            const w = Math.max(1, panelSize);
            const h = Math.max(1, w / viewportRatio);
            setStageSize({ w, h });
        });
        obs.observe(el);
        return () => obs.disconnect();
    }, [panelSize, viewportRatio]);

    // Ensure width updates immediately when panelSize changes, even if height is unchanged
    useEffect(() => {
        setStageSize((prev) => ({ w: Math.max(1, panelSize), h: Math.max(1, panelSize / viewportRatio || prev.h) }));
    }, [panelSize, viewportRatio]);



    // DnD monitor: track hover-over state and handle drop selection
    useDndMonitor({
        onDragStart: () => {
            setIsOverDropZone(false);
        },
        onDragMove: (event) => {
            const data = event.active?.data?.current as MediaItem | undefined;
            const overId = event.over?.id as string | undefined;
            const isValid = !!data && (data.type === 'image' || data.type === 'video');
            setIsOverDropZone(isValid && overId === 'image-input');
        },
        onDragCancel: () => {
            setIsOverDropZone(false);
        },
        onDragEnd: (event) => {
            const overId = event.over?.id as string | undefined;
            const data = event.active?.data?.current as MediaItem | undefined;
            const isValid = !!data && (data.type === 'image' || data.type === 'video');
            if (isValid && overId === 'image-input') {
                const next = { kind: 'media', assetUrl: (data as MediaItem).assetUrl } as const;
                const isSame = value && value.kind === 'media' && value.assetUrl === (data as MediaItem).assetUrl;
                emitSelection(isSame ? null : { ...next, frame: 0 });
            }
            setIsOverDropZone(false);
        }
    });


    // Render poster preview into the trigger area whenever the value changes
    useEffect(() => {
        let cancelled = false;
        const requestedFocusFrame = Math.max(0, Math.round(value?.frame ?? 0));

        if (value?.kind === 'media' && mediaClip?.clipId === `media:${value.assetUrl}`) {
            return;
        }
        if (value?.kind === 'clip' && mediaClip?.clipId === value.clipId) {
            return;
        }
        (async () => {
            try {
                const canvasEl = canvasRef.current;
                if (!value) {
                    if (canvasEl) {
                        const ctx = canvasEl.getContext('2d');
                        if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
                    }
                    setStageSize({ w: 0, h: 0 });
                    setMediaClip(null);
                    const storeState = useInputControlsStore.getState();
                    const currentDuration = storeState.getTimelineDuration(inputId);
                    const currentFocus = storeState.getFocusFrame(inputId);
                    if (currentDuration[0] !== 0 || currentDuration[1] !== 1) {
                        setInputTimelineDuration(0, 1, inputId);
                    }
                    if ((currentFocus ?? 0) !== 0) {
                        setInputFocusFrame(0, inputId);
                    }
                    return;
                }
                if (value.kind === 'media') {
                    try {
                        const ext = getLowercaseExtension(value.assetUrl);
                        const isVideo = VIDEO_EXTS.includes(ext);
                        if (isVideo) {
                            const prelim: VideoClipProps = {
                                type: 'video',
                                clipId: `media:${value.assetUrl}`,
                                src: value.assetUrl,
                                startFrame: 0,
                                endFrame: 1,
                                preprocessors: [],
                                masks: [],
                            } as any;
                            setMediaClip(prelim as AnyClipProps);
                            const info = await getMediaInfo(value.assetUrl);
                            const fps = Math.max(1, Math.floor(info?.stats?.video?.averagePacketRate || DEFAULT_FPS));
                            const duration = Math.max(0, Math.floor((info?.duration || 0) * fps));
                            const finalEnd = Math.max(1, duration);
                            if (!cancelled) {
                                const clipForTimeline = { ...prelim, endFrame: finalEnd } as AnyClipProps;
                                setMediaClip(clipForTimeline);
                                const clampedFocus = Math.max(0, Math.min(finalEnd - 1, requestedFocusFrame));
                                const storeState = useInputControlsStore.getState();
                                const currentFocus = storeState.getFocusFrame(inputId);
                                const currentDuration = storeState.getTimelineDuration(inputId);
                                if (currentDuration[0] !== (prelim.startFrame ?? 0) || currentDuration[1] !== finalEnd) {
                                    setInputTimelineDuration(prelim.startFrame ?? 0, finalEnd, inputId);
                                }
                                if (currentFocus !== clampedFocus) {
                                    setInputFocusFrame(clampedFocus, inputId);
                                }
                            }
                        } else {
                            const imgClip: ImageClipProps = {
                                type: 'image',
                                clipId: `media:${value.assetUrl}`,
                                src: value.assetUrl,
                                startFrame: 0,
                                endFrame: 1,
                                preprocessors: [],
                                masks: [],
                            } as any;
                            setMediaClip(imgClip as AnyClipProps);
                            const storeState = useInputControlsStore.getState();
                            const currentDuration = storeState.getTimelineDuration(inputId);
                            const currentFocus = storeState.getFocusFrame(inputId);
                            if (currentDuration[0] !== (imgClip.startFrame ?? 0) || currentDuration[1] !== (imgClip.endFrame ?? 0)) {
                                setInputTimelineDuration(imgClip.startFrame ?? 0, imgClip.endFrame ?? 0, inputId);
                            }
                            if ((currentFocus ?? 0) !== 0) {
                                setInputFocusFrame(0, inputId);
                            }
                        }
                    } catch {}
                } else if (value.kind === 'clip') {
                    const clip = getClipById(value.clipId) as AnyClipProps | undefined;
                    if (clip && clip.type !== 'audio') {
                        const duration = Math.max(1, (clip.endFrame ?? 0) - (clip.startFrame ?? 0));
                        const projectedClip = {
                            ...clip,
                            startFrame: 0,
                            endFrame: duration
                        } as AnyClipProps;
                        setMediaClip(projectedClip);
                        const storeState = useInputControlsStore.getState();
                        const currentDuration = storeState.getTimelineDuration(inputId);
                        const currentFocus = storeState.getFocusFrame(inputId);
                        if (currentDuration[0] !== 0 || currentDuration[1] !== duration) {
                            setInputTimelineDuration(0, duration, inputId);
                        }
                        const clampedFocus = Math.max(0, Math.min(duration - 1, requestedFocusFrame));
                        if (currentFocus !== clampedFocus) {
                            setInputFocusFrame(clampedFocus, inputId);
                        }
                    }
                }
            } catch {}
        })();
        return () => { cancelled = true };
    }, [value, getClipById, inputId, setInputFocusFrame, setInputTimelineDuration]);

    useEffect(() => {
        const prevKey = lastSelectionKeyRef.current;
        lastSelectionKeyRef.current = selectionKey;
        if (!value) return;
        if (prevKey !== selectionKey) {
            // Selection changed; allow state setters to establish initial values before syncing back
            return;
        }
        const normalizedFocus = Math.max(0, Math.round(focusFrameForInput ?? 0));
        const currentFrame = Math.max(0, Math.round(value?.frame ?? 0));
        if (normalizedFocus === currentFrame) return;
        emitSelection({ ...value, frame: normalizedFocus });
    }, [focusFrameForInput, selectionKey, value, emitSelection]);

    const handleToggleApply = React.useCallback(() => {
        const next = !applyPreprocessor;
        setApplyPreprocessor(next);
        if (onChangeComposite) {
            onChangeComposite({ selection: value ?? null, preprocessor_ref: preprocessorRef, preprocessor_name: resolvedPreprocessorName, apply_preprocessor: next });
        }
    }, [applyPreprocessor, onChangeComposite, value, preprocessorRef, resolvedPreprocessorName]);


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
            const before = await listConvertedMedia();
            const existingNames = new Set(before.map(it => it.name));
            await importMediaPaths([path]);
            const after = await listConvertedMedia();
            const newlyAdded = after
                .filter(it => !existingNames.has(it.name))
                .filter(it => it.type === 'image' || it.type === 'video');
            const first = newlyAdded[0] ?? null;
            if (first) {
                clearSelectedAsset();
                emitSelection({ kind: 'media', assetUrl: first.assetUrl, frame: 0 });
            }
            bumpMediaLibraryVersion();
        } catch {}
    };

    const timelineClip = useMemo<AnyClipProps | null>(() => {
        if (!value) return null;
        if (mediaClip) return mediaClip;
        if (value.kind === 'media') {
            const ext = getLowercaseExtension(value.assetUrl);
            const isVideo = VIDEO_EXTS.includes(ext);
            if (!isVideo) return null;
            const storeState = useInputControlsStore.getState();
            const [, persistedEnd] = storeState.getTimelineDuration(inputId);
            const end = Math.max(1, persistedEnd || 1);
            return {
                type: 'video',
                clipId: `media:${value.assetUrl}`,
                src: value.assetUrl,
                startFrame: 0,
                endFrame: end,
                preprocessors: [],
                masks: [],
            } as AnyClipProps;
        }
        if (value.kind === 'clip') {
            const clip = getClipById(value.clipId) as AnyClipProps | undefined;
            if (clip && clip.type !== 'audio') {
                const duration = Math.max(1, (clip.endFrame ?? 0) - (clip.startFrame ?? 0));
                return { ...clip, startFrame: 0, endFrame: duration } as AnyClipProps;
            }
        }
        return null;
    }, [value, mediaClip, inputId, getClipById]);

  return (
    <Droppable className="w-full h-full" id="image-input" accepts={['media']}>
        
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
        <div ref={stageContainerRef} onDragEnter={handleExternalDragEnter} onDragOver={handleExternalDragOver} onDragLeave={handleExternalDragLeave} onDrop={handleExternalDrop} style={{ height: stageSize.h }} className={cn("w-full flex flex-col items-center justify-center gap-y-3  shadow-accent  hover:opacity-70  cursor-pointer relative overflow-hidden", 
            value ? '': 'border-dashed',
            value ? '' : 'p-4 border-brand-light/10 border bg-brand-background-light/50'
            )}>
            {isOverDropZone && (
                <div className="absolute inset-0 z-30 bg-brand-background-light/40 backdrop-blur-sm pointer-events-none transition-opacity duration-150" />
            )}
            {value ? (
                (stageSize.w > 0 && stageSize.h > 0) ? (
                    (() => {
                        if (value.kind === 'media') {
                            const ext = getLowercaseExtension(value.assetUrl);
                            const isVideo = VIDEO_EXTS.includes(ext);
                            const fallbackClip: AnyClipProps = isVideo ? ({
                                type: 'video',
                                clipId: `media:${value.assetUrl}`,
                                src: value.assetUrl,
                                startFrame: 0,
                                endFrame: 1,
                                preprocessors: [],
                                masks: [],
                            } as unknown as VideoClipProps) : ({
                                type: 'image',
                                clipId: `media:${value.assetUrl}`,
                                src: value.assetUrl,
                                startFrame: 0,
                                endFrame: 1,
                                preprocessors: [],
                                masks: [],
                            } as unknown as ImageClipProps);
                            const clipToRender = mediaClip || fallbackClip;
                            return <TimelineClipPosterPreview key={clipToRender.clipId} clip={clipToRender} width={stageSize.w} height={stageSize.h} inputId={inputId} />
                        } else {
                            const clip = getClipById(value.clipId) as AnyClipProps | undefined;
                            if (clip && clip.type !== 'audio') {
                                return <TimelineClipPosterPreview key={value.clipId} clipId={value.clipId} width={stageSize.w} height={stageSize.h} inputId={inputId} />
                            }
                            return (
                              <div className="w-full h-full flex items-center justify-center text-brand-light/70 text-[12px]">
                                Unable to preview clip.
                              </div>
                            );
                        }
                    })()
                ) : (
                    <canvas ref={canvasRef} className="w-full h-auto rounded-[6px]" />
                )
            ) : (
                <>
                    <IoImageOutline className="w-10 h-10 text-brand-light" />
                    <span className="text-brand-light text-[11px] w-full text-center font-medium">Click or drag and drop an image here.</span>
                </>
            )}
        </div>
        </PopoverTrigger>
        <PopoverImage value={value} onChange={emitSelection} clipId={clipId} />
    </Popover> 
    {timelineClip && (timelineClip.type === 'video' || timelineClip.type === 'group') && value && (
    <div className="w-full h-full mt-3">
      <TimelineSelector
        inputId={inputId}
        clip={timelineClip}
        width={stageSize.w}
        height={44}
        mode="frame"
      />
      </div>
    )}
    {preprocessorRef && (
            <div className=" flex flex-row-reverse items-center gap-x-3 justify-between">
              <span className="text-brand-light text-[10px] font-medium">Apply {resolvedPreprocessorName}</span>
              <button
                type="button"
                aria-pressed={applyPreprocessor}
                aria-label={applyPreprocessor ? 'Disable' : 'Enable'}
                onClick={handleToggleApply}
                className={cn(
                  'relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none',
                  applyPreprocessor ? 'bg-blue-600' : 'bg-brand-background border border-brand-light/10'
                )}
              >
                <span
                  className={cn(
                    'inline-block h-4 w-4 transform rounded-full bg-brand-light shadow transition-transform',
                    applyPreprocessor ? 'translate-x-4.5' : 'translate-x-0.5'
                  )}
                />
              </button>
            </div>
          )}
    </div>
    </div>
    
    
    </Droppable>
    )
}

export default ImageInput
