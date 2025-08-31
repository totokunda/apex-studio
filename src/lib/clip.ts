import { create } from "zustand";
import { ClipProps } from "./types";
import { v4 as uuidv4 } from 'uuid';
import { useControlsStore } from "./control";



interface ClipStore {  
    clipDuration: number;
    _setClipDuration: (duration: number) => void;
    clips: ClipProps[];
    setClips: (clips: ClipProps[]) => void;
    addClip: (clip: ClipProps) => void;
    removeClip: (clipId: string) => void;
    updateClip: (clipId: string, clipToUpdate: Partial<ClipProps>) => void;
    resizeClip: (clipId: string, side: 'left' | 'right', newFrame: number) => void;
    splitClip: (cutFrame: number) => void;
    mergeClips: (clipIds: string[]) => void;
    moveClipToEnd: (clipId: string) => void;
    clipboard: ClipProps[];
    copyClips: (clipIds: string[]) => void;
    cutClips: (clipIds: string[]) => void;
    pasteClips: (atFrame?: number) => void;
}

// Helper function to calculate total duration of all clips
const calculateTotalClipDuration = (clips: ClipProps[]): number => {
    const maxEndFrame = Math.max(...clips.map(clip => clip.endFrame || 0));
    return maxEndFrame;
};


// Helper function to resolve overlaps by shifting clips to maintain frame gaps
const resolveOverlaps = (clips: ClipProps[]): ClipProps[] => {
    if (clips.length === 0) return clips;
    
    // Sort clips by start frame
    const sortedClips = [...clips].sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
    const resolvedClips: ClipProps[] = [];
    
    for (let i = 0; i < sortedClips.length; i++) {
        const currentClip = { ...sortedClips[i] };
        const currentStart = currentClip.startFrame || 0;
        const currentEnd = currentClip.endFrame || 0;
        
        // Check for overlap with previous clip
        if (resolvedClips.length > 0) {
            const previousClip = resolvedClips[resolvedClips.length - 1];
            const previousEnd = previousClip.endFrame || 0;
            
            // If current clip overlaps with previous clip, shift it to start after previous clip ends
            if (currentStart < previousEnd) {
                const clipDuration = currentEnd - currentStart;
                currentClip.startFrame = previousEnd;
                currentClip.endFrame = previousEnd + clipDuration;
            }
        }
        
        resolvedClips.push(currentClip);
    }
    
    return resolvedClips;
};

export const getCorrectedClip = (clipId: string, clips: ClipProps[]): ClipProps | null => {
    // find the clip in the clips array
    const resolvedClips = resolveOverlaps(clips);
    const clip = resolvedClips.find((clip) => clip.clipId === clipId);
    if (!clip) return null;
    return clip;
}

export const useClipStore = create<ClipStore>((set) => ({       
    clipDuration: 0,
    _setClipDuration: (duration) => set({ clipDuration: duration }),
    clips: [],  
    clipboard: [],
    setClips: (clips: ClipProps[]) => {
        const resolvedClips = resolveOverlaps(clips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        set({ clips: resolvedClips, clipDuration });
    },
    addClip: (clip: ClipProps) => set((state) => {
        const newClips = [...state.clips, clip];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    removeClip: (clipId: string) => set((state) => {
        const newClips = state.clips.filter((clip) => clip.clipId !== clipId);
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    updateClip: (clipId: string, clipToUpdate: Partial<ClipProps>) => set((state) => {
        const newClips = state.clips.map((clip) => clip.clipId === clipId ? { ...clip, ...clipToUpdate } : clip);
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    resizeClip: (clipId: string, side: 'left' | 'right', newFrame: number) => set((state) => {
        const sortedClips = [...state.clips].sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
        const currentIndex = sortedClips.findIndex(c => c.clipId === clipId);
        if (currentIndex === -1) return { clips: state.clips };
        
        const currentClip = sortedClips[currentIndex];
        const newClips = [...state.clips];
        
        if (side === 'right') {
            // Resize right edge - adjust current clip's end and shift all clips after it
            const oldEndFrame = currentClip.endFrame || 0;
            const newEndFrame = Math.max((currentClip.startFrame || 0) + 1, newFrame);
            const frameDelta = newEndFrame - oldEndFrame;
            
            if (frameDelta + (currentClip.framesToGiveEnd || 0) > 0) {
                return { clips: state.clips };
            }

            const currentClipIndex = newClips.findIndex(c => c.clipId === clipId);
            newClips[currentClipIndex] = { ...currentClip, endFrame: newEndFrame, framesToGiveEnd: frameDelta + (currentClip.framesToGiveEnd || 0) };

        } else if (side === 'left') {
            // Resize left edge - adjust current clip's start and shift all clips before it
            const oldStartFrame = currentClip.startFrame || 0;
            const newStartFrame = Math.min((currentClip.endFrame || 0) - 1, newFrame);
            let frameDelta = newStartFrame - oldStartFrame;

            if (frameDelta + (currentClip.framesToGiveStart || 0) < 0) {
                return { clips: state.clips };
            }

            if (frameDelta == 0 && (currentClip.framesToGiveStart || 0) >  0) {
                frameDelta = Math.max(0, Math.min(1, (currentClip.framesToGiveStart || 0) - 1)); 

            } else {
                const currentClipIndex = newClips.findIndex(c => c.clipId === clipId);
                newClips[currentClipIndex] = { ...currentClip, startFrame: newStartFrame, framesToGiveStart: frameDelta + (currentClip.framesToGiveStart || 0) };

            }
        }

        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    // Create two new clips from the original clip at the cut frame
    splitClip: (cutFrame: number) => set((state) => {
        // Find the clip that contains the cut frame
        const clip = state.clips.find((clip) => {
            const startFrame = clip.startFrame || 0;
            const endFrame = clip.endFrame || 0;
            return cutFrame > startFrame && cutFrame < endFrame;
        });
        
        if (!clip) return { clips: state.clips };
        
        // remove the clip from the array 
        const filteredClips = state.clips.filter((c) => c.clipId !== clip.clipId);
        
        // create new clip ids
        const newClipId1 = uuidv4();
        const newClipId2 = uuidv4();
        const newClip1 = { ...clip, endFrame: cutFrame, clipId: newClipId1 };
        const newClip2 = { ...clip, startFrame: cutFrame, clipId: newClipId2};
        const newClips = [...filteredClips, newClip1, newClip2];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    mergeClips: (clipIds: string[]) => set((state) => {
        if (clipIds.length < 2) return { clips: state.clips };
        
        // Find all clips to merge
        const clipsToMerge = clipIds.map(id => state.clips.find(clip => clip.clipId === id)).filter(Boolean) as ClipProps[];
        
        if (clipsToMerge.length < 2) return { clips: state.clips };
        
        // Sort clips by start frame to check adjacency
        const sortedClips = clipsToMerge.sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
        
        // Check if all clips are frame-adjacent (no gaps between them)
        for (let i = 0; i < sortedClips.length - 1; i++) {
            const currentEnd = sortedClips[i].endFrame || 0;
            const nextStart = sortedClips[i + 1].startFrame || 0;
            
            // If there's a gap between clips, don't merge
            if (currentEnd !== nextStart) {
                return { clips: state.clips };
            }
        }
        
        // Remove the clips from the array
        const filteredClips = state.clips.filter((clip) => !clipIds.includes(clip.clipId));
        
        // Find the bounds of all clips to merge
        const clipStart = sortedClips[0].startFrame || 0;
        const clipEnd = sortedClips[sortedClips.length - 1].endFrame || 0;
        
        // Use the first clip as the base and merge all others into it
        const baseClip = sortedClips[0];
        const newClip = { ...baseClip, startFrame: clipStart, endFrame: clipEnd };
        const newClips = [...filteredClips, newClip];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    copyClips: (clipIds: string[]) => set((state) => {
        if (!clipIds || clipIds.length === 0) return { clipboard: state.clipboard };
        const toCopy = clipIds
            .map(id => state.clips.find(c => c.clipId === id))
            .filter(Boolean) as ClipProps[];
        return { clipboard: toCopy.map(c => ({ ...c })) };
    }),
    cutClips: (clipIds: string[]) => set((state) => {
        if (!clipIds || clipIds.length === 0) return { clips: state.clips, clipboard: state.clipboard };
        const toCut = clipIds
            .map(id => state.clips.find(c => c.clipId === id))
            .filter(Boolean) as ClipProps[];
        const remaining = state.clips.filter(c => !clipIds.includes(c.clipId));
        const resolvedClips = resolveOverlaps(remaining);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration, clipboard: toCut.map(c => ({ ...c })) };
    }),
    pasteClips: (atFrame?: number) => set((state) => {
        const clipboardItems = state.clipboard || [];
        if (clipboardItems.length === 0) return { clips: state.clips };
        const baseStart = Math.min(...clipboardItems.map(c => c.startFrame || 0));
        const insertionFrame = Math.max(0, Math.round(atFrame || 0));
        const newIds: string[] = [];
        const pasted = clipboardItems.map(template => {
            const templateStart = template.startFrame || 0;
            const templateEnd = template.endFrame || 0;
            const duration = Math.max(1, templateEnd - templateStart);
            const offset = templateStart - baseStart;
            const start = insertionFrame + offset;
            const end = start + duration;
            const newId = uuidv4();
            newIds.push(newId);
            return { ...template, clipId: newId, startFrame: start, endFrame: end, framesToGiveEnd: 0, framesToGiveStart: 0 } as ClipProps;
        });
        const newClips = [...state.clips, ...pasted];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        // Select newly pasted clips
        try {
            const controls = useControlsStore.getState();
            controls.setSelectedClipIds(newIds);
        } catch {}
        return { clips: resolvedClips, clipDuration };
    }),
    moveClipToEnd: (clipId: string) => set((state) => {
        const clip = state.clips.find((clip) => clip.clipId === clipId);
        if (!clip) return { clips: state.clips };
        const newClips = [...state.clips.filter((c) => c.clipId !== clipId), clip];
        const clipDuration = calculateTotalClipDuration(newClips);
        return { clips: newClips, clipDuration };
    }),
}));




export const getClipWidth = (startFrame:number, endFrame:number, timelineWidth:number, timelineDuration:number[]) => {
    const [timelineStartFrame, timelineEndFrame] = timelineDuration;
    const percentage = (endFrame - startFrame) / (timelineEndFrame - timelineStartFrame);
    return timelineWidth * percentage;
}

export const getClipX = (startFrame:number | null, endFrame:number | null, timelineWidth:number | null, timelineDuration:number[]) => {
    if (startFrame === null || endFrame === null || timelineWidth === null) return 0;

    const [timelineStartFrame, timelineEndFrame] = timelineDuration;
    const relativePosition = (startFrame - timelineStartFrame) / (timelineEndFrame - timelineStartFrame); 
    return relativePosition * timelineWidth;
}   

export const getClipImage = (src:string | undefined) => {
    // get an image from the video file if it exists
    if (!src) return Promise.resolve(null);

    // Simple memoization to avoid regenerating thumbnails for the same source
    const globalAny = globalThis as unknown as { __clipImageCache?: Map<string, Promise<HTMLCanvasElement | null>> };
    if (!globalAny.__clipImageCache) {
        globalAny.__clipImageCache = new Map<string, Promise<HTMLCanvasElement | null>>();
    }
    const cache = globalAny.__clipImageCache;
    if (cache.has(src)) {
        return cache.get(src)!;
    }

    const promise = new Promise<HTMLCanvasElement | null>(async (resolve) => {
        try {
            const videoElement = document.createElement('video');
            // Attempt to avoid tainting the canvas when possible
            videoElement.crossOrigin = 'anonymous';
            videoElement.muted = true;
            (videoElement as any).playsInline = true;
            videoElement.preload = 'auto';


            const isProbablyLocalPath = (value: string) => {
                if (!value) return false;
                if (value.startsWith('file://')) return true;
                if (value.startsWith('/')) return true; // unix-like absolute path
                if (/^[a-zA-Z]:[\\\/]/.test(value)) return true; // windows absolute path
                return false;
            };
            const normalizeLocalPath = (value: string) => {
                if (value.startsWith('file://')) {
                    try {
                        return new URL(value).pathname;
                    } catch (_) {
                        return value.replace(/^file:\/\//, '');
                    }
                }
                return value;
            };
            const resolveSrc = async (original: string): Promise<string> => {
               
                if (!isProbablyLocalPath(original)) return original;
                const localPath = normalizeLocalPath(original);
                try {
                    const core = await import('@tauri-apps/api/core');
                    
                    
                    if (core && typeof core.convertFileSrc === 'function') {
                        return core.convertFileSrc(localPath);
                    }
                } catch (_) {
                    // Not in Tauri or API not available
                    console.log('error', _);
                }
                // Fallback to file URL if possible
                if (localPath.startsWith('/')) {
                    return `file://${localPath}`;
                }
                return original;
            };
            resolveSrc(src).then((resolvedSrc) => {
                try {
                   
                    videoElement.src = resolvedSrc;
                } catch (e) {
                    console.log('error setting src', e);
                }
            });

            const handleError = () => {
                console.log('error', videoElement.error);
                cleanup();
                resolve(null);
            };

            const handleLoadedData = () => {
                // Ensure we have dimensions to draw
                const width = Math.max(1, videoElement.videoWidth || 1);
                const height = Math.max(1, videoElement.videoHeight || 1);
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const context = canvas.getContext('2d');

                
                if (!context) {
                    cleanup();
                    resolve(null);
                    return;
                }

                const drawFrame = () => {
                    try {
                        context.drawImage(videoElement, 0, 0, width, height);
                        cleanup();
                        resolve(canvas);
                    } catch (e) {
                        cleanup();
                        resolve(null);
                    }
                };

                // Some browsers require a seek to render the very first frame reliably
                const onSeeked = () => {
                    videoElement.removeEventListener('seeked', onSeeked);
                    drawFrame();
                };

                // Try drawing immediately; if it fails, seek then draw
                try {
                    context.drawImage(videoElement, 0, 0, width, height);
                    cleanup();
                    resolve(canvas);
                } catch {
                    videoElement.addEventListener('seeked', onSeeked);
                    try {
                        videoElement.currentTime = 0;
                    } catch {
                        // If seeking throws, fall back to draw after canplay
                        videoElement.addEventListener('canplay', drawFrame, { once: true });
                    }
                }
            };

            const cleanup = () => {
                videoElement.removeEventListener('loadeddata', handleLoadedData);
                videoElement.removeEventListener('error', handleError);
                try {
                    videoElement.pause();
                } catch (_) {}
                videoElement.removeAttribute('src');
                try {
                    videoElement.load();
                } catch (_) {}
            };

            videoElement.addEventListener('loadeddata', handleLoadedData, { once: true });
            videoElement.addEventListener('error', handleError, { once: true });
            // Kick off loading explicitly
            try {
                videoElement.load();
            } catch (_) {}
        } catch {
            resolve(null);
        }
    });

    cache.set(src, promise);
    return promise;
}
