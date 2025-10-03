import { getZoomLevelConfig } from '@/lib/zoom';
import { useControlsStore } from '@/lib/control';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Stage, Layer,  Text, Line, Rect} from 'react-konva';
import Konva from 'konva';
import { ScrubControl } from './Scrubber';
import Timeline from './Timeline';
import { useClipStore, getClipWidth, getClipX, isValidTimelineForClip, getTimelineTypeForClip } from '@/lib/clip';
import { GoFileMedia } from "react-icons/go";
import Droppable from '../dnd/Droppable';
import {useDndMonitor} from '@dnd-kit/core';
import { MediaItem } from '../media/Item';
import {v4 as uuidv4} from 'uuid';
import { AnyClipProps, TimelineType } from '@/lib/types';
import TimelineSidebar from './TimelineSidebar';
import Scrollbar from './Scrollbar';

interface TimelineEditorProps {

}


interface TimelineMomentsProps {
    stageWidth:number;
    startPadding:number;
    maxScroll:number;
    thumbY: () => number;
}

interface TickMark {
    x:number;
    text:string;
    type: 'major' | 'minor';
    format: 'frame' | 'second';
}

const SCROLLBAR_HW = 8;
const SCROLL_BOTTOM_PADDING = 48; // extra space at bottom for easier drag-in

const getMajorZoomConfigFormat = (zoomConfig:{
    majorTickFormat: 'frame' | 'second';
    minorTickFormat: 'frame' | 'second';
}, fps:number, frameInterval:number) => {
    if (zoomConfig.majorTickFormat == 'second') {
        return 'second';
    } else if (zoomConfig.majorTickFormat == 'frame' && frameInterval % fps == 0) {
        return 'second';
    } else {
        return 'frame';
    }
}

const TimelineMoments:React.FC<TimelineMomentsProps> = React.memo(({stageWidth, startPadding, maxScroll, thumbY}) => {
    const { timelineDuration, fps, zoomLevel, focusFrame, shiftTimelineDuration, maxZoomLevel, minZoomLevel } = useControlsStore();
    const [startFrame, endFrame] = timelineDuration;

    // We will basically render from startDuration to startDuration + duration. 
    useEffect(() => {
            // ensure focusframe is always within the timeline duration
            if (focusFrame < startFrame) {
                shiftTimelineDuration(focusFrame - startFrame);
            }
            if (focusFrame > endFrame) {
                let duration = endFrame - startFrame;
                const additionalDuration = duration + endFrame > focusFrame ? 0 : focusFrame - endFrame;
                shiftTimelineDuration(duration + additionalDuration);
        }
    }, [startFrame, endFrame, focusFrame]);
    // Convert duration to milliseconds if needed for consistent calculations
    const tickMark:TickMark[] = useMemo(() => {
        let ticks:TickMark[] = [];
        const zoomConfig = getZoomLevelConfig(zoomLevel, timelineDuration, fps, maxZoomLevel, minZoomLevel);
        const majorTickInterval = zoomConfig.majorTickInterval * (zoomConfig.majorTickFormat === 'second' ? fps : 1);
        const minorTickInterval = zoomConfig.minorTickInterval * (zoomConfig.minorTickFormat === 'second' ? fps : 1);
        const [startFrame, endFrame] = timelineDuration;

        for (let i = startFrame; i <= endFrame; i += majorTickInterval) {
                ticks.push({
                    x: Math.round(i),
                    text: Math.round(i).toString(),
                    type: 'major',
                    format: getMajorZoomConfigFormat(zoomConfig, fps, majorTickInterval)
                });
            }
        
        for (let i = startFrame; i <= endFrame; i += minorTickInterval) {
            
            ticks.push({
                    x: Math.round(i),
                    text: Math.round(i).toString(),
                    type: 'minor',
                    format: zoomConfig.minorTickFormat
            });
            
        }
        ticks.sort((a, b) => a.x - b.x);
        // remove duplicates where x is the same and minor is true
        ticks = ticks.filter((tick, index, self) =>
            index === 0 || tick.x !== self[index - 1].x || tick.type !== 'minor'
        );

        return ticks;
    }, [timelineDuration, zoomLevel, fps]);
    
    
    // Helper function to format tick labels
    const formatTickLabel = (value: number, format: 'frame' | 'second'): string => {
        if (format === 'frame') {
            return `${value}f`;
        } else {
            // Convert frames to seconds
            const seconds = value / fps;
            const totalSeconds = Math.floor(seconds);
            const minutes = Math.floor(totalSeconds / 60);
            const remainingSeconds = totalSeconds % 60;
            return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
        }
    };


  
    return (
        <>
        <Rect x={0} y={0} width={stageWidth} height={28} fill={maxScroll > 0 && thumbY() > 24 ? "#222124" : undefined} listening={false} />
            {tickMark.map((tick, index) => {
                // Calculate x position based on timeline progress
                const progress = (tick.x - startFrame) / (endFrame - startFrame);
                const xPosition = progress * stageWidth;
                
                // Only render ticks that are visible on screen
                if (xPosition < -50 || xPosition > stageWidth + 50) {
                    return null;
                }
                
                const isMajor = tick.type === 'major';
                const tickHeight = isMajor ? 21 : 7;
                const tickY = 0;
                
                return (
                    <React.Fragment key={`${tick.x}-${tick.type}-${index}`}>
                        {/* Tick line */}
                        <Line
                            points={[xPosition + startPadding, tickY, xPosition + startPadding, tickY + tickHeight]}
                            stroke={isMajor ? "rgba(255, 255, 255, 0.3)" : "rgba(255, 255, 255, 0.1)"}
                            strokeWidth={1}
                            listening={false}
                        />
                        
                        {/* Label for major ticks only */}
                        {isMajor && (
                            <Text
                                x={xPosition + 4 + startPadding}
                                y={tickY + tickHeight - 8}
                                text={formatTickLabel(tick.x, tick.format)}
                                fontSize={8.5}
                                fill="rgba(255, 255, 255, 0.4)"
                                fontFamily="Poppins, system-ui, sans-serif"
                                listening={false}
                            />
                        )}
                    </React.Fragment>
                );
            })}
        </>
    )
});

const TimelineEditor:React.FC<TimelineEditorProps> = React.memo(() => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  // Always mount timeline elements; avoid conditional unmounts that can
  // cause rendering glitches in Tauri's WebView. Toggle visibility via data.
  
  const controlStore = useControlsStore();
  const {clips, timelines, getClipsForTimeline, addTimeline, addClip, setGhostStartEndFrame, setGhostX, setGhostTimelineId, setActiveMediaItem, removeTimeline, setGhostInStage, setHoveredTimelineId, hoveredTimelineId, snapGuideX} = useClipStore();
  // const scrollBarRef = useRef<any>(null);
  // const isSyncingScrollRef = useRef(false);
  const [isRulerDragging, setIsRulerDragging] = useState(false);
  const panStateRef = useRef({ startX: 0, lastX: 0, startFrame: 0, fractionalFrames: 0 });
  const wheelRemainderRef = useRef(0);
  const timelinesLayerRef = useRef<Konva.Layer | null>(null);
  
  const [verticalScroll, setVerticalScroll] = useState(0);
  const verticalScrollRef = useRef(0);
  const [isScrollbarHovered, setIsScrollbarHovered] = useState(false);
  const [canScrollHorizontal, setCanScrollHorizontal] = useState(false);
  const {totalTimelineFrames, timelineDuration} = useControlsStore();

  useEffect(() => {
    const [startFrame, endFrame] = timelineDuration;
    const totalDuration = endFrame - startFrame;
    if (totalDuration < totalTimelineFrames) {
      setCanScrollHorizontal(true);
    } else {
      setCanScrollHorizontal(false);
    }  }, [totalTimelineFrames, timelineDuration]);
  


  // Adjust zoom so that the first incoming clip spans roughly half of the visible window.
  // Only runs when there are no existing clips (timeline length is effectively zero).
  

  // Removed unused snapZoomToNearestLevelIfNeeded to avoid lints; handled elsewhere.

  useDndMonitor({
    onDragStart: (event) => {
      const data = event.active?.data?.current as unknown as MediaItem | undefined;
      if (!data) return;
      const mediaInfo = data.mediaInfo;
      if (!mediaInfo) return;
      setActiveMediaItem(data);
      const clipFrames = (() => {
        if (data.type === 'image') return controlStore.fps * 5;
        if (data.type === 'video') {
          const fps = mediaInfo.stats.video?.averagePacketRate ?? 0;
          return Math.round((mediaInfo.duration ?? 0) * fps);
        }
        if (data.type === 'audio') {
          const fps = mediaInfo.stats.audio?.averagePacketRate ?? 0;
          return Math.round((mediaInfo.duration ?? 0) * fps);
        }
        return 0;
      })();
      setGhostStartEndFrame(0, clipFrames);
    },
    onDragMove: (event) => {
      const container = containerRef.current;
      let pointerX: number | null = null;
      let pointerY: number | null = null;
      const data = event.active?.data?.current as unknown as MediaItem | undefined;
      if (clips.length === 0) return;

    if (container) {
        const rect = container.getBoundingClientRect();
        const activeRect = (event as any)?.active?.rect?.current?.translated || (event as any)?.active?.rect?.current;
        if (activeRect) {
          const centerX = (activeRect.left ?? 0) + ((activeRect.width ?? 0) / 2);
          const centerY = (activeRect.top ?? 0) + ((activeRect.height ?? 0) / 2);
          pointerX = centerX - rect.left;
          pointerY = centerY - rect.top;
        }
      }

      if (pointerX == null || pointerY == null) {
        return;
      }

      // Edge auto-scroll for external DnD moves
      edgeAutoScroll(pointerY);

      // check if the pointer is over a dashed line
      const stage = timelinesLayerRef.current;
      const children = stage?.children || [];
      let hoveredTimelineIdCurrent: string | null = null;
      for (const child of children) {
        const id = child.id();
        if (id.startsWith('dashed-')) {
         // get the rect of the child
         const rect = child.getClientRect();

          const rectY = rect.y;
          const rectX = rect.x;
          const rectWidth = rect.width;
          const boundNumberTop = id?.endsWith('-top') ? 36 : 16;
          const boundNumberBottom = 16;
          const boundsY = [rectY - boundNumberTop, rectY + boundNumberBottom];

          const boundsX = [rectX, rectX + rectWidth];
          if (pointerY >= boundsY[0] && pointerY <= boundsY[1] && pointerX >= boundsX[0] && pointerX <= boundsX[1]) {
            hoveredTimelineIdCurrent = id;
            break;
          }
        } 
      }

      setHoveredTimelineId(hoveredTimelineIdCurrent);

      if (hoveredTimelineIdCurrent) {
        // make everything else null
        setGhostTimelineId(null);
        setGhostInStage(false);
        return;
      }
      
      const timelinePadding = 24;
      const stageWidth = dimensions.stageWidth;
      const timelines = useClipStore.getState().timelines;

      let activeTimelineId: string | null = null;
      for (let i = 0; i < timelines.length; i++) {
        const t = timelines[i];
        const top = (t.timelineY ?? 0) + 8 - (verticalScrollRef.current || 0);
        const height = t.timelineHeight ?? 64;
        const left = timelinePadding;
        const right = left + stageWidth;
        const bottom = top + height;

        const isInside = pointerY >= top && pointerY <= bottom && pointerX >= left && pointerX <= right;
        if (isInside) {
          // check for same type 
          if (isValidTimelineForClip(t, data!)) {
            activeTimelineId = t.timelineId!;
            break;
          }
        }
      }

      if (!activeTimelineId) {
        setGhostTimelineId(null);
        setGhostInStage(false);
        return;
      } 

      // Center ghost under pointer and validate against bounds/overlaps
      const pointerLocalX = pointerX - timelinePadding; // pointer relative to inner timeline
      let [visibleStartFrame, visibleEndFrame] = useControlsStore.getState().timelineDuration;
      const ghostFrames = useClipStore.getState().ghostStartEndFrame;
      const ghostFramesLen = Math.max(1, (ghostFrames[1] ?? 0) - (ghostFrames[0] ?? 0));

      // Map pointer directly to ghost's left edge in PX to keep exact alignment
      const desiredLeftPx = pointerLocalX;
      const ghostWidthPx = getClipWidth(0, ghostFramesLen, stageWidth, [visibleStartFrame, visibleEndFrame]);
      let desiredLeft = desiredLeftPx;

      // Build occupied intervals (in px, inner coordinates)
      const getClipsForTimeline = useClipStore.getState().getClipsForTimeline;
      const existingClips = getClipsForTimeline(activeTimelineId);
      const occupied = existingClips
        .map((c) => {
          const sx = getClipX(c.startFrame || 0, c.endFrame || 0, stageWidth, [visibleStartFrame, visibleEndFrame]);
          const sw = getClipWidth(c.startFrame || 0, c.endFrame || 0, stageWidth, [visibleStartFrame, visibleEndFrame]);
          const lo = Math.max(0, Math.min(stageWidth, sx));
          const hi = Math.max(0, Math.min(stageWidth, sx + sw));
          return hi > lo ? [lo, hi] as [number, number] : null;
        })
        .filter(Boolean) as [number, number][];

      occupied.sort((a, b) => a[0] - b[0]);
      // Merge overlapping/touching intervals
      const merged: [number, number][] = [];
      for (const [lo, hi] of occupied) {
        if (merged.length === 0) {
          merged.push([lo, hi]);
        } else {
          const last = merged[merged.length - 1];
          if (lo <= last[1]) {
            last[1] = Math.max(last[1], hi);
          } else {
            merged.push([lo, hi]);
          }
        }
      }

      // Compute gaps within [0, stageWidth]
      const gaps: [number, number][] = [];
      let prev = 0;
      for (const [lo, hi] of merged) {
        if (lo > prev) gaps.push([prev, lo]);
        prev = Math.max(prev, hi);
      }
      if (prev < stageWidth) gaps.push([prev, stageWidth]);
      const validGaps = gaps.filter(([lo, hi]) => hi - lo >= ghostWidthPx);

      // Choose the gap that can contain the ghost with its LEFT edge under the pointer
      const pointerLeft = pointerLocalX;
      let chosenGap: [number, number] | null = null;
      for (const gap of validGaps) {
        if (pointerLeft >= gap[0] && pointerLeft <= gap[1] - ghostWidthPx) { chosenGap = gap; break; }
      }

      let validatedLeft = desiredLeft;
      if (chosenGap) {
        const [gLo] = chosenGap;
        // keep pointer alignment; only prevent crossing into the occupied region on the left
        validatedLeft = Math.max(desiredLeft, gLo);
      } else {
        // No chosen gap: prevent overriding existing clips on the left edge only
        // Compute gLo as:
        // - if pointer is inside an occupied interval [lo, hi], use hi
        // - else, use the last hi to the left of pointer (or 0 if none)
        let gLo = 0;
        for (let i = 0; i < merged.length; i++) {
          const [lo, hi] = merged[i];
          if (pointerLeft < lo) {
            // Pointer is in the gap before this interval; keep current gLo
            break;
          }
          if (pointerLeft <= hi) {
            // Pointer is inside this occupied interval; set gLo to its right edge
            gLo = Math.max(gLo, hi);
            break;
          }
          // Pointer is to the right of this interval; advance gLo
          gLo = Math.max(gLo, hi);
        }
        // gHi is conceptually Infinity; just clamp to stage bounds later
        validatedLeft = Math.max(desiredLeft, gLo);
      }


      setGhostTimelineId(activeTimelineId);
      setGhostInStage(true);
      setGhostX(validatedLeft);
    },
    onDragEnd: (event) => {
      
      const data = event.active?.data?.current as unknown as MediaItem | undefined;
      if (!event.over && !useClipStore.getState().ghostTimelineId && !useClipStore.getState().hoveredTimelineId) return;
      if (!data) return;
      
      // check if any timelines exist 
      const timelines = useClipStore.getState().timelines;
      let timelineId:string | undefined = undefined;

      console.log('timelines', timelines);

      if (timelines.length === 0) {
        timelineId = uuidv4();
        const newTimeline = {
          type: getTimelineTypeForClip(data),
          timelineId,
          timelineWidth: size.width,
          timelineY: 0,
          timelineHeight: 64,
        };
        addTimeline(newTimeline);
      } else if (hoveredTimelineId) {
         // we add a new timeline at the hovered timeline id
         const timelines = useClipStore.getState().timelines;
         const hoveredTimelineIdx = timelines.findIndex((t) => t.timelineId === hoveredTimelineId.replace('dashed-', ''));
         const hoveredTimeline = hoveredTimelineIdx !== -1 ? timelines[hoveredTimelineIdx] : null;
         timelineId = uuidv4();
          const newTimeline = {
          type: data.type === 'image' || data.type === 'video' ? 'media' : data.type as TimelineType,
          timelineId,
          timelineWidth: size.width,
          timelineY: (hoveredTimeline?.timelineY ?? 0) + 64,
          timelineHeight: 64,
        };
        addTimeline(newTimeline, hoveredTimelineIdx);
      } else if (timelines.length === 1) {
        timelineId = timelines[0].timelineId;
      } 

        const mediaInfo = data.mediaInfo;
        if (!mediaInfo) {
          setActiveMediaItem(null);
          setGhostTimelineId(null);
          setGhostStartEndFrame(0, 0);
          setGhostX(0);
          return;
        }
        
        let numFrames:number = 0;
        let framesToGiveEnd = 0;
        let framesToGiveStart  = 0;
        let height:number | undefined = undefined;
        let width:number | undefined = undefined;

        if (data.type === 'video') {
          const duration = mediaInfo.duration ?? 0; 
          const fps = mediaInfo.stats.video?.averagePacketRate ?? 0;
          height = mediaInfo.video?.codedHeight;
          width = mediaInfo.video?.codedWidth;
          numFrames = Math.round(duration * fps);
        } else if (data.type === 'audio') {
            const duration = mediaInfo.duration ?? 0; 
            const fps = controlStore.fps;
            numFrames = Math.round(duration * fps);
        } else if (data.type === 'image') {
          numFrames = controlStore.fps * 5;
          framesToGiveEnd = -Infinity;
          framesToGiveStart = Infinity;
          height = mediaInfo.image?.height;
          width = mediaInfo.image?.width;
        } 

        // Use validated ghost position to compute frames
        const state = useClipStore.getState();
        const ghostTimelineId = state.ghostTimelineId;
        const ghostX = state.ghostTimelineId ? state.ghostX : 0;
        
        const dropTimelineId = hoveredTimelineId ? timelineId : (ghostTimelineId || timelineId);
        setHoveredTimelineId(null);

        if (!dropTimelineId) {
          setActiveMediaItem(null);
          setGhostTimelineId(null);
          setGhostStartEndFrame(0, 0);
          setGhostX(0);
          return;
        }

        let [tStart, tEnd] = controlStore.timelineDuration;
        const stageWidth = dimensions.stageWidth;
        const visibleDuration = tEnd - tStart;
        const clipLen = Math.max(1, numFrames);
        // Map ghostX (left edge) to frame in visible window
   
        let startFrame = Math.round(tStart + (Math.max(0, Math.min(stageWidth, ghostX)) / stageWidth) * visibleDuration);
        let endFrame = startFrame + clipLen;

        // Validate against overlaps on the target timeline across ALL frames
        // and choose a feasible placement. If no gap at desired position, append at end.
        const getClipsForTimeline = state.getClipsForTimeline;
        const existingClips = getClipsForTimeline(dropTimelineId)
          .map(c => ({ lo: c.startFrame || 0, hi: c.endFrame || 0 }))
          .filter(iv => iv.hi > iv.lo)
          .sort((a, b) => a.lo - b.lo);

        // Merge intervals globally
        const merged: {lo:number, hi:number}[] = [];
        for (const iv of existingClips) {
          if (merged.length === 0) merged.push({ ...iv });
          else {
            const last = merged[merged.length - 1];
            if (iv.lo <= last.hi) last.hi = Math.max(last.hi, iv.hi);
            else merged.push({ ...iv });
          }
        }

        // Try to place at or after requested startFrame without overlap
        let placementStart = Math.max(0, startFrame);
        if (merged.length === 0) {
          // No existing clips; keep startFrame as selected (0 if first clip logic below)
        } else {
          // Advance placementStart past any overlapping intervals
          for (const iv of merged) {
            if (placementStart + clipLen <= iv.lo) {
              // Fits before this interval
              break;
            }
            if (placementStart < iv.hi) {
              // Overlaps; move right after this interval and continue
              placementStart = iv.hi;
            }
          }
        }
        startFrame = existingClips.length === 0 ? (existingClips.length === 0 ? 0 : placementStart) : placementStart;
        endFrame = startFrame + clipLen;

        const newClip = {
          timelineId: dropTimelineId,
          clipId: uuidv4(),
          startFrame: existingClips.length === 0 ? 0 : startFrame,
          endFrame,
          src: data.assetUrl,
          type: data.type,
          framesToGiveEnd: framesToGiveEnd,
          framesToGiveStart: framesToGiveStart,
          height: height,
          width:width,
          speed: 1.0
        };

        addClip(newClip as AnyClipProps);

        setActiveMediaItem(null);
        setGhostTimelineId(null);
        setGhostStartEndFrame(0, 0);
        setGhostX(0);

        // Baseline and zoom level recalibration happen within addClip -> _updateZoomLevel

      }
    
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    
    // Throttle resize updates to improve performance
    let timeoutId: NodeJS.Timeout;
    const observer = new ResizeObserver((entries) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const entry = entries[0];
        if (!entry) return;
        const { width, height } = entry.contentRect;
        setSize({ width, height });
      }, 16); // ~60fps throttle
    });
    
    observer.observe(el);
    return () => {
      clearTimeout(timeoutId);
      observer.disconnect();
    };
  }, []);

  // monitor all timelines and if any are empty, remove them
  useEffect(() => {
    const emptyTimelines = timelines.filter((t) => getClipsForTimeline(t.timelineId).length === 0);
      emptyTimelines.forEach((t) => {
      removeTimeline(t.timelineId);
    });
  }, [clips]);

  // Memoize dimensions to prevent unnecessary recalculations
  const dimensions = useMemo(() => {
    const stageWidth = Math.max(1, size.width);
    const stageHeight = Math.max(1, size.height);
    const dropZonePadding = 24;
    const dropZoneHeight = 80;
    const dropZoneX = dropZonePadding;
    const dropZoneY = Math.max(0, stageHeight / 2 - dropZoneHeight / 2);
    const dropZoneWidth = Math.max(0, stageWidth - dropZonePadding * 2);
    
    return {
      stageWidth,
      stageHeight,
      dropZoneX,
      dropZoneY,
      dropZoneWidth,
      dropZoneHeight
    };
  }, [size.width, size.height]);

  // Compute vertical content height for scrollbar logic
  const contentHeight = useMemo(() => {
    if (!timelines || timelines.length === 0) return dimensions.stageHeight;
    const bottoms = timelines.map((t) => {
      const y = t.timelineY || 0;
      const h = t.timelineHeight || 64;
      return y + h + 32; // include padding/offsets in Timeline rendering
    });
    return Math.max(...bottoms, 0) + SCROLL_BOTTOM_PADDING;
  }, [timelines, dimensions.stageHeight]);

  const maxScroll = useMemo(() => Math.max(0, contentHeight - dimensions.stageHeight), [contentHeight, dimensions.stageHeight]);
  const clampedScroll = useMemo(() => Math.max(0, Math.min(verticalScroll, maxScroll)), [verticalScroll, maxScroll]);

  useEffect(() => {
    setVerticalScroll((prev) => Math.max(0, Math.min(prev, maxScroll)));
  }, [maxScroll]);

  useEffect(() => {
    verticalScrollRef.current = clampedScroll;
  }, [clampedScroll]);

  // Auto-scroll when dragging near top/bottom edges (both external DnD and internal clip drags)
  const edgeAutoScroll = useCallback((pointerY: number) => {
    if (maxScroll <= 0) return;
    const stageHeight = dimensions.stageHeight || 0;
    const TOP_MARGIN = 24; // ruler height
    const EDGE_ZONE = 36;  // px threshold near edges
    const MAX_DELTA = 28;  // px per invocation

    // Scroll up when within top edge zone (below the ruler)
    const topEdgeY = TOP_MARGIN + EDGE_ZONE;
    if (pointerY <= topEdgeY) {
      const intensity = Math.max(0, (topEdgeY - pointerY) / EDGE_ZONE);
      const delta = Math.max(6, Math.round(intensity * MAX_DELTA));
      setVerticalScroll((prev) => Math.max(0, prev - delta));
      return;
    }

    // Scroll down when within bottom edge zone
    const bottomEdgeStart = Math.max(0, stageHeight - EDGE_ZONE);
    if (pointerY >= bottomEdgeStart) {
      const intensity = Math.max(0, (pointerY - bottomEdgeStart) / EDGE_ZONE);
      const delta = Math.max(6, Math.round(intensity * MAX_DELTA));
      setVerticalScroll((prev) => Math.min(maxScroll, prev + delta));
    }
  }, [dimensions.stageHeight, maxScroll]);

  // Listen for internal clip drag auto-scroll events
  useEffect(() => {
    const handler = (e: Event) => {
      try {
        const anyEvt = e as any;
        const y = Number(anyEvt?.detail?.y);
        if (!Number.isFinite(y)) return;
        edgeAutoScroll(y);
      } catch {}
    };
    window.addEventListener('timeline-editor-autoscroll', handler as any);
    return () => {
      window.removeEventListener('timeline-editor-autoscroll', handler as any);
    };
  }, [edgeAutoScroll]);

  // Horizontal scroll/pan with mouse wheel or trackpad
  const handleWheelScroll = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    if (!dimensions.stageWidth) return;
    const absX = Math.abs(e.deltaX);
    const absY = Math.abs(e.deltaY);
    const isHorizontalIntent = absX >= absY || e.shiftKey;

    if (isHorizontalIntent) {
      const [startFrame, endFrame] = controlStore.timelineDuration;
      const duration = endFrame - startFrame;
      const framesPerPixel = duration / dimensions.stageWidth;
      const speedMultiplier = e.shiftKey ? 3 : 1;
      const deltaFramesFloat = e.deltaX * framesPerPixel * speedMultiplier + wheelRemainderRef.current;
      const integerShift = Math.trunc(deltaFramesFloat);
      wheelRemainderRef.current = deltaFramesFloat - integerShift;
      if (integerShift !== 0) {
        if (controlStore.canTimelineDurationBeShifted(integerShift)) {
          controlStore.shiftTimelineDuration(integerShift, true, true);
        }
      }
    } else if (maxScroll > 0) {
      const pxDelta = e.deltaY;
      if (pxDelta !== 0) {
        setVerticalScroll((prev) => Math.max(0, Math.min(prev + pxDelta, maxScroll)));
      }
    }
  }, [dimensions.stageWidth, controlStore.timelineDuration, maxScroll]);

  // Drag to pan when clicking the top ruler area
  const handleStageMouseDown = useCallback((e: any) => {
    const stage = e?.target?.getStage?.();
    if (!stage) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;
    const tickAreaHeight = 24; // top ruler area height
    if (pos.y <= tickAreaHeight) {
      setIsRulerDragging(true);
      panStateRef.current.startX = pos.x;
      panStateRef.current.lastX = pos.x;
      panStateRef.current.startFrame = controlStore.timelineDuration[0];
      panStateRef.current.fractionalFrames = 0;
      if (e?.evt?.preventDefault) e.evt.preventDefault();
    }
  }, [controlStore.timelineDuration]);

  const handleStageMouseMove = useCallback((e: any) => {
    if (!isRulerDragging) return;
    const stage = e?.target?.getStage?.();
    if (!stage) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;
    const deltaX = pos.x - panStateRef.current.lastX;
    const [startFrame, endFrame] = controlStore.timelineDuration;
    const duration = endFrame - startFrame;
    const framesPerPixel = duration / Math.max(1, dimensions.stageWidth);
    const deltaFramesFloat = deltaX * framesPerPixel + panStateRef.current.fractionalFrames;
    const integerShift = Math.trunc(deltaFramesFloat);
    panStateRef.current.fractionalFrames = deltaFramesFloat - integerShift;
    if (integerShift !== 0) {
      // update focus frame
      
      if (controlStore.canTimelineDurationBeShifted(integerShift)) {
          controlStore.shiftTimelineDuration(integerShift, true);
      }
    }
    panStateRef.current.lastX = pos.x;
  }, [isRulerDragging, controlStore.timelineDuration, dimensions.stageWidth]);

  const endRulerDrag = useCallback(() => {
    if (isRulerDragging) setIsRulerDragging(false);
  }, [isRulerDragging]);


  const hasClips = useMemo(() => clips.length > 0, [clips]);

  const handleStageClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    // Check if the click target is the stage itself (background click)
    if (e.target === e.target.getStage()) {
      controlStore.clearSelection();
      const stage = e.target.getStage();
      const pos = stage?.getPointerPosition();
      if (!pos) return;
      // Map pointer X to frame within the visible window [startFrame, endFrame]
      const timelinePadding = 24; // left padding used by timeline
      const [startFrame, endFrame] = controlStore.timelineDuration;
      const innerX = Math.max(0, Math.min((pos.x - timelinePadding), dimensions.stageWidth));
      const progress = dimensions.stageWidth > 0 ? innerX / dimensions.stageWidth : 0;
      const targetFrame = Math.round(startFrame + progress * (endFrame - startFrame));

      // Pause if playing, then set focus without auto-resume
      if (controlStore.isPlaying) controlStore.pause();
      controlStore.setFocusAnchorRatio(Math.max(0, Math.min(1, progress)));
      controlStore.setFocusFrame(targetFrame, false);
    }
    
  }, [controlStore]);


  return (
    <div  className='relative h-full flex flex-row overflow-hidden'>
      {hasClips && <TimelineSidebar clampedScroll={clampedScroll} />}
      <div className='relative h-full w-full overflow-hidden' ref={containerRef} onWheel={handleWheelScroll}>
        {dimensions.stageWidth > 0 && dimensions.stageHeight > 0 && (
            <>
            {hasClips && (
             <>
            <Stage 
              width={dimensions.stageWidth} 
              height={dimensions.stageHeight} 
              className='border-b border-brand-light/10 bg-brand z-10 relative'
              onClick={handleStageClick}
              onMouseDown={handleStageMouseDown}
              onMouseMove={handleStageMouseMove}
              onMouseUp={endRulerDrag}
              onMouseLeave={endRulerDrag}
              style={{ cursor: isRulerDragging ? 'grabbing' : 'default' }}
            >
                <Layer ref={timelinesLayerRef} visible={hasClips} y={-clampedScroll}>
                    {timelines.map((timeline, index) => (
                        <Timeline key={timeline.timelineId} scrollY={clampedScroll} index={index} type={timeline.type} muted={timeline.muted} hidden={timeline.hidden} timelineWidth={dimensions.stageWidth} timelineY={timeline.timelineY} timelineHeight={timeline.timelineHeight} timelineId={timeline.timelineId} />
                    ))}
                </Layer>
                <Layer listening={false} visible={hasClips}>
                    {/* Time labels along the top */}
                    <TimelineMoments stageWidth={dimensions.stageWidth} startPadding={24} maxScroll={maxScroll} thumbY={() => {
                      const trackTop = 24;
                      const trackBottomPad = 8;
                      const trackHeight = Math.max(0, dimensions.stageHeight - trackTop - trackBottomPad);
                      const ratio = Math.max(0, Math.min(1, dimensions.stageHeight / Math.max(1, contentHeight)));
                      const minThumb = 24;
                      const thumbHeight = Math.max(minThumb, Math.round(trackHeight * ratio));
                      const maxThumbY = Math.max(0, trackHeight - thumbHeight);
                      const thumbY = trackTop + (maxScroll > 0 ? Math.round((clampedScroll / maxScroll) * maxThumbY) : 0);
                      return thumbY;
                    }} />
                </Layer>
                {/* Snap guideline overlay */}
                <Layer listening={false}>
                  {typeof snapGuideX === 'number' && (
                    <Line
                      points={[snapGuideX, 0, snapGuideX, dimensions.stageHeight]}
                      stroke={'#AE81CE'}
                      strokeWidth={1.5}
                      dash={[4, 4]}
                      shadowColor={'#AE81CE'}
                      shadowBlur={6}
                      shadowOpacity={0.4}
                    />
                  )}
                  
                </Layer>
                {/* Virtual vertical scrollbar */}
                {hasClips && maxScroll > 0 && (
                  <Layer>
                    {/* Invisible hover track to detect mouseover across full height on the right margin */}
                    <Rect
                      x={Math.max(0, dimensions.stageWidth - SCROLLBAR_HW)}
                      y={0}
                      width={SCROLLBAR_HW}
                      height={dimensions.stageHeight}
                      fill={'transparent'}
                      listening
                      onMouseEnter={() => setIsScrollbarHovered(true)}
                      onMouseLeave={() => setIsScrollbarHovered(false)}
                    />
                    {(() => {
                       const scrollbarWidth = SCROLLBAR_HW;
                       const trackTop = 24;
                       const trackBottomPad = 8;
                       const trackHeight = Math.max(0, dimensions.stageHeight - trackTop - trackBottomPad);
                       const ratio = Math.max(0, Math.min(1, dimensions.stageHeight / Math.max(1, contentHeight)));
                       const minThumb = 24;
                       const thumbHeight = Math.max(minThumb, Math.round(trackHeight * ratio));
                       const maxThumbY = Math.max(0, trackHeight - thumbHeight);
                       const thumbY = trackTop + (maxScroll > 0 ? Math.round((clampedScroll / maxScroll) * maxThumbY) : 0);
                       const thumbX = Math.max(0, dimensions.stageWidth - scrollbarWidth);

                      return (
                        <Rect
                          x={thumbX}
                          y={thumbY}
                          width={scrollbarWidth}
                          height={thumbHeight}
                          cornerRadius={scrollbarWidth}
                          fill={isScrollbarHovered ? 'rgba(227,227,227,0.4)' : 'rgba(227,227,227,0.1)'}
                          draggable
                          dragBoundFunc={(pos) => {
                            const clampedY = Math.max(trackTop, Math.min(trackTop + maxThumbY, pos.y));
                            return { x: thumbX, y: clampedY };
                          }}
                          onDragMove={(e) => {
                            const y = e.target.y();
                            const rel = maxThumbY > 0 ? (y - trackTop) / maxThumbY : 0;
                            const next = rel * maxScroll;
                            setVerticalScroll(next);
                          }}
                          onDragEnd={(e) => {
                            const y = e.target.y();
                            const rel = maxThumbY > 0 ? (y - trackTop) / maxThumbY : 0;
                            const next = rel * maxScroll;
                            setVerticalScroll(next);
                          }}
                          onMouseEnter={() => setIsScrollbarHovered(true)}
                          onMouseLeave={() => setIsScrollbarHovered(false)}
                        />
                      );
                    })()}
                    
                  </Layer>
                )}
                {canScrollHorizontal && (
                  <Layer>
                    <Scrollbar 
                      stageWidth={dimensions.stageWidth}
                      stageHeight={dimensions.stageHeight}
                      isScrollbarHovered={isScrollbarHovered}
                      setIsScrollbarHovered={setIsScrollbarHovered}
                    />
                  </Layer>
                )}
            </Stage>
            </>
        )}
            {(!hasClips) && (
                <>
          {/* Overlay helper text/icon centered */}
          
            <div style={{display: clips.length > 0 ? 'none' : 'flex'}} className='h-full items-center justify-center w-full p-8'>
            <Droppable id='timeline' className='w-full rounded-lg bg-brand-background/70 text-brand-light/70 duration-100 ease-in-out' accepts={['media']} highlight={{borderColor: '#A477C4', textColor: '#E8E8E8', bgColor: '#A477C4'}}>
              <div className='w-full group py-6 px-10    border-brand-light/30 rounded-lg  flex items-center'>
              <div className=' mx-auto w-full flex items-center font-sans pointer-events-auto'>
                <h4 className=" flex items-center flex-row leading-none gap-x-3.5">
                    <span className='flex items-center justify-center leading-none'>
                        <GoFileMedia className='w-5 h-5 ' />
                    </span>
                    <span className='text-[12.5px] font-poppins leading-none '>Drag and drop media, tracks and models</span>
                </h4>
            </div>
              </div>
              </Droppable>
            </div>
            
          </>
        )}
          </>
        )}
        <div style={{display: hasClips ? undefined : 'none'}}>
        <ScrubControl stageHeight={dimensions.stageHeight - 30} stageWidth={dimensions.stageWidth} />
      </div>
      </div>
      
    </div>
  )
});

export default TimelineEditor; 