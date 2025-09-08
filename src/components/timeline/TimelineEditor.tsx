import { getZoomLevelConfig } from '@/lib/zoom';
import { useControlsStore } from '@/lib/control';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Stage, Layer, Rect, Text, Line} from 'react-konva';
import { ScrubControl } from './Scrubber';
import Timeline from './Timeline';
import { TimelineProps, VideoClipProps } from '@/lib/types';
import { useClipStore } from '@/lib/clip';
// import { Scrollbar } from 'react-scrollbars-custom';

interface TimelineEditorProps {
    name:string;
    icon:React.ReactNode;
}

interface AddContentProps {
    name:string;
    icon:React.ReactNode;
    onMouseEnter?: () => void;
    onMouseLeave?: () => void;
}

const AddContent = ({ name, icon, onMouseEnter, onMouseLeave}:AddContentProps) => {
    return (
        <div className=' mx-auto w-full h-20 flex items-center justify-center cursor-pointer text-brand-light/50 font-sans hover:text-brand-light pointer-events-auto' onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
            <h4 className=" flex items-center leading-none">
                <span className='w-4 h-4 mr-2 shrink-0 transform-gpu inline-flex items-center justify-center leading-none'>
                    {icon}
                </span>
                <span className='text-xs font-medium leading-none'>Drag {name} here</span>
            </h4>
        </div>
    )
}

interface TimelineMomentsProps {
    stageWidth:number;
    startPadding:number;
}

interface TickMark {
    x:number;
    text:string;
    type: 'major' | 'minor';
    format: 'frame' | 'second';
}

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

const TimelineMoments:React.FC<TimelineMomentsProps> = React.memo(({stageWidth, startPadding}) => {
    const { timelineDuration, fps, zoomLevel, focusFrame, isPlaying, shiftTimelineDuration } = useControlsStore();
    const [startFrame, endFrame] = timelineDuration;

    // We will basically render from startDuration to startDuration + duration. 
    useEffect(() => {
            // ensure focusframe is alwys within the timeline duration
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
        const zoomConfig = getZoomLevelConfig(zoomLevel, timelineDuration, fps);
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

const TimelineEditor:React.FC<TimelineEditorProps> = React.memo(({name, icon}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  // Always mount timeline elements; avoid conditional unmounts that can
  // cause rendering glitches in Tauri's WebView. Toggle visibility via data.
  const [isDropZoneHover, setIsDropZoneHover] = useState(false);
  const controlStore = useControlsStore();
  const {clips, setClips, setTimelines, timelines} = useClipStore();
  // const scrollBarRef = useRef<any>(null);
  // const isSyncingScrollRef = useRef(false);
  const [isRulerDragging, setIsRulerDragging] = useState(false);
  const panStateRef = useRef({ startX: 0, lastX: 0, startFrame: 0, fractionalFrames: 0 });
  const wheelRemainderRef = useRef(0);
  
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

  // Horizontal scroll/pan with mouse wheel or trackpad
  const handleWheelScroll = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    if (!dimensions.stageWidth) return;
    const dominantDelta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
    if (!isFinite(dominantDelta) || dominantDelta === 0) return;
    const [startFrame, endFrame] = controlStore.timelineDuration;
    const duration = endFrame - startFrame;
    const framesPerPixel = duration / dimensions.stageWidth;
    const speedMultiplier = e.shiftKey ? 3 : 1;
    const deltaFramesFloat = dominantDelta * framesPerPixel * speedMultiplier + wheelRemainderRef.current;
    const integerShift = Math.trunc(deltaFramesFloat);
    wheelRemainderRef.current = deltaFramesFloat - integerShift;
    if (integerShift !== 0) {
      // update focus frame
      if (controlStore.canTimelineDurationBeShifted(integerShift)) {
        controlStore.shiftTimelineDuration(integerShift);
          controlStore.setFocusFrame(controlStore.focusFrame + integerShift);
      }
      
    }
  }, [dimensions.stageWidth, controlStore.timelineDuration]);

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
          controlStore.shiftTimelineDuration(integerShift);
          controlStore.setFocusFrame(controlStore.focusFrame + integerShift);
      }
    }
    panStateRef.current.lastX = pos.x;
  }, [isRulerDragging, controlStore.timelineDuration, dimensions.stageWidth]);

  const endRulerDrag = useCallback(() => {
    if (isRulerDragging) setIsRulerDragging(false);
  }, [isRulerDragging]);


  // In a real app, clips would come from state/store. Keeping inline for now.
  const initialClips: VideoClipProps[] = [
    {
      clipId: '1',
      startFrame: 0,
      endFrame: 81,
      src: '/Users/tosinkuye/apex-studio/public/whipping_pose.mp4',
      framesToGiveEnd: 0,
      framesToGiveStart: 0,
      height: 832,
      width: 480,
      type: 'video',
      timelineId: '1',
    },
     {
         clipId: '2',
         startFrame: 81,
         endFrame: 162,
         src: '/Users/tosinkuye/apex-studio/public/whipping_pose.mp4',
         framesToGiveEnd: 0,
         framesToGiveStart: 0,
         height: 832,
         width: 480,
         timelineId: '1',
         type: 'video',
       },
  ];

  const initialTimelines: TimelineProps[] = [
    {
      timelineId: '1',
      timelineWidth: dimensions.stageWidth,
      timelineY: 0,
      timelineHeight: 64,
    },
  ];

  useEffect(() => {
    setClips(initialClips);
    setTimelines(initialTimelines);
  }, []);

  const hasClips = clips.length > 0;

  const handleStageClick = useCallback((e: any) => {
    // Check if the click target is the stage itself (background click)
    if (e.target === e.target.getStage()) {
      controlStore.clearSelection();
    }
  }, [controlStore]);

  return (
    <div  className='relative h-full flex flex-col'>
      <div className='relative h-full w-full overflow-hidden' ref={containerRef} onWheel={handleWheelScroll}>
      
        {dimensions.stageWidth > 0 && dimensions.stageHeight > 0 && (
            <>
            {hasClips && (
                <>
            <Stage 
              width={dimensions.stageWidth} 
              height={dimensions.stageHeight} 
              className='border-b border-brand-light/10 bg-brand-background/30 z-10 relative'
              onClick={handleStageClick}
              onMouseDown={handleStageMouseDown}
              onMouseMove={handleStageMouseMove}
              onMouseUp={endRulerDrag}
              onMouseLeave={endRulerDrag}
              style={{ cursor: isRulerDragging ? 'grabbing' : 'default' }}
            >
                <Layer listening={false} visible={hasClips}>
                    {/* Time labels along the top */}
                    <TimelineMoments stageWidth={dimensions.stageWidth} startPadding={24} />
                </Layer>
                <Layer visible={hasClips}>
                    {timelines.map((timeline) => (
                        <Timeline key={timeline.timelineId} timelineWidth={dimensions.stageWidth} timelineY={timeline.timelineY} timelineHeight={timeline.timelineHeight} timelineId={timeline.timelineId} />
                    ))}
                </Layer>
            </Stage>
            </>
        )}
            {(!hasClips) && (
                <>
                <div  onMouseMove={(e) => {
              const el = containerRef.current;
              if (!el) return;
              const rect = el.getBoundingClientRect();
              const localX = e.clientX - rect.left;
              const localY = e.clientY - rect.top;
              const withinX = localX >= dimensions.dropZoneX && localX <= dimensions.dropZoneX + dimensions.dropZoneWidth;
              const withinY = localY >= dimensions.dropZoneY && localY <= dimensions.dropZoneY + dimensions.dropZoneHeight;
              setIsDropZoneHover(withinX && withinY);
            }} onMouseLeave={() => setIsDropZoneHover(false)} >
            <Stage width={dimensions.stageWidth} height={dimensions.stageHeight - 30} className='w-full h-full'>
              <Layer>
                 <Rect
                x={dimensions.dropZoneX}
                y={dimensions.dropZoneY}
                width={dimensions.dropZoneWidth}
                height={dimensions.dropZoneHeight}
                cornerRadius={6}
                stroke={isDropZoneHover ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.15)'}
                dash={[3, 3]}
                fill={'rgba(0,0,0,0.0001)'}
              />
            </Layer>
          </Stage>
          {/* Overlay helper text/icon centered */}
            <div style={{display: hasClips ? 'none' : 'flex'}} className='h-full items-center justify-center absolute inset-0 pointer-events-none'>
              <AddContent name={name} icon={icon} onMouseEnter={() => setIsDropZoneHover(true)} onMouseLeave={() => setIsDropZoneHover(false)} />
            </div>
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