import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { X, ChevronDown, Check } from 'lucide-react';
import { useControlsStore } from '@/lib/control';
import { useClipStore } from '@/lib/clip';
import type { ModelClipProps } from '@/lib/types';

interface FloatingTextPanelProps {
  id?: string;
  clipId: string;
  initialPosition: { x: number; y: number };
  onDelete?: () => void;
}

const FloatingTextPanel: React.FC<FloatingTextPanelProps> = ({ 
  id = 'floating-text-panel',
  clipId,
  initialPosition,
  onDelete
}) => {
  const setFloatingPanelData = useControlsStore((s) => s.setFloatingPanelData);
  const getFloatingPanelData = useControlsStore((s) => s.getFloatingPanelData);
  const setActiveClipId = useControlsStore((s) => s.setActiveClipId);
  const getClipById = useClipStore((s) => s.getClipById);
  const clips = useClipStore((s) => s.clips);
  
  const [position, setPosition] = useState(initialPosition);
  const [size, setSize] = useState({ width: 300, height: 150 });
  const [promptText, setPromptText] = useState<string>('');
  const [isResizing, setIsResizing] = useState(false);
  const [showModelMenu, setShowModelMenu] = useState<boolean>(false);
  const [hoveredModelClipId, setHoveredModelClipId] = useState<string | null>(null);
  
  // Get clip data for model info
  const clip = getClipById(clipId) as ModelClipProps | undefined;
  const modelName = clip?.name || 'Model';
  const trackName = clip?.trackName || '';
  
  // Format track name to abbreviation (e.g., "Text to Video" -> "T2V")
  const formatTrackAbbreviation = (track: string): string => {
    const trackMap: { [key: string]: string } = {
      'Text to Image': 'T2I',
      'Image to Image': 'I2I',
      'Text to Video': 'T2V',
      'Image to Video': 'I2V',
      'Audio-Image to Video': 'AI2V',
      'Image-Control to Video': 'IC2V',
      'Image-Mask to Image': 'IM2I',
    };
    return trackMap[track] || track;
  };
  
  const modelPillText = trackName 
    ? `${modelName} - ${formatTrackAbbreviation(trackName)}`
    : modelName;
  
  // Get all model clips from timeline
  const modelClips = useMemo(() => {
    return clips.filter(c => c.type === 'model') as ModelClipProps[];
  }, [clips]);
  
  // Handler for switching to a different model clip
  const handleModelSwitch = (targetClipId: string) => {
    setShowModelMenu(false);
    
    // Get the target clip to determine its panel type
    const targetClip = getClipById(targetClipId) as ModelClipProps | undefined;
    const isTextModel = targetClip?.trackName?.startsWith('Text') || false;
    
    // Update both the active clip and panel type
    const { setFloatingPanelType } = useControlsStore.getState();
    setFloatingPanelType(isTextModel ? 'text' : 'input');
    setActiveClipId(targetClipId);
  };
  
  // Track if component has fully initialized to prevent saving on mount
  const hasLoadedRef = useRef(false);
  
  // Load saved data when clipId changes
  useEffect(() => {
    const savedData = getFloatingPanelData(clipId);
    console.log('[TextPanel] Loading for clipId:', clipId, 'savedData:', savedData);
    setPromptText(savedData.promptText);
    
    // Mark as loaded after a short delay to ensure state has settled
    setTimeout(() => {
      hasLoadedRef.current = true;
    }, 0);
  }, [clipId, getFloatingPanelData]);
  const panelRef = useRef<HTMLDivElement>(null);
  const lastTransform = useRef({ x: 0, y: 0 });
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0, posX: 0, posY: 0 });
  const resizeEdgeRef = useRef<string>('');

  const { attributes, listeners, setNodeRef: setDraggableRef, transform, isDragging } = useDraggable({
    id: `${id}-draggable`,
    data: { type: 'floating-panel' },
  });

  const { setNodeRef: setDroppableRef, isOver } = useDroppable({
    id: `${id}-droppable`,
    data: {
      accepts: ['media', 'model', 'audio', 'video', 'image', 'timeline-clip', 'preprocessor'],
    },
  });

  useEffect(() => {
    if (isDragging && transform) {
      lastTransform.current = { x: transform.x, y: transform.y };
    }
  }, [isDragging, transform]);

  useEffect(() => {
    if (!isDragging && (lastTransform.current.x !== 0 || lastTransform.current.y !== 0)) {
      setPosition(prev => ({
        x: prev.x + lastTransform.current.x,
        y: prev.y + lastTransform.current.y,
      }));
      lastTransform.current = { x: 0, y: 0 };
    }
  }, [isDragging]);

  // Handle window resize to keep panel in bounds
  useEffect(() => {
    const handleWindowResize = () => {
      setPosition(prev => {
        const maxX = window.innerWidth - size.width;
        const maxY = window.innerHeight - size.height;
        return {
          x: Math.max(0, Math.min(prev.x, maxX)),
          y: Math.max(0, Math.min(prev.y, maxY))
        };
      });
    };

    window.addEventListener('resize', handleWindowResize);
    return () => window.removeEventListener('resize', handleWindowResize);
  }, [size.width, size.height]);

  // Save prompt text when it changes
  useEffect(() => {
    if (!hasLoadedRef.current) {
      console.log('[TextPanel] Skipping save - not loaded yet');
      return;
    }
    console.log('[TextPanel] Saving for clipId:', clipId, 'promptText:', promptText);
    setFloatingPanelData(clipId, { promptText });
  }, [promptText, clipId, setFloatingPanelData]);

  const setRefs = (element: HTMLDivElement | null) => {
    panelRef.current = element;
    setDroppableRef(element);
  };

  const handleResizeStart = (e: React.MouseEvent, edge: string) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);
    resizeEdgeRef.current = edge;
    resizeStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      width: size.width,
      height: size.height,
      posX: position.x,
      posY: position.y,
    };
  };

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - resizeStartRef.current.x;
      const deltaY = e.clientY - resizeStartRef.current.y;
      const edge = resizeEdgeRef.current;
      
      let newWidth = resizeStartRef.current.width;
      let newHeight = resizeStartRef.current.height;
      let newX = resizeStartRef.current.posX;
      let newY = resizeStartRef.current.posY;

      // Handle horizontal resizing
      if (edge.includes('right')) {
        newWidth = Math.min(500, Math.max(300, resizeStartRef.current.width + deltaX));
      } else if (edge.includes('left')) {
        newWidth = Math.min(500, Math.max(300, resizeStartRef.current.width - deltaX));
        newX = resizeStartRef.current.posX + (resizeStartRef.current.width - newWidth);
      }

      // Handle vertical resizing
      if (edge.includes('bottom')) {
        newHeight = Math.min(250, Math.max(150, resizeStartRef.current.height + deltaY));
      } else if (edge.includes('top')) {
        newHeight = Math.min(250, Math.max(150, resizeStartRef.current.height - deltaY));
        newY = resizeStartRef.current.posY + (resizeStartRef.current.height - newHeight);
      }

      setSize({ width: newWidth, height: newHeight });
      setPosition({ x: newX, y: newY });
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      resizeEdgeRef.current = '';
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, size, position]);

  const style: React.CSSProperties = {
    position: 'fixed',
    left: `${position.x}px`,
    top: `${position.y}px`,
    transform: transform ? `translate(${transform.x}px, ${transform.y}px)` : undefined,
    width: `${size.width}px`,
    height: `${size.height}px`,
    zIndex: isDragging || isResizing ? 10000 : 9998,
    pointerEvents: 'auto',
  };

  return (
    <div
      ref={setRefs}
      style={style}
      className={`
        bg-brand rounded-xl shadow-2xl
        flex flex-col
        font-poppins
        ${isOver ? 'ring-2 ring-brand-accent' : ''}

      `}
    >
      {/* Top Tab - Draggable Area */}
      <div 
        ref={setDraggableRef}
        {...attributes}
        {...listeners}
        className="relative flex items-center justify-end px-3 py-2 cursor-grab active:cursor-grabbing bg-brand border-b border-brand-light/10 rounded-t-xl"
      >
        {/* Close Button */}
        <button
          onPointerDown={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            onDelete?.();
          }}
          className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 flex items-center justify-center transition-colors pointer-events-auto"
          title="Close panel"
        >
          <X size={10} className="text-black" />
        </button>
      </div>

      {/* Panel Content Area */}
      <div className="flex-1 px-4 pt-4 pb-2 overflow-auto flex flex-col">
        {/* Text Input Field - Full Height, No Border */}
        <textarea
          placeholder="Write your prompt..."
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          className="w-full flex-1 bg-transparent text-white font-poppins text-sm resize-none focus:outline-none placeholder:text-brand-light/40"
        />
      </div>

      {/* Bottom Bar with Model Pill */}
      <div className="relative flex items-center justify-start px-3 py-2 bg-brand rounded-b-xl">
        {/* Model Pill Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowModelMenu(!showModelMenu);
          }}
          className="flex items-center gap-2 px-3 py-1 bg-white/10 hover:bg-white/15 rounded-full transition-colors cursor-pointer"
        >
          <span className="text-white text-xs font-poppins font-medium whitespace-nowrap">
            {modelPillText}
          </span>
          <ChevronDown 
            size={14} 
            className={`text-white transition-transform ${showModelMenu ? 'rotate-180' : ''}`}
          />
        </button>
        
        {/* Model Selector Menu */}
        {showModelMenu && modelClips.length > 0 && (
          <>
            {/* Dimmed overlay */}
            <div
              className="fixed inset-0 bg-transparent"
              style={{ zIndex: 9997 }}
              onClick={() => setShowModelMenu(false)}
            />
            
            {/* Menu Dropdown */}
            <div 
              className="absolute bottom-full left-3 mb-2 bg-brand-background border border-brand-light/10 rounded-lg shadow-2xl overflow-hidden"
              style={{ 
                pointerEvents: 'auto',
                minWidth: '220px',
                maxHeight: '300px',
                overflowY: 'auto',
                zIndex: 9998
              }}
            >
              {/* Function Header */}
              <div className="px-4 py-2 bg-brand-light/5 border-b border-brand-light/10">
                <span className="text-brand-light/60 text-xs font-poppins font-medium uppercase tracking-wider">
                  Function
                </span>
              </div>
              
              {/* Model Clips List */}
              {modelClips.map((modelClip) => {
                const isActive = modelClip.clipId === clipId;
                const displayName = modelClip.name || 'Model';
                const displayTrack = modelClip.trackName || '';
                
                return (
                  <button
                    key={modelClip.clipId}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleModelSwitch(modelClip.clipId);
                    }}
                    onMouseEnter={() => setHoveredModelClipId(modelClip.clipId)}
                    onMouseLeave={() => setHoveredModelClipId(null)}
                    className={`w-full px-4 py-3 text-left font-poppins transition-colors flex items-center justify-between ${
                      hoveredModelClipId === modelClip.clipId ? 'bg-white/10' : ''
                    } ${isActive ? 'bg-white/5' : ''}`}
                  >
                    <div className="flex flex-col gap-0.5">
                      <span className="text-white text-sm font-medium">
                        {displayName}
                      </span>
                      {displayTrack && (
                        <span className="text-brand-light/60 text-xs">
                          {displayTrack}
                        </span>
                      )}
                    </div>
                    {isActive && (
                      <Check size={16} className="text-brand-accent" />
                    )}
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Resize Handles */}
      {/* Edges */}
      <div
        onMouseDown={(e) => handleResizeStart(e, 'left')}
        className="absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize"
        style={{ pointerEvents: 'auto' }}
      />
      <div
        onMouseDown={(e) => handleResizeStart(e, 'right')}
        className="absolute right-0 top-0 bottom-0 w-1 cursor-ew-resize"
        style={{ pointerEvents: 'auto' }}
      />
      <div
        onMouseDown={(e) => handleResizeStart(e, 'top')}
        className="absolute left-0 right-0 top-0 h-1 cursor-ns-resize"
        style={{ pointerEvents: 'auto' }}
      />
      <div
        onMouseDown={(e) => handleResizeStart(e, 'bottom')}
        className="absolute left-0 right-0 bottom-0 h-1 cursor-ns-resize"
        style={{ pointerEvents: 'auto' }}
      />
      
      {/* Corners */}
      <div
        onMouseDown={(e) => handleResizeStart(e, 'top-left')}
        className="absolute left-0 top-0 w-3 h-3 cursor-nwse-resize"
        style={{ pointerEvents: 'auto' }}
      />
      <div
        onMouseDown={(e) => handleResizeStart(e, 'top-right')}
        className="absolute right-0 top-0 w-3 h-3 cursor-nesw-resize"
        style={{ pointerEvents: 'auto' }}
      />
      <div
        onMouseDown={(e) => handleResizeStart(e, 'bottom-left')}
        className="absolute left-0 bottom-0 w-3 h-3 cursor-nesw-resize"
        style={{ pointerEvents: 'auto' }}
      />
      <div
        onMouseDown={(e) => handleResizeStart(e, 'bottom-right')}
        className="absolute right-0 bottom-0 w-3 h-3 cursor-nwse-resize"
        style={{ pointerEvents: 'auto' }}
      />
    </div>
  );
};

export default FloatingTextPanel;
