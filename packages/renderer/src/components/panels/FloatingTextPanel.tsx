import React, { useState, useRef, useEffect } from 'react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { X, Minus } from 'lucide-react';
import { useControlsStore } from '@/lib/control';

interface FloatingTextPanelProps {
  id?: string;
  clipId: string;
  initialPosition: { x: number; y: number };
  onDelete?: () => void;
  onMinimize?: () => void;
}

const FloatingTextPanel: React.FC<FloatingTextPanelProps> = ({ 
  id = 'floating-text-panel',
  clipId,
  initialPosition,
  onDelete,
  onMinimize
}) => {
  const setFloatingPanelData = useControlsStore((s) => s.setFloatingPanelData);
  const getFloatingPanelData = useControlsStore((s) => s.getFloatingPanelData);
  
  const [position, setPosition] = useState(initialPosition);
  const [size, setSize] = useState({ width: 300, height: 150 });
  const [promptText, setPromptText] = useState<string>('');
  const [isResizing, setIsResizing] = useState(false);
  
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
        ${isDragging ? 'opacity-70' : 'opacity-100'}
      `}
    >
      {/* Chrome-style Tab */}
      <div 
        ref={setDraggableRef}
        {...attributes}
        {...listeners}
        className="relative flex items-center justify-end px-3 py-2 cursor-grab active:cursor-grabbing bg-brand border-b border-brand-light/10 rounded-t-xl"
      >
        {/* Title - Centered */}
        <span className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-brand-light text-sm font-poppins font-medium">Apex Studio</span>
        
        {/* Buttons */}
        <div className="flex items-center gap-2">
          {/* Minimize Button */}
          <button
            onPointerDown={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onMinimize?.();
            }}
            className="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 flex items-center justify-center transition-colors pointer-events-auto font-poppins"
            title="Minimize panel"
          >
            <Minus size={10} className="text-black" />
          </button>
          
          {/* Close Button */}
          <button
            onPointerDown={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onDelete?.();
            }}
            className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 flex items-center justify-center transition-colors pointer-events-auto font-poppins"
            title="Close panel"
          >
            <X size={10} className="text-black" />
          </button>
        </div>
      </div>

      {/* Panel Content Area */}
      <div className="flex-1 p-4 overflow-auto flex flex-col">
        {/* Text Input Field - Full Height */}
        <textarea
          placeholder="Write your prompt..."
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          className="w-full flex-1 bg-brand border border-brand-light/20 rounded-lg p-3 text-white font-poppins text-sm resize-none focus:outline-none focus:border-brand-accent/50 placeholder:text-brand-light/40"
        />
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
