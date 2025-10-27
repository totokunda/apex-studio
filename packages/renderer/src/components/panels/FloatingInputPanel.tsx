import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useDraggable, useDroppable, useDndMonitor } from '@dnd-kit/core';
import { X, Minus, ImagePlus, PencilLine, ChevronDown, Check } from 'lucide-react';
import { MediaItem } from '@/components/media/Item';
import { readFileBuffer, fileURLToPath } from '@app/preload';
import { useControlsStore } from '@/lib/control';
import { useClipStore } from '@/lib/clip';
import type { ModelClipProps } from '@/lib/types';

interface FloatingInputPanelProps {
  id?: string;
  clipId: string;
  initialPosition: { x: number; y: number };
  onDelete?: () => void;
  onMinimize?: () => void;
}

const FloatingInputPanel: React.FC<FloatingInputPanelProps> = ({ 
  id = 'floating-input-panel',
  clipId,
  initialPosition,
  onDelete,
  onMinimize
}) => {
  const setFloatingPanelData = useControlsStore((s) => s.setFloatingPanelData);
  const getFloatingPanelData = useControlsStore((s) => s.getFloatingPanelData);
  const setActiveClipId = useControlsStore((s) => s.setActiveClipId);
  const getClipById = useClipStore((s) => s.getClipById);
  const clips = useClipStore((s) => s.clips);
  
  const [position, setPosition] = useState(initialPosition);
  const [size, setSize] = useState({ width: 500, height: 300 });
  const [isResizing, setIsResizing] = useState(false);
  const [droppedImage, setDroppedImage] = useState<MediaItem | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [imageObjectUrl, setImageObjectUrl] = useState<string | null>(null);
  const [promptText, setPromptText] = useState<string>('');
  const [hoveredEditIcon, setHoveredEditIcon] = useState<boolean>(false);
  const [hoveredInputArea, setHoveredInputArea] = useState<boolean>(false);
  const [showContextMenu, setShowContextMenu] = useState<boolean>(false);
  const [hoveredMenuItem, setHoveredMenuItem] = useState<'replace' | 'remove' | null>(null);
  const [showModelMenu, setShowModelMenu] = useState<boolean>(false);
  const [hoveredModelClipId, setHoveredModelClipId] = useState<string | null>(null);
  
  // Get clip data for model info
  const clip = getClipById(clipId) as ModelClipProps | undefined;
  const modelName = clip?.name || 'Model';
  const trackName = clip?.trackName || '';
  
  // Format track name to abbreviation (e.g., "Image to Video" -> "I2V")
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
    console.log('[InputPanel] Loading for clipId:', clipId, 'savedData:', savedData);
    setImageBlob(savedData.imageBlob);
    setPromptText(savedData.promptText);
    // Reset temp states when switching clips
    setDroppedImage(null);
    setSelectedFile(null);
    setShowContextMenu(false);
    setHoveredInputArea(false);
    setHoveredEditIcon(false);
    
    // Mark as loaded after a short delay to ensure state has settled
    setTimeout(() => {
      hasLoadedRef.current = true;
    }, 0);
  }, [clipId, getFloatingPanelData]);
  
  // Load image from dropped/selected files and convert to blob
  useEffect(() => {
    let cancelled = false;
    
    (async () => {
      try {
        if (droppedImage?.assetUrl) {
          // For dropped images from media panel, load via readFileBuffer
          const filePath = droppedImage.assetUrl.startsWith('file://') 
            ? fileURLToPath(droppedImage.assetUrl) 
            : droppedImage.assetUrl;
          
          const buffer = await readFileBuffer(filePath);
          if (cancelled) return;
          
          const blob = new Blob([buffer as unknown as ArrayBuffer], { type: 'image/png' });
          setImageBlob(blob);
        } else if (selectedFile) {
          // For file input selection, use the file directly as a blob
          setImageBlob(selectedFile);
        }
      } catch (error) {
        console.error('Failed to load image:', error);
        if (!cancelled) {
          setImageBlob(null);
        }
      }
    })();
    
    return () => {
      cancelled = true;
    };
  }, [droppedImage, selectedFile]);

  // Create object URL from blob for rendering
  useEffect(() => {
    if (imageBlob) {
      const objectUrl = URL.createObjectURL(imageBlob);
      setImageObjectUrl(objectUrl);
      
      // Cleanup function
      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    } else {
      setImageObjectUrl(null);
    }
  }, [imageBlob]);
  const panelRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastTransform = useRef({ x: 0, y: 0 });
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0, posX: 0, posY: 0 });

  // Draggable setup for moving the panel around
  const { attributes, listeners, setNodeRef: setDraggableRef, transform, isDragging } = useDraggable({
    id: `${id}-draggable`,
    data: { type: 'floating-panel' },
  });

  // Droppable setup for accepting media/timeline items
  const { setNodeRef: setDroppableRef, isOver } = useDroppable({
    id: `${id}-droppable`,
    data: {
      accepts: ['media', 'model', 'audio', 'video', 'image', 'timeline-clip', 'preprocessor'],
    },
  });

  // Separate droppable for image input - only accepts image type
  const { setNodeRef: setImageDropRef, isOver: isImageOver } = useDroppable({
    id: `${id}-image-droppable`,
    data: {
      accepts: ['image'],
    },
  });

  // Handle image drop from DndContext
  useDndMonitor({
    onDragEnd: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id;
      
      // Check if dropped on the image droppable and it's an image type
      if (overId === `${id}-image-droppable` && data && data.type === 'image') {
        setDroppedImage(data);
        setSelectedFile(null); // Clear any file input selection
      }
    },
  });

  // Handle file input change
//   const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     const file = e.target.files?.[0];
//     if (file && file.type.startsWith('image/')) {
//       setSelectedFile(file);
//       setDroppedImage(null); // Clear any dropped image
//     }
//   };

  // Handle replace media - opens file picker
//   const handleReplaceMedia = () => {
//     setShowContextMenu(false);
//     fileInputRef.current?.click();
//   };

    const handleReplaceMedia = (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
    
        const input = fileInputRef.current;
        if (!input) return;
    
        // Prefer showPicker when available
        const anyInput = input as HTMLInputElement & { showPicker?: () => void };
        try {
        if (anyInput.showPicker) anyInput.showPicker();
        else input.click();
        } catch {
        input.click();
        }
        setShowContextMenu(false);
    };
    
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && file.type.startsWith('image/')) {
        setSelectedFile(file);
        setDroppedImage(null);
        }
         // close after selection
    };

  // Handle remove media - clears the image and resets to upload state
  const handleRemoveMedia = () => {
    setShowContextMenu(false);
    setImageBlob(null);
    setImageObjectUrl(null);
    setSelectedFile(null);
    setDroppedImage(null);
    // Reset the file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Toggle context menu
  const handleEditClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setShowContextMenu(!showContextMenu);
  };

  // Cleanup object URL when component unmounts or selectedFile changes
  useEffect(() => {
    if (selectedFile) {
      const objectUrl = URL.createObjectURL(selectedFile);
      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    }
  }, [selectedFile]);

  // Store transform while dragging
  useEffect(() => {
    if (isDragging && transform) {
      lastTransform.current = { x: transform.x, y: transform.y };
    }
  }, [isDragging, transform]);

  // Update position when dragging ends
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

  // Show pencil briefly when image loads
  useEffect(() => {
    if (imageObjectUrl) {
      setHoveredInputArea(true);
      const timer = setTimeout(() => {
        setHoveredInputArea(false);
      }, 2000); // Show for 2 seconds after image loads
      
      return () => clearTimeout(timer);
    }
  }, [imageObjectUrl]);

  // Save image blob when it changes (NOT the object URL, as URLs are component-scoped)
  useEffect(() => {
    if (!hasLoadedRef.current) {
      console.log('[InputPanel] Skipping image save - not loaded yet');
      return;
    }
    console.log('[InputPanel] Saving image for clipId:', clipId, 'hasBlob:', !!imageBlob);
    setFloatingPanelData(clipId, { 
      imageBlob,
      imageSource: droppedImage ? 'dropped' : selectedFile ? 'selected' : null
    });
  }, [imageBlob, clipId, setFloatingPanelData, droppedImage, selectedFile]);

  // Save prompt text when it changes
  useEffect(() => {
    if (!hasLoadedRef.current) {
      console.log('[InputPanel] Skipping text save - not loaded yet');
      return;
    }
    console.log('[InputPanel] Saving text for clipId:', clipId, 'promptText:', promptText);
    setFloatingPanelData(clipId, { promptText });
  }, [promptText, clipId, setFloatingPanelData]);

  // Combine droppable ref with panel ref
  const setRefs = (element: HTMLDivElement | null) => {
    panelRef.current = element;
    setDroppableRef(element);
  };

  // Handle edge resizing
  const resizeEdgeRef = useRef<string>('');
  
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
            newWidth = Math.min(800, Math.max(300, resizeStartRef.current.width + deltaX));
          } else if (edge.includes('left')) {
            newWidth = Math.min(800, Math.max(300, resizeStartRef.current.width - deltaX));
            newX = resizeStartRef.current.posX + (resizeStartRef.current.width - newWidth);
          }

      // Handle vertical resizing
      if (edge.includes('bottom')) {
        newHeight = Math.min(600, Math.max(250, resizeStartRef.current.height + deltaY));
      } else if (edge.includes('top')) {
        newHeight = Math.min(600, Math.max(250, resizeStartRef.current.height - deltaY));
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
    <>
      {/* Dimmed overlay when context menu is open */}
      {showContextMenu && (
        <div
          className="fixed inset-0 bg-black/20"
          style={{ zIndex: 9997 }}
          onClick={() => setShowContextMenu(false)}
        />
      )}
      
      <div
        ref={setRefs}
        style={{
          ...style,
          pointerEvents: showContextMenu ? 'none' : 'auto'
        }}
        className={`
          bg-brand rounded-xl shadow-2xl
          flex flex-col
          font-poppins
          ${isOver ? 'ring-2 ring-brand-accent' : ''}

        `}
      >
      {/* Chrome-style Tab */}
      <div 
        ref={showContextMenu ? null : setDraggableRef}
        {...(showContextMenu ? {} : attributes)}
        {...(showContextMenu ? {} : listeners)}
        className="relative flex items-center justify-end px-3 py-2 cursor-grab active:cursor-grabbing bg-brand border-b border-brand-light/10 rounded-t-xl"
      >
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
      <div className="flex-1 px-4 pt-4 overflow-auto flex flex-col gap-4">
        {/* Text Input Field - Top Half */}
        <textarea
          placeholder="Write your prompt..."
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          className="w-full flex-1 min-h-10 bg-brand border border-brand-light/20 rounded-lg p-3 text-white font-poppins text-sm resize-none focus:outline-none focus:border-brand-accent/50 placeholder:text-brand-light/40"
          />
        
        {/* Image Upload Area */}
        <div className="flex items-center justify-center" style={{ minHeight: '140px' }}>
          <label 
            ref={setImageDropRef}
            htmlFor={imageObjectUrl ? undefined : "image-upload"}
            onMouseEnter={() => {
              if (imageObjectUrl) {
                setHoveredInputArea(true);
              }
            }}
            onMouseLeave={() => {
              if (imageObjectUrl) {
                setHoveredInputArea(false);
              }
            }}
            onClick={(e) => {
              // When image is present, prevent any clicks on the label itself
              if (imageObjectUrl) {
                e.preventDefault();
                e.stopPropagation();
              }
            }}
            className={`relative flex flex-col items-center justify-center gap-0 px-3 py-1 bg-brand-light/5 backdrop-blur-md rounded-3xl ${imageObjectUrl ? '' : 'cursor-pointer hover:bg-brand-light/10'} transition-all overflow-hidden ${isImageOver ? 'ring-2 ring-brand-accent bg-brand-accent/20' : ''}`}
            style={{
              width: size.width > 500 ? `${160 + ((size.width - 500) / (800 - 500)) * (230 - 160)}px` : '160px',
              height: size.height > 300 ? `${132 + ((size.height - 300) / (600 - 300)) * (198 - 132)}px` : '132px',
              transition: isResizing ? 'none' : 'width 0.15s ease-out, height 0.15s ease-out',
              pointerEvents: 'auto'
            }}
          >
            {/* Icon and text behind the image */}
            <div 
              className="flex items-center justify-center relative"
              style={{
                width: size.width > 500 ? `${64 + ((size.width - 500) / (800 - 500)) * (96 - 64)}px` : '64px',
                height: size.width > 500 ? `${64 + ((size.width - 500) / (800 - 500)) * (96 - 64)}px` : '64px',
                marginTop: size.height > 300 ? `${-5 + ((size.height - 300) / (600 - 300)) * (7 - 1)}px` : '-5px',
                transition: isResizing ? 'none' : 'width 0.15s ease-out, height 0.15s ease-out, margin-top 0.15s ease-out',
                zIndex: 0
              }}
            >
              <ImagePlus 
                size={size.width > 500 ? 30 + ((size.width - 500) / (800 - 500)) * (48 - 30) : 30} 
                className="text-white"
                style={{
                  transition: isResizing ? 'none' : 'all 0.15s ease-out'
                }}
              />
            </div>
            <span 
              className="text-brand-light/70 font-poppins font-medium -mt-1 relative"
              style={{
                fontSize: size.width > 500 ? `${14 + ((size.width - 500) / (800 - 500)) * (20 - 14)}px` : '14px',
                transition: isResizing ? 'none' : 'font-size 0.15s ease-out',
                zIndex: 0
              }}
            >
              Add Image
            </span>
            
            {/* Absolute positioned image on top */}
            {imageObjectUrl && (
              <>
                <div 
                  className="absolute inset-0 rounded-3xl"
                  style={{
                    backgroundImage: `url("${imageObjectUrl}")`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat',
                    zIndex: 10
                  }}
                />
                
                {/* Edit overlay icon - show on hover or when context menu is open */}
                {(hoveredInputArea || showContextMenu) && (
                  <div 
                    className="absolute inset-0 flex items-center justify-center" 
                    style={{ zIndex: 20, pointerEvents: 'none' }}
                    onClick={(e) => {
                      // Prevent clicks on the overlay from propagating
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                  >
                    {/* Edit button */}
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        handleEditClick(e);
                      }}
                      onMouseEnter={() => setHoveredEditIcon(true)}
                      onMouseLeave={() => setHoveredEditIcon(false)}
                      className={`flex items-center justify-center w-12 h-12 rounded-lg transition-all ${
                        showContextMenu
                          ? 'bg-brand-accent' 
                          : hoveredEditIcon
                            ? 'bg-white/40'
                            : 'bg-white/20'
                      }`}
                      style={{ pointerEvents: showContextMenu ? 'none' : 'auto' }}
                      title="Edit image"
                    >
                      <PencilLine size={24} className="text-white" />
                    </button>
                  </div>
                )}
              </>
            )}
            
            <input 
              ref={fileInputRef}
              id="image-upload" 
              type="file" 
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
          
          {/* Context Menu - Positioned with top-left corner at edit icon center */}
          {showContextMenu && imageObjectUrl && (
            <div 
              className="absolute bg-brand-background border border-brand-light/10 rounded-lg shadow-2xl overflow-hidden"
              style={{ 
                pointerEvents: 'auto',
                left: '50%',
                top: '50%',
                minWidth: size.width > 600 ? '200px' : '180px',
                zIndex: 50
              }}
            >
              {/* Replace option */}
              <button
                onClick={handleReplaceMedia}
                onMouseEnter={() => setHoveredMenuItem('replace')}
                onMouseLeave={() => setHoveredMenuItem(null)}
                className={`w-full px-4 py-2.5 text-left font-poppins text-white transition-colors whitespace-nowrap ${
                  hoveredMenuItem === 'replace' ? 'bg-white/10' : ''
                }`}
                style={{
                  fontSize: size.width > 600 ? '13px' : '12px'
                }}
              >
                Replace 
              </button>
              
              {/* Remove option */}
              <button
                onClick={handleRemoveMedia}
                onMouseEnter={() => setHoveredMenuItem('remove')}
                onMouseLeave={() => setHoveredMenuItem(null)}
                className={`w-full px-4 py-2.5 text-left font-poppins text-white transition-colors ${
                  hoveredMenuItem === 'remove' ? 'bg-white/10' : ''
                }`}
                style={{
                  fontSize: size.width > 600 ? '13px' : '12px'
                }}
              >
                Remove
              </button>
            </div>
          )}
        </div>
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

      {/* Resize Handles - disabled when context menu is open */}
      {!showContextMenu && (
        <>
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
        </>
      )}
    </div>
    </>
  );
};

export default FloatingInputPanel;

