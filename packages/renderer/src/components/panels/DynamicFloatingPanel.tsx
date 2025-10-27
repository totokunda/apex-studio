import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useDraggable, useDroppable, useDndMonitor } from '@dnd-kit/core';
import { X, Check, PencilLine } from 'lucide-react';
import { getFloatingPanelUI, type InputSpec, type FloatingPanelUIData } from '@/lib/manifest';
import { ImagePlus, Video } from 'lucide-react';
import { useControlsStore } from '@/lib/control';
import { useClipStore } from '@/lib/clip';
import type { ModelClipProps } from '@/lib/types';
import { importMediaPaths, listConvertedMedia, pickMediaPaths, fileURLToPath } from '@app/preload';
import { toast } from 'sonner';
import type { MediaItem } from '@/components/media/Item';

interface DynamicFloatingPanelProps {
  id?: string;
  clipId: string;
  manifestId: string;
  modelName?: string;
  trackName?: string;
  initialPosition: { x: number; y: number };
  onDelete?: () => void;
}

const DynamicFloatingPanel: React.FC<DynamicFloatingPanelProps> = ({
  id = 'dynamic-floating-panel',
  clipId,
  manifestId,
  modelName = 'Panel',
  trackName = '',
  initialPosition,
  onDelete
}) => {
  const setActiveClipId = useControlsStore((s) => s.setActiveClipId);
  const clips = useClipStore((s) => s.clips);
  const setFloatingPanelData = useControlsStore((s) => s.setFloatingPanelData);
  const getFloatingPanelData = useControlsStore((s) => s.getFloatingPanelData);
  
  const [position, setPosition] = useState(initialPosition);
  const [panelData, setPanelData] = useState<FloatingPanelUIData | null>(null);
  const [loading, setLoading] = useState(true);
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [showModelMenu, setShowModelMenu] = useState<boolean>(false);
  const [hoveredModelClipId, setHoveredModelClipId] = useState<string | null>(null);
  
  const [openShortcutMenu, setOpenShortcutMenu] = useState<string | null>(null);
  const [activeDropTarget, setActiveDropTarget] = useState<string | null>(null);
  const [showContextMenu, setShowContextMenu] = useState<string | null>(null); // inputId of active context menu
  const [contextMenuPosition, setContextMenuPosition] = useState<{ top: number; left: number } | null>(null);
  const [hoveredMenuItem, setHoveredMenuItem] = useState<'replace' | 'remove' | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const lastTransform = useRef({ x: 0, y: 0 });
  const hasLoadedRef = useRef(false);
  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});
  const pencilButtonRefs = useRef<{ [key: string]: HTMLButtonElement | null }>({});

  const modelPillButtonRef = useRef<HTMLButtonElement>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; left: number } | null>(null);
  
  const shortcutButtonRefs = useRef<{ [key: string]: HTMLButtonElement | null }>({});
  const [shortcutMenuPosition, setShortcutMenuPosition] = useState<{ top: number; left: number } | null>(null);

  // Format track name to abbreviation
  const formatTrackAbbreviation = (track: string): string => {
    const trackMap: { [key: string]: string } = {
      'Text to Image': 'T2I',
      'Image to Image': 'I2I',
      'Text to Video': 'T2V',
      'Image to Video': 'I2V',
      'Video to Video': 'V2V',
      'Audio-Image to Video': 'AI2V',
      'Image-Control to Video': 'IC2V',
      'Image-Control to Image': 'IC2I',
      'Image-Mask to Image': 'IM2I',
      'Video Animation Control': 'VACE',
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
    setActiveClipId(targetClipId);
  };

  const { attributes, listeners, setNodeRef: setDraggableRef, transform, isDragging } = useDraggable({
    id: `${id}-draggable`,
    data: { type: 'floating-panel' },
  });

  // Load panel configuration from API
  useEffect(() => {
    const fetchPanelConfig = async () => {
      // Reset loaded flag when switching clips
      hasLoadedRef.current = false;
      
      setLoading(true);
      const data = await getFloatingPanelUI(manifestId);
      console.log('[DynamicPanel] Loaded config for', manifestId, data);
      setPanelData(data);
      
      // Load saved form values for this clip
      const savedData = getFloatingPanelData(clipId) as Record<string, any>;
      console.log('[DynamicPanel] Loading for clipId:', clipId, 'savedData:', savedData);
      
      // Initialize form values with saved data or defaults
      if (data) {
        const initialValues: Record<string, any> = {};
        data.inputs.forEach(input => {
          // Only use saved value if it exists AND matches this input's ID
          if (savedData && savedData[input.id] !== undefined) {
            initialValues[input.id] = savedData[input.id];
          } else if (input.default !== undefined) {
            initialValues[input.id] = input.default;
          }
        });
        setFormValues(initialValues);
      }
      
      setLoading(false);
      
      // Mark as loaded after a short delay
      setTimeout(() => {
        hasLoadedRef.current = true;
      }, 0);
    };

    fetchPanelConfig();
  }, [manifestId, clipId, getFloatingPanelData]);

  // Save form values when they change
  useEffect(() => {
    if (!hasLoadedRef.current) {
      console.log('[DynamicPanel] Skipping save - not loaded yet');
      return;
    }
    
    if (!panelData) {
      console.log('[DynamicPanel] Skipping save - panel data not loaded');
      return;
    }
    
    // Only save values for inputs that exist in the current panel
    const validInputIds = new Set(panelData.inputs.map(input => input.id));
    const filteredValues: Record<string, any> = {};
    Object.keys(formValues).forEach(key => {
      if (validInputIds.has(key)) {
        filteredValues[key] = formValues[key];
      }
    });
    
    console.log('[DynamicPanel] Saving formValues for clipId:', clipId, 'values:', filteredValues);
    setFloatingPanelData(clipId, filteredValues);
  }, [formValues, clipId, setFloatingPanelData, panelData]);

  // Convert file:// URL to app://user-data/ protocol URL (like MediaMenu does)
  const convertToProtocolUrl = (assetUrl: string): string => {
    try {
      const filePath = assetUrl.startsWith('file://') ? fileURLToPath(assetUrl) : assetUrl;
      const url = new URL(`app://user-data/${filePath}`);
      return url.toString();
    } catch (error) {
      console.error('[DynamicPanel] Failed to convert URL:', assetUrl, error);
      return assetUrl;
    }
  };

  // Handle drag-drop from media panel
  useDndMonitor({
    onDragOver: (event) => {
      const overId = event.over?.id;
      
      // Check if hovering over a media input drop zone
      if (overId && typeof overId === 'string' && overId.startsWith(`${id}-media-`)) {
        const inputId = overId.replace(`${id}-media-`, '');
        setActiveDropTarget(inputId);
      } else {
        setActiveDropTarget(null);
      }
    },
    onDragEnd: (event) => {
      const data = event.active?.data?.current as MediaItem | undefined;
      const overId = event.over?.id;
      
      // Check if dropped on a media input
      if (overId && typeof overId === 'string' && overId.startsWith(`${id}-media-`)) {
        const inputId = overId.replace(`${id}-media-`, '');
        
        // Find the input spec to check its required type
        const targetInput = panelData?.inputs.find(i => i.id === inputId);
        const requiredType = targetInput?.type === 'image' || targetInput?.type === 'image+preprocessor' ? 'image' : 'video';
        
        // Only accept media if type matches
        if (data && data.type === requiredType) {
          console.log('[DynamicPanel] Media dropped:', data, 'on input:', inputId);
          setFormValues(prev => ({
            ...prev,
            [inputId]: {
              assetUrl: data.assetUrl,
              type: data.type,
              name: data.name
            }
          }));
        } else if (data) {
          console.warn('[DynamicPanel] Media type mismatch. Expected:', requiredType, 'Got:', data.type);
          toast.error(`Cannot drop ${data.type} into ${requiredType} input`, { position: "bottom-right", duration: 2000 });
        }
      }
      
      setActiveDropTarget(null);
    },
  });

  // Handle dragging
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

  const style: React.CSSProperties = {
    position: 'fixed',
    left: `${position.x}px`,
    top: `${position.y}px`,
    transform: transform ? `translate(${transform.x}px, ${transform.y}px)` : undefined,
    width: '600px',
    minHeight: '400px',
    zIndex: isDragging ? 45 : 40,
    pointerEvents: 'auto',
    overflow: 'visible',
  };

  // Get input specs for a specific region
  const getInputsForRegion = (regionName: 'media' | 'text' | 'shortcuts'): InputSpec[] => {
    if (!panelData) return [];
    
    const region = panelData.floating_panel.regions[regionName];
    if (!region) return [];

    return region.inputs
      .map(inputId => panelData.inputs.find(inp => inp.id === inputId))
      .filter(Boolean) as InputSpec[];
  };

  // Handle file picker for media inputs
  const handlePickMedia = async (inputId: string, type: 'image' | 'video') => {
    try {
      const filters = type === 'image' 
        ? [{ name: 'Image Files', extensions: ['jpg', 'jpeg', 'png', 'gif', 'webp'] }]
        : [{ name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] }];
      
      const title = `Choose ${type === 'image' ? 'an image' : 'a video'}`;
      const res = await pickMediaPaths({ directory: false, filters, title });
      
      if (!res || res.length === 0) return;
      
      const selectedPath = res[0];
      
      // Import the media to the library
      const loadingId = toast.loading(`Importing media...`, { position: "bottom-right" });
      await importMediaPaths([selectedPath]);
      
      // Get the converted path from the media library
      const convertedMedia = await listConvertedMedia();
      const basename = selectedPath.split(/[/\\]/).pop() || '';
      const nameWithoutExt = basename.replace(/\.[^.]+$/, '');
      
      // Find the converted media item (most recent with matching name)
      const convertedItem = convertedMedia
        .filter(item => item.name.startsWith(nameWithoutExt))
        .sort((a, b) => (b.dateAddedMs || 0) - (a.dateAddedMs || 0))[0];
      
      toast.dismiss(loadingId);
      
      if (convertedItem) {
        setFormValues(prev => ({
          ...prev,
          [inputId]: {
            assetUrl: convertedItem.assetUrl,
            type: convertedItem.type,
            name: convertedItem.name
          }
        }));
        toast.success('Media imported successfully', { position: "bottom-right", duration: 2000 });
      }
    } catch (error) {
      console.error('Failed to pick media:', error);
      toast.error('Failed to import media', { position: "bottom-right", duration: 3000 });
    }
  };

  // Handle replace media - opens file picker for specific input
  const handleReplaceMedia = (inputId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const input = fileInputRefs.current[inputId];
    if (!input) return;
    
    const anyInput = input as HTMLInputElement & { showPicker?: () => void };
    try {
      if (anyInput.showPicker) anyInput.showPicker();
      else input.click();
    } catch {
      input.click();
    }
    setShowContextMenu(null);
  };

  // Handle remove media - clears the media for specific input
  const handleRemoveMedia = (inputId: string) => {
    setShowContextMenu(null);
    setFormValues(prev => {
      const newValues = { ...prev };
      delete newValues[inputId];
      return newValues;
    });
    // Reset the file input
    if (fileInputRefs.current[inputId]) {
      fileInputRefs.current[inputId]!.value = '';
    }
  };

  // Handle file input change for specific input
  const handleFileChange = async (inputId: string, type: 'image' | 'video', e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const expectedType = type === 'image' ? 'image/' : 'video/';
    if (!file.type.startsWith(expectedType)) {
      toast.error(`Please select a valid ${type} file`, { position: "bottom-right", duration: 3000 });
      return;
    }
    
    try {
      // Get file path - in Electron, files from input have a path property
      const tempPath = (file as any).path || file.name;
      
      const loadingId = toast.loading(`Importing ${type}...`, { position: "bottom-right" });
      await importMediaPaths([tempPath]);
      
      // Get the converted path from the media library
      const convertedMedia = await listConvertedMedia();
      const basename = file.name;
      const nameWithoutExt = basename.replace(/\.[^.]+$/, '');
      
      // Find the converted media item (most recent with matching name)
      const convertedItem = convertedMedia
        .filter(item => item.name.startsWith(nameWithoutExt))
        .sort((a, b) => (b.dateAddedMs || 0) - (a.dateAddedMs || 0))[0];
      
      toast.dismiss(loadingId);
      
      if (convertedItem) {
        setFormValues(prev => ({
          ...prev,
          [inputId]: {
            assetUrl: convertedItem.assetUrl,
            type: convertedItem.type,
            name: convertedItem.name
          }
        }));
        toast.success(`${type.charAt(0).toUpperCase() + type.slice(1)} imported successfully`, { position: "bottom-right", duration: 2000 });
      }
    } catch (error) {
      console.error('Failed to import file:', error);
      toast.error(`Failed to import ${type}`, { position: "bottom-right", duration: 3000 });
    }
  };

  // Render media drop zone component
  const MediaDropZone: React.FC<{ input: InputSpec; type: 'image' | 'video' }> = ({ input, type }) => {
    const { setNodeRef } = useDroppable({
      id: `${id}-media-${input.id}`,
      data: { accepts: [type] },
    });
    
    const [isHovered, setIsHovered] = useState(false);
    const [hoveredEditIcon, setHoveredEditIcon] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const mediaValue = formValues[input.id];
    const isActive = activeDropTarget === input.id;
    const Icon = type === 'image' ? ImagePlus : Video;
    
    // Validate that media type matches input type requirement
    const mediaTypeMatches = mediaValue?.type === type;
    const hasMedia = !!mediaValue?.assetUrl && mediaTypeMatches;
    const isContextMenuOpen = showContextMenu === input.id;
    
    // Convert file:// URL to app://user-data/ protocol URL (like MediaMenu)
    const protocolUrl = mediaValue?.assetUrl ? convertToProtocolUrl(mediaValue.assetUrl) : null;
    
    // Handle video play/pause on hover (but not when context menu is open)
    useEffect(() => {
      if (videoRef.current && type === 'video') {
        if (isHovered && !isContextMenuOpen) {
          videoRef.current.play().catch(err => console.error('[MediaDropZone] Play error:', err));
        } else {
          videoRef.current.pause();
          videoRef.current.currentTime = 0; // Reset to first frame
        }
      }
    }, [isHovered, isContextMenuOpen, type]);
    
    // Immediately stop hover state when context menu opens
    useEffect(() => {
      if (isContextMenuOpen) {
        setIsHovered(false);
      }
    }, [isContextMenuOpen]);
    
    console.log('[MediaDropZone]', input.id, 'protocolUrl:', protocolUrl, 'type:', type);
    
    // Handle click - if media exists, prevent default click
    const handleClick = (e: React.MouseEvent) => {
      if (hasMedia) {
        e.preventDefault();
        e.stopPropagation();
      } else {
        handlePickMedia(input.id, type);
      }
    };
    
    // Handle edit icon click
    const handleEditClick = (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      
      if (isContextMenuOpen) {
        setShowContextMenu(null);
        setContextMenuPosition(null);
      } else {
        // Get the pencil button position
        const buttonEl = pencilButtonRefs.current[input.id];
        if (buttonEl) {
          const rect = buttonEl.getBoundingClientRect();
          // Position menu with top-left corner at center of pencil button
          setContextMenuPosition({
            top: rect.top + rect.height / 2,
            left: rect.left + rect.width / 2
          });
        }
        setShowContextMenu(input.id);
      }
    };
    
    return (
      <div
        ref={setNodeRef}
        onClick={handleClick}
        onMouseEnter={() => hasMedia && setIsHovered(true)}
        onMouseLeave={() => hasMedia && setIsHovered(false)}
        className={`relative flex flex-col items-center justify-center gap-0 px-3 py-1 backdrop-blur-md rounded-3xl ${hasMedia ? '' : 'cursor-pointer hover:bg-brand-light/10'} transition-all overflow-hidden ${
          isActive ? 'bg-brand-light/20 ring-2 ring-brand-light' : 'bg-brand-light/5'
        }`}
        style={{ width: '160px', height: '132px' }}
      >
        {protocolUrl ? (
          <>
            {/* Hover label at top - only when media is loaded */}
            {isHovered && (
              <div className="absolute top-2 left-0 right-0 flex justify-center z-10 pointer-events-none">
                <span className="text-brand-light/70 text-sm font-poppins font-medium px-2 py-1 rounded ">
                  {input.label}
                </span>
              </div>
            )}
            
            {type === 'image' ? (
              <img 
                src={protocolUrl} 
                alt={input.label}
                className="absolute inset-0 w-full h-full object-cover rounded-3xl"
                onError={(e) => {
                  console.error('[MediaDropZone] Image load error:', protocolUrl, e);
                }}
              />
            ) : (
              <video 
                ref={videoRef}
                src={protocolUrl}
                className="absolute inset-0 w-full h-full object-cover rounded-3xl"
                muted
                loop
                playsInline
                preload="metadata"
                onError={(e) => {
                  console.error('[MediaDropZone] Video load error:', protocolUrl, e);
                }}
                onLoadedData={() => console.log('[MediaDropZone] Video loaded successfully:', protocolUrl)}
              />
            )}
            
            {/* Dim overlay on hover */}
            {isHovered && (
              <div className="absolute inset-0 bg-black/30 rounded-3xl pointer-events-none z-5" />
            )}
            
            {/* Edit overlay icon - show on hover or when context menu is open */}
            {(isHovered || isContextMenuOpen) && (
              <div 
                className="absolute inset-0 flex items-center justify-center" 
                style={{ zIndex: 20, pointerEvents: 'none' }}
              >
                <button
                  ref={(el) => { pencilButtonRefs.current[input.id] = el; }}
                  onClick={handleEditClick}
                  onMouseEnter={() => setHoveredEditIcon(true)}
                  onMouseLeave={() => setHoveredEditIcon(false)}
                  className={`flex items-center justify-center w-12 h-12 rounded-lg transition-all ${
                    isContextMenuOpen
                      ? 'bg-brand-accent' 
                      : hoveredEditIcon
                        ? 'bg-brand-accent'
                        : 'bg-white/20'
                  }`}
                  style={{ pointerEvents: isContextMenuOpen ? 'none' : 'auto' }}
                  title="Edit media"
                >
                  <PencilLine size={24} className="text-white" />
                </button>
              </div>
            )}
            
            {/* Hidden file input */}
            <input 
              ref={(el) => { fileInputRefs.current[input.id] = el; }}
              type="file" 
              accept={type === 'image' ? 'image/*' : 'video/*'}
              onChange={(e) => handleFileChange(input.id, type, e)}
              className="hidden"
            />
          </>
        ) : (
          <>
            <div className="flex items-center justify-center relative" style={{ width: '64px', height: '64px', marginTop: '-5px', zIndex: 0 }}>
              <Icon size={30} className="text-white" />
            </div>
            <span className="text-brand-light/70 font-poppins font-medium -mt-1 relative text-sm" style={{ zIndex: 0 }}>
              {input.label}
            </span>
          </>
        )}
      </div>
    );
  };

  // Render input based on type
  const renderInput = (input: InputSpec) => {
    const value = formValues[input.id] || '';

    switch (input.type) {
      case 'image':
      case 'image+preprocessor':
        return <MediaDropZone key={input.id} input={input} type="image" />;

      case 'video':
        return <MediaDropZone key={input.id} input={input} type="video" />;

      case 'text':
      case 'textarea':
        return (
          <textarea
            key={input.id}
            placeholder={input.label}
            value={value}
            onChange={(e) => setFormValues(prev => ({ ...prev, [input.id]: e.target.value }))}
            className="w-full bg-transparent text-white font-poppins text-sm resize-none focus:outline-none placeholder:text-brand-light/40 min-h-[80px] p-3 border border-brand-light/20 rounded-lg"
          />
        );

      case 'select':
        const options = (input.options || []) as Array<{ name: string; value: any }>;
        const currentValue = formValues[input.id] || input.default || (options[0]?.value);
        const currentOption = options.find((opt: { name: string; value: any }) => opt.value === currentValue);
        const isMenuOpen = openShortcutMenu === input.id;
        
        return (
          <div key={input.id}>
            <button
              ref={(el) => { shortcutButtonRefs.current[input.id] = el; }}
              onClick={(e) => {
                e.stopPropagation();
                if (isMenuOpen) {
                  setOpenShortcutMenu(null);
                  setShortcutMenuPosition(null);
                } else {
                  const buttonEl = shortcutButtonRefs.current[input.id];
                  if (buttonEl) {
                    const rect = buttonEl.getBoundingClientRect();
                    setShortcutMenuPosition({
                      top: rect.top - 8,
                      left: rect.left
                    });
                  }
                  setOpenShortcutMenu(input.id);
                }
              }}
              className="flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/15 rounded-full transition-colors cursor-pointer"
            >
              <span className="text-white text-xs font-poppins font-medium whitespace-nowrap">
                {currentOption?.name || input.label}
              </span>
            </button>
            
            {isMenuOpen && shortcutMenuPosition && (
              <>
                <div
                  className="fixed inset-0 bg-transparent"
                  style={{ zIndex: 9997 }}
                  onClick={() => {
                    setOpenShortcutMenu(null);
                    setShortcutMenuPosition(null);
                  }}
                />
                
                <div 
                  className="fixed bg-brand-background border border-brand-light/10 rounded-lg shadow-2xl overflow-hidden"
                  style={{ 
                    pointerEvents: 'auto',
                    minWidth: '180px',
                    maxWidth: '300px',
                    maxHeight: '250px',
                    overflowY: 'auto',
                    zIndex: 9998,
                    top: `${shortcutMenuPosition.top}px`,
                    left: `${shortcutMenuPosition.left}px`,
                    transform: 'translateY(-100%)'
                  }}
                >
                  {options.map((option: { name: string; value: any }) => {
                    const isSelected = currentValue === option.value;
                    
                    return (
                      <button
                        key={option.value}
                        onClick={(e) => {
                          e.stopPropagation();
                          setFormValues(prev => ({ ...prev, [input.id]: option.value }));
                          setOpenShortcutMenu(null);
                          setShortcutMenuPosition(null);
                        }}
                        className={`w-full px-4 py-2.5 text-left font-poppins transition-colors flex items-center justify-between gap-3 hover:bg-white/10 ${
                          isSelected ? 'bg-white/5' : ''
                        }`}
                      >
                        <span className="text-white text-sm flex-1 min-w-0">
                          {option.name}
                        </span>
                        {isSelected && (
                          <Check size={14} className="text-brand-accent flex-shrink-0" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div ref={panelRef} style={style} className="bg-brand rounded-xl shadow-2xl flex items-center justify-center font-poppins">
        <div className="text-brand-light/50 text-sm">Loading panel...</div>
      </div>
    );
  }

  if (!panelData) {
    return (
      <div ref={panelRef} style={style} className="bg-brand rounded-xl shadow-2xl flex items-center justify-center font-poppins">
        <div className="text-brand-light/50 text-sm">No panel configuration found</div>
      </div>
    );
  }

  const mediaInputs = getInputsForRegion('media');
  const textInputs = getInputsForRegion('text');
  const shortcutInputs = getInputsForRegion('shortcuts');

  return (
    <>
      {/* Dimmed overlay when context menu is open */}
      {showContextMenu && (
        <div
          className="fixed inset-0 bg-black/20"
          style={{ zIndex: 41 }}
          onClick={() => {
            setShowContextMenu(null);
            setContextMenuPosition(null);
          }}
        />
      )}
      
      {/* Context Menu for Media Inputs - Fixed position above overlay */}
      {showContextMenu && contextMenuPosition && (
        <div 
          className="fixed bg-brand-background border border-brand-light/10 rounded-lg shadow-2xl overflow-hidden"
          style={{ 
            pointerEvents: 'auto',
            left: `${contextMenuPosition.left}px`,
            top: `${contextMenuPosition.top}px`,
            minWidth: '180px',
            zIndex: 45
          }}
        >
          {/* Replace option */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleReplaceMedia(showContextMenu, e);
            }}
            onMouseEnter={() => setHoveredMenuItem('replace')}
            onMouseLeave={() => setHoveredMenuItem(null)}
            className={`w-full px-4 py-2.5 text-left font-poppins text-white transition-colors whitespace-nowrap ${
              hoveredMenuItem === 'replace' ? 'bg-white/10' : ''
            }`}
            style={{ fontSize: '12px' }}
          >
            Replace 
          </button>
          
          {/* Remove option */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleRemoveMedia(showContextMenu);
            }}
            onMouseEnter={() => setHoveredMenuItem('remove')}
            onMouseLeave={() => setHoveredMenuItem(null)}
            className={`w-full px-4 py-2.5 text-left font-poppins text-white transition-colors ${
              hoveredMenuItem === 'remove' ? 'bg-white/10' : ''
            }`}
            style={{ fontSize: '12px' }}
          >
            Remove
          </button>
        </div>
      )}
      
      <div ref={panelRef} style={style} className="bg-brand rounded-xl shadow-2xl flex flex-col font-poppins">
      {/* Top Tab - Draggable */}
      <div
        ref={setDraggableRef}
        {...attributes}
        {...listeners}
        className="relative flex items-center justify-end px-3 py-2 cursor-grab active:cursor-grabbing bg-brand border-b border-brand-light/10 rounded-t-xl"
      >
        <span className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-brand-light text-sm font-poppins font-medium">
          {modelName}
        </span>
        
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

      {/* Content Area */}
      <div className="flex-1 p-4 overflow-auto flex flex-col gap-4">
        {/* Text Region - Text inputs in a column */}
        {textInputs.length > 0 && (
          <div className="flex flex-col gap-3">
            {textInputs.map((input) => renderInput(input))}
          </div>
        )}

        {/* Media Region - Images in a row at bottom left */}
        {mediaInputs.length > 0 && (
          <div className="flex flex-row gap-3 items-center justify-start mt-auto">
            {mediaInputs.map((input) => renderInput(input))}
          </div>
        )}
      </div>

      {/* Bottom Bar with Model Pill and Shortcuts - Scrollable */}
      <div className="relative flex items-center justify-start gap-2 px-3 py-2 bg-brand rounded-b-xl overflow-x-auto" style={{ maxWidth: '100%' }}>
        {/* Model Pill Button */}
        <button
          ref={modelPillButtonRef}
          onClick={(e) => {
            e.stopPropagation();
            if (modelPillButtonRef.current) {
              const rect = modelPillButtonRef.current.getBoundingClientRect();
              setMenuPosition({
                top: rect.top - 8,
                left: rect.left
              });
            }
            setShowModelMenu(!showModelMenu);
          }}
          className="flex-shrink-0 flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/15 rounded-full transition-colors cursor-pointer"
        >
          <span className="text-white text-xs font-poppins font-medium whitespace-nowrap">
            {modelPillText}
          </span>
        </button>
        
        {/* Shortcuts Pills */}
        {shortcutInputs.map((input) => (
          <div key={input.id} className="flex-shrink-0">
            {renderInput(input)}
          </div>
        ))}
        
        {/* Model Selector Menu */}
        {showModelMenu && modelClips.length > 0 && menuPosition && (
          <>
            {/* Dimmed overlay */}
            <div
              className="fixed inset-0 bg-transparent"
              style={{ zIndex: 9997 }}
              onClick={() => setShowModelMenu(false)}
            />
            
            {/* Menu Dropdown */}
            <div 
              className="fixed bg-brand-background border border-brand-light/10 rounded-lg shadow-2xl overflow-hidden"
              style={{ 
                pointerEvents: 'auto',
                minWidth: '280px',
                maxWidth: '400px',
                maxHeight: '300px',
                overflowY: 'auto',
                zIndex: 9998,
                top: `${menuPosition.top}px`,
                left: `${menuPosition.left}px`,
                transform: 'translateY(-100%)'
              }}
            >
              {/* Function Header */}
              <div className="px-4 py-2 bg-brand-light/5 border-b border-brand-light/10">
                <span className="text-brand-light/60 text-xs font-poppins font-medium uppercase tracking-wider">
                  Active Models
                </span>
              </div>
              
              {/* Model Clips List */}
              {modelClips.map((clip) => {
                const isActive = clip.clipId === clipId;
                const displayName = clip.name || 'Model';
                const displayTrack = clip.trackName || '';
                
                return (
                  <button
                    key={clip.clipId}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleModelSwitch(clip.clipId);
                    }}
                    onMouseEnter={() => setHoveredModelClipId(clip.clipId)}
                    onMouseLeave={() => setHoveredModelClipId(null)}
                    className={`w-full px-4 py-3 text-left font-poppins transition-colors flex items-center justify-between gap-3 ${
                      hoveredModelClipId === clip.clipId ? 'bg-white/10' : ''
                    } ${isActive ? 'bg-white/5' : ''}`}
                  >
                    <div className="flex flex-col gap-0.5 flex-1 min-w-0">
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
                      <Check size={16} className="text-brand-accent flex-shrink-0" />
                    )}
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
    </>
  );
};

export default DynamicFloatingPanel;

