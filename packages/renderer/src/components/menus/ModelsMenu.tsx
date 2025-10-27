import React, { useEffect, useRef, useState } from 'react'
import { LuChevronDown, LuCheck } from "react-icons/lu";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import Draggable from '@/components/dnd/Draggable';
import { getAllManifests, ManifestInfo } from '@/lib/manifest';

// Model/Track types
type ModelTrackType = 'diffusion';

interface ModelItem {
    id: string;
    name: string;
    type: ModelTrackType;
    dateAdded: number;
    description?: string;
    track?: string;
}

// Map API model_type to track names used in the UI
const MODEL_TYPE_TO_TRACK: Record<string, string> = {
    't2v': 'text-to-video',
    'i2v': 'image-to-video',
    'v2v': 'video-to-video',
    'x2v': 'any-to-video',
    't2i': 'text-to-image',
    'i2i': 'image-to-image',
    'edit': 'image-to-image',
    'ic2i': 'image-control-to-image',
    'vace': 'video-animation-control',
    'control': 'image-control-to-video', // Fallback for any control types
};

const ModelCard: React.FC<{ modelItem: ModelItem }> = ({ modelItem }) => {
    // Convert model to draggable data format
    const draggableData = {
        name: modelItem.name,
        type: 'model' as const,
        absPath: '',
        assetUrl: modelItem.id,
        dateAddedMs: Date.now(),
        track: modelItem.track,
        modelId: modelItem.id,
    };

    // Get display name for track
    const getTrackDisplayName = (track?: string) => {
        const trackMap: { [key: string]: string } = {
            'text-to-image': 'Text to Image',
            'image-to-image': 'Image to Image',
            'text-to-video': 'Text to Video',
            'image-to-video': 'Image to Video',
            'any-to-video': 'Any to Video',
            'video-to-video': 'Video to Video',
            'video-animation-control': 'Video Animation Control',
            'image-control-to-image': 'Image-Control to Image',
            'image-control-to-video': 'Image-Control to Video',
            'image-mask-to-image': 'Image-Mask to Image',
        };
        return track && trackMap[track] ? trackMap[track] : '';
    };

    return (
        <Draggable id={`model-${modelItem.id}`} data={draggableData}>
            <div className="bg-brand border border-brand-light/10 rounded-md p-3 hover:bg-brand-light/5 hover:border-brand-accent/30 transition-all duration-200 cursor-grab active:cursor-grabbing group">
                <div className="flex flex-col gap-y-2">
                    <div className="flex flex-row items-start justify-between gap-x-2">
                        <div className="flex-1 min-w-0">
                            <div className="text-brand-light text-[13px] font-medium truncate group-hover:text-brand-accent transition-colors">
                                {modelItem.name}
                            </div>
                            {modelItem.description && (
                                <div className="text-brand-light/50 text-[11px] mt-1">
                                    {modelItem.description}
                                </div>
                            )}
                        </div>
                    </div>
                    <div className="flex flex-row items-center justify-between mt-1">
                        <span className="text-brand-accent text-[10px] font-medium px-2 py-0.5 bg-brand-accent/10 rounded">
                            {getTrackDisplayName(modelItem.track)}
                        </span>
                        <div className="w-2 h-2 rounded-full bg-brand-accent group-hover:scale-125 transition-transform" />
                    </div>
                </div>
            </div>
        </Draggable>
    );
};

/**
 * Convert ManifestInfo from API to ModelItem for UI
 */
function convertManifestToModelItem(manifest: ManifestInfo): ModelItem {
    // Each manifest now has only ONE model_type
    const modelType = Array.isArray(manifest.model_type) ? manifest.model_type[0] : manifest.model_type;
    const track = MODEL_TYPE_TO_TRACK[modelType.toLowerCase()] || modelType;
    
    return {
        id: manifest.id,  // ← Simplified! Just use metadata.id directly
        name: manifest.name,
        type: 'diffusion' as ModelTrackType,
        dateAdded: Date.now(),
        description: manifest.description,
        track,
    };
}

const ModelsMenu: React.FC = () => {
    const [selectedTrack, setSelectedTrack] = useState<string>('popular');
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [models, setModels] = useState<ModelItem[]>([]);
    const [loading, setLoading] = useState(true);
    
    const uploadBarRef = useRef<HTMLDivElement | null>(null);
    const panelRef = useRef<HTMLDivElement | null>(null);
    const [menuWidth, setMenuWidth] = useState<number>(0);
    const [panelHeight, setPanelHeight] = useState<number>(0);

    const tracks = [
        { id: 'popular', name: 'Popular' },
        { id: 'text-to-image', name: 'Text to Image' },
        { id: 'image-to-image', name: 'Image to Image' },
        { id: 'text-to-video', name: 'Text to Video' },
        { id: 'image-to-video', name: 'Image to Video' },
        { id: 'any-to-video', name: 'Any to Video' },
        { id: 'video-to-video', name: 'Video to Video' },
        { id: 'video-animation-control', name: 'Video Animation Control' },
        { id: 'image-control-to-image', name: 'Image-Control to Image' },
        { id: 'image-control-to-video', name: 'Image-Control to Video' },
        { id: 'image-mask-to-image', name: 'Image-Mask to Image' },
    ];

    // Fetch models from API on mount
    useEffect(() => {
        const fetchModels = async () => {
            setLoading(true);
            try {
                const manifests = await getAllManifests();
                console.log('Fetched manifests from API:', manifests);
                
                // Convert manifests to model items
                const modelItems: ModelItem[] = manifests.map(manifest => 
                    convertManifestToModelItem(manifest)
                );
                
                setModels(modelItems);
                console.log('Converted to model items:', modelItems);
            } catch (error) {
                console.error('Failed to fetch models:', error);
                setModels([]);
            } finally {
                setLoading(false);
            }
        };

        fetchModels();
    }, []);

    useEffect(() => {
        const el = uploadBarRef.current;
        if (!el) return;
        const update = () => setMenuWidth(el.clientWidth);
        update();
        const ro = new ResizeObserver(update);
        ro.observe(el);
        window.addEventListener('resize', update);
        return () => {
            ro.disconnect();
            window.removeEventListener('resize', update);
        };
    }, []);

    // Track panel height and update on resize to size the ScrollArea dynamically
    useEffect(() => {
        const el = panelRef.current;
        if (!el) return;
        const update = () => setPanelHeight(el.clientHeight);
        update();
        const ro = new ResizeObserver(update);
        ro.observe(el);
        window.addEventListener('resize', update);
        return () => {
            ro.disconnect();
            window.removeEventListener('resize', update);
        };
    }, []);

    return (
        <div ref={panelRef} className="h-full w-full duration-200 ease-out">
            <div className="flex flex-col gap-y-3 px-5 pb-3 pt-3">
                {/* Track Selector Dropdown */}
                <DropdownMenu open={dropdownOpen} onOpenChange={setDropdownOpen}>
                    <div ref={uploadBarRef} className="bg-brand-accent-shade shadow flex flex-row items-center text-brand-lighter w-full text-[12.5px] font-medium rounded transition-all duration-200 cursor-pointer">
                        <span 
                            onClick={() => setDropdownOpen(!dropdownOpen)} 
                            className='px-4 py-2 flex flex-row gap-x-2.5 rounded-l items-center justify-center hover:bg-brand-accent-two-shade border-r border-brand-light/30 w-full'
                        >
                            <span className='text-[12.5px]'>
                                {tracks.find(t => t.id === selectedTrack)?.name || 'Popular'}
                            </span>
                        </span>
                        <DropdownMenuTrigger className='px-1 py-2 flex flex-row gap-x-2.5 items-center justify-center w-10 hover:bg-brand-accent-two-shade rounded-r-md'>
                            <LuChevronDown className='w-4 h-4 cursor-pointer stroke-2' />
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align='end' sideOffset={2} style={{ width: menuWidth, maxWidth: menuWidth }} className='dark mt-1 flex flex-col text-brand-light bg-brand-background font-poppins'>
                            {tracks.map((track) => (
                                <DropdownMenuItem
                                    key={track.id}
                                    className='dark text-[12px] font-medium flex flex-row items-center cursor-pointer gap-x-3 w-full'
                                    onClick={() => {
                                        setSelectedTrack(track.id);
                                        setDropdownOpen(false);
                                    }}
                                >
                                    <span>{track.name}</span>
                                    {selectedTrack === track.id && (
                                        <LuCheck className='w-4 h-4 ml-auto text-brand-accent' />
                                    )}
                                </DropdownMenuItem>
                            ))}
                        </DropdownMenuContent>
                    </div>
                </DropdownMenu>
            </div>

            <div className="border-t border-brand-light/5"></div>

            {/* Models Content Area with ScrollArea */}
            <ScrollArea style={{ height: Math.max(0, panelHeight - 120) }} className="px-5 py-4">
                {loading ? (
                    <div className="flex items-center justify-center h-32">
                        <div className="text-brand-light/50 text-sm">Loading models from API...</div>
                    </div>
                ) : models.length === 0 ? (
                    <div className="flex items-center justify-center h-32">
                        <div className="text-brand-light/50 text-sm">No models available</div>
                    </div>
                ) : (
                    <div className="grid gap-3 w-full" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))' }}>
                        {models
                            .filter(modelItem => selectedTrack === 'popular' || modelItem.track === selectedTrack)
                            .map((modelItem) => (
                                <ModelCard key={modelItem.id} modelItem={modelItem} />
                            ))}
                    </div>
                )}
            </ScrollArea>
        </div>
    )
}

export default ModelsMenu


