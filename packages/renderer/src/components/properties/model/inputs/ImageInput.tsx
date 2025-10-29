import React, { useEffect, useMemo, useRef, useState     } from 'react'
import Droppable from '@/components/dnd/Droppable';
import { IoImageOutline } from "react-icons/io5";
import { cn } from '@/lib/utils';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@radix-ui/react-tabs';
import { LuCheck, LuSearch, LuUpload } from 'react-icons/lu';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { MediaCache, useMediaCache } from '@/lib/media/cache';
import { MediaItem, MediaThumb } from '@/components/media/Item';
import { getMediaInfo } from '@/lib/media/utils';
import { listConvertedMedia } from '@app/preload';
import { ScrollArea } from '@/components/ui/scroll-area';
import TimelineSearch from './timeline/TimelineSearch';


interface ImageInputProps {
  label?: string;
  description?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  defaultValue?: string;
  clipId: string;
}

interface PopoverImageProps {
    value: string;
    onChange: (value: string) => void;
    clipId: string | null;
}

const PopoverImage: React.FC<PopoverImageProps> = ({ value, onChange, clipId }) => {
    const isUserInteractingRef = useRef(false);
    const [selectedTab, setSelectedTab] = useState<'timeline' | 'library'>('library');
    const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [filteredMediaItems, setFilteredMediaItems] = useState<MediaItem[]>([]);
    const {media} = useMediaCache();
    const  [selectedMediaItem, setSelectedMediaItem] = useState<MediaItem | null>(null);
    useEffect(() => {
        setFilteredMediaItems(mediaItems.filter((media) => media.name.toLowerCase().includes(searchQuery.toLowerCase())));
    }, [searchQuery, mediaItems]);

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
    }, [media]);

    return (
        <PopoverContent 
            side='left' 
            align='start'
            sideOffset={20} 
            className={cn('p-2 z-[90] dark h-full flex flex-col gap-y-3 border border-brand-light/10 rounded-[7px] font-poppins transition-all duration-250', selectedTab === 'timeline' ? 'w-[600px]' : 'w-96')}
            onOpenAutoFocus={() => { isUserInteractingRef.current = true; }} onCloseAutoFocus={() => { isUserInteractingRef.current = false; }}>
                <Tabs className='' value={selectedTab} onValueChange={(value) => setSelectedTab(value as 'timeline' | 'library')}>
                    <div className='w-full flex flex-row items-center justify-between gap-x-2'>
                <TabsList className='w-full border cursor-pointer border-brand-light/5 bg-brand  text-brand-light text-[10.5px] rounded font-medium text-start flex flex-row divide-x divide-brand-light/5 overflow-hidden'>
                    <TabsTrigger value="library" className={cn('px-4 w-full py-1.5 cursor-pointer flex items-center justify-center', selectedTab === 'library' ? 'bg-brand-light/10' : '')}>
                        Media Library
                    </TabsTrigger>
                    <TabsTrigger value="timeline" className={cn('px-4 w-full py-1.5 cursor-pointer flex items-center justify-center', selectedTab === 'timeline' ? 'bg-brand-light/10' : '')}>
                        Timeline Assets
                    </TabsTrigger>
                </TabsList>
                <button className='w-fit px-3 h-full flex flex-row items-center justify-center gap-x-1.5 bg-brand-background-light hover:bg-brand-light/5 transition-all duration-200 cursor-pointer rounded border border-brand-light/10 py-1.5'>
                        <LuUpload className='w-3.5 h-3.5 text-brand-light' />
                        <span className='text-brand-light text-[10.5px] font-medium'>Upload</span>
                    </button>
                </div>
                <TabsContent value="library" className='w-full h-full flex flex-col py-2 gap-y-4'>
                    <div className='w-full flex flex-row items-center justify-between gap-x-2'>
                    <span className='relative w-full'>
                    <LuSearch className='w-3.5 h-3.5 text-brand-light/50 absolute left-2 top-1/2 -translate-y-1/2' />
                    <input type="text" placeholder='Search for media' className="w-full h-full pl-8 text-brand-light text-[10.5px] font-normal bg-brand rounded-[7px] border border-brand-light/10 p-2 outline-none" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
                    </span>
                    </div>
                    <ScrollArea  className='w-full h-96 px-3'>
                        <div className='w-full h-full grid grid-cols-2 gap-3'>
                            {filteredMediaItems.map((media) => (
                                <div key={media.name} onClick={() => {
                                    if (selectedMediaItem?.name === media.name) {
                                        setSelectedMediaItem(null);
                                    } else {
                                        setSelectedMediaItem(media);
                                    }
                                }} className={cn('w-full flex flex-col items-center justify-center gap-y-1.5 cursor-pointer group relative',
                                    selectedMediaItem?.name === media.name ? '' : ''

                                )}>
                                    <div className='relative'>
                                    <div className={cn('absolute top-0 left-0 w-full h-full bg-brand-background-light/50 backdrop-blur-sm rounded-md z-20 group-hover:opacity-100 transition-all duration-200 flex items-center justify-center', selectedMediaItem?.name === media.name ? 'opacity-100' : 'opacity-0')}>
                                      <div className={cn('rounded-full py-1 px-3  bg-brand-light/10 flex items-center justify-center font-medium text-[10.5px] w-fit', selectedMediaItem?.name === media.name ? 'bg-brand-light/20' : '')}>
                                        {selectedMediaItem?.name === media.name ? 'Selected' : 'Use as Input'}
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
                <TabsContent value="timeline">
                    <TimelineSearch types={['image', 'video']} excludeClipId={clipId} />
                </TabsContent>
             </Tabs>
        </PopoverContent>
    )
}



const ImageInput: React.FC<ImageInputProps> = ({ label, description, value, onChange, placeholder, defaultValue, clipId }) => {
    const isUserInteractingRef = useRef(false);
  return (
    <Droppable className="w-full h-full" id="image-input" accepts={['media']}>
        
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0 bg-brand-background/50 rounded-[7px] border border-brand-light/10 h-64 shadow  ">
    <div className="w-full h-full flex flex-col items-start justify-start p-3">
    <div className="w-full flex flex-col items-start justify-start mb-3">
        {label && <label className="text-brand-light text-[10.5px] w-full font-medium text-start">{label}</label>}
        {description && <span className="text-brand-light/80 text-[9.5px] w-full text-start">{description}</span>}
    </div>
    <Popover>
        <PopoverTrigger className="w-full h-full">
        <div className="w-full h-full flex flex-col items-center justify-center gap-y-3 border-dashed border-brand-light/10 border bg-brand-background-light/50 shadow-accent rounded-md p-4 hover:bg-brand-light/5 transition-all duration-200 cursor-pointer">
            <IoImageOutline className="w-10 h-10 text-brand-light" />
            <span className="text-brand-light text-[11px] w-full text-center font-medium">Click or drag and drop an image here.</span>
        </div>
        </PopoverTrigger>
        <PopoverImage value={value} onChange={onChange} clipId={clipId} />
    </Popover>
    </div>
    </div>
    </Droppable>
    )
}

export default ImageInput