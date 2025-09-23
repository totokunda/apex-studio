import React, { useCallback, useRef, useState } from 'react'
import { ResizablePanel } from '@/components/ui/resizable'
import { cn } from '@/lib/utils'
import { useSidebarStore } from '@/lib/sidebar'
import MediaMenu from '@/components/media/MediaMenu'
import { ImperativePanelHandle } from 'react-resizable-panels'
import { HiFilm } from "react-icons/hi";
import { LuTrainTrack } from "react-icons/lu";
import { HiOutlineTemplate } from "react-icons/hi";
import { HiOutlineLightBulb } from "react-icons/hi";
import { MdOutlineMovieFilter } from 'react-icons/md'
import MediaModelTrigger from '../media/MediaModelTrigger'

interface MediaModelPanelProps { order?: number; defaultSize?: number; minSize?: number }

const MediaModelPanel: React.FC<MediaModelPanelProps> = ({ order, defaultSize = 25, minSize = 20 }) => {
    const panelRef = useRef<ImperativePanelHandle | null>(null);
    const [isPanelOpenAnimation, setIsPanelOpenAnimation] = useState(false);
    const { section, closeSection } = useSidebarStore();

    const resizePanelOpen = useCallback(() => {
        setIsPanelOpenAnimation(true);
        const panel = panelRef.current;
          panel?.resize(20);
          panel?.expand?.();
          setTimeout(() => {
            setIsPanelOpenAnimation(false);
          }, 100);
          
      }, [panelRef.current]);
    
      const resizePanelClose = useCallback(() => {
        setIsPanelOpenAnimation(true);
        const panel = panelRef.current;
        setTimeout(() => {
        panel?.resize(0);
        panel?.collapse?.();
        }, 10);
        setTimeout(() => {
          setIsPanelOpenAnimation(false);
        }, 100);
      }, [panelRef.current]);   
  return (
    <ResizablePanel
    ref={panelRef}
    minSize={minSize}
    defaultSize={defaultSize}
    order={order}
    className={cn("bg-brand-background rounded-lg  overflow-hidden", {
      "transition-all duration-300 ease-in-out": isPanelOpenAnimation,
    })}
  >
    <div className="flex flex-row gap-x-2 p-3 px-5 pb-1 w-full">
        <MediaModelTrigger icon={<HiFilm className="h-4 w-4" />} title="Media" section="media" onOpen={resizePanelOpen} onClose={resizePanelClose} />
        <MediaModelTrigger icon={<HiOutlineLightBulb className="h-4 w-4 stroke-2" />} title="Models" section="models" onOpen={resizePanelOpen} onClose={resizePanelClose} />
        <MediaModelTrigger icon={<LuTrainTrack className="h-4 w-4 stroke-2" />} title="Tracks" section="tracks" onOpen={resizePanelOpen} onClose={resizePanelClose} />
        <MediaModelTrigger icon={<MdOutlineMovieFilter className="h-4 w-4 " />} title="LoRAs" section="loras" onOpen={resizePanelOpen} onClose={resizePanelClose} />
        <MediaModelTrigger icon={<HiOutlineTemplate className="h-4 w-4 stroke-2" />} title="Templates" section="templates" onOpen={resizePanelOpen} onClose={resizePanelClose} />
    </div>
    {section === 'media' && <MediaMenu onClose={() => closeSection(resizePanelClose)} />}
  </ResizablePanel>
  )
}

export default MediaModelPanel