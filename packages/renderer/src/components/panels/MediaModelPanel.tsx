import React, { useCallback, useRef, useState, useEffect } from 'react'
import { ResizablePanel } from '@/components/ui/resizable'
import { cn } from '@/lib/utils'
import { useSidebarStore } from '@/lib/sidebar'
import MediaMenu from '@/components/menus/MediaMenu'
import PreprocessorMenu from '@/components/menus/PreprocessorMenu'
import { ImperativePanelHandle } from 'react-resizable-panels'
import { HiFilm } from "react-icons/hi";
import { MdPhotoFilter } from "react-icons/md";
import { HiOutlineTemplate } from "react-icons/hi";
import { MdOutlineMovieFilter } from 'react-icons/md'
import { TbWand } from "react-icons/tb";
import { LuChevronRight, LuChevronLeft } from 'react-icons/lu'
import MediaModelTrigger from '../media/MediaModelTrigger'
import FilterMenu from '../menus/FilterMenu'
import { VscServerProcess } from "react-icons/vsc";

interface MediaModelPanelProps { order?: number; defaultSize?: number; minSize?: number; maxSize?: number }

const MediaModelPanel: React.FC<MediaModelPanelProps> = ({ order, defaultSize = 25, minSize = 20, maxSize }) => {
    const panelRef = useRef<ImperativePanelHandle | null>(null);
    const triggersRef = useRef<HTMLDivElement>(null);
    const [isPanelOpenAnimation, setIsPanelOpenAnimation] = useState(false);
    const [panelSize, setPanelSize] = useState(0);
    const [canScrollLeft, setCanScrollLeft] = useState(false);
    const [canScrollRight, setCanScrollRight] = useState(false);
    const { section } = useSidebarStore();

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

    const checkScrollButtons = () => {
      if (triggersRef.current) {
        const { scrollLeft, scrollWidth, clientWidth } = triggersRef.current;
        setCanScrollLeft(scrollLeft > 0);
        setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 1);
      }
    };

    useEffect(() => {
      checkScrollButtons();
      const handleScroll = () => checkScrollButtons();
      const triggerElement = triggersRef.current;
      if (triggerElement) {
        triggerElement.addEventListener('scroll', handleScroll);
        return () => triggerElement.removeEventListener('scroll', handleScroll);
      }
    }, [panelSize]);

  return (
    <ResizablePanel
    ref={panelRef}
    minSize={minSize}
    maxSize={maxSize}
    defaultSize={defaultSize}
    order={order}
    onResize={() => {
      const element = panelRef.current?.getSize();
      if (element !== undefined) {
        const container = panelRef.current?.getSize();
        setPanelSize((container ?? 0) * window.innerWidth / 100);
      }
    }}
    className={cn("bg-brand-background rounded-lg  overflow-hidden", {
      "transition-all duration-300 ease-in-out": isPanelOpenAnimation,
    })}
  >
    <div className="relative p-3 px-5 pb-1 w-full">
      <LuChevronLeft onClick={() => {
        if (triggersRef.current) {
          triggersRef.current.scrollBy({ left: -80, behavior: 'smooth' });
        }
      }} className={cn("text-brand-light h-6 w-6 mt-1 bg-brand-background/90 border border-brand-light/10 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute left-2 top-1/2 -translate-y-1/2 p-1 cursor-pointer", canScrollLeft ? "block" : "hidden")} />
      <div ref={triggersRef} style={{scrollbarWidth: 'none', msOverflowStyle: 'none'}} className="overflow-x-auto [&::-webkit-scrollbar]:hidden">
        <div className="flex flex-row gap-x-2 w-fit">
          <MediaModelTrigger icon={<HiFilm className="h-4 w-4" />} title="Media" section="media" onOpen={resizePanelOpen} onClose={resizePanelClose} />
          <MediaModelTrigger icon={<TbWand className="h-4 w-4 stroke-2" />} title="Models" section="models" onOpen={resizePanelOpen} onClose={resizePanelClose} />
          <MediaModelTrigger icon={<VscServerProcess className="h-4 w-4 " />} title="Preprocessors" section="preprocessors" onOpen={resizePanelOpen} onClose={resizePanelClose} />
          <MediaModelTrigger icon={<MdPhotoFilter className="h-4 w-4 " />} title="Filters" section="filters" onOpen={resizePanelOpen} onClose={resizePanelClose} />
          <MediaModelTrigger icon={<MdOutlineMovieFilter className="h-4 w-4 " />} title="LoRAs" section="loras" onOpen={resizePanelOpen} onClose={resizePanelClose} />
          {/*<MediaModelTrigger icon={<HiOutlineTemplate className="h-4 w-4 stroke-2" />} title="Templates" section="templates" onOpen={resizePanelOpen} onClose={resizePanelClose} />*/}
        </div>
      </div>
      <LuChevronRight onClick={() => {
        if (triggersRef.current) {
          triggersRef.current.scrollBy({ left: 80, behavior: 'smooth' });
        }
      }} className={cn("text-brand-light h-6 w-6 mt-1 border border-brand-light/10 bg-brand-background/90 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute right-2 top-1/2 -translate-y-1/2 p-1 cursor-pointer", canScrollRight ? "block" : "hidden")} />
    </div>
    {section === 'media' && <MediaMenu />}
    {section === 'filters' && <FilterMenu />}
    {section === 'preprocessors' && <PreprocessorMenu />}
  </ResizablePanel>
  )
}

export default MediaModelPanel