import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useManifestTypes, useManifests, type ManifestInfo } from '@/lib/manifest';
import { cn } from '@/lib/utils';
import { ScrollArea } from '../ui/scroll-area';
import { LuChevronLeft, LuChevronRight, LuArrowRight, LuSearch, LuInfo, LuDownload, LuCheck } from "react-icons/lu";
import Draggable from '../dnd/Draggable';
import { v4 as uuidv4 } from 'uuid';
import { useManifestStore } from '@/lib/manifest/store';
import ModelPage from '../models/ModelPage';
// check 


export const ModelItem:React.FC<{ manifest: ManifestInfo, isDragging?: boolean }> = ({ manifest, isDragging }) => {
  const { setSelectedManifestId } = useManifestStore();
  const isVideoDemo = React.useMemo(() => {
    const value = (manifest.demo_path || '').toLowerCase();
    try {
      const url = new URL(value);
      const pathname = url.pathname;
      const ext = pathname.split('.').pop() || '';
      return ['mp4', 'webm', 'mov', 'm4v', 'ogg', 'm3u8'].includes(ext);
    } catch {
      return value.endsWith('.mp4') || value.endsWith('.webm') || value.endsWith('.mov') || value.endsWith('.m4v') || value.endsWith('.ogg') || value.endsWith('.m3u8');
    }
  }, [manifest.demo_path]);

  return (
    <div className={cn("flex flex-col transition-all font-poppins duration-200 rounded-md relative bg-brand border border-brand-light/5 shadow-md cursor-grab active:cursor-grabbing w-52", {
    })}>
    <Draggable id={uuidv4()} data={{
      type: 'model',
      ...manifest,
    }}>
      <div className="flex flex-col items-center relative w-full ">

        <div className="h-32 rounded-t-md overflow-hidden  flex items-center justify-center w-full aspect-square">
          {isVideoDemo ? (
            <video
              src={manifest.demo_path}
              className="h-full w-full object-cover rounded-t-md"
              autoPlay
              muted
              loop
              playsInline
            />
          ) : (
            <img src={manifest.demo_path} alt={manifest.name} className="h-full object-cover rounded-t-md" />
          )}
        </div>
      </div>
        
      <div className='flex flex-col gap-y-1.5 py-3.5 pb-2 px-3 border-t border-brand-light/5 w-full '>
      <div className="w-full truncate leading-tight font-semibold text-brand-light text-[12px] text-start">{manifest.name}</div>
      <div className='flex items-center gap-x-1 w-full flex-wrap justify-start gap-y-1'>
          {manifest.tags.map((tag) => (
            <span key={tag} className="text-[8px] text-brand-light bg-brand-background border shadow border-brand-light/10 rounded px-2 py-0.5 ">{tag}</span>
          ))}
        </div>
      </div>
    
    </Draggable>
    {!isDragging && (
          <div className='flex items-center gap-x-1 w-full p-3 pt-0'>
            <button
              onClick={() => {
                setSelectedManifestId(manifest.id);
              }}
              type='button'
              className='text-[10px] font-medium flex items-center transition-all duration-200 justify-center gap-x-1.5 text-brand-light flex-1 hover:text-brand-light/80 bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded px-2 py-1'
              title='Show more info'
            >
              <LuInfo className='w-3 h-3' />
              <span>More info</span>
            </button>
            <button
              type='button'
              disabled={!!manifest.downloaded}
              className={cn(
                'text-[10px] font-medium flex items-center transition-all duration-200 justify-center gap-x-1.5 rounded px-2 py-1 border flex-1',
                manifest.downloaded
                  ? 'text-brand-light/80 bg-brand-background border-brand-light/10 cursor-default'
                  : 'text-brand-light hover:text-brand-light/90 bg-brand-background hover:bg-brand-background/70 border-brand-light/10'
              )}
              title={manifest.downloaded ? 'Already downloaded' : 'Download model'}
            >
              {manifest.downloaded ? <LuCheck className='w-3 h-3' /> : <LuDownload className='w-3 h-3' />}
              <span>{manifest.downloaded ? 'Downloaded' : 'Download'}</span>
            </button>
          </div>
      )}
    </div>
  );
}

const ModelCategory:React.FC<{ category: string, manifests: ManifestInfo[], width: number, onViewAll: () => void }> = ({ category, manifests, width, onViewAll }) => {
  const carouselRef = useRef<HTMLDivElement>(null);
  const [showLeftArrow, setShowLeftArrow] = useState(false);
  const [showRightArrow, setShowRightArrow] = useState(false);

  const checkScroll = () => {
    if (carouselRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = carouselRef.current;
      const hasOverflow = scrollWidth > clientWidth;
      setShowLeftArrow(scrollLeft > 5);
      setShowRightArrow(hasOverflow && scrollLeft + clientWidth < scrollWidth - 5);
    } else {
      setShowLeftArrow(false);
      setShowRightArrow(false);
    }
  };

  useEffect(() => {
    const timeouts = [
      setTimeout(checkScroll, 0),
      setTimeout(checkScroll, 100),
      setTimeout(checkScroll, 300),
      setTimeout(checkScroll, 500),
    ];
    const carousel = carouselRef.current;
    if (carousel) {
      carousel.addEventListener('scroll', checkScroll);
      window.addEventListener('resize', checkScroll);
      return () => {
        timeouts.forEach(clearTimeout);
        carousel.removeEventListener('scroll', checkScroll);
        window.removeEventListener('resize', checkScroll);
      };
    }
    return () => timeouts.forEach(clearTimeout);
  }, [manifests]);

  const scroll = (direction: 'left' | 'right') => {
    if (carouselRef.current) {
      const scrollAmount = 300;
      carouselRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth',
      });
    }
  };

  return (
    <div className="flex flex-col gap-y-1 w-full px-4">
      <div className="flex items-center justify-between py-2" style={{ maxWidth: width }}>
        <span className="text-brand-light text-[13px] font-medium">{category}</span>
        <button onClick={onViewAll} className="flex items-center gap-x-1.5 text-brand-light hover:text-brand-light/70 text-[12px] font-medium cursor-pointer transition-colors rounded-md flex-shrink-0">
          <span>View all</span>
          <LuArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="relative w-full" style={{ width: width }}>
        {showLeftArrow && (
          <button onClick={() => scroll('left')} className="absolute -left-3 top-1/2 cursor-pointer -translate-y-1/2 z-50 bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20">
            <LuChevronLeft className="w-4 h-4 text-brand-light" />
          </button>
        )}
        {showRightArrow && (
          <button onClick={() => scroll('right')} className="absolute -right-3 top-1/2 cursor-pointer -translate-y-1/2 z-50 bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20">
            <LuChevronRight className="w-4 h-4 text-brand-light" />
          </button>
        )}
        <div ref={carouselRef} className="carousel-container flex gap-x-2 overflow-x-auto rounded-md" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none', WebkitOverflowScrolling: 'touch' }}>
          {manifests.map((manifest) => (
            <div key={manifest.id} className="flex-shrink-0">
              <ModelItem manifest={manifest}  />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const CategoryDetailView:React.FC<{ category: string, manifests: ManifestInfo[], onBack: () => void }> = ({ category, manifests, onBack }) => {
  return (
    <div className="flex flex-col h-full w-full">
      <div className="px-7 pt-4 pb-4 border-b border-brand/20">
        <div className="flex items-center gap-x-3">
          <button onClick={onBack} className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded-md transition-colors cursor-pointer">
            <LuChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-brand-light text-[14px] font-medium">{category}</span>
        </div>
      </div>
      <ScrollArea className="flex-1 pb-16">
        <div className="px-7 pt-6">
          <div className="grid gap-x-2 gap-y-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            {manifests.map((manifest) => (
              <div key={manifest.id} className="flex justify-center">
                <ModelItem manifest={manifest} />
              </div>
            ))}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

const ModelMenu:React.FC = () => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { data: manifestsData } = useManifests();
  const { data: modelTypesData } = useManifestTypes();
  const { selectedManifestId, clearSelectedManifestId } = useManifestStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [scrollWidth, setScrollWidth] = useState(0);
  const categorySectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [activeCategory, setActiveCategory] = useState<string | null>(null);

  const manifestTypeKeyToLabel = useMemo(() => {
    const map = new Map<string, string>();
    (modelTypesData || []).forEach((t) => map.set(t.key, t.label));
    return map;
  }, [modelTypesData]);

  const manifests: ManifestInfo[] = useMemo(() => manifestsData ?? [], [manifestsData]);

  const filteredManifests = useMemo(() => {
    if (!searchQuery.trim()) return manifests;
    const query = searchQuery.toLowerCase();
    return manifests.filter((m) => {
      const typeKeys: string[] = Array.isArray(m.model_type) ? m.model_type : [m.model_type];
      const typeLabels = typeKeys.map((k) => manifestTypeKeyToLabel.get(k) || k);
      return (
        m.name.toLowerCase().includes(query) ||
        (m.description?.toLowerCase().includes(query) ?? false) ||
        m.model.toLowerCase().includes(query) ||
        typeKeys.some((k) => k.toLowerCase().includes(query)) ||
        typeLabels.some((l) => l.toLowerCase().includes(query)) ||
        (m.tags || []).some((t) => t.toLowerCase().includes(query))
      );
    });
  }, [manifests, searchQuery, manifestTypeKeyToLabel]);

  const categories = useMemo(() => {
    const set = new Set<string>();
    filteredManifests.forEach((m) => {
      const typeKeys: string[] = Array.isArray(m.model_type) ? m.model_type : [m.model_type];
      typeKeys.forEach((k) => set.add(manifestTypeKeyToLabel.get(k) || k));
    });
    return Array.from(set);
  }, [filteredManifests, manifestTypeKeyToLabel]);

  const handleCategoryClick = (category: string) => {
    setActiveCategory(category);
    const section = categorySectionRefs.current[category];
    const viewport = viewportRef.current;
    if (section && viewport) {
      const containerTop = viewport.getBoundingClientRect().top;
      const sectionTop = section.getBoundingClientRect().top;
      const scrollOffset = sectionTop - containerTop + viewport.scrollTop;
      viewport.scrollTo({ top: scrollOffset, behavior: 'smooth' });
    } else if (section) {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  useEffect(() => {
    const updateWidth = () => {
      if (scrollRef.current) {
        const viewport = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLDivElement | null;
        viewportRef.current = viewport;
        const newWidth = (viewport || scrollRef.current).clientWidth;
        if (newWidth > 0) setScrollWidth(newWidth);
      }
    };
    const timeouts = [
      setTimeout(updateWidth, 0),
      setTimeout(updateWidth, 50),
      setTimeout(updateWidth, 100),
      setTimeout(updateWidth, 200),
    ];
    const resizeObserver = new ResizeObserver(updateWidth);
    if (scrollRef.current) resizeObserver.observe(scrollRef.current);
    window.addEventListener('resize', updateWidth);
    return () => {
      timeouts.forEach(clearTimeout);
      resizeObserver.disconnect();
      window.removeEventListener('resize', updateWidth);
    };
  }, [selectedCategory, selectedManifestId]);

  if (selectedManifestId) {
    return (
      <ModelPage manifestId={selectedManifestId} />
    );
  }

  if (selectedCategory) {
    return (
      <>
        <style>{`
          .carousel-container::-webkit-scrollbar { display: none; }
        `}</style>
        <CategoryDetailView
          category={selectedCategory}
          manifests={filteredManifests.filter((m) => {
            const keys = Array.isArray(m.model_type) ? m.model_type : [m.model_type];
            const labels = keys.map((k) => manifestTypeKeyToLabel.get(k) || k);
            return labels.includes(selectedCategory);
          })}
          onBack={() => setSelectedCategory(null)}
        />
      </>
    );
  }

  return (
    <>
      <style>{`
        .carousel-container::-webkit-scrollbar { display: none; }
      `}</style>
      <div className="flex flex-col h-full w-full border-t border-brand-light/5 mt-2">
        <div className="flex flex-1 min-h-0 w-full">
          <div className="flex flex-col border-r border-brand-light/5 min-w-36 w-36 gap-y-1 bg-brand-background">
            <span className="text-[8.5px] px-2 pt-2.5 mb-1 text-brand-light/60 text-start font-medium">CATEGORIES</span>
            <div className="flex flex-col gap-y-1 px-1">
              {categories.map((category) => (
                <button
                  key={category}
                  onClick={() => handleCategoryClick(category)}
                  className={cn(
                    "text-start w-full p-[5.5px] px-2 rounded text-[10.5px] font-medium text-brand-light/80 hover:text-brand-light hover:bg-brand/60 transition-colors truncate",
                    { 'bg-brand/60 text-brand-light': activeCategory === category }
                  )}
                  title={category}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="w-full p-3 flex-shrink-0">
              <div className="relative bg-brand text-brand-light rounded-md placeholder:text-brand-light/50 items-center flex w-full p-3 space-x-2 text-[11px] focus:outline-none focus:ring-2 focus:ring-brand-light/30 transition-all">
                <LuSearch className="w-4 h-4 text-brand-light/60" />
                <input
                  type="text"
                  placeholder="Search models..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full outline-none bg-brand"
                />
              </div>
            </div>
            <ScrollArea className="flex-1" ref={scrollRef}>
              <div className="flex flex-col gap-y-5 pt-1 pb-28">
                {categories.map((category) => (
                  <div key={category} ref={(el) => { categorySectionRefs.current[category] = el; }} className="w-full">
                    <ModelCategory
                      width={scrollWidth - 36}
                      category={category}
                      manifests={filteredManifests.filter((m) => {
                        const keys = Array.isArray(m.model_type) ? m.model_type : [m.model_type];
                        const labels = keys.map((k) => manifestTypeKeyToLabel.get(k) || k);
                        return labels.includes(category);
                      })}
                      onViewAll={() => setSelectedCategory(category)}
                    />
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>
    </>
  );
}

export default ModelMenu;