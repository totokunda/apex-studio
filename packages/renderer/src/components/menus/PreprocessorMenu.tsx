import React, { useEffect, useMemo, useRef, useState } from 'react'
import { listPreprocessors, Preprocessor } from '@/lib/preprocessor/api'
import Draggable from '../dnd/Draggable'
import { ScrollArea } from '../ui/scroll-area'
import { LuInfo, LuChevronLeft, LuChevronRight, LuArrowRight, LuSearch, LuDownload } from "react-icons/lu";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip'
import { cn } from '@/lib/utils'
import { useClipStore } from '@/lib/clip'

export const PreprocessorItem:React.FC<{preprocessor: Preprocessor, isDragging?: boolean}> = ({preprocessor, isDragging}) => {
    const isDownloaded = preprocessor.is_downloaded ?? true;
    const {clips} = useClipStore();

    const disabled = useMemo(() => {
        return clips.length === 0;
    }, [clips]);

    return (
        <Draggable data={{
            ...preprocessor,
            type: 'preprocessor',
            processor_url: `/preprocessors/${preprocessor.id}.png`,
        }} id={preprocessor.id} disabled={disabled}>
           <div className={cn("flex flex-col gap-y-2.5 cursor-pointer w-28 transition-all duration-200 rounded-md ", {
            'w-28': !isDragging,
            'w-24': isDragging,
            'opacity-50': disabled,
            'cursor-not-allowed': disabled,
            'pointer-events-none': disabled,
           })}>
            <div className="flex items-center gap-x-1 relative">
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild className={cn('absolute bottom-1.5 left-1.5 bg-brand/90 rounded-md', {
                            'hidden': isDragging,
                        })}>
                            <LuInfo className="w-4 h-4 text-brand-light" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-[280px] p-2.5  bg-brand font-poppins border border-brand-light/10 rounded-md w-full text-wrap">
                            <div className="flex flex-col gap-y-1.5 w-full">
                                <div className="flex flex-col gap-y-0.5">
                                    <span className="font-medium text-[12px]">{preprocessor.name}</span>
                                    <span className="text-[11px] text-brand-light/70 bg-brand/30 rounded">{preprocessor.category}</span>
                                </div>
                                <p className="text-[10.5px] text-brand-light w-full">{preprocessor.description}</p>
                            </div>
                        </TooltipContent>
                    </Tooltip>
                </TooltipProvider>
                <img src={`/preprocessors/${preprocessor.id}.png`} alt={preprocessor.name} className=" h-full object-cover rounded-md" />
                <div className="absolute bottom-1.5 right-1.5 rounded-full p-1">
                    {isDownloaded ? (
                        null
                    ) : (
                        <LuDownload className="w-3 h-3 text-brand-light/60" />
                    )}
                </div>
            </div>
            {!isDragging && <div className="w-full truncate leading-tight font-medium text-brand-light text-[10px] text-start ">{preprocessor.name}
            </div>} 
            </div>
        </Draggable>
    )
}

const PreprocessorCategory:React.FC<{category: string, preprocessors: Preprocessor[], width: number, onViewAll: () => void}> = ({category, preprocessors, width, onViewAll}) => {
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
        // Multiple checks to ensure we catch the content after it's rendered
        const timeouts = [
            setTimeout(checkScroll, 0),
            setTimeout(checkScroll, 100),
            setTimeout(checkScroll, 300),
            setTimeout(checkScroll, 500)
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
    }, [preprocessors]);

    const scroll = (direction: 'left' | 'right') => {
        if (carouselRef.current) {
            const scrollAmount = 300;
            carouselRef.current.scrollBy({
                left: direction === 'left' ? -scrollAmount : scrollAmount,
                behavior: 'smooth'
            });
        }
    };

    return (
        <div className="flex flex-col gap-y-1 px-7">
            <div className="flex items-center justify-between py-2" style={{
                width: width,
            }}>
                <span className="text-brand-light text-[13px] font-medium">{category}</span>
                <button 
                    onClick={onViewAll}
                    className="flex items-center gap-x-1.5 text-brand-light hover:text-brand-light/70 text-[12px] font-medium cursor-pointer transition-colors rounded-md flex-shrink-0"
                >
                    <span>View all</span>
                    <LuArrowRight className="w-3.5 h-3.5" />
                </button>
            </div>
            <div className="relative" style={{
                width: width,
            }}>
                {showLeftArrow && (
                    <button
                        onClick={() => scroll('left')}
                        className="absolute -left-3 top-1/2 cursor-pointer -translate-y-1/2 z-[9999] bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20"
                    >
                        <LuChevronLeft className="w-4 h-4 text-brand-light" />
                    </button>
                )}
                {showRightArrow && (
                    <button
                        onClick={() => scroll('right')}
                        className="absolute -right-3 top-1/2 cursor-pointer -translate-y-1/2 z-[9999] bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20"
                    >
                        <LuChevronRight className="w-4 h-4 text-brand-light" />
                    </button>
                )}
                <div 
                    ref={carouselRef}
                    className="carousel-container flex gap-x-2 overflow-x-auto rounded-md"
                    style={{ 
                        scrollbarWidth: 'none', 
                        msOverflowStyle: 'none',
                        WebkitOverflowScrolling: 'touch'
                    }}
                >
                    {preprocessors.map((preprocessor) => (
                        <div key={preprocessor.name} className="flex-shrink-0">
                            <PreprocessorItem preprocessor={preprocessor} />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

const CategoryDetailView: React.FC<{category: string, preprocessors: Preprocessor[], onBack: () => void}> = ({category, preprocessors, onBack}) => {
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
                    <div className="grid gap-x-2 gap-y-3" style={{gridTemplateColumns: 'repeat(auto-fit, minmax(112px, 1fr))'}}>
                        {preprocessors.map((preprocessor) => (
                            <div key={preprocessor.name} className="flex justify-center">
                                <PreprocessorItem preprocessor={preprocessor} />
                            </div>
                        ))}
                    </div>
                </div>
            </ScrollArea>
        </div>
    );
};

const PreprocessorMenu:React.FC = () => {
    const scrollRef = useRef<HTMLDivElement>(null)
    const [preprocessors, setPreprocessors] = useState<Preprocessor[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
    const [scrollWidth, setScrollWidth] = useState(0);
    
    const filteredPreprocessors = useMemo(() => {
        if (!searchQuery.trim()) return preprocessors;
        const query = searchQuery.toLowerCase();
        return preprocessors.filter(preprocessor => 
            preprocessor.name.toLowerCase().includes(query) ||
            preprocessor.description?.toLowerCase().includes(query) ||
            preprocessor.category.toLowerCase().includes(query)
        );
    }, [preprocessors, searchQuery]);

    const categories = useMemo(() => {
        return [...new Set(filteredPreprocessors.map((preprocessor) => preprocessor.category))]
    }, [filteredPreprocessors]);

    useEffect(() => {
        listPreprocessors().then((res) => {
            const preprocessors = res.data?.preprocessors || []
            setPreprocessors(preprocessors)
        })
    }, []);

    useEffect(() => {
        const updateWidth = () => {
            if (scrollRef.current) {
                const newWidth = scrollRef.current.clientWidth;
                if (newWidth > 0) {
                    setScrollWidth(newWidth);
                }
            }
        };

        // Initial update with multiple attempts to catch the width after layout
        const timeouts = [
            setTimeout(updateWidth, 0),
            setTimeout(updateWidth, 50),
            setTimeout(updateWidth, 100),
            setTimeout(updateWidth, 200)
        ];

        // Use ResizeObserver for more reliable resize tracking
        const resizeObserver = new ResizeObserver(updateWidth);
        if (scrollRef.current) {
            resizeObserver.observe(scrollRef.current);
        }

        // Also listen to window resize as fallback
        window.addEventListener('resize', updateWidth);

        return () => {
            timeouts.forEach(clearTimeout);
            resizeObserver.disconnect();
            window.removeEventListener('resize', updateWidth);
        };
    }, [selectedCategory]);

    if (selectedCategory) {
        return (
            <>
                <style>{`
                    .carousel-container::-webkit-scrollbar {
                        display: none;
                    }
                `}</style>
                <CategoryDetailView 
                    category={selectedCategory}
                    preprocessors={filteredPreprocessors.filter(p => p.category === selectedCategory)}
                    onBack={() => setSelectedCategory(null)}
                />
            </>
        );
    }

  return (
    <>
        <style>{`
            .carousel-container::-webkit-scrollbar {
                display: none;
            }
        `}</style>
        <div className="flex flex-col h-full w-full">
            <div className="px-7 pt-4  pb-2 ">
                <div className="relative">
                    <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-brand-light/60" />
                    <input
                        type="text"
                        placeholder="Search preprocessors..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full bg-brand text-brand-light placeholder:text-brand-light/50  rounded-md pl-10 pr-4 py-2.5 text-[12px] focus:outline-none focus:ring-2 focus:ring-brand-light/30 transition-all"
                    />
                </div>
            </div>
            <ScrollArea className="flex-1 pb-16" ref={scrollRef}>
                <div className="flex flex-col gap-y-5 pt-4">
                {categories.map((category) => (
                    <PreprocessorCategory 
                        width={scrollWidth - 48} 
                        key={category} 
                        category={category} 
                        preprocessors={filteredPreprocessors.filter((preprocessor) => preprocessor.category === category)}
                        onViewAll={() => setSelectedCategory(category)}
                    />
                ))}
                </div>
            </ScrollArea>
        </div>
    </>
  )
}

export default PreprocessorMenu