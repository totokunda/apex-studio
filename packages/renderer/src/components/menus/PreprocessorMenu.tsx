import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Preprocessor } from '@/lib/preprocessor/api'
import Draggable from '../dnd/Draggable'
import { ScrollArea } from '../ui/scroll-area'
import { LuInfo, LuChevronLeft, LuChevronRight, LuArrowRight, LuSearch, LuDownload, LuImage, LuVideo, LuLoader } from "react-icons/lu";
import { cn } from '@/lib/utils'
import { usePreprocessorsListStore } from '@/lib/preprocessor/list-store'
import PreprocessorPage from '../preprocessors/PreprocessorPage'
import { downloadPreprocessor as downloadPreprocessorApi, usePreprocessorJob, useJobProgress, getPreprocessorStatus, usePreprocessorJobStore } from '@/lib/preprocessor/api'

export const PreprocessorItem:React.FC<{preprocessor: Preprocessor, isDragging?: boolean, onMoreInfo?: (id: string) => void}> = ({preprocessor, isDragging, onMoreInfo}) => {
    const isDownloaded = !!preprocessor.is_downloaded;
    const [jobId, setJobId] = useState<string | null>(null);
    const [starting, setStarting] = useState(false);
    // Subscribe to the job we started (same mechanism as details page)
    const { isProcessing, isComplete } = usePreprocessorJob(jobId, true);
    // Read-only global job state for this preprocessor; do not auto-start tracking to avoid false pending state
    const globalJob = useJobProgress(preprocessor.id);
    const isGlobalProcessing = !!(globalJob && (globalJob.status === 'running' || globalJob.status === 'pending'));
    const { load } = usePreprocessorsListStore();
    const startTracking = usePreprocessorJobStore((s) => s.startTracking);
    const isActivelyTracked = usePreprocessorJobStore((s) => s.activeJobs.has(preprocessor.id));

    useEffect(() => {
        if (isComplete && jobId) {
            (async () => {
                try { await load(true); } catch {}
                setJobId(null);
                setStarting(false);
            })();
        }
    }, [isComplete, jobId, load]);

    // If a job already exists in the store for this preprocessor, adopt it locally
    useEffect(() => {
        if (!jobId && globalJob && (globalJob.status === 'running' || globalJob.status === 'pending')) {
            setJobId(preprocessor.id);
            setStarting(false);
        }
    }, [jobId, globalJob, preprocessor.id]);

    // Adopt downloads started elsewhere (e.g., details page) without creating phantom jobs
    useEffect(() => {
        let cancelled = false;
        if (isDownloaded) return;
        // If store says it's processing but we are not actively tracking, reattach tracking
        if (isGlobalProcessing && !isActivelyTracked) {
            (async () => { try { await startTracking(preprocessor.id); } catch {} })();
            setJobId(preprocessor.id);
            setStarting(false);
            return;
        }
        if (isGlobalProcessing) return; // already tracked
        if (jobId && isProcessing) return; // local job already tracked
        (async () => {
            try {
                const res = await getPreprocessorStatus(preprocessor.id);
                const st = res?.data?.status;
                if (!cancelled && res.success && (st === 'running' || st === 'pending')) {
                    try { await startTracking(preprocessor.id); } catch {}
                    // Also adopt locally so the UI updates immediately
                    setJobId(preprocessor.id);
                    setStarting(false);
                }
            } catch {}
        })();
        return () => { cancelled = true; };
    }, [preprocessor.id, isDownloaded, isGlobalProcessing, isActivelyTracked, jobId, isProcessing, startTracking]);

    const handleDownload = async () => {
        // Mirror gating from details page: don't block on stale global processing state
        if (starting || (jobId && isProcessing)) return;
        setStarting(true);
        try {
            const res = await downloadPreprocessorApi(preprocessor.id, preprocessor.id);
            if (res.success) {
                setJobId(preprocessor.id);
            } else {
                setStarting(false);
            }
        } catch {
            setStarting(false);
        }
    };

    const formatSize = (bytes: number): string | null => {
        if (bytes === 0) {
            return null
        }
        if (bytes < 1024) {
            return `${bytes} B`;
        } else if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(0)} KB`;
        } else if (bytes < 1024 * 1024 * 1024) {
            return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
        } else {
            return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
        }
    };

    const totalDownloadSize = useMemo(() => {
        const bytes = preprocessor.files?.reduce((acc, file) => acc + file.size_bytes, 0) ?? 0;
        return formatSize(bytes);
    }, [preprocessor.files]);


    return (
        <div className={cn("flex flex-col cursor-pointer w-28 transition-all duration-200 rounded-md border shadow border-brand-light/10", {
            'w-36': !isDragging,
            'w-32': isDragging
        })}>
            <Draggable data={{
                ...preprocessor,
                type: 'preprocessor',
                processor_url: `/preprocessors/${preprocessor.id}.png`,
            }} id={preprocessor.id}>
                <div className="flex flex-col">
                    <div className="flex items-center gap-x-1 relative">
                        <img src={`/preprocessors/${preprocessor.id}.png`} alt={preprocessor.name} className={cn(" h-48 aspect-square object-cover rounded-t-md", {
                            'h-48': !isDragging,
                            'h-44': isDragging
                        })} />
                    </div>
                    <div className="w-full bg-brand p-2 rounded-b-md">
                        <div className="w-full flex flex-col gap-y-1 ">
                            <div className="w-full flex flex-col gap-y-1">
                                <div className="truncate leading-tight font-medium text-brand-light text-[10px] text-start">{preprocessor.name}</div>
                                <div className="w-full flex items-center justify-between">
                                    {totalDownloadSize && (
                                        <span className="text-brand-light/50 text-[9px] w-full text-start">{totalDownloadSize}</span>
                                    )}
                                    <div className={cn("w-full flex items-center gap-x-1 text-brand-light/50", {
                                        'justify-end': totalDownloadSize,
                                        'justify-start': !totalDownloadSize,
                                    })}>
                                        {preprocessor.supports_image && (
                                            <LuImage className="w-3 h-3" />
                                        )}
                                        {preprocessor.supports_video && (
                                            <LuVideo className="w-3 h-3" />
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Draggable>
            {!isDragging && (
                <div className='flex items-center gap-x-1 w-full p-2 pt-0 bg-brand'>
                    <button
                        onClick={() => onMoreInfo?.(preprocessor.id)}
                        type='button'
                        className='text-[10px] font-medium flex items-center transition-all duration-200 justify-center gap-x-1.5 text-brand-light flex-1 hover:text-brand-light/80 bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded px-2 py-1'
                        title='Show more info'
                    >
                        <LuInfo className='w-3 h-3' />
                        <span>More info</span>
                    </button>
                    {isDownloaded ? null : (
                        <button
                            type="button"
                            onClick={handleDownload}
                            disabled={starting || (!!jobId && isProcessing) || isGlobalProcessing}
                            className={cn(
                                "inline-flex items-center justify-center gap-x-1 text-[9px] bg-brand-background border border-brand-light/10 rounded py-1 px-1 h-[25px] min-w-7",
                                {
                                    'text-brand-light/60 cursor-default! opacity-70': jobId,
                                    'text-brand-light/80 hover:text-brand-light hover:bg-brand-background/70': !jobId
                                }
                            )}
                            title={starting ? 'Starting…' : 'Download'}
                        >
                            {(jobId)
                                ? <LuLoader className="w-3 h-3 animate-spin" />
                                : <LuDownload className="w-3 h-3" />}
                        </button>
                    )}
                </div>
            )}
        </div>
    )
}

const PreprocessorCategory:React.FC<{category: string, preprocessors: Preprocessor[], width: number, onViewAll: () => void, onMoreInfo: (id: string) => void}> = ({category, preprocessors, width, onViewAll, onMoreInfo}) => {
    const carouselRef = useRef<HTMLDivElement>(null);
    const [showLeftArrow, setShowLeftArrow] = useState(false);
    const [showRightArrow, setShowRightArrow] = useState(false);
    void width; // width is no longer used; maintain param to avoid refactor

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
        <div className="flex flex-col gap-y-1 w-full px-4">
            <div className="flex items-center justify-between py-2" style={{maxWidth: width}}>
                <span className="text-brand-light text-[13px] font-medium">{category}</span>
                <button 
                    onClick={onViewAll}
                    className="flex items-center gap-x-1.5 text-brand-light hover:text-brand-light/70 text-[12px] font-medium cursor-pointer transition-colors rounded-md flex-shrink-0"
                >
                    <span>View all</span>
                    <LuArrowRight className="w-3.5 h-3.5" />
                </button>
            </div>
            <div className="relative w-full" style={{width: width}}>
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
                            <PreprocessorItem preprocessor={preprocessor} onMoreInfo={onMoreInfo} />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

const CategoryDetailView: React.FC<{category: string, preprocessors: Preprocessor[], onBack: () => void, onMoreInfo: (id: string) => void}> = ({category, preprocessors, onBack, onMoreInfo}) => {
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
                    <div className="grid gap-x-2 gap-y-3" style={{gridTemplateColumns: 'repeat(auto-fit, minmax(132px, 1fr))'}}>
                        {preprocessors.map((preprocessor) => (
                            <div key={preprocessor.name} className="flex justify-center">
                                <PreprocessorItem preprocessor={preprocessor} onMoreInfo={onMoreInfo} />
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
    const viewportRef = useRef<HTMLDivElement | null>(null)
    const { preprocessors, load } = usePreprocessorsListStore();
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
    const [selectedPreprocessorId, setSelectedPreprocessorId] = useState<string | null>(null);
    const [scrollWidth, setScrollWidth] = useState(0);
    const categorySectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
    const [activeCategory, setActiveCategory] = useState<string | null>(null);
    const handleCategoryClick = (category: string) => {
        setActiveCategory(category);
        const section = categorySectionRefs.current[category];
        const viewport = viewportRef.current;
        
        if (section && viewport) {
            // Calculate offset from the top of the scrollable container
            const containerTop = viewport.getBoundingClientRect().top;
            const sectionTop = section.getBoundingClientRect().top;
            const scrollOffset = sectionTop - containerTop + viewport.scrollTop;
            
            viewport.scrollTo({ top: scrollOffset, behavior: 'smooth' });
        } else if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };
    
    const filteredPreprocessors = useMemo(() => {
        const list = preprocessors ?? [];
        if (!searchQuery.trim()) return list;
        const query = searchQuery.toLowerCase();
        return list.filter(preprocessor => 
            preprocessor.name.toLowerCase().includes(query) ||
            preprocessor.description?.toLowerCase().includes(query) ||
            preprocessor.category.toLowerCase().includes(query)
        );
    }, [preprocessors, searchQuery]);

    const categories = useMemo(() => {
        setActiveCategory(preprocessors?.[0]?.category || null);
        return [...new Set(filteredPreprocessors.map((preprocessor) => preprocessor.category))]
    }, [filteredPreprocessors]);

    useEffect(() => {
        // trigger a single idempotent load; store prevents refetching
        load();
    }, [load]);

    useEffect(() => {
        const updateWidth = () => {
            if (scrollRef.current) {
                const viewport = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLDivElement | null;
                viewportRef.current = viewport;
                const newWidth = (viewport || scrollRef.current).clientWidth;
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
    }, [selectedCategory, selectedPreprocessorId]);

    // Sync active category to manual scroll position
    useEffect(() => {
        if (selectedCategory || selectedPreprocessorId) return; // Only in overview mode
        const viewport = viewportRef.current;
        if (!viewport) return;

        let rafId = 0;
        const handleScroll = () => {
            cancelAnimationFrame(rafId);
            rafId = requestAnimationFrame(() => {
                const viewportTop = viewport.getBoundingClientRect().top;
                let nearestCategory: string | null = null;
                let nearestDelta = Infinity;
                for (const category of categories) {
                    const section = categorySectionRefs.current[category];
                    if (!section) continue;
                    const sectionTop = section.getBoundingClientRect().top;
                    const delta = Math.abs(sectionTop - viewportTop);
                    if (delta < nearestDelta) {
                        nearestDelta = delta;
                        nearestCategory = category;
                    }
                }
                if (nearestCategory && nearestCategory !== activeCategory) {
                    setActiveCategory(nearestCategory);
                }
            });
        };

        viewport.addEventListener('scroll', handleScroll, { passive: true });
        window.addEventListener('resize', handleScroll);
        handleScroll();

        return () => {
            viewport.removeEventListener('scroll', handleScroll as EventListener);
            window.removeEventListener('resize', handleScroll);
            cancelAnimationFrame(rafId);
        };
    }, [categories, selectedCategory, selectedPreprocessorId, activeCategory]);

    if (selectedPreprocessorId) {
        return (
            <PreprocessorPage preprocessorId={selectedPreprocessorId} onBack={() => setSelectedPreprocessorId(null)} />
        );
    }

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
                    onMoreInfo={(id) => setSelectedPreprocessorId(id)}
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
                                placeholder="Search preprocessors..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full outline-none bg-brand"
                            />
                        </div>
                    </div>
                    <ScrollArea className="flex-1" ref={scrollRef}>
                        <div className="flex flex-col gap-y-5 pt-1 pb-28">
                            {categories.map((category) => (
                                <div
                                    key={category}
                                    ref={(el) => { categorySectionRefs.current[category] = el; }}
                                    className="w-full"
                                >
                                    <PreprocessorCategory 
                                        width={scrollWidth - 36} 
                                        category={category} 
                                        preprocessors={filteredPreprocessors.filter((preprocessor) => preprocessor.category === category)}
                                        onViewAll={() => setSelectedCategory(category)}
                                        onMoreInfo={(id) => setSelectedPreprocessorId(id)}
                                    />
                                </div>
                            ))}
                        </div>
                    </ScrollArea>
                </div>
            </div>
        </div>
    </>
  )
}

export default PreprocessorMenu