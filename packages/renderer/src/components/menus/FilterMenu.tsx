import React, { useEffect, useState, useRef, useMemo } from 'react'
import { fetchFilters} from '@app/preload';
import { Filter } from '@/lib/types';
import Draggable from '@/components/dnd/Draggable';
import { ScrollArea } from '@/components/ui/scroll-area';
import { LuSearch, LuChevronLeft, LuChevronRight, LuArrowRight } from "react-icons/lu";

const FilterItem = ({ filter }: { filter: Filter }) => {
  return (
    <Draggable id={filter.id} data={{...filter, type: 'filter', absPath: filter.examplePath, assetUrl: filter.exampleAssetUrl, fillCanvas: true }}>
      <div className="cursor-pointer w-32 transition-all duration-200 rounded-md">
        <div className="flex flex-col gap-y-2.5">
          <div className="flex items-center gap-x-1 relative">
            <img src={filter.examplePath} alt={filter.name} className="h-full object-cover rounded-md" />
          </div>
          <div className="w-full truncate leading-tight font-medium text-brand-light text-[10px] text-start">
            {filter.name}
          </div>
        </div>
      </div>
    </Draggable>
  )
}

const FilterCategory: React.FC<{category: string, filters: Filter[], width: number, onViewAll: () => void}> = ({category, filters, width, onViewAll}) => {
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
    }, [filters]);

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
                    {filters.map((filter) => (
                        <div key={filter.id} className="flex-shrink-0">
                            <FilterItem filter={filter} />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

const CategoryDetailView: React.FC<{category: string, filters: Filter[], onBack: () => void}> = ({category, filters, onBack}) => {
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
                    <div className="grid gap-x-2 gap-y-3" style={{gridTemplateColumns: 'repeat(auto-fit, minmax(128px, 1fr))'}}>
                        {filters.map((filter) => (
                            <div key={filter.id} className="flex justify-center">
                                <FilterItem filter={filter} />
                            </div>
                        ))}
                    </div>
                </div>
            </ScrollArea>
        </div>
    );
};

const FilterMenu = () => {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [filters, setFilters] = useState<Filter[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [scrollWidth, setScrollWidth] = useState(0);

  const filteredFilters = useMemo(() => {
    if (!searchQuery.trim()) return filters;
    const query = searchQuery.toLowerCase();
    return filters.filter(filter => 
      filter.name.toLowerCase().includes(query) ||
      filter.category.toLowerCase().includes(query)
    );
  }, [filters, searchQuery]);

  const categories = useMemo(() => {
    return [...new Set(filteredFilters.map((filter) => filter.category))]
  }, [filteredFilters]);

  useEffect(() => {
    fetchFilters().then((filters) => {
      setFilters(filters);
    });
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
          filters={filteredFilters.filter(f => f.category === selectedCategory)}
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
        <div className="px-7 pt-4 pb-2">
          <div className="relative">
            <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-brand-light/60" />
            <input
              type="text"
              placeholder="Search filters..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-brand text-brand-light placeholder:text-brand-light/50 rounded-md pl-10 pr-4 py-2.5 text-[12px] focus:outline-none focus:ring-2 focus:ring-brand-light/30 transition-all"
            />
          </div>
        </div>
        <ScrollArea className="flex-1 pb-16" ref={scrollRef}>
          <div className="flex flex-col gap-y-6 pt-4">
            {categories.map((category) => (
              <FilterCategory 
                width={scrollWidth - 48} 
                key={category} 
                category={category} 
                filters={filteredFilters.filter((filter) => filter.category === category)}
                onViewAll={() => setSelectedCategory(category)}
              />
            ))}
          </div>
        </ScrollArea>
      </div>
    </>
  )
}

export default FilterMenu
