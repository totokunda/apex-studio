import React, { useEffect, useState, forwardRef, useRef, useMemo } from 'react'
import { FixedSizeGrid as Grid } from 'react-window';
import { fetchFilters} from '@app/preload';
import { Filter } from '@/lib/types';
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"
import Draggable from '@/components/dnd/Draggable';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Search } from 'lucide-react';

const FilterItem = ({ filter }: { filter: Filter }) => {
  return (
    <Draggable id={filter.id} data={{...filter, type: 'filter', absPath: filter.examplePath, assetUrl: filter.exampleAssetUrl, fillCanvas: true }}>
      <div className="w-full h-full flex flex-col p-2  rounded-md hover:bg-brand-light/5 transition-colors cursor-pointer">
        <div className="flex-1 overflow-hidden rounded-md mb-2">
          <img src={filter.examplePath} alt={filter.name} className="w-full h-full object-cover" />
         
        </div>
        <div className="text-[10px] flex  flex-col gap-x-1 w-full  truncate">
          <span className="text-brand-light/80 text-[10px] truncate">{filter.name}</span>
          <span className="text-brand-light/25 text-[9.5px] truncate font-light">{filter.category}</span>
        </div>
      </div>
    </Draggable>
  )
}

const CustomScrollbarsVirtualList = forwardRef<HTMLDivElement, React.HTMLProps<HTMLDivElement>>(({ ...props }, ref) => (
  <ScrollAreaPrimitive.Root className="relative overflow-hidden h-full  flex flex-col items-center">
    <ScrollAreaPrimitive.Viewport 
      ref={ref} 
      className="h-full w-full rounded-[inherit]" 
      {...props} 
    />
    <ScrollAreaPrimitive.Scrollbar
      orientation="vertical"
      className="flex touch-none select-none transition-colors h-full w-2 border-l border-l-transparent p-[1px]"
    >
      <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-brand-light/10 hover:bg-brand-light/20 transition-colors" />
    </ScrollAreaPrimitive.Scrollbar>
    <ScrollAreaPrimitive.Corner />
  </ScrollAreaPrimitive.Root>
));

CustomScrollbarsVirtualList.displayName = 'CustomScrollbarsVirtualList';

const FilterMenu = () => {
  const [filters, setFilters] = useState<Filter[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [search, setSearch] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [containerWidth, setContainerWidth] = useState<number>(600);
  const [containerHeight, setContainerHeight] = useState<number>(600);
  const containerRef = useRef<HTMLDivElement>(null);
  const filterBarRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<any>(null);

  useEffect(() => {
    fetchFilters().then((filters) => {
      const uniqueCategories = Array.from(new Set(filters.map((filter) => filter.category)));
      setCategories(uniqueCategories);
      setFilters(filters);
    });
  }, []);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth);
        setContainerHeight(containerRef.current.offsetHeight);
      }
    };
    
    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    
    return () => resizeObserver.disconnect();
  }, []);

  const filteredFilters = useMemo(() => {
    return filters.filter((filter) => {
      const matchesSearch = search === '' || 
        filter.name.toLowerCase().includes(search.toLowerCase()) ||
        filter.category.toLowerCase().includes(search.toLowerCase());
      const matchesCategory = selectedCategory === 'all' || filter.category === selectedCategory;
      return matchesSearch && matchesCategory;
    });
  }, [filters, search, selectedCategory]);

  const ITEM_MIN_WIDTH = 140;
  const ROW_GAP = 24;
  const COLUMN_GAP = 12;
  const FILTER_BAR_HEIGHT = 80;
  const HORIZONTAL_PADDING = 32; // px-4 (left + right) = 16px * 2 = 32px
  
  const availableWidth = containerWidth - HORIZONTAL_PADDING;
  const columnCount = Math.max(1, Math.floor(availableWidth / (ITEM_MIN_WIDTH + COLUMN_GAP)));
  const columnWidth = Math.floor(availableWidth / columnCount);
  
  // Calculate dynamic item height based on column width
  // Image takes most of the space, text/padding takes ~40px
  const itemHeight = Math.floor(columnWidth * 0.8);
  const rowHeight = itemHeight + ROW_GAP;
  
  const rowCount = Math.ceil(filteredFilters.length / columnCount);
  const gridHeight = Math.max(0, containerHeight - FILTER_BAR_HEIGHT - 40);

  const Cell = ({ columnIndex, rowIndex, style }: { columnIndex: number; rowIndex: number; style: React.CSSProperties }) => {
    const index = rowIndex * columnCount + columnIndex;
    if (index >= filteredFilters.length) return null;
    
    const isLastRow = rowIndex === rowCount - 1;
    const extraBottomPadding = isLastRow ? 16 : 0;
    
    // Add offset for top spacing by adjusting the top position
    const topOffset = 12;
    const adjustedTop = (style.top as number) + topOffset;
    
    return (
      <div style={{
        ...style,
        top: adjustedTop,
        paddingRight: COLUMN_GAP,
        paddingBottom: ROW_GAP + extraBottomPadding,
      }} className="box-border">
        <div className="w-full h-full">
          <FilterItem filter={filteredFilters[index]} />
        </div>
      </div>
    );
  };

  return (
    <div ref={containerRef} className="flex flex-col w-full h-full">
      <div ref={filterBarRef} className="flex flex-col gap-3 p-4 border-b border-brand-light/5">
        <div className="flex flex-row gap-2 w-full">
          <div className="relative w-3/5">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-brand-light/40" />
            <Input
              placeholder="Search filters..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9  border-0 text-brand-light text-[12px]! bg-brand placeholder:text-brand-light/40"
            />
          </div>
          <div className="relative w-2/5">
          <Select value={selectedCategory} onValueChange={setSelectedCategory}> 
            <SelectTrigger className="w-full bg-brand border-0 text-brand-light text-[11.5px]!">
              <SelectValue placeholder="All Categories" />
            </SelectTrigger>
            <SelectContent className="text-[11px]! dark font-poppins">
              <SelectItem value="all" className="text-[11.5px]! text-brand-light">All Categories</SelectItem>
              {categories.map((category) => (
                <SelectItem key={category} value={category} className="text-[11.5px]! text-brand-light">
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          </div>
        </div>
      </div>
      <Grid
        ref={gridRef}
        columnCount={columnCount}
        columnWidth={columnWidth}
        height={gridHeight}
        rowCount={rowCount}
        rowHeight={rowHeight}
        width={availableWidth}
        outerElementType={CustomScrollbarsVirtualList}
      >
        {Cell}
      </Grid>

    </div>
  )
}

export default FilterMenu
