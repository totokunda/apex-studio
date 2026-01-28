import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { LuFolder, LuArrowUpDown, LuRefreshCw } from "react-icons/lu";
import { cn } from "@/lib/utils";
import { MediaItem, MediaThumb } from "@/components/media/Item";
import { getMediaInfo } from "@/lib/media/utils";
import { deleteFile, listServerMediaPage, revealPathInFolder } from "@app/preload";
import { useProjectsStore } from "@/lib/projects";
import Draggable from "@/components/dnd/Draggable";
import { RiAiGenerate } from "react-icons/ri";
import { FixedSizeList } from "react-window";
import chunk from "lodash/chunk";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { BsFilter } from "react-icons/bs";
import { TbDots, TbFolderOpen, TbTrash } from "react-icons/tb";
import { toast } from "sonner";
import { IoFileTrayOutline } from "react-icons/io5";

type GenerationMediaType = "video" | "image";
type SortKey = "name" | "date";
type SortOrder = "asc" | "desc";

const getGenerationDedupeKey = (it: MediaItem): string => {
  // Prefer stable filesystem identity when available, otherwise fall back to assetUrl.
  // As a last resort, use a composite key to avoid rendering duplicates.
  return (
    it.absPath ||
    it.assetUrl ||
    `${it.type ?? "unknown"}:${it.name ?? "unknown"}:${it.dateAddedMs ?? "0"}`
  );
};

const dedupeItems = (items: MediaItem[]): MediaItem[] => {
  const seen = new Set<string>();
  const out: MediaItem[] = [];
  for (const it of items) {
    const k = getGenerationDedupeKey(it);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(it);
  }
  return out;
};

interface DeleteAlertDialogProps {
  onDelete: () => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const DeleteAlertDialog: React.FC<
  React.PropsWithChildren<DeleteAlertDialogProps>
> = ({ onDelete, open, onOpenChange }) => {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="dark bg-brand font-poppins">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-brand-light text-base">
            Delete Generation
          </AlertDialogTitle>
          <AlertDialogDescription className="text-brand-light/70 py-0 text-sm">
            Are you sure you want to delete this generated media? This action
            cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => onOpenChange(false)}
            className="bg-brand text-brand-light cursor-pointer text-[12.5px]"
          >
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              onDelete();
              onOpenChange(false);
            }}
            className="cursor-pointer bg-brand-lighter text-[12.5px]"
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

const filterItems = (
  items: MediaItem[],
  selectedTypes: Set<GenerationMediaType>,
): MediaItem[] => {
  if (!selectedTypes || selectedTypes.size === 0) return items;
  return items.filter((it) =>
    selectedTypes.has(it.type as GenerationMediaType),
  );
};

const nameComparator = (a: MediaItem, b: MediaItem) =>
  a.name.toLowerCase().localeCompare(b.name.toLowerCase());
const dateComparator = (a: MediaItem, b: MediaItem) =>
  (a.dateAddedMs ?? 0) - (b.dateAddedMs ?? 0);

const GENERATIONS_PAGE_SIZE = 15;

const sortItems = (
  items: MediaItem[],
  sortKey: SortKey,
  sortOrder: SortOrder,
): MediaItem[] => {
  const sorted = [...items];
  let cmp: (a: MediaItem, b: MediaItem) => number;
  switch (sortKey) {
    case "date":
      cmp = dateComparator;
      break;
    case "name":
    default:
      cmp = nameComparator;
      break;
  }
  sorted.sort(cmp);
  if (sortOrder === "desc") sorted.reverse();
  return sorted;
};

interface RowData {
  rows: MediaItem[][];
  itemWidth: number;
  gap: number;
  paddingX: number;
  setDeleteItem: (item: MediaItem) => void;
  setDeleteAlertOpen: (open: boolean) => void;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
}

const GenerationsGridRow = ({
  index,
  style,
  data,
}: {
  index: number;
  style: React.CSSProperties;
  data: RowData;
}) => {
  const {
    rows,
    itemWidth,
    gap,
    paddingX,
    setDeleteItem,
    setDeleteAlertOpen,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = data;

  // Loader row
  if (index >= rows.length) {
    return (
      <div
        style={{ ...style, paddingLeft: paddingX, paddingRight: paddingX }}
        className="w-full pt-4 flex flex-col items-center gap-y-2"
      >
        {hasNextPage && (
          <button
            className={cn(
              "px-3 py-1.5 rounded-md text-brand-light/90 text-[12px] flex bg-brand flex-row items-center gap-x-2 transition-colors",
              "hover:bg-brand-light/10",
              isFetchingNextPage && "opacity-60 cursor-not-allowed",
            )}
            onClick={() => void fetchNextPage()}
            disabled={isFetchingNextPage}
          >
            <LuRefreshCw
              className={cn(
                "w-[14px] h-[14px]",
                isFetchingNextPage && "animate-spin",
              )}
            />
            <span>{isFetchingNextPage ? "Loading..." : "Load more"}</span>
          </button>
        )}
      </div>
    );
  }

  const rowItems = rows[index];

  return (
    <div
      style={{ ...style, paddingLeft: paddingX, paddingRight: paddingX }}
      className="flex flex-row"
    >
      {rowItems.map((item, idx) => (
        <div
          key={item.assetUrl}
          style={{
            width: itemWidth,
            marginRight: idx < rowItems.length - 1 ? gap : 0,
          }}
          className={cn(
            "group rounded-md transition-colors p-2.5 hover:bg-brand-light/5",
          )}
        >
          <div className="relative">
            <Draggable id={item.assetUrl} data={item}>
              <div className="w-full aspect-video overflow-hidden rounded-md bg-brand relative">
                <MediaThumb item={item} />
              </div>
            </Draggable>
            <DropdownMenu>
              <div
                className="absolute top-2 right-2 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity"
                title="Actions"
              >
                <DropdownMenuTrigger className="bg-black/50 cursor-pointer hover:bg-black/70 duration-300 text-white p-1 rounded">
                  <TbDots className="w-4 h-4" />
                </DropdownMenuTrigger>

                <DropdownMenuContent
                  align="start"
                  className="dark w-40 flex flex-col text-brand-light bg-brand-background font-poppins"
                >
                  <DropdownMenuItem
                    className="py-1 rounded"
                    onClick={() => {
                      const target = item.absPath || item.assetUrl;
                      if (!target) return;
                      void revealPathInFolder(target).catch((e) => {
                        console.error("Failed to reveal generation path", e);
                        toast.error("Failed to open file location", {
                          position: "bottom-right",
                          duration: 3000,
                          style: { width: "fit-content" },
                        });
                      });
                    }}
                  >
                    <TbFolderOpen className="w-3.5 h-3.5" />
                    <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                      Open File Location
                    </span>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="py-1 rounded"
                    onClick={() => {
                      setDeleteItem(item);
                      setDeleteAlertOpen(true);
                    }}
                  >
                    <TbTrash className="w-3.5 h-3.5" />
                    <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                      Delete
                    </span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </div>
            </DropdownMenu>
          </div>
        </div>
      ))}
    </div>
  );
};

const ScrollAreaOuter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ style, children, ...props }, ref) => {
  // Extract dimensions from react-window style to apply to Root
  const { width, height, ...viewportStyle } = style || {};

  return (
    <ScrollAreaPrimitive.Root
      className="relative overflow-hidden"
      style={{ width, height }}
    >
      <ScrollAreaPrimitive.Viewport
        ref={ref}
        style={{ ...viewportStyle, width: "100%", height: "100%" }}
        className="h-full w-full rounded-[inherit]"
        {...props}
      >
        {children}
      </ScrollAreaPrimitive.Viewport>
      <ScrollAreaPrimitive.Scrollbar
        orientation="vertical"
        className="flex touch-none select-none transition-colors h-full w-2 border-l border-l-transparent p-px"
      >
        <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-brand-light/10 hover:bg-brand-light/20 transition-colors" />
      </ScrollAreaPrimitive.Scrollbar>
    </ScrollAreaPrimitive.Root>
  );
});
ScrollAreaOuter.displayName = "ScrollAreaOuter";

const GenerationsMenu: React.FC = () => {
  const queryClient = useQueryClient();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [panelSize, setPanelSize] = useState({ width: 0, height: 0 });
  const [selectedTypes, setSelectedTypes] = useState<Set<GenerationMediaType>>(
    new Set(),
  );
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [filterOpen, setFilterOpen] = useState(false);
  const [sortOpen, setSortOpen] = useState(false);
  const [deleteAlertOpen, setDeleteAlertOpen] = useState(false);
  const [deleteItem, setDeleteItem] = useState<MediaItem | null>(null);
  const activeProject = useProjectsStore((s) => s.getActiveProject());
  const folderUuid = activeProject?.folderUuid ?? null;
  const [activeMediaType, setActiveMediaType] = useState<
    "generations" | "processors"
  >("generations");
  const [mediaTypeOpen, setMediaTypeOpen] = useState(false);

  const generationsQueryKey = useMemo(
    () => ["media", activeMediaType, folderUuid, sortKey, sortOrder] as const,
    [folderUuid, activeMediaType, sortKey, sortOrder],
  );

  const {
    data,
    isLoading,
    isFetching,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
    refetch,
  } = useInfiniteQuery({
    queryKey: generationsQueryKey,
    initialPageParam: null as string | null,
    queryFn: async ({ pageParam }) => {
      const page = await listServerMediaPage({
        folderUuid: folderUuid ?? undefined,
        cursor: pageParam,
        limit: GENERATIONS_PAGE_SIZE,
        type: activeMediaType,
        sortKey,
        sortOrder,
      });

      const infoPromises = page.items.map((it) =>
        getMediaInfo(it.assetUrl, { sourceDir: "apex-cache" }),
      );
      const infos = await Promise.allSettled(infoPromises);
      // filter out the rejected promises
      const filteredInfos = infos.filter((it) => it.status === "fulfilled").map((it) => it.value);

      const results: MediaItem[] = filteredInfos.map((it, idx) => ({
        name: page.items[idx].name,
        type: page.items[idx].type,
        absPath: page.items[idx].absPath,
        assetUrl: page.items[idx].assetUrl,
        dateAddedMs: page.items[idx].dateAddedMs,
        mediaInfo: it,
        hasProxy: false,
      }));

      return { items: results, nextCursor: page.nextCursor };
    },
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    placeholderData: (prev) => {
      if (prev) return prev;
    },
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: Infinity, // 5 minutes
  });

  const items = useMemo(() => {
    const pages = data?.pages ?? [];
    const merged = pages.flatMap((p) => p.items ?? []);
    return dedupeItems(merged);
  }, [data]);

  // Track panel size to size the List dynamically
  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const update = () => {
      setPanelSize({ width: el.clientWidth, height: el.clientHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    window.addEventListener("resize", update);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);

  // Refresh generations whenever an engine job completes
  useEffect(() => {
    const handler = async () => {
      // Always reload from disk so we pick up any newly written engine_results
      queryClient.removeQueries({ queryKey: generationsQueryKey, exact: true });
      await refetch();
    };

    try {
      window.addEventListener(
        "generations-menu-reload",
        handler as EventListener,
      );
    } catch {
      // In non-browser environments this may fail; ignore.
    }

    return () => {
      try {
        window.removeEventListener(
          "generations-menu-reload",
          handler as EventListener,
        );
      } catch {
        // ignore
      }
    };
  }, [queryClient, generationsQueryKey, refetch]);

  const displayItems = useMemo(() => {
    const filtered = filterItems(items, selectedTypes);
    return sortItems(filtered, sortKey, sortOrder);
  }, [items, selectedTypes, sortKey, sortOrder]);

  const handleDelete = useCallback(
    async (item: MediaItem) => {
      try {
        // Use absolute filesystem path for deletion, mirroring MediaMenu's behavior
        const target = item.absPath || item.assetUrl;
        if (!target) return;

        await deleteFile(target);

        // Reload from disk to stay in sync with actual engine_results contents
        queryClient.removeQueries({
          queryKey: generationsQueryKey,
          exact: true,
        });
        await refetch();

        toast.success("Generation deleted", {
          position: "bottom-right",
          duration: 3000,
          style: { width: "fit-content" },
        });
      } catch (e) {
        console.error(e);
        toast.error("Failed to delete generation", {
          position: "bottom-right",
          duration: 3000,
          style: { width: "fit-content" },
        });
      }
    },
    [queryClient, generationsQueryKey, refetch],
  );

  const loading = isLoading;

  // Layout Calculations
  const PADDING_X = 20; // px-5
  const GAP = 8; // gap-2
  const MIN_ITEM_WIDTH = 140;

  const availableWidth = Math.max(0, panelSize.width - PADDING_X * 2);
  const numCols = Math.max(
    1,
    Math.floor((availableWidth + GAP) / (MIN_ITEM_WIDTH + GAP)),
  );
  // Re-calculate precise item width to fill space
  const itemWidth = (availableWidth - (numCols - 1) * GAP) / numCols;
  // Item Card Height:
  // Wrapper padding p-2.5 = 10px each side.
  // Inner content width = itemWidth - 20.
  // Inner content height = innerWidth * (9/16).
  // Wrapper height = Inner height + 20.
  const itemHeight = (itemWidth - 20) * (9 / 16) + 20;
  const rowHeight = itemHeight + GAP;

  const rows = useMemo(
    () => chunk(displayItems, numCols),
    [displayItems, numCols],
  );

  const onItemsRendered = ({ visibleStopIndex }: any) => {
    if (
      visibleStopIndex >= rows.length - 1 &&
      hasNextPage &&
      !isFetchingNextPage
    ) {
      void fetchNextPage();
    }
  };

  const itemData = useMemo<RowData>(
    () => ({
      rows,
      itemWidth,
      gap: GAP,
      paddingX: PADDING_X,
      setDeleteItem,
      setDeleteAlertOpen,
      hasNextPage,
      isFetchingNextPage,
      fetchNextPage,
    }),
    [
      rows,
      itemWidth,
      setDeleteItem,
      setDeleteAlertOpen,
      hasNextPage,
      isFetchingNextPage,
      fetchNextPage,
    ],
  );

  return (
    <div ref={panelRef} className="h-full w-full duration-200 ease-out">
      <div className="border-t border-brand-light/5 mt-2" />
      <div className="px-5 py-2">
        <div className="flex flex-row items-center gap-x-1.5">
          <DropdownMenu open={filterOpen} onOpenChange={setFilterOpen}>
            <DropdownMenuTrigger asChild>
              <button
                className={cn(
                  "px-2.5 py-1.5 rounded-[6px] text-brand-light/90 text-[11px] font-medium flex bg-brand flex-row items-center gap-x-2 transition-colors",
                  "hover:bg-brand-light/10",
                  (filterOpen || selectedTypes.size > 0) && "bg-brand-light/10",
                )}
                title="Filter"
              >
                <BsFilter className="w-[16px] h-[16px]" />
                <span>Filter</span>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="start"
              className="dark w-44 flex flex-col text-brand-light bg-brand-background font-poppins text-[12px]"
            >
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={selectedTypes.size === 0}
                onCheckedChange={(checked) => {
                  if (checked) setSelectedTypes(new Set());
                }}
              >
                All
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={selectedTypes.has("video")}
                onCheckedChange={(checked) => {
                  setSelectedTypes((prev) => {
                    const next = new Set(prev);
                    if (checked) next.add("video");
                    else next.delete("video");
                    return next;
                  });
                }}
              >
                Videos
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={selectedTypes.has("image")}
                onCheckedChange={(checked) => {
                  setSelectedTypes((prev) => {
                    const next = new Set(prev);
                    if (checked) next.add("image");
                    else next.delete("image");
                    return next;
                  });
                }}
              >
                Images
              </DropdownMenuCheckboxItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu open={sortOpen} onOpenChange={setSortOpen}>
            <DropdownMenuTrigger asChild>
              <button
                className={cn(
                  "px-2.5 py-1.5 rounded-[6px] text-brand-light/90 text-[11px] font-medium flex bg-brand flex-row items-center gap-x-2 transition-colors",
                  "hover:bg-brand-light/10",
                  sortOpen && "bg-brand-light/10",
                )}
                title="Sort"
              >
                <LuArrowUpDown className="w-[16px] h-[16px]" />
                <span>Sort</span>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="start"
              className="dark w-48 flex flex-col text-brand-light bg-brand-background font-poppins"
            >
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={sortKey === "date"}
                onCheckedChange={(checked) => {
                  if (checked) setSortKey("date");
                }}
              >
                Date Added
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={sortKey === "name"}
                onCheckedChange={(checked) => {
                  if (checked) setSortKey("name");
                }}
              >
                Name
              </DropdownMenuCheckboxItem>
              <DropdownMenuSeparator />
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={sortOrder === "desc"}
                onCheckedChange={(checked) => {
                  if (checked) setSortOrder("desc");
                }}
              >
                Newest First
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={sortOrder === "asc"}
                onCheckedChange={(checked) => {
                  if (checked) setSortOrder("asc");
                }}
              >
                Oldest First
              </DropdownMenuCheckboxItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu open={mediaTypeOpen} onOpenChange={setMediaTypeOpen}>
            <DropdownMenuTrigger asChild>
              <button
                className={cn(
                  "px-2.5 py-1.5 rounded-[6px] text-brand-light/90 text-nowrap text-[11px] font-medium flex bg-brand flex-row items-center gap-x-2 transition-colors",
                  "hover:bg-brand-light/10",
                  mediaTypeOpen && "bg-brand-light/10",
                )}
                title="Media Type"
              >
                <IoFileTrayOutline className="w-[16px] h-[16px]" />
                <span>Media Type</span>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="start"
              className="dark w-48 flex flex-col text-brand-light bg-brand-background font-poppins"
            >
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={activeMediaType === "generations"}
                onCheckedChange={(checked) => {
                  if (checked) setActiveMediaType("generations");
                }}
              >
                Generations
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                className="text-[11px] font-medium"
                checked={activeMediaType === "processors"}
                onCheckedChange={(checked) => {
                  if (checked) setActiveMediaType("processors");
                }}
              >
                Processors
              </DropdownMenuCheckboxItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <button
            className={cn(
              "px-2.5 py-1.5 rounded-[6px] text-brand-light/90 text-[11px] font-medium flex bg-brand flex-row items-center gap-x-2 transition-colors",
              "hover:bg-brand-light/10",
              isFetching && "opacity-60 cursor-not-allowed",
            )}
            title="Refresh"
            onClick={() => {
              if (isFetching) return;
              queryClient.removeQueries({
                queryKey: generationsQueryKey,
                exact: true,
              });
              void refetch();
            }}
            disabled={isFetching}
          >
            <LuRefreshCw
              className={cn(
                "w-[16px] h-[16px]",
                isFetching && "animate-spin",
              )}
            />
            <span>Refresh</span>
          </button>
        </div>
      </div>
      <div className="relative">
        {loading && displayItems.length === 0 && (
          <div className="text-brand-lighter/70 text-[13px] px-5 py-8 justify-center flex flex-row items-center gap-x-2">
            <div className="rounded-md border bg-brand border-brand-lighter/30 animate-ripple font-medium px-3.5 py-1 flex flex-row items-center gap-x-2">
              <RiAiGenerate className="w-5 h-5 animate-pulse" />
              <div className="relative z-10 flex flex-row items-center gap-x-2">
                Loading generations...
              </div>
            </div>
          </div>
        )}
        {!loading && displayItems.length === 0 && (
          <div
            style={{ height: Math.max(0, panelSize.height - 72) }}
            className="flex flex-col items-center justify-center w-full text-sm gap-y-2 text-brand-lighter/60"
          >
            <LuFolder className="w-7 h-7 text-brand-light" />
            <div className="flex flex-col items-center justify-center gap-y-1">
              <span className="text-brand-lighter text-base font-medium">
                No generations yet
              </span>
              <span className="text-brand-lighter/50 text-xs">
                Run a model to create some outputs.
              </span>
            </div>
          </div>
        )}
        {displayItems.length > 0 && (
          <FixedSizeList
            height={Math.max(0, panelSize.height - 72)}
            width={panelSize.width}
            itemCount={rows.length + (hasNextPage ? 1 : 0)}
            itemSize={rowHeight}
            itemData={itemData}
            onItemsRendered={onItemsRendered}
            outerElementType={ScrollAreaOuter}
            className="pb-12"
            style={{
              overflowX: "hidden",
            }}
          >
            {GenerationsGridRow}
          </FixedSizeList>
        )}
      </div>
      {/* Local overlay to dim the app when the confirmation dialog is open */}
      {deleteAlertOpen && (
        <div className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm pointer-events-none" />
      )}
      <DeleteAlertDialog
        open={deleteAlertOpen}
        onOpenChange={setDeleteAlertOpen}
        onDelete={() => {
          if (deleteItem) {
            void handleDelete(deleteItem);
            setDeleteItem(null);
          }
        }}
      />
    </div>
  );
};
export default GenerationsMenu;
