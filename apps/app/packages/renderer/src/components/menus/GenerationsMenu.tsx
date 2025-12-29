import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LuFolder, LuArrowUpDown, LuRefreshCw } from "react-icons/lu";
import { cn } from "@/lib/utils";
import { MediaItem, MediaThumb } from "@/components/media/Item";
import { getMediaInfo } from "@/lib/media/utils";
import { deleteFile, listGeneratedMediaPage } from "@app/preload";
import { useProjectsStore } from "@/lib/projects";
import Draggable from "@/components/dnd/Draggable";
import { RiAiGenerate } from "react-icons/ri";
import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { ModelClipProps } from "@/lib/types";
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
import { TbDots, TbTrash } from "react-icons/tb";
import { toast } from "sonner";

type GenerationMediaType = "video" | "image";
type SortKey = "name" | "date";
type SortOrder = "asc" | "desc";

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

const GenerationsMenu: React.FC = () => {
  const queryClient = useQueryClient();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [panelHeight, setPanelHeight] = useState<number>(0);
  const scrollAreaRootRef = useRef<HTMLDivElement | null>(null);
  const loadMoreSentinelRef = useRef<HTMLDivElement | null>(null);
  const [selectedTypes, setSelectedTypes] = useState<
    Set<GenerationMediaType>
  >(new Set());
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [filterOpen, setFilterOpen] = useState(false);
  const [sortOpen, setSortOpen] = useState(false);
  const [deleteAlertOpen, setDeleteAlertOpen] = useState(false);
  const [deleteItem, setDeleteItem] = useState<MediaItem | null>(null);
  const activeProject = useProjectsStore((s) => s.getActiveProject());
  const folderUuid = activeProject?.folderUuid ?? null;
  const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
  const getClipById = useClipStore((s) => s.getClipById);
  const updateClip = useClipStore((s) => s.updateClip);
  const addAsset = useClipStore((s) => s.addAsset);

  const generationsQueryKey = useMemo(
    () => ["media", "generated", folderUuid] as const,
    [folderUuid],
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
      const page = await listGeneratedMediaPage({
        folderUuid: folderUuid ?? undefined,
        cursor: pageParam,
        limit: GENERATIONS_PAGE_SIZE,
      });

      const infoPromises = page.items.map((it) =>
        getMediaInfo(it.assetUrl, { sourceDir: "apex-cache" }),
      );
      const infos = await Promise.all(infoPromises);

      const results: MediaItem[] = page.items.map((it, idx) => ({
        name: it.name,
        type: it.type,
        absPath: it.absPath,
        assetUrl: it.assetUrl,
        dateAddedMs: it.dateAddedMs,
        mediaInfo: infos[idx],
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
    return pages.flatMap((p) => p.items ?? []);
  }, [data]);

  // Track panel height to size the ScrollArea dynamically
  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const update = () => setPanelHeight(el.clientHeight);
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

  // Infinite scroll: when sentinel becomes visible, fetch the next page.
  useEffect(() => {
    const rootEl = scrollAreaRootRef.current;
    const sentinelEl = loadMoreSentinelRef.current;
    if (!rootEl || !sentinelEl) return;

    const viewport = rootEl.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLElement | null;
    if (!viewport) return;

    const obs = new IntersectionObserver(
      (entries) => {
        const hit = entries.some((e) => e.isIntersecting);
        if (!hit) return;
        if (!hasNextPage) return;
        if (isFetchingNextPage) return;
        void fetchNextPage();
      },
      {
        root: viewport,
        rootMargin: "250px",
        threshold: 0.01,
      },
    );

    obs.observe(sentinelEl);
    return () => obs.disconnect();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage, items.length]);

  const displayItems = useMemo(() => {
    const filtered = filterItems(items, selectedTypes);
    return sortItems(filtered, sortKey, sortOrder);
  }, [items, selectedTypes, sortKey, sortOrder]);

  const handleApplyToSelectedClip = useCallback(
    (item: MediaItem) => {
      try {
        if (!selectedClipIds || selectedClipIds.length === 0) return;
        const currentClipId = selectedClipIds[selectedClipIds.length - 1];
        if (!currentClipId) return;

        const clip = getClipById(currentClipId) as ModelClipProps | null;
        if (!clip || clip.type !== "model") return;

        const assetUrl = item.assetUrl;
        if (!assetUrl) return;

        const asset = addAsset({ path: assetUrl }, "apex-cache");
        const prevAssetId = clip.assetId;
        let assetIdHistory = Array.isArray(clip.assetIdHistory)
          ? [...clip.assetIdHistory]
          : [];

        if (
          typeof prevAssetId === "string" &&
          prevAssetId &&
          prevAssetId !== asset.id &&
          !assetIdHistory.includes(prevAssetId)
        ) {
          assetIdHistory = [...assetIdHistory, prevAssetId];
        }

        const patch: Partial<ModelClipProps> = {
          assetId: asset.id,
          previewPath: assetUrl,
        };

        if (assetIdHistory.length > 0) {
          patch.assetIdHistory = assetIdHistory;
        }

        updateClip(clip.clipId, patch as any);
      } catch (e) {
        console.error(e);
      }
    },
    [selectedClipIds, getClipById, addAsset, updateClip],
  );

  const handleDelete = useCallback(
    async (item: MediaItem) => {
      try {
        // Use absolute filesystem path for deletion, mirroring MediaMenu's behavior
        const target = item.absPath || item.assetUrl;
        if (!target) return;

        await deleteFile(target);

        // Reload from disk to stay in sync with actual engine_results contents
        queryClient.removeQueries({ queryKey: generationsQueryKey, exact: true });
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

  return (
    <div ref={panelRef} className="h-full w-full duration-200 ease-out">
      <div className="border-t border-brand-light/5 mt-2" />
      <div className="px-5 py-2">
        <div className="flex flex-row items-center gap-x-1.5">
          <DropdownMenu open={filterOpen} onOpenChange={setFilterOpen}>
            <DropdownMenuTrigger asChild>
              <button
                className={cn(
                  "px-2.5 py-1.5 rounded-md text-brand-light/90 text-[12px] flex bg-brand flex-row items-center gap-x-2 transition-colors",
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
                  "px-2.5 py-1.5 rounded-md text-brand-light/90 text-[12px] flex bg-brand flex-row items-center gap-x-2 transition-colors",
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
          <button
            className={cn(
              "px-2.5 py-1.5 rounded-md text-brand-light/90 text-[12px] flex bg-brand flex-row items-center gap-x-2 transition-colors",
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
      <div className="overflow-y-auto relative">
        {loading && displayItems.length === 0 && (
          <div className="text-brand-lighter/70 text-[13px] px-5 py-8 justify-center flex flex-row items-center gap-x-2">
              <div className="rounded-md border bg-brand border-brand-lighter/30 animate-ripple font-medium px-3.5 py-1 flex flex-row items-center gap-x-2">
                <RiAiGenerate className="w-5 h-5 animate-pulse" />
                <div className="relative z-10 flex flex-row items-center gap-x-2">Loading generations...</div>
            </div>
          </div>
        )}
        {!loading && displayItems.length === 0 && (
          <div
            style={{ height: Math.max(0, panelHeight - 72) }}
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
          <ScrollArea
            ref={scrollAreaRootRef as any}
            style={{ height: Math.max(0, panelHeight - 72) }}
            className="px-5 py-4 pt-1 dark pb-10"
          >
            <div
              className="grid gap-2 w-full"
              style={{
                gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
              }}
            >
              {displayItems.map((item) => (
                <div
                  key={item.assetUrl}
                  className={cn(
                    "group rounded-md transition-colors p-2.5 hover:bg-brand-light/5",
                  )}
                >
                  <div className="relative">
                    <Draggable id={item.assetUrl} data={item}>
                      <div className="w-full aspect-video overflow-hidden rounded-md bg-brand relative">
                        <MediaThumb item={item}  />
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
                            onClick={() => handleApplyToSelectedClip(item)}
                          >
                            <RiAiGenerate className="w-3.5 h-3.5" />
                            <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                              Use in selected clip
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

            <div className="w-full pt-4 flex flex-col items-center gap-y-2">
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
              <div ref={loadMoreSentinelRef} className="h-4 w-full" />
            </div>
          </ScrollArea>
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
