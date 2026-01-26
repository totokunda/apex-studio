import React, { useEffect, useMemo, useRef } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { LuRefreshCw } from "react-icons/lu";

import { cn } from "@/lib/utils";
import { useProjectsStore } from "@/lib/projects";
import { getMediaInfo } from "@/lib/media/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MediaItem, MediaThumb } from "@/components/media/Item";
import { listServerMediaPage } from "@app/preload";

export type ServerMediaType = "generations" | "processors";

type ServerMediaPickerGridProps = {
  mediaType: ServerMediaType;
  enabled?: boolean;
  allowedTypes?: Array<"image" | "video" | "audio">;
  filterItem?: (item: MediaItem) => boolean;
  showItemName?: boolean;
  isSelected: (item: MediaItem) => boolean;
  onSelect: (item: MediaItem) => void;
  emptyTitle?: string;
  emptySubtitle?: string;
  scrollAreaClassName?: string;
};

const PAGE_SIZE = 30;

const dedupeKey = (it: MediaItem): string =>
  it.absPath ||
  it.assetUrl ||
  `${it.type ?? "unknown"}:${it.name ?? "unknown"}:${it.dateAddedMs ?? "0"}`;

const dedupe = (items: MediaItem[]): MediaItem[] => {
  const seen = new Set<string>();
  const out: MediaItem[] = [];
  for (const it of items) {
    const k = dedupeKey(it);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(it);
  }
  return out;
};

const defaultEmpty = (mediaType: ServerMediaType) => {
  if (mediaType === "processors") {
    return {
      title: "No processor outputs yet",
      subtitle: "Run a processor to create some outputs.",
    };
  }
  return {
    title: "No generations yet",
    subtitle: "Run a model to create some outputs.",
  };
};

const ServerMediaPickerGrid: React.FC<ServerMediaPickerGridProps> = ({
  mediaType,
  enabled = true,
  allowedTypes,
  filterItem,
  showItemName = true,
  isSelected,
  onSelect,
  emptyTitle,
  emptySubtitle,
  scrollAreaClassName,
}) => {
  const queryClient = useQueryClient();
  const activeProject = useProjectsStore((s) => s.getActiveProject());
  const folderUuid = activeProject?.folderUuid ?? undefined;

  const queryKey = useMemo(
    () => ["server-media-picker", mediaType, folderUuid] as const,
    [mediaType, folderUuid],
  );

  const {
    data,
    isLoading,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
    refetch,
  } = useInfiniteQuery({
    queryKey,
    initialPageParam: null as string | null,
    enabled: !!enabled,
    queryFn: async ({ pageParam }) => {
      const page = await listServerMediaPage({
        folderUuid,
        cursor: pageParam,
        limit: PAGE_SIZE,
        type: mediaType,
        sortKey: "date",
        sortOrder: "desc",
      });

      const settled = await Promise.allSettled(
        (page.items ?? []).map((it) =>
          getMediaInfo(it.assetUrl, { sourceDir: "apex-cache" }),
        ),
      );

      const items: MediaItem[] = (page.items ?? []).map((it, idx) => ({
        name: it.name,
        type: it.type,
        absPath: it.absPath,
        assetUrl: it.assetUrl,
        dateAddedMs: it.dateAddedMs,
        mediaInfo: settled[idx]?.status === "fulfilled" ? settled[idx].value : undefined,
        hasProxy: false,
      }));

      return { items, nextCursor: page.nextCursor };
    },
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 60_000,
  });

  const items = useMemo(() => {
    const pages = data?.pages ?? [];
    const merged = pages.flatMap((p) => p.items ?? []);
    const deduped = dedupe(merged);
    const filteredByType =
      Array.isArray(allowedTypes) && allowedTypes.length > 0
        ? deduped.filter((it) => allowedTypes.includes(it.type as any))
        : deduped;
    const filtered = typeof filterItem === "function"
      ? filteredByType.filter(filterItem)
      : filteredByType;
    return filtered;
  }, [data, allowedTypes, filterItem]);

  // Keep in sync with GenerationsMenu refreshes (engine job completion).
  useEffect(() => {
    if (!enabled) return;
    const handler = async () => {
      queryClient.removeQueries({ queryKey, exact: true });
      await refetch();
    };
    try {
      window.addEventListener(
        "generations-menu-reload",
        handler as unknown as EventListener,
      );
    } catch {
      // ignore
    }
    return () => {
      try {
        window.removeEventListener(
          "generations-menu-reload",
          handler as unknown as EventListener,
        );
      } catch {
        // ignore
      }
    };
  }, [enabled, queryClient, queryKey, refetch]);

  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  // Auto-fetch next page when scrolling near the bottom.
  useEffect(() => {
    if (!enabled) return;
    if (!hasNextPage) return;
    if (isFetchingNextPage) return;
    const rootEl = scrollAreaRef.current;
    const sentinel = sentinelRef.current;
    if (!rootEl || !sentinel) return;
    const viewport = rootEl.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLElement | null;
    if (!viewport) return;

    const obs = new IntersectionObserver(
      (entries) => {
        const first = entries[0];
        if (!first?.isIntersecting) return;
        if (!hasNextPage || isFetchingNextPage) return;
        void fetchNextPage();
      },
      { root: viewport, rootMargin: "100px" },
    );
    obs.observe(sentinel);
    return () => obs.disconnect();
  }, [enabled, hasNextPage, isFetchingNextPage, fetchNextPage, items.length]);

  const empty = defaultEmpty(mediaType);
  const title = emptyTitle ?? empty.title;
  const subtitle = emptySubtitle ?? empty.subtitle;

  return (
    <ScrollArea
      ref={scrollAreaRef as any}
      className={cn("w-full h-96 ", scrollAreaClassName)}
    >
      {isLoading && items.length === 0 ? (
        <div className="w-full h-40 flex items-center justify-center text-brand-light/70 text-[10.5px]">
          Loading...
        </div>
      ) : items.length === 0 ? (
        <div className="w-full h-40 flex flex-col items-center justify-center text-brand-light/70 text-[10.5px] gap-y-1">
          <div className="text-brand-light/90 font-medium">{title}</div>
          <div className="text-brand-light/50">{subtitle}</div>
        </div>
      ) : (
        <div className="w-full px-3">
          <div className="w-full h-full grid grid-cols-2 gap-3">
            {items.map((item) => {
              const selected = isSelected(item);
              return (
                <div
                  key={dedupeKey(item)}
                  onClick={() => onSelect(item)}
                  className={cn(
                    "w-full flex flex-col items-center justify-center cursor-pointer group relative",
                    showItemName ? "gap-y-1.5" : "gap-y-0",
                  )}
                >
                  <div className="relative">
                    <div
                      className={cn(
                        "absolute top-0 left-0 w-full h-full bg-brand-background-light/50 backdrop-blur-sm rounded-md z-20 group-hover:opacity-100 transition-all duration-200 flex items-center justify-center",
                        selected ? "opacity-100" : "opacity-0",
                      )}
                    >
                      <div
                        className={cn(
                          "rounded-full py-1 px-3 bg-brand-light/10 flex items-center justify-center font-medium text-[10.5px] w-fit",
                          selected ? "bg-brand-light/20" : "",
                        )}
                      >
                        {selected ? "Selected" : "Use as Input"}
                      </div>
                    </div>
                    <MediaThumb item={item} />
                  </div>
                  {showItemName && (
                    <div className="text-brand-light/90 text-[9.5px] text-start truncate w-full text-ellipsis overflow-hidden group-hover:text-brand-light transition-all duration-200">
                      {item.name}
                    </div>
                  )}
                </div>
              );
            })}
            <div className="col-span-2 flex flex-col items-center justify-center gap-y-2 pt-1 pb-2">
              {hasNextPage && (
                <button
                  type="button"
                  onClick={() => void fetchNextPage()}
                  disabled={isFetchingNextPage}
                  className={cn(
                    "px-3 py-1.5 rounded-md text-brand-light/90 text-[11px] font-medium flex bg-brand flex-row items-center gap-x-2 transition-colors",
                    "hover:bg-brand-light/10",
                    isFetchingNextPage && "opacity-60 cursor-not-allowed",
                  )}
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
              <div ref={sentinelRef} className="h-1 w-full" />
            </div>
          </div>
        </div>
      )}
    </ScrollArea>
  );
};

export default ServerMediaPickerGrid;

