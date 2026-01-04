import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type ManifestDocument,
  type ModelTypeInfo,
} from "@/lib/manifest";
import { cn } from "@/lib/utils";
import { ScrollArea } from "../ui/scroll-area";
import {
  LuChevronLeft,
  LuChevronRight,
  LuArrowRight,
  LuSearch,
  LuInfo,
  LuDownload,
  LuLoader,
  LuPlus,
  LuRefreshCw,
  LuCircleSlash,
} from "react-icons/lu";
import { TbWorldDownload } from "react-icons/tb";
import Draggable from "../dnd/Draggable";
import { useManifestStore } from "@/lib/manifest/store";
import { extractAllDownloadablePaths } from "@/lib/manifest/api";

import ModelPage from "../models/ModelPage";
// check
import CategorySidebar from "./CategorySidebar";
import { useControlsStore } from "@/lib/control";
import {
  useClipStore,
  getTimelineHeightForClip,
  getTimelineTypeForClip,
  isValidTimelineForClip,
} from "@/lib/clip";
import { v4 as uuidv4 } from "uuid";
import { useQuery, useQueryClient} from "@tanstack/react-query";
import {
  fetchManifestsAndPrimeCache,
  fetchModelTypes,
  useManifestQuery,
} from "@/lib/manifest/queries";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";

export const ModelItem: React.FC<{
  manifest: ManifestDocument;
  isDragging?: boolean;
  category?: string;
}> = ({ manifest:initialManifest, isDragging, category }) => {
  const { setSelectedManifestId } = useManifestStore();
  const tagsContainerRef = useRef<HTMLDivElement>(null);
  const hiddenMeasureRef = useRef<HTMLDivElement>(null);
  const [visibleTagCount, setVisibleTagCount] = useState<number | null>(null);
  const [isStartingDownload, setIsStartingDownload] = useState(false);
  const { data: manifestData } = useManifestQuery(initialManifest.metadata?.id || "");
  const manifest = manifestData ?? initialManifest;
  const {getSourceToJobId, getJobUpdates} = useDownloadJobIdStore();

  const isVideoDemo = React.useMemo(() => {
    const value = (manifest.metadata?.demo_path || "").toLowerCase();
    try {
      const url = new URL(value);
      const pathname = url.pathname;
      const ext = pathname.split(".").pop() || "";
      return ["mp4", "webm", "mov", "m4v", "ogg", "m3u8"].includes(ext);
    } catch {
      return (
        value.endsWith(".mp4") ||
        value.endsWith(".webm") ||
        value.endsWith(".mov") ||
        value.endsWith(".m4v") ||
        value.endsWith(".ogg") ||
        value.endsWith(".m3u8")
      );
    }
  }, [manifest.metadata?.demo_path]);

  const allDownloadablePaths = useMemo(() => {
    return extractAllDownloadablePaths(manifest);
  }, [manifest]);


  
  const isDownloading = allDownloadablePaths.some((path) => {
      const jobId = getSourceToJobId(path.path);
      return jobId && (getJobUpdates(jobId)?.length ?? 0) > 0;
    });


  const allDownloaded = useMemo(() => {
    return (manifestData?.downloaded ?? manifest.downloaded) && !isDownloading;
  }, [manifest.downloaded, isDownloading]);

  // Compute how many tags fit on a single line
  useLayoutEffect(() => {
    const computeVisibleTags = () => {
      const container = tagsContainerRef.current;
      const measure = hiddenMeasureRef.current;
      if (!container || !measure) return;
      // Match the measurement container width to the visible container
      const width = container.clientWidth;
      if (width <= 0) return;
      measure.style.width = width + "px";

      // Force layout read after width set
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      measure.offsetWidth;

      const children = Array.from(measure.children) as HTMLElement[];
      if (children.length === 0) {
        setVisibleTagCount(0);
        return;
      }
      let firstTop = Infinity;
      let count = 0;
      for (const child of children) {
        const top = child.offsetTop;
        if (firstTop === Infinity) firstTop = top;
        if (top === firstTop) {
          count += 1;
        }
      }
      setVisibleTagCount(count);
    };

    computeVisibleTags();

    const ro = new ResizeObserver(() => computeVisibleTags());
    if (tagsContainerRef.current) ro.observe(tagsContainerRef.current);
    window.addEventListener("resize", computeVisibleTags);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", computeVisibleTags);
    };
  }, [manifest.metadata?.tags]);



  useEffect(() => {
    if (isDownloading) {
      setIsStartingDownload(false);
    }
  }, [isDownloading]);



  const card = (
    <div className="flex flex-col items-center relative w-full ">
      <div
        className={cn(
          "rounded-t-md overflow-hidden  flex items-center justify-center w-full aspect-square h-28",
          {},
        )}
      >
        {isVideoDemo ? (
          <video
            src={manifest.metadata?.demo_path}
            className="h-full w-full object-cover rounded-t-md"
            autoPlay
            muted
            loop
            playsInline
          />
        ) : (
          <img
            src={manifest.metadata?.demo_path}
            alt={manifest.metadata?.name}
            className="h-full w-full object-cover rounded-t-md"
          />
        )}
      </div>
    </div>
  );

  const details = (
    <div className="flex flex-col gap-y-1.5 py-3.5 pb-2 px-3 border-t border-brand-light/5 w-full ">
      <div className="w-full truncate leading-tight font-semibold text-brand-light text-[12px] text-start">
        {manifest.metadata?.name}
      </div>
      <div
        ref={tagsContainerRef}
        className="flex items-center gap-x-1 w-full justify-start gap-y-1 overflow-hidden"
      >
        {(visibleTagCount == null
          ? manifest.metadata?.tags
          : manifest.metadata?.tags?.slice(0, visibleTagCount)
        )?.map((tag: string) => (
          <span
            key={tag}
            className="text-[8px] text-brand-light bg-brand-background border shadow border-brand-light/10 rounded px-2 py-0.5 "
          >
            {tag}
          </span>
        ))}
      </div>
      <div
        ref={hiddenMeasureRef}
        aria-hidden
        style={{
          position: "fixed",
          top: -10000,
          left: -10000,
          visibility: "hidden",
        }}
        className="flex items-center gap-x-1 flex-wrap justify-start gap-y-1"
      >
        {manifest?.metadata?.tags?.map((tag: string) => (
          <span
            key={tag}
            className="text-[8px] text-brand-light bg-brand-background border shadow border-brand-light/10 rounded px-2 py-0.5 "
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  );

  const stableId = `model-${manifest.metadata?.id}-${category}`;

  return (
    <div
      className={cn(
        "flex flex-col transition-all font-poppins duration-200 rounded-md relative bg-brand border border-brand-light/5 shadow-md cursor-grab active:cursor-grabbing",
        {
          "w-60": true,
          "opacity-[0.975]": isDragging,
        },
      )}
    >
      {isDragging ? (
        <>
          {card}
          {details}
        </>
      ) : (
        <Draggable
          id={stableId}
          data={{
            type: "model",
            category: category,
            ...manifest,
          }}
        >
          {card}
          {details}
        </Draggable>
      )}
      {!isDragging && (
        <div className="flex items-center gap-x-1 w-full p-3 pt-0 justify-between">
          <button
            onClick={() => {
              setSelectedManifestId(manifest.metadata?.id || "");
            }}
            type="button"
            className="text-[10px] font-medium w-1/2 flex items-center transition-all duration-200 justify-center gap-x-1.5 text-brand-light  hover:text-brand-light/80 bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded px-2 py-1.5"
            title="Show more info"
          >
            {allDownloaded ? <LuInfo className="w-3 h-3" /> : <LuDownload className="w-3 h-3" />}
            <span>
              {allDownloaded ? "Info" : "Download"}
            </span>
          </button>
          {<button
            onClick={async () => {
              try {
                const controls = useControlsStore.getState();
                const clipStore = useClipStore.getState();
                const fps = Math.max(1, controls.fps || 1);
                const focusFrame = Math.max(0, controls.focusFrame || 0);
                const desiredFrames = Math.max(
                  1,
                  (manifest.spec?.default_duration_secs ?? controls.defaultClipLength) * fps,
                );
                const startFrame = focusFrame;
                const endFrame = startFrame + desiredFrames;

                // Choose an existing compatible timeline with free space at [startFrame, endFrame)
                const mediaTimelines = clipStore.timelines.filter((t) =>
                  isValidTimelineForClip(t, { type: "model" } as any),
                );
                const intervalOverlaps = (
                  loA: number,
                  hiA: number,
                  loB: number,
                  hiB: number,
                ) => loA < hiB && hiA > loB;
                let targetTimelineId: string | undefined;
                for (const t of mediaTimelines) {
                  const existing = clipStore
                    .getClipsForTimeline(t.timelineId)
                    .map((c) => ({
                      lo: c.startFrame || 0,
                      hi: c.endFrame || 0,
                    }))
                    .filter((iv) => iv.hi > iv.lo);
                  const hasConflict = existing.some((iv) =>
                    intervalOverlaps(startFrame, endFrame, iv.lo, iv.hi),
                  );
                  if (!hasConflict) {
                    targetTimelineId = t.timelineId;
                    break;
                  }
                }

                // If no space found, create a new media timeline
                if (!targetTimelineId) {
                  const timelineId = uuidv4();
                  const last =
                    clipStore.timelines[clipStore.timelines.length - 1];
                  clipStore.addTimeline({
                    timelineId,
                    type: getTimelineTypeForClip("model"),
                    timelineHeight: getTimelineHeightForClip("model"),
                    timelineWidth: last?.timelineWidth ?? 0,
                    timelineY:
                      (last?.timelineY ?? 0) + (last?.timelineHeight ?? 54),
                    timelinePadding: last?.timelinePadding ?? 24,
                    muted: false,
                    hidden: false,
                  });
                  targetTimelineId = timelineId;
                }

                // Build the new clip and fetch manifest before adding
                const newClipId = uuidv4();
                const clipBase: any = {
                  timelineId: targetTimelineId,
                  clipId: newClipId,
                  startFrame,
                  endFrame,
                  // @ts-ignore
                  type: "model",
                  trimEnd: -Infinity,
                  trimStart: Infinity,
                  speed: 1.0,
                  category,
                };
                clipBase.manifest = manifest;
                useClipStore.getState().addClip(clipBase);
              } catch {}
            }}
            type="button"
            disabled={!allDownloaded}
            className={cn(
              "text-[10px] font-medium disabled:opacity-50 disabled:cursor-default! w-1/2 flex items-center transition-all duration-200 justify-center gap-x-1.5 rounded px-2 py-1.5 border text-brand-light hover:text-brand-light/90 bg-brand-background hover:bg-brand-background/70 border-brand-light/10",
            )}
            title={
              allDownloaded ? "Add clip at playhead" : "No Weights"  
            }
          >
            {allDownloaded ? (
              <LuPlus className="w-3 h-3" />
            ) : isStartingDownload || isDownloading ? (
              <LuLoader className="w-3 h-3 animate-spin" />
            ) : (
              <LuCircleSlash className="w-3 h-3" />
            )}
            <span>
              {allDownloaded ? "Add Clip" : "No Weights"}
            </span>
          </button>}
        </div>
      )}
    </div>
  );
};

const ModelCategory: React.FC<{
  category: string;
  manifests: ManifestDocument[];
  width: number;
  onViewAll: () => void;
}> = ({ category, manifests, width, onViewAll }) => {
  const carouselRef = useRef<HTMLDivElement>(null);
  const [showLeftArrow, setShowLeftArrow] = useState(false);
  const [showRightArrow, setShowRightArrow] = useState(false);

  const checkScroll = () => {
    if (carouselRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = carouselRef.current;
      const hasOverflow = scrollWidth > clientWidth;
      setShowLeftArrow(scrollLeft > 5);
      setShowRightArrow(
        hasOverflow && scrollLeft + clientWidth < scrollWidth - 5,
      );
    } else {
      setShowLeftArrow(false);
      setShowRightArrow(false);
    }
  };

  useLayoutEffect(() => {
    const carousel = carouselRef.current;
    if (!carousel) return;

    checkScroll();
    let raf = requestAnimationFrame(checkScroll);

    const ro = new ResizeObserver(() => checkScroll());
    ro.observe(carousel);

    carousel.addEventListener("scroll", checkScroll, { passive: true });
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      carousel.removeEventListener("scroll", checkScroll as EventListener);
    };
  }, [manifests, width]);

  const scroll = (direction: "left" | "right") => {
    if (carouselRef.current) {
      const scrollAmount = 300;
      carouselRef.current.scrollBy({
        left: direction === "left" ? -scrollAmount : scrollAmount,
        behavior: "smooth",
      });
    }
  };

  return (
    <div className="flex flex-col gap-y-1 w-full px-4">
      <div
        className="flex items-center justify-between py-2"
        style={{ maxWidth: width }}
      >
        <span className="text-brand-light text-[13px] font-medium">
          {category}
        </span>
        <button
          onClick={onViewAll}
          className="flex items-center gap-x-1.5 text-brand-light hover:text-brand-light/70 text-[12px] font-medium cursor-pointer transition-colors rounded-md shrink-0"
        >
          <span>View all</span>
          <LuArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="relative w-full" style={{ width: width }}>
        {showLeftArrow && (
          <button
            onClick={() => scroll("left")}
            className="absolute -left-3 top-1/2 cursor-pointer -translate-y-1/2 z-50 bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20"
          >
            <LuChevronLeft className="w-4 h-4 text-brand-light" />
          </button>
        )}
        {showRightArrow && (
          <button
            onClick={() => scroll("right")}
            className="absolute -right-3 top-1/2 cursor-pointer -translate-y-1/2 z-50 bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20"
          >
            <LuChevronRight className="w-4 h-4 text-brand-light" />
          </button>
        )}
        <div
          ref={carouselRef}
          className="carousel-container flex gap-x-2 overflow-x-auto rounded-md"
          style={{
            scrollbarWidth: "none",
            msOverflowStyle: "none",
            WebkitOverflowScrolling: "touch",
          }}
        >
          {manifests.map((manifest) => (
            <div key={manifest.metadata?.name || manifest.metadata?.id} className="shrink-0">
              <ModelItem manifest={manifest} category={category} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const CategoryDetailView: React.FC<{
  category: string;
  manifests: ManifestDocument[];
  onBack: () => void;
  scrollCache: Map<string, number>;
}> = ({ category, manifests, onBack, scrollCache }) => {
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const root = scrollAreaRef.current;
    if (!root) return;

    const key = `modelMenu:category:${category}`;
    const viewport = root.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLDivElement | null;
    if (!viewport) return;

    const saved = scrollCache.get(key);
    if (typeof saved === "number") {
      viewport.scrollTop = saved;
    }

    const onScroll = () => {
      scrollCache.set(key, viewport.scrollTop);
    };

    viewport.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      viewport.removeEventListener("scroll", onScroll as EventListener);
    };
  }, [category, scrollCache]);

  return (
    <div className="flex flex-col h-full w-full">
      <div className="px-7 pt-4 pb-4 border-b border-brand/20">
        <div className="flex items-center gap-x-3">
          <button
            onClick={onBack}
            className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded-md transition-colors cursor-pointer"
          >
            <LuChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-brand-light text-[14px] font-medium">
            {category}
          </span>
        </div>
      </div>
      <ScrollArea className="flex-1 pb-16" ref={scrollAreaRef}>
        <div className="px-7 pt-6">
          <div
            className="grid gap-x-2 gap-y-3"
            style={{
              gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            }}
          >
            {manifests.map((manifest) => (
              <div
                key={manifest.metadata?.id || ""}
                className="flex justify-center"
              >
                <ModelItem manifest={manifest} />
              </div>
            ))}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};

const ModelMenu: React.FC = () => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const scrollCacheRef = useRef<Map<string, number>>(new Map());
  const {selectedManifestId} = useManifestStore();
  const queryClient = useQueryClient();

  const manifestsQuery = useQuery<ManifestDocument[]>({
    queryKey: ["manifest"],
    queryFn: () => fetchManifestsAndPrimeCache(queryClient),
    initialData: () =>
      queryClient.getQueryData<ManifestDocument[]>(["manifest"]),
    placeholderData: (prev) => prev,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 30000,
  });
  
  const modelTypesQuery = useQuery<ModelTypeInfo[]>({
    queryKey: ["modelTypes"],
    queryFn: fetchModelTypes,
    initialData: () => queryClient.getQueryData<ModelTypeInfo[]>(["modelTypes"]),
    placeholderData: (prev) => prev,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 30000,
  });

  const manifestsData = manifestsQuery.data;
  const modelTypesData = modelTypesQuery.data;

  const backendUnavailable = manifestsQuery.isError;

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [scrollWidth, setScrollWidth] = useState(0);
  const categorySectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const DOWNLOADED_CATEGORY = "Downloaded";

  // Map from category key -> human-friendly label (backend /manifest/categories).
  const manifestCategoryKeyToLabel = useMemo(() => {
    const map = new Map<string, string>();
    (modelTypesData || []).forEach((t) => map.set(t.key, t.label));
    return map;
  }, [modelTypesData]);

  const manifests: ManifestDocument[] = useMemo(
    () => manifestsData ?? [],
    [manifestsData],
  );

  const filteredManifests = useMemo(() => {
    if (!searchQuery.trim()) return manifests;
    const query = searchQuery.toLowerCase();
    return manifests.filter((m) => {
      const categoryKeys: string[] = m.metadata?.categories || [];
      const categoryLabels = categoryKeys.map(
        (k) => manifestCategoryKeyToLabel.get(k) || k,
      );
      return (
        m.metadata?.name.toLowerCase().includes(query) ||
        (m.metadata?.description?.toLowerCase().includes(query) ?? false) ||
        m.metadata?.model?.toLowerCase().includes(query) ||
        categoryKeys.some((k) => k.toLowerCase().includes(query)) ||
        categoryLabels.some((l) => l.toLowerCase().includes(query)) ||
        (m.metadata?.tags || []).some((t: string) =>
          t.toLowerCase().includes(query),
        )
      );
    });
  }, [manifests, searchQuery, manifestCategoryKeyToLabel]);

  const categories = useMemo(() => {
    const set = new Set<string>();
    filteredManifests.forEach((m) => {
      const categoryKeys: string[] = m.metadata?.categories || [];
      categoryKeys.forEach((k) => {
        const label =
          manifestCategoryKeyToLabel.get(k) ||
          k.replace(/[_-]/g, " ").replace(/\s+/g, " ").trim();
        set.add(label);
      });
    });
    return Array.from(set);
  }, [filteredManifests, manifestCategoryKeyToLabel]);

  const hasDownloaded = useMemo(() => {
    return filteredManifests.some((m) => !!m.downloaded);
  }, [filteredManifests]);

  // Keep a sensible active category when data arrives/changes.
  useEffect(() => {
    if (!selectedCategory && !activeCategory && categories.length > 0) {
      setActiveCategory(categories[0]);
    }
  }, [categories, selectedCategory, activeCategory]);

  const handleCategoryClick = (category: string) => {
    setActiveCategory(category);
    if (category === DOWNLOADED_CATEGORY) {
      setSelectedCategory(category);
      return;
    }
    const section = categorySectionRefs.current[category];
    const viewport = viewportRef.current;
    if (section && viewport) {
      const containerTop = viewport.getBoundingClientRect().top;
      const sectionTop = section.getBoundingClientRect().top;
      const scrollOffset = sectionTop - containerTop + viewport.scrollTop;
      viewport.scrollTo({ top: scrollOffset, behavior: "smooth" });
    } else if (section) {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  useLayoutEffect(() => {
    const root = scrollRef.current;
    if (!root) return;

    let observed: Element | null = null;
    let raf = 0;

    const updateWidth = () => {
      const viewport = root.querySelector(
        "[data-radix-scroll-area-viewport]",
      ) as HTMLDivElement | null;
      viewportRef.current = viewport;
      const target = viewport ?? root;
      const newWidth = target.clientWidth;
      if (newWidth > 0) setScrollWidth(newWidth);

      if (observed !== target) {
        try {
          if (observed) ro.unobserve(observed);
        } catch {}
        observed = target;
        ro.observe(target);
      }
    };

    const ro = new ResizeObserver(() => updateWidth());

    updateWidth();
    raf = requestAnimationFrame(updateWidth);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [selectedCategory, selectedManifestId]);

  // Remember & restore scroll position for the overview list.
  useLayoutEffect(() => {
    if (selectedCategory || selectedManifestId) return;
    const root = scrollRef.current;
    if (!root) return;
    const viewport = root.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLDivElement | null;
    if (!viewport) return;
    viewportRef.current = viewport;

    const key = "modelMenu:overview";

    const saved = scrollCacheRef.current.get(key);
    if (typeof saved === "number") {
      viewport.scrollTop = saved;
    }

    const onScroll = () => {
      scrollCacheRef.current.set(key, viewport.scrollTop);
    };
    viewport.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      viewport.removeEventListener("scroll", onScroll as EventListener);
    };
  }, [selectedCategory, selectedManifestId]);

  // If Downloaded is selected but none exist (e.g., after search), exit that view
  useEffect(() => {
    if (selectedCategory === DOWNLOADED_CATEGORY && !hasDownloaded) {
      setSelectedCategory(null);
      if (activeCategory === DOWNLOADED_CATEGORY) {
        setActiveCategory(categories[0] ?? null);
      }
    }
  }, [selectedCategory, hasDownloaded, activeCategory, categories]);

  // Sync active category to manual scroll position
  useEffect(() => {
    if (selectedCategory) return; // Only in overview mode
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

    viewport.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);
    handleScroll();

    return () => {
      viewport.removeEventListener("scroll", handleScroll as EventListener);
      window.removeEventListener("resize", handleScroll);
      cancelAnimationFrame(rafId);
    };
  }, [categories, selectedCategory, activeCategory]);

  if (selectedManifestId) {
    return (
      <ModelPage
        manifestId={selectedManifestId}
        scrollCache={scrollCacheRef.current}
        scrollKey={`modelMenu:model:${selectedManifestId}`}
      />
    );
  }

  if (selectedCategory) {
    if (selectedCategory === DOWNLOADED_CATEGORY) {
      return (
        <>
          <style>{`
            .carousel-container::-webkit-scrollbar { display: none; }
          `}</style>
          <CategoryDetailView
            category={DOWNLOADED_CATEGORY}
            manifests={filteredManifests.filter((m) => m.downloaded)}
            onBack={() => setSelectedCategory(null)}
            scrollCache={scrollCacheRef.current}
          />
        </>
      );
    } else {
      return (
        <>
          <style>{`
            .carousel-container::-webkit-scrollbar { display: none; }
          `}</style>
          <CategoryDetailView
            category={selectedCategory}
            manifests={filteredManifests.filter((m) => {
              const keys: string[] = m.metadata?.categories || [];
              const labels = keys.map(
                (k) =>
                  manifestCategoryKeyToLabel.get(k) ||
                  k.replace(/[_-]/g, " ").replace(/\s+/g, " ").trim(),
              );
              return labels.includes(selectedCategory);
            })}
            onBack={() => setSelectedCategory(null)}
            scrollCache={scrollCacheRef.current}
          />
        </>
      );
    }
  }

  const hasAnyManifests = manifests.length > 0;
  const hasAnyFiltered = filteredManifests.length > 0;
  const showEmptyState = manifestsQuery.isFetched && !hasAnyFiltered;

  return (
    <>
      <style>{`
        .carousel-container::-webkit-scrollbar { display: none; }
      `}</style>
      <div className="flex flex-col h-full w-full border-t border-brand-light/5 mt-2">
        <div className="flex flex-1 min-h-0 w-full">
          <CategorySidebar
            categories={categories}
            activeCategory={activeCategory}
            onCategoryClick={handleCategoryClick}
            title="MODELS"
            persistenceKey="sidebar:model"
            downloadedItem={
              hasDownloaded
                ? {
                    key: DOWNLOADED_CATEGORY,
                    label: "Downloaded",
                    icon: <TbWorldDownload className="w-3 h-3" />,
                  }
                : undefined
            }
          />
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="w-full p-3 shrink-0">
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
            {backendUnavailable ? (
              <div className="px-3 pb-2">
                <div className="bg-brand border border-brand-light/10 rounded-lg shadow-md p-3 flex items-center gap-x-2 text-[11px] text-brand-light/80">
                  <LuInfo className="w-4 h-4 text-brand-light/80" />
                  <div className="text-brand-light/80">Backend unavailable — start/connect the server to load models.</div>
                </div>
              </div>
            ) : null}
            <ScrollArea className="flex-1" ref={scrollRef}>
              {showEmptyState ? (
                <div className="flex flex-col items-center justify-center h-full w-full px-3">
                  <div className="bg-brand border border-brand-light/10 rounded-lg shadow-md p-4 w-full ">
                    <div className="flex items-center justify-between gap-x-3">
                      <div className="flex flex-col text-start">
                        <div className="text-brand-light text-[13px] font-semibold leading-tight">
                          No Models Found
                        </div>
                        <div className="text-brand-light/70 text-[11px] leading-snug">
                          {hasAnyManifests
                            ? "Try adjusting your search, or clear filters."
                            : "Connect a backend or refresh to load available models."}
                        </div>
                      </div>
                      {!hasAnyManifests ? (
                        <button
                          type="button"
                          title="Refresh models"
                          aria-label="Refresh models"
                          disabled={manifestsQuery.isFetching}
                          onClick={() => manifestsQuery.refetch()}
                          className="text-[11px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
                        >
                          <LuRefreshCw
                            className={`w-3.5 h-3.5 ${manifestsQuery.isFetching ? "animate-spin" : ""}`}
                          />
                          <span>Refresh</span>
                        </button>
                      ) : null}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-y-5 pt-1 pb-28">
                  {categories.map((category) => (
                    <div
                      key={category}
                      ref={(el) => {
                        categorySectionRefs.current[category] = el;
                      }}
                      className="w-full"
                    >
                      <ModelCategory
                        width={Math.max(0, scrollWidth - 36)}
                        category={category}
                        manifests={filteredManifests.filter((m) => {
                          const keys: string[] = m.metadata?.categories || [];
                          const labels = keys.map(
                            (k) =>
                              manifestCategoryKeyToLabel.get(k) ||
                              k
                                .replace(/[_-]/g, " ")
                                .replace(/\s+/g, " ")
                                .trim(),
                          );
                          return labels.includes(category);
                        })}
                        onViewAll={() => setSelectedCategory(category)}
                      />
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>
        </div>
      </div>
    </>
  );
};

export default ModelMenu;
