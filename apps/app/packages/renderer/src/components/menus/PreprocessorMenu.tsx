import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Preprocessor } from "@/lib/preprocessor/api";
import Draggable from "../dnd/Draggable";
import { ScrollArea } from "../ui/scroll-area";
import {
  LuInfo,
  LuChevronLeft,
  LuChevronRight,
  LuArrowRight,
  LuSearch,
  LuDownload,
  LuImage,
  LuVideo,
  LuLoader,
  LuPlus,
} from "react-icons/lu";
import { TbWorldDownload } from "react-icons/tb";
import { cn } from "@/lib/utils";
import { ensureExternalAssetUrl } from "@/lib/externalAssets";
import {
  PREPROCESSOR_QUERY_KEY,
  PREPROCESSORS_LIST_QUERY_KEY,
  usePreprocessorQuery,
  usePreprocessorsListQuery,
} from "@/lib/preprocessor/queries";
import PreprocessorPage from "../preprocessors/PreprocessorPage";
import CategorySidebar from "./CategorySidebar";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";
import { useStartUnifiedDownloadMutation } from "@/lib/download/mutations";
import { fetchRayJobs, type RayJobStatus } from "@/lib/jobs/api";

export const PreprocessorItem: React.FC<{
  preprocessor: Preprocessor;
  isDragging?: boolean;
  onMoreInfo?: (id: string) => void;
  onAdd?: (preprocessor: Preprocessor) => void;
  addDisabled?: boolean;
  dimmed?: boolean;
  polledJobs?: RayJobStatus[];
}> = ({
  preprocessor:initialPreprocessor,
  isDragging,
  onMoreInfo,
  onAdd,
  addDisabled,
  dimmed,
  polledJobs,
}) => {
  const queryClient = useQueryClient();
  const {data:preprocessorData} = usePreprocessorQuery(initialPreprocessor.id);
  const preprocessor = preprocessorData ?? initialPreprocessor;
  const [isDownloaded, setIsDownloaded] = useState<boolean>(
    !!preprocessor.is_downloaded,
  );
  const [starting, setStarting] = useState(false);
  const [localJobId, setLocalJobId] = useState<string | null>(null);
  const {
    addSourceToJobId,
    getSourceToJobId,
    getJobUpdates,
    removeJobUpdates,
    removeSourceToJobId,
    removeSourceByJobId,
  } =
    useDownloadJobIdStore();


  const mappedJobId = getSourceToJobId(preprocessor.id) || null;
  const jobId = mappedJobId || localJobId;
  const jobUpdates = getJobUpdates(jobId ?? undefined) ?? [];

  const defaultProcessorUrl = useMemo(
    () => `/preprocessors/${preprocessor.id}.png`,
    [preprocessor.id],
  );
  const [processorUrl, setProcessorUrl] = useState<string>(defaultProcessorUrl);
  const triedProcessorAssetFallbackRef = useRef(false);

  useEffect(() => {
    triedProcessorAssetFallbackRef.current = false;
    setProcessorUrl(defaultProcessorUrl);
  }, [defaultProcessorUrl]);

  const ensureProcessorImageFallback = async () => {
    if (triedProcessorAssetFallbackRef.current) return;
    triedProcessorAssetFallbackRef.current = true;
    const url = await ensureExternalAssetUrl({
      folder: "preprocessors",
      filePath: `${preprocessor.id}.png`,
    });
    if (url) setProcessorUrl(url);
  };

  const { mutateAsync: startDownload } = useStartUnifiedDownloadMutation({
    onSuccess(data, variables) {
      addSourceToJobId(variables.source, data.job_id);
      setLocalJobId(data.job_id);
    },
  });

  useEffect(() => {
    setIsDownloaded(!!preprocessor.is_downloaded);
  }, [preprocessor.is_downloaded]);

  const handleComplete = useCallback(async () => {
    setStarting(false);
    setIsDownloaded(true);
    try {
      // Update caches optimistically so the UI reflects the new state immediately,
      // even when `usePreprocessorQuery` is configured to be cache-only.
      queryClient.setQueryData(
        PREPROCESSOR_QUERY_KEY(preprocessor.id),
        (prev: Preprocessor | undefined) => {
          if (!prev) return prev;
          return { ...prev, is_downloaded: true };
        },
      );
      queryClient.setQueryData(
        PREPROCESSORS_LIST_QUERY_KEY,
        (prev: Preprocessor[] | undefined) => {
          if (!Array.isArray(prev)) return prev;
          return prev.map((p) =>
            p?.id === preprocessor.id ? { ...p, is_downloaded: true } : p,
          );
        },
      );
    } catch {}
  }, [queryClient, preprocessor.id]);

  const terminalStatuses = useMemo(
    () => new Set(["complete", "completed", "cancelled", "canceled", "error", "failed"]),
    [],
  );
  const polledJob = useMemo(() => {
    if (!jobId) return null;
    return (polledJobs || []).find((j) => j.job_id === jobId) ?? null;
  }, [jobId, polledJobs]);
  const polledStatus = (polledJob?.status || "").toLowerCase();
  const isJobActive =
    !!jobId &&
    !!polledJob &&
    polledJob.category === "download" &&
    !terminalStatuses.has(polledStatus);
  const isDownloading = starting || (jobUpdates?.length ?? 0) > 0 || isJobActive;

  useEffect(() => {
    if (!jobId) return;

    const latestWsStatus =
      typeof jobUpdates?.[jobUpdates.length - 1]?.status === "string"
        ? jobUpdates[jobUpdates.length - 1].status.toLowerCase()
        : "";

    const terminalFromPolling = !!polledJob && terminalStatuses.has(polledStatus);
    const terminalFromWs = !!latestWsStatus && terminalStatuses.has(latestWsStatus);

    // If the job disappears from polling quickly, fall back to WS terminal updates.
    if (!terminalFromPolling && !terminalFromWs) return;

    // Finalize: update UI + clear download tracking so we don't get stuck in "downloading".
    try {
      removeJobUpdates(jobId);
    } catch {}
    try {
      // Best-effort: clear both the explicit source mapping and any mapping by job id.
      removeSourceToJobId(preprocessor.id);
    } catch {}
    try {
      removeSourceByJobId(jobId);
    } catch {}
    setLocalJobId(null);
    void handleComplete();
  }, [
    jobId,
    polledJob,
    polledStatus,
    terminalStatuses,
    handleComplete,
    jobUpdates,
    removeJobUpdates,
    removeSourceToJobId,
    removeSourceByJobId,
    preprocessor.id,
  ]);

  // If the preprocessor flips to downloaded via cache updates, ensure we clear
  // any lingering download tracking so the loader/button don't get stuck.
  useEffect(() => {
    if (!jobId) return;
    if (!preprocessor.is_downloaded) return;
    try {
      removeJobUpdates(jobId);
    } catch {}
    try {
      removeSourceToJobId(preprocessor.id);
    } catch {}
    try {
      removeSourceByJobId(jobId);
    } catch {}
    setLocalJobId(null);
    setStarting(false);
    setIsDownloaded(true);
  }, [
    jobId,
    preprocessor.is_downloaded,
    preprocessor.id,
    removeJobUpdates,
    removeSourceToJobId,
    removeSourceByJobId,
  ]);

  const formatSize = (bytes: number): string | null => {
    if (bytes === 0) {
      return null;
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

  const handleDownload = async () => {
    if (starting || isDownloading || isDownloaded) return;
    setStarting(true);
    try {
      const res = await startDownload({
        item_type: "preprocessor",
        source: preprocessor.id,
      });
      setLocalJobId(res.job_id);
    } catch {
      setStarting(false);
    }
  };

  const totalDownloadSize = useMemo(() => {
    const bytes =
      preprocessor.files?.reduce((acc, file) => acc + file.size_bytes, 0) ?? 0;
    return formatSize(bytes);
  }, [preprocessor.files]);

  return (
    <div
      className={cn(
        "flex flex-col transition-all duration-200 rounded-md border shadow border-brand-light/10",
        {
          "w-44": !isDragging,
          "w-32": isDragging,
        },
        dimmed ? "opacity-100" : "",
        addDisabled ? "cursor-default" : "cursor-pointer",
      )}
    >
      {isDragging ? (
        <div className="flex flex-col">
          <div
            className={cn("flex items-center gap-x-1 relative h-full w-full", {
              "opacity-50": addDisabled,
            })}
          >
            <div
              className={cn(
                "absolute top-0 left-0 h-full w-full rounded-t-md",
                {
                  "backdrop-blur-sm": addDisabled,
                },
              )}
            />
            <img
              src={processorUrl}
              alt={preprocessor.name}
              className={cn(" h-48 aspect-square object-cover rounded-t-md", {
                "h-48": !isDragging,
                "h-44": isDragging,
              })}
              onError={() => {
                void ensureProcessorImageFallback();
              }}
            />
          </div>
          <div className="w-full bg-brand p-2 rounded-b-md">
            <div className="w-full flex flex-col gap-y-1 ">
              <div className="w-full flex flex-col gap-y-1">
                <div className="truncate leading-tight font-medium text-brand-light text-[10px] text-start">
                  {preprocessor.name}
                </div>
                <div className="w-full flex items-center justify-between">
                  {totalDownloadSize && (
                    <span className="text-brand-light/50 text-[9px] w-full text-start">
                      {totalDownloadSize}
                    </span>
                  )}
                  <div
                    className={cn(
                      "w-full flex items-center gap-x-1 text-brand-light/50",
                      {
                        "justify-end": totalDownloadSize,
                        "justify-start": !totalDownloadSize,
                      },
                    )}
                  >
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
      ) : (
        <Draggable
          data={{
            ...preprocessor,
            type: "preprocessor",
            processor_url: processorUrl,
          }}
          id={preprocessor.id}
          disabled={addDisabled || !isDownloaded}
        >
          <div className="flex flex-col">
            <div
              className={cn(
                "flex items-center gap-x-1 relative h-full w-full",
                {
                  "opacity-50": addDisabled,
                },
              )}
            >
              <div
                className={cn(
                  "absolute top-0 left-0 h-full w-full rounded-t-md",
                  {
                    "backdrop-blur-sm": addDisabled,
                  },
                )}
              />
              <img
                src={processorUrl}
                alt={preprocessor.name}
                className={cn(" h-48 aspect-square object-cover rounded-t-md", {
                  "h-48": !isDragging,
                  "h-44": isDragging,
                })}
                onError={() => {
                  void ensureProcessorImageFallback();
                }}
              />
            </div>
            <div className="w-full bg-brand p-2 rounded-b-md">
              <div className="w-full flex flex-col gap-y-1 ">
                <div className="w-full flex flex-col gap-y-1">
                  <div className="truncate leading-tight font-medium text-brand-light text-[10px] text-start">
                    {preprocessor.name}
                  </div>
                  <div className="w-full flex items-center justify-between">
                    {totalDownloadSize && (
                      <span className="text-brand-light/50 text-[9px] w-full text-start">
                        {totalDownloadSize}
                      </span>
                    )}
                    <div
                      className={cn(
                        "w-full flex items-center gap-x-1 text-brand-light/50",
                        {
                          "justify-end": totalDownloadSize,
                          "justify-start": !totalDownloadSize,
                        },
                      )}
                    >
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
      )}
      {!isDragging && (
        <div className="bg-brand p-2 gap-y-1 flex flex-col">
          <div className="flex items-center gap-x-1 w-full pt-0 bg-brand">
            <button
              onClick={() => onMoreInfo?.(preprocessor.id)}
              type="button"
              className="text-[10px] font-medium whitespace-nowrap w-3/8 flex items-center transition-all duration-200 justify-center gap-x-1.5 text-brand-light flex-1 hover:text-brand-light/80 bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded px-2 py-1"
              title="Show more info"
            >
              <LuInfo className="min-w-3 min-h-3 inline-block" />
              <span>Info</span>
            </button>
            {!!onAdd && isDownloaded && (
              <button
                type="button"
                onClick={() => !addDisabled && onAdd?.(preprocessor)}
                disabled={addDisabled}
                className={cn(
                  "text-[10px] font-medium w-1/2 flex items-center transition-all duration-200 justify-center gap-x-1.5 text-brand-light bg-brand-background border disabled:cursor-default! disabled:opacity-50 border-brand-light/10 rounded px-2 py-1",
                )}
                title="Add to clip"
              >
                <LuPlus className="w-2.5 h-2.5" />
                <span>Add</span>
              </button>
            )}
            {isDownloaded ? null : (
              <button
                type="button"
                onClick={handleDownload}
                disabled={starting || isDownloading}
                className={cn(
                  "inline-flex items-center justify-center gap-x-1 text-[10px] font-medium w-5/8 bg-brand-background border border-brand-light/10 rounded py-1 px-1 h-[25px] min-w-7",
                  {
                    "text-brand-light/60 cursor-default opacity-70":
                      isDownloading || starting,
                    "text-brand-light hover:text-brand-light hover:bg-brand-background/70":
                      !isDownloading && !starting,
                  },
                )}
                title={starting ? "Starting…" : "Download"}
              >
                {isDownloading || starting ? (
                  <LuLoader className="min-w-2.5! min-h-2.5! inline-block animate-spin" />
                ) : (
                  <LuDownload className="min-w-2.5! min-h-2.5! inline-block" />
                )}
                {isDownloading || starting ? "Download" : "Download"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const PreprocessorCategory: React.FC<{
  category: string;
  preprocessors: Preprocessor[];
  width: number;
  onViewAll: () => void;
  onMoreInfo: (id: string) => void;
  polledJobs: RayJobStatus[];
}> = ({ category, preprocessors, width, onViewAll, onMoreInfo, polledJobs }) => {
  const carouselRef = useRef<HTMLDivElement>(null);
  const [showLeftArrow, setShowLeftArrow] = useState(false);
  const [showRightArrow, setShowRightArrow] = useState(false);
  void width; // width is no longer used; maintain param to avoid refactor

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

  useEffect(() => {
    // Multiple checks to ensure we catch the content after it's rendered
    const timeouts = [
      setTimeout(checkScroll, 0),
      setTimeout(checkScroll, 100),
      setTimeout(checkScroll, 300),
      setTimeout(checkScroll, 500),
    ];

    const carousel = carouselRef.current;
    if (carousel) {
      carousel.addEventListener("scroll", checkScroll);
      window.addEventListener("resize", checkScroll);
      return () => {
        timeouts.forEach(clearTimeout);
        carousel.removeEventListener("scroll", checkScroll);
        window.removeEventListener("resize", checkScroll);
      };
    }
    return () => timeouts.forEach(clearTimeout);
  }, [preprocessors]);

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
            className="absolute -left-3 top-1/2 cursor-pointer -translate-y-1/2 z-9999 bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20"
          >
            <LuChevronLeft className="w-4 h-4 text-brand-light" />
          </button>
        )}
        {showRightArrow && (
          <button
            onClick={() => scroll("right")}
            className="absolute -right-3 top-1/2 cursor-pointer -translate-y-1/2 z-9999 bg-brand hover:bg-brand/80 rounded-full p-1.5 transition-colors shadow-lg border border-brand-light/20"
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
          {preprocessors.map((preprocessor) => (
            <div key={preprocessor.name} className="shrink-0">
              <PreprocessorItem
                preprocessor={preprocessor}
                onMoreInfo={onMoreInfo}
                polledJobs={polledJobs}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const CategoryDetailView: React.FC<{
  category: string;
  preprocessors: Preprocessor[];
  onBack: () => void;
  onMoreInfo: (id: string) => void;
  polledJobs: RayJobStatus[];
}> = ({ category, preprocessors, onBack, onMoreInfo, polledJobs }) => {
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
      <ScrollArea className="flex-1 pb-16">
        <div className="px-7 pt-6">
          <div
            className="grid gap-x-2 gap-y-3"
            style={{
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            }}
          >
            {preprocessors.map((preprocessor) => (
              <div key={preprocessor.name} className="flex justify-center">
                <PreprocessorItem
                  preprocessor={preprocessor}
                  onMoreInfo={onMoreInfo}
                  polledJobs={polledJobs}
                />
              </div>
            ))}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};

const PreprocessorMenu: React.FC = () => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { data: preprocessors = [] } = usePreprocessorsListQuery();
  const { data: polledJobs = [] } = useQuery<RayJobStatus[]>({
    queryKey: ["rayJobs"],
    queryFn: fetchRayJobs,
    placeholderData: (prev) => prev ?? [],
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: 4000,
    refetchIntervalInBackground: true,
  });
  const [selectedPreprocessorId, setSelectedPreprocessorId] = useState<
    string | null
  >(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [scrollWidth, setScrollWidth] = useState(0);
  const categorySectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const DOWNLOADED_CATEGORY = "Downloaded";
  const handleCategoryClick = (category: string) => {
    setActiveCategory(category);
    if (category === DOWNLOADED_CATEGORY) {
      setSelectedCategory(category);
      return;
    }
    const section = categorySectionRefs.current[category];
    const viewport = viewportRef.current;

    if (section && viewport) {
      // Calculate offset from the top of the scrollable container
      const containerTop = viewport.getBoundingClientRect().top;
      const sectionTop = section.getBoundingClientRect().top;
      const scrollOffset = sectionTop - containerTop + viewport.scrollTop;

      viewport.scrollTo({ top: scrollOffset, behavior: "smooth" });
    } else if (section) {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const filteredPreprocessors = useMemo(() => {
    const list = preprocessors ?? [];
    if (!searchQuery.trim()) return list;
    const query = searchQuery.toLowerCase();
    return list.filter(
      (preprocessor) =>
        preprocessor.name.toLowerCase().includes(query) ||
        preprocessor.description?.toLowerCase().includes(query) ||
        preprocessor.category.toLowerCase().includes(query),
    );
  }, [preprocessors, searchQuery]);

  const hasDownloaded = useMemo(() => {
    return filteredPreprocessors.some((p) => !!p.is_downloaded);
  }, [filteredPreprocessors]);

  const categories = useMemo(() => {
    setActiveCategory(preprocessors?.[0]?.category || null);
    return [
      ...new Set(
        filteredPreprocessors.map((preprocessor) => preprocessor.category),
      ),
    ];
  }, [filteredPreprocessors]);
  // `usePreprocessorsListQuery()` handles fetching/caching.

  useEffect(() => {
    const updateWidth = () => {
      if (scrollRef.current) {
        const viewport = scrollRef.current.querySelector(
          "[data-radix-scroll-area-viewport]",
        ) as HTMLDivElement | null;
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
      setTimeout(updateWidth, 200),
    ];

    // Use ResizeObserver for more reliable resize tracking
    const resizeObserver = new ResizeObserver(updateWidth);
    if (scrollRef.current) {
      resizeObserver.observe(scrollRef.current);
    }

    // Also listen to window resize as fallback
    window.addEventListener("resize", updateWidth);

    return () => {
      timeouts.forEach(clearTimeout);
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateWidth);
    };
  }, [selectedCategory, selectedPreprocessorId, setSelectedPreprocessorId]);

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

    viewport.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);
    handleScroll();

    return () => {
      viewport.removeEventListener("scroll", handleScroll as EventListener);
      window.removeEventListener("resize", handleScroll);
      cancelAnimationFrame(rafId);
    };
  }, [categories, selectedCategory, selectedPreprocessorId, activeCategory]);

  // If Downloaded is selected but none exist (e.g., after search), exit that view
  useEffect(() => {
    if (selectedCategory === DOWNLOADED_CATEGORY && !hasDownloaded) {
      setSelectedCategory(null);
      if (activeCategory === DOWNLOADED_CATEGORY) {
        setActiveCategory(categories[0] ?? null);
      }
    }
  }, [selectedCategory, hasDownloaded, activeCategory, categories]);

  if (selectedPreprocessorId) {
    return (
      <PreprocessorPage
        preprocessorId={selectedPreprocessorId}
        onBack={() => setSelectedPreprocessorId(null)}
      />
    );
  }

  if (selectedCategory) {
    if (selectedCategory === DOWNLOADED_CATEGORY) {
      return (
        <>
          <style>{`
                        .carousel-container::-webkit-scrollbar {
                            display: none;
                        }
                    `}</style>
          <CategoryDetailView
            category={DOWNLOADED_CATEGORY}
            preprocessors={filteredPreprocessors.filter(
              (p) => !!p.is_downloaded,
            )}
            onBack={() => setSelectedCategory(null)}
            onMoreInfo={(id) => setSelectedPreprocessorId(id)}
            polledJobs={polledJobs}
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
        <CategoryDetailView
          category={selectedCategory}
          preprocessors={filteredPreprocessors.filter(
            (p) => p.category === selectedCategory,
          )}
          onBack={() => setSelectedCategory(null)}
          onMoreInfo={(id) => setSelectedPreprocessorId(id)}
          polledJobs={polledJobs}
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
          <CategorySidebar
            categories={categories}
            activeCategory={activeCategory}
            onCategoryClick={handleCategoryClick}
            title="PREPROCESSORS"
            persistenceKey="sidebar:preprocessor"
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
                    ref={(el) => {
                      categorySectionRefs.current[category] = el;
                    }}
                    className="w-full"
                  >
                    <PreprocessorCategory
                      width={scrollWidth - 32}
                      category={category}
                      preprocessors={filteredPreprocessors.filter(
                        (preprocessor) => preprocessor.category === category,
                      )}
                      onViewAll={() => setSelectedCategory(category)}
                      onMoreInfo={(id) => setSelectedPreprocessorId(id)}
                      polledJobs={polledJobs}
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
};

export default PreprocessorMenu;
