import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  useManifestTypes,
  useManifests,
  type ManifestDocument,
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
} from "react-icons/lu";
import { TbWorldDownload } from "react-icons/tb";
import Draggable from "../dnd/Draggable";
import { useManifestStore } from "@/lib/manifest/store";
import { useDownloadStore } from "@/lib/download/store";
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

export const ModelItem: React.FC<{
  manifest: ManifestDocument;
  isDragging?: boolean;
  category?: string;
}> = ({ manifest, isDragging, category }) => {
  const { setSelectedManifestId, refreshManifestPart } = useManifestStore();
  const tagsContainerRef = useRef<HTMLDivElement>(null);
  const hiddenMeasureRef = useRef<HTMLDivElement>(null);
  const [visibleTagCount, setVisibleTagCount] = useState<number | null>(null);
  const [isStartingDownload, setIsStartingDownload] = useState(false);
  const {
    startAndTrackDownload,
    wsFilesByPath,
    downloadingPaths,
    resolveDownloadBatch,
    subscribeToJob,
  } = useDownloadStore();
  const onCompleteDownload = useCallback(
    async (paths: string | string[]) => {
      // dispatch a custom event to refresh the model menu
      window.dispatchEvent(
        new CustomEvent("component-card-reload", {
          detail: {
            paths: Array.isArray(paths) ? paths : [paths],
            manifestId: manifest.metadata?.id,
          },
        }),
      );
      await refreshManifestPart(manifest.metadata?.id || "", `spec.components`);
    },
    [manifest.metadata?.id],
  );

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

  const getAllComponentPaths = useMemo(() => {
    const paths:string[] = [];
    const components = manifest?.spec?.components || [];
    for (const comp of components) {
      const modelPaths = Array.isArray(comp.model_path)
        ? comp.model_path
        : comp.model_path
          ? [{ path: comp.model_path }]
          : [];
      // add config_paths too
      if (comp?.config_path) {
        paths.push(comp.config_path as string);
      }
      if (
        comp?.type === "scheduler" &&
        Array.isArray(comp?.scheduler_options)
      ) {
        for (const opt of comp.scheduler_options as any[]) {
          if (opt?.config_path) {
            paths.push(opt.config_path as string);
          }
        }
      }
      // add extra_model_paths too
      if (Array.isArray(comp?.extra_model_paths)) {
        for (const p of comp.extra_model_paths) {
          if (typeof p === "string") {
            paths.push(p);
          } else {
            paths.push(p.path);
          }
        }
      }
      for (const modelPath of modelPaths) {
        if (typeof modelPath === "string") {
          paths.push(modelPath);
        } else {
          paths.push(modelPath.path);
        }
      }
    }
    return paths;
  }, []);

  const getAllLoraPaths = useMemo(() => {
    const paths = [];
    const loras = manifest?.spec?.loras || [];
    for (const lora of loras) {
      if (typeof lora === "string") {
        paths.push(lora);
      } else {
        if (lora.source) {
          paths.push(lora.source);
        }
      }
    }
    return paths;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      try {
        const response = await resolveDownloadBatch({
          item_type: "component",
          sources: getAllComponentPaths,
        });

        if (response?.results?.some((r) => r.running) && !cancelled) {
          for (const r of response.results) {
            if (r.running) {
              await subscribeToJob(
                r.job_id || "",
                r.source as string,
                onCompleteDownload,
              );
            }
          }
        }
      } catch {}

      try {
        const response = await resolveDownloadBatch({
          item_type: "lora",
          sources: getAllLoraPaths,
        });
        if (response?.results?.some((r) => r.running) && !cancelled) {
          for (const r of response.results) {
            if (r.running) {
              await subscribeToJob(
                r.job_id || "",
                r.source as string,
                onCompleteDownload,
              );
            }
          }
        }
      } catch {}
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [manifest.metadata?.id, getAllComponentPaths, getAllLoraPaths]);

  const isDownloading = useMemo(() => {
    return (
      getAllComponentPaths.some((p) => downloadingPaths.has(p)) ||
      getAllComponentPaths.some(
        (p) =>
          wsFilesByPath[p] &&
          Object.values(wsFilesByPath[p]).some(
            (v) => v.status === "processing" || v.status === "pending",
          ),
      ) ||
      getAllLoraPaths.some((p) => downloadingPaths.has(p)) ||
      getAllLoraPaths.some(
        (p) =>
          wsFilesByPath[p] &&
          Object.values(wsFilesByPath[p]).some(
            (v) => v.status === "processing" || v.status === "pending",
          ),
      ) ||
      Array.from(downloadingPaths).some((p) => getAllComponentPaths.includes(p))
    );
  }, [getAllComponentPaths, wsFilesByPath, downloadingPaths, getAllLoraPaths]);

  const allDownloaded = useMemo(() => {
    return manifest.downloaded && !isDownloading;
  }, [manifest.downloaded, isDownloading]);

  // Compute how many tags fit on a single line
  useEffect(() => {
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

  const handleDownloadAllDefault = async () => {
    try {
      setIsStartingDownload(true);

      const components = manifest?.spec?.components || [];
      for (const comp of components) {
        const modelPaths = Array.isArray(comp.model_path)
          ? comp.model_path
          : comp.model_path
            ? [{ path: comp.model_path }]
            : [];
        // Only default variants (treat string paths as default)
        const filtered = modelPaths.filter((item: any) => {
          if (typeof item === "string") return true;
          const v = (item?.variant ?? "").toLowerCase();
          return v === "" || v.toLowerCase() === "default";
        });
        for (const item of filtered) {
          const p = typeof item === "string" ? item : item?.path;
          await startAndTrackDownload(
            {
              item_type: "component",
              source: p,
            },
            onCompleteDownload,
          );
        }
        // Ensure scheduler configs (base and options) are downloaded too
        const configPathsSet = new Set<string>();
        if (comp?.config_path) {
          configPathsSet.add(comp.config_path as string);
        }
        if (
          comp?.type === "scheduler" &&
          Array.isArray(comp?.scheduler_options)
        ) {
          for (const opt of comp.scheduler_options as any[]) {
            const cp = opt?.config_path as string | undefined;
            if (cp) configPathsSet.add(cp);
          }
        }
        for (const cp of configPathsSet) {
          await startAndTrackDownload(
            {
              item_type: "component",
              source: cp,
              save_path: comp?.save_path,
            },
            onCompleteDownload,
          );
        }
      }
    } catch {}
    try {
      const loras = manifest?.spec?.loras || [];
      for (const lora of loras) {
        const source = typeof lora === "string" ? lora : lora.source;
        if (source) {
          await startAndTrackDownload(
            {
              item_type: "lora",
              source: source,
            },
            onCompleteDownload,
          );
        }
      }
    } catch {
    } finally {
      // Fallback clear in case store entries are delayed
      setTimeout(() => setIsStartingDownload(false), 1200);
    }
  };

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
          "w-48": true,
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
        <div className="flex items-center gap-x-1 w-full p-3 pt-0">
          <button
            onClick={() => {
              setSelectedManifestId(manifest.metadata?.id || "");
            }}
            type="button"
            className="text-[10px] font-medium flex items-center transition-all duration-200 justify-center gap-x-1.5 text-brand-light flex-1 hover:text-brand-light/80 bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded px-2 py-1"
            title="Show more info"
          >
            <LuInfo className="w-3 h-3" />
            <span>Info</span>
          </button>
          <button
            onClick={async () => {
              if (!allDownloaded) {
                await handleDownloadAllDefault();
                return;
              }
              try {
                const controls = useControlsStore.getState();
                const clipStore = useClipStore.getState();
                const fps = Math.max(1, controls.fps || 1);
                const focusFrame = Math.max(0, controls.focusFrame || 0);
                const desiredFrames = Math.max(
                  1,
                  (manifest as any)?.desired_duration ?? controls.defaultClipLength * fps,
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
                  height: 540,
                  width: 540,
                  speed: 1.0,
                  category,
                };
                const store = useManifestStore.getState();
                const existing = store.getLoadedManifest(
                  manifest.metadata?.id || "",
                );
                if (existing) {
                  clipBase.manifest = existing;
                  useClipStore.getState().addClip(clipBase);
                } else {
                  clipBase.manifest = manifest;
                  useClipStore.getState().addClip(clipBase);
                }
              } catch {}
            }}
            type="button"
            disabled={isStartingDownload || isDownloading}
            className={cn(
              "text-[10px] font-medium flex items-center transition-all duration-200 justify-center gap-x-1.5 rounded px-2 py-1 border flex-1 text-brand-light hover:text-brand-light/90 bg-brand-background hover:bg-brand-background/70 border-brand-light/10",
            )}
            title={
              allDownloaded
                ? "Add clip at playhead"
                : isStartingDownload || isDownloading
                  ? "Downloading"
                  : "Download default variant"
            }
          >
            {allDownloaded ? (
              <LuPlus className="w-3 h-3" />
            ) : isStartingDownload || isDownloading ? (
              <LuLoader className="w-3 h-3 animate-spin" />
            ) : (
              <LuDownload className="w-3 h-3" />
            )}
            <span>
              {allDownloaded
                ? "Add Clip"
                : isStartingDownload || isDownloading
                  ? "Downloading"
                  : "Download"}
            </span>
          </button>
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

  useEffect(() => {
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
  }, [manifests]);

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
          className="flex items-center gap-x-1.5 text-brand-light hover:text-brand-light/70 text-[12px] font-medium cursor-pointer transition-colors rounded-md flex-shrink-0"
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
            <div key={manifest.metadata?.id || ""} className="flex-shrink-0">
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
}> = ({ category, manifests, onBack }) => {
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
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
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
  const { data: manifestsData } = useManifests();
  const { data: modelTypesData } = useManifestTypes();
  const { selectedManifestId } = useManifestStore();

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

  useEffect(() => {
    const updateWidth = () => {
      if (scrollRef.current) {
        const viewport = scrollRef.current.querySelector(
          "[data-radix-scroll-area-viewport]",
        ) as HTMLDivElement | null;
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
    window.addEventListener("resize", updateWidth);
    return () => {
      timeouts.forEach(clearTimeout);
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateWidth);
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
    return <ModelPage manifestId={selectedManifestId} />;
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
          />
        </>
      );
    }
  }

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
                    <ModelCategory
                      width={scrollWidth - 36}
                      category={category}
                      manifests={filteredManifests.filter((m) => {
                        const keys: string[] = m.metadata?.categories || [];
                        const labels = keys.map(
                          (k) =>
                            manifestCategoryKeyToLabel.get(k) ||
                            k.replace(/[_-]/g, " ").replace(/\s+/g, " ").trim(),
                        );
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
};

export default ModelMenu;
