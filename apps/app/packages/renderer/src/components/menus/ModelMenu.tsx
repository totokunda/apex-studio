import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type ManifestDocument,
  type ManifestGroup,
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
  LuPlus,
  LuRefreshCw,
  LuSettings,
} from "react-icons/lu";
import { TbWorldDownload } from "react-icons/tb";
import Draggable from "../dnd/Draggable";
import { useManifestStore } from "@/lib/manifest/store";

import {
  ensureExternalAssetUrl,
  getLastPathSegment,
  inferExternalFolderFromPath,
} from "@/lib/externalAssets";

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
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchManifestsAndPrimeCache,
  fetchManifestGroups,
  fetchModelTypes,
  useManifestQuery,
} from "@/lib/manifest/queries";
import { resolveManifestVariantId } from "@/lib/manifest/variantStorageKey";

import { getBackendUrl, getOffloadDefaultsForManifest } from "@app/preload";

const MIN_MANIFEST_GROUPS_VERSION = "0.1.2";

const parseSemver = (value: unknown): [number, number, number] | null => {
  if (typeof value !== "string") return null;
  const match = value.trim().match(/^v?(\d+)\.(\d+)\.(\d+)$/i);
  if (!match) return null;
  return [
    Number.parseInt(match[1], 10),
    Number.parseInt(match[2], 10),
    Number.parseInt(match[3], 10),
  ];
};

const isSemverAtLeast = (version: string, minimum: string): boolean => {
  const lhs = parseSemver(version);
  const rhs = parseSemver(minimum);
  if (!lhs || !rhs) return false;
  if (lhs[0] !== rhs[0]) return lhs[0] > rhs[0];
  if (lhs[1] !== rhs[1]) return lhs[1] > rhs[1];
  return lhs[2] >= rhs[2];
};

const fetchSupportsManifestGroups = async (): Promise<boolean> => {
  try {
    const backendUrlRes = await getBackendUrl();
    const rawBackendUrl =
      backendUrlRes?.success && backendUrlRes?.data?.url
        ? String(backendUrlRes.data.url)
        : "";
    const backendUrl = rawBackendUrl.replace(/\/+$/, "");
    if (!backendUrl) return false;

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 3_000);
    try {
      const response = await fetch(`${backendUrl}/manifest/version`, {
        method: "GET",
        signal: controller.signal,
      });
      if (!response.ok) return false;
      const payload = (await response.json()) as
        | {
            version?: string;
            manifest_version?: string;
            manifest_api_version?: string;
            supports_groups?: boolean;
          }
        | undefined;

      if (typeof payload?.supports_groups === "boolean") {
        return payload.supports_groups;
      }

      const version =
        payload?.version ??
        payload?.manifest_version ??
        payload?.manifest_api_version;
      return (
        typeof version === "string" &&
        isSemverAtLeast(version, MIN_MANIFEST_GROUPS_VERSION)
      );
    } finally {
      window.clearTimeout(timeoutId);
    }
  } catch {
    return false;
  }
};

export const ModelItem: React.FC<{
  manifest: ManifestDocument;
  isDragging?: boolean;
  category?: string;
}> = ({ manifest:initialManifest, isDragging, category }) => {
  const { setSelectedManifestId } = useManifestStore();
  const tagsContainerRef = useRef<HTMLDivElement>(null);
  const hiddenMeasureRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [visibleTagCount, setVisibleTagCount] = useState<number | null>(null);
  const { data: manifestData } = useManifestQuery(initialManifest.metadata?.id || "");
  const manifest = manifestData ?? initialManifest;
  const ctrlToggleClipSelection = useControlsStore(
    (s) => s.toggleClipSelection,
  );

  // Keep group metadata stable even when manifestData refreshes and does not
  // carry the transient `_group` field.
  const group =
    ((initialManifest as any)?._group as ManifestGroup | undefined) ??
    ((manifestData as any)?._group as ManifestGroup | undefined);
  const displayName = group?.name ?? manifest.metadata?.name;
  const displayTags = group?.tags ?? manifest.metadata?.tags;
  const demoPath = group?.demo_path || manifest.metadata?.demo_path;

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

  const posterPath = useMemo(() => {
    if (!demoPath) return undefined;

    const replaceExt = (s: string) =>
      s.replace(/\.(mp4|webm|mov|m4v|ogg|m3u8)$/i, ".poster.jpg");

    try {
      const url = new URL(demoPath);
      url.pathname = replaceExt(url.pathname);
      return url.toString();
    } catch {
      const m = demoPath.match(/^([^?#]+)(\?[^#]*)?(#.*)?$/);
      const base = m?.[1] ?? demoPath;
      const search = m?.[2] ?? "";
      const hash = m?.[3] ?? "";
      return replaceExt(base) + search + hash;
    }
  }, [demoPath]);

  const [resolvedDemoPath, setResolvedDemoPath] = useState<string | undefined>(
    demoPath,
  );
  const [resolvedPosterPath, setResolvedPosterPath] = useState<
    string | undefined
  >(posterPath);
  const triedDemoFallbackRef = useRef(false);

  useEffect(() => {
    triedDemoFallbackRef.current = false;
    setResolvedDemoPath(demoPath);
    setResolvedPosterPath(posterPath);
  }, [demoPath, posterPath]);

  const ensureDemoFallback = async () => {
    if (triedDemoFallbackRef.current) return;
    triedDemoFallbackRef.current = true;
    if (!demoPath) return;

    const folder = inferExternalFolderFromPath(demoPath);
    const demoSeg = getLastPathSegment(demoPath);
    const posterSeg =
      isVideoDemo && posterPath ? getLastPathSegment(posterPath) : null;

    // For video previews, prioritize the poster (fast) so UI updates quickly,
    // while the (potentially large) video downloads in the background.
    const posterPromise = posterSeg
      ? ensureExternalAssetUrl({ folder, filePath: posterSeg })
      : Promise.resolve<string | null>(null);
    const demoPromise = demoSeg
      ? ensureExternalAssetUrl({ folder, filePath: demoSeg })
      : Promise.resolve<string | null>(null);

    try {
      const posterUrl = await posterPromise;
      if (posterUrl) setResolvedPosterPath(posterUrl);
    } catch {
      // ignore
    }

    try {
      const demoUrl = await demoPromise;
      if (demoUrl) setResolvedDemoPath(demoUrl);
    } catch {
      // ignore
    }
  };

  const playDemo = () => {
    const v = videoRef.current;
    if (!v) return;
    const p = v.play();
    if (p) p.catch(() => {});
  };

  const stopDemo = () => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    try {
      v.currentTime = 0;
    } catch {
      // ignore
    }
  };


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
  }, [displayTags]);


  const card = (
    <div className={cn("flex flex-col items-center relative w-full ", {
      "rounded-md": isDragging,
      "rounded-t-md": !isDragging,
    })}>
      <div
        className={cn(
          " overflow-hidden flex items-center justify-center aspect-square relative",
          {
            "rounded-md": isDragging,
            "rounded-t-md": !isDragging,
          },
        )}
      >
        {isVideoDemo ? (
          <div className="relative h-full w-full">
            {resolvedPosterPath ? (
              <img
                src={resolvedPosterPath}
                alt={displayName}
                className="w-48 h-48 object-cover rounded-t-md"
                onError={() => {
                  void ensureDemoFallback();
                }}
              />
            ) : null}
            <video
              key={`${resolvedDemoPath ?? ""}|${resolvedPosterPath ?? ""}`}
              ref={videoRef}
              src={resolvedDemoPath}
              poster={resolvedPosterPath}
              className="w-48 h-48 object-cover rounded-t-md"
              preload="none"
              muted
              loop
              playsInline
              onMouseEnter={playDemo}
              onMouseLeave={stopDemo}
              onError={() => {
                void ensureDemoFallback();
              }}
            />
          </div>
        ) : (
          <div className="relative w-full h-full">
            <img
              src={resolvedDemoPath}
              alt={displayName}
              className="w-48 h-48 object-cover rounded-t-md"
              onError={() => {
                void ensureDemoFallback();
              }}
            />
          </div>
        )}
        <div
          className="absolute inset-0 rounded-t-sm bg-linear-to-b from-black/80 via-black/30 to-transparent pointer-events-none"
          aria-hidden
        />
        <div className="absolute top-1 left-1 right-0 px-3 py-2.5 pointer-events-none min-w-0 text-start">
          <span className="block truncate text-[13px] font-semibold text-white drop-shadow-md">
            {displayName}
          </span>
        </div>
      </div>
    </div>
  );

  const details = (
    <div className="flex flex-col gap-y-1.5  px-3  w-full ">
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
        {displayTags?.map((tag: string) => (
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

  const tags = (
    <div
    ref={tagsContainerRef}
    className="flex items-center gap-x-1 w-full overflow-hidden absolute bottom-1.5 left-1.5 right-0"
  >
    {(visibleTagCount == null
      ? displayTags
      : displayTags?.slice(0, visibleTagCount)
    )?.map((tag: string) => (
      <span
        key={tag}
        className="text-[8px] text-brand-light backdrop-blur-sm  shadow  rounded-[4px] px-2 py-0.5 bg-brand/60"
      >
        {tag}
      </span>
    ))}
  </div>
  );

  const buttons = (
        <div className=" w-full flex flex-row items-center rounded-b-lg shadow-lg">
          
          <button
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
                if (group) {
                  clipBase.group = group;
                  clipBase.variantId = resolveManifestVariantId({
                    group,
                    manifest,
                  });
                }
                try {
                  const mfId = String(manifest?.metadata?.id || "").trim();
                  if (mfId) {
                    const defaults = await getOffloadDefaultsForManifest(mfId);
                    if (defaults) {
                      clipBase.offload = defaults;
                    }
                  }
                } catch {
                  // ignore; defaults are best-effort
                }
                useClipStore.getState().addClip(clipBase);
                ctrlToggleClipSelection(newClipId);
              } catch {}
            }}
            type="button"
            className={cn(
              "text-[10px] w-full font-semibold disabled:opacity-50 z-20 disabled:cursor-default! flex items-center transition-all duration-200 justify-center gap-x-1 rounded-bl-md px-2 py-2.5 border-0 text-brand-light hover:text-white backdrop-blur-sm bg-brand-background-light hover:bg-brand/90 border-r border-brand-light/5",
            )}
            title={
             "Add clip at playhead"
            }
          >
            <LuPlus className="w-3 h-3" />
            <span>
              Add Clip
            </span>
          </button>
          <button
            onClick={() => {
              setSelectedManifestId(manifest.metadata?.id || "");
            }}
            type="button"
            className="text-[10px] font-semibold w-full flex items-center transition-all duration-200 border-0 justify-center gap-x-1 text-brand-light hover:text-white backdrop-blur-sm bg-brand-background-light hover:bg-brand/90  border-brand-light/5 rounded-br-md px-2 py-2.5"
            title="Show more info"
          >
            <LuSettings className="w-3 h-3" />
            <span>Manage</span>
          </button>
        </div>
      );
  
  return (
    <div
      className={cn(
        "group flex flex-col transition-all font-inter duration-200 rounded-md relative shadow-md cursor-grab active:cursor-grabbing",
        {
          "w-48": true,
          "opacity-[0.975]": isDragging,
        },
      )}
    >
      {isDragging ? (
        <div className="flex flex-col items-center relative w-full rounded-md">
          {card}
          {tags}
          {details}
        </div>
      ) : (
        <Draggable
          id={stableId}
          data={{
            ...manifest,
            type: "model",
            category: category,
            _group: group,
          }}
        >
          {card}
          {tags}
          {details}
        </Draggable>
      )}
      {!isDragging && (
        <div className="w-full flex flex-col items-center  ">

          {buttons}
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

const ModelMenu: React.FC<{ panelSize?: number }> = ({ panelSize = 0 }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const scrollCacheRef = useRef<Map<string, number>>(new Map());
  const { selectedManifestId } = useManifestStore();
  const queryClient = useQueryClient();

  useEffect(() => {
    void queryClient.prefetchQuery({
      queryKey: ["manifestSupportsGroups"],
      queryFn: fetchSupportsManifestGroups,
      staleTime: 300_000,
    });
    void queryClient.prefetchQuery({
      queryKey: ["modelTypes"],
      queryFn: fetchModelTypes,
      staleTime: 30_000,
    });
  }, [queryClient]);
  // Keep showing the last successfully rendered data while queries refetch/error,
  // to avoid the menu ever flashing "nothing" after it has rendered once.
  const lastGoodManifestsRef = useRef<ManifestDocument[] | null>(null);
  const lastGoodModelTypesRef = useRef<ModelTypeInfo[] | null>(null);

  const manifestVersionGateQuery = useQuery<boolean>({
    queryKey: ["manifestSupportsGroups"],
    queryFn: fetchSupportsManifestGroups,
    placeholderData: (prev) => prev,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 300_000,
    gcTime: Infinity,
  });

  const versionResolved = manifestVersionGateQuery.isFetched;
  const useGroupedManifestEndpoint =
    versionResolved && manifestVersionGateQuery.data === true;
  const useLegacyManifestListEndpoint =
    versionResolved && manifestVersionGateQuery.data === false;

  // Group manifests are the primary source, but only when the version gate
  // confirms the backend supports group APIs.
  const groupsQuery = useQuery<ManifestGroup[]>({
    queryKey: ["manifestGroups"],
    queryFn: () => fetchManifestGroups(queryClient),
    initialData: () =>
      queryClient.getQueryData<ManifestGroup[]>(["manifestGroups"]),
    placeholderData: (prev) => prev,
    retry: true,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
    gcTime: Infinity,
    enabled: useGroupedManifestEndpoint,
  });

  // Legacy flat manifest list is only used when the version endpoint is
  // unavailable/too old.
  const manifestsQuery = useQuery<ManifestDocument[]>({
    queryKey: ["manifest"],
    queryFn: () => fetchManifestsAndPrimeCache(queryClient),
    initialData: () =>
      queryClient.getQueryData<ManifestDocument[]>(["manifest"]),
    placeholderData: (prev) => prev,
    retry: true,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
    gcTime: Infinity,
    enabled: useLegacyManifestListEndpoint,
  });

  const modelTypesQuery = useQuery<ModelTypeInfo[]>({
    queryKey: ["modelTypes"],
    queryFn: fetchModelTypes,
    initialData: () => queryClient.getQueryData<ModelTypeInfo[]>(["modelTypes"]),
    placeholderData: (prev) => prev,
    retry: true,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
    gcTime: Infinity,
  });

  let manifestsData: ManifestDocument[] | undefined;
  const groupsData = useGroupedManifestEndpoint ? groupsQuery.data : undefined;
  const modelTypesData = modelTypesQuery.data;

  useEffect(() => {
    if (manifestsData && manifestsData.length > 0) {
      lastGoodManifestsRef.current = manifestsData;
    }
  }, [manifestsData]);

  useEffect(() => {
    if (Array.isArray(modelTypesData) && modelTypesData.length > 0) {
      lastGoodModelTypesRef.current = modelTypesData;
    }
  }, [modelTypesData]);

  // Backend is unavailable when we are on the legacy path and the list call fails.
  const backendUnavailable =
    useLegacyManifestListEndpoint && manifestsQuery.isFetched && manifestsQuery.isError;

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [scrollWidth, setScrollWidth] = useState(0);
  const categorySectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const DOWNLOADED_CATEGORY = "Downloaded";
  const isSearching = searchQuery.trim().length > 0;

  const stableModelTypes: ModelTypeInfo[] = useMemo(() => {
    if (Array.isArray(modelTypesData) && modelTypesData.length > 0) return modelTypesData;
    if ((modelTypesQuery.isFetching || modelTypesQuery.isError) && lastGoodModelTypesRef.current) {
      return lastGoodModelTypesRef.current;
    }
    return modelTypesData ?? lastGoodModelTypesRef.current ?? [];
  }, [modelTypesData, modelTypesQuery.isFetching, modelTypesQuery.isError]);

  // Map from category key -> human-friendly label (backend /manifest/categories).
  const manifestCategoryKeyToLabel = useMemo(() => {
    const map = new Map<string, string>();
    stableModelTypes.forEach((t) => map.set(t.key, t.label));
    return map;
  }, [stableModelTypes]);

  // Groups are the only source when the version gate enables grouped manifests.
  // The flat manifest list is used only on the legacy path.
  const manifests: ManifestDocument[] = useMemo(() => {
    if (useGroupedManifestEndpoint) {
      const groupManifests: ManifestDocument[] = [];
      const seenIds = new Set<string>();

      for (const group of groupsData ?? []) {
        const variants = group.variants ?? [];
        const defaultVariant = variants.find((v) => v.default) ?? variants[0];
        if (!defaultVariant) continue;

        let manifest = defaultVariant.manifest as ManifestDocument | null | undefined;

        if (manifest) {
          const id = manifest.metadata?.id;
          if (id && !seenIds.has(id)) {
            seenIds.add(id);
            // Carry the parent group so downstream components can read variants
            (manifest as any)._group = group;
            groupManifests.push(manifest);
          }
        }
      }

      return groupManifests;
    }

    if (useLegacyManifestListEndpoint) {
      if (Array.isArray(manifestsData) && manifestsData.length > 0) return manifestsData;
      if (
        (manifestsQuery.isFetching || manifestsQuery.isError) &&
        lastGoodManifestsRef.current
      ) {
        return lastGoodManifestsRef.current;
      }
      return manifestsData ?? lastGoodManifestsRef.current ?? [];
    }

    return [];
  }, [
    useGroupedManifestEndpoint,
    useLegacyManifestListEndpoint,
    groupsData,
    manifestsData,
    manifestsQuery.isFetching,
    manifestsQuery.isError,
  ]);

  // Category keys should reflect both the group definition (authoritative for
  // grouped model families) and the manifest itself (fallback/compatibility).
  const getManifestCategoryKeys = (manifest: ManifestDocument): string[] => {
    const groupCategories =
      ((manifest as any)?._group as ManifestGroup | undefined)?.metadata
        ?.categories ?? [];
    const manifestCategories = manifest.metadata?.categories ?? [];
    return Array.from(new Set([...groupCategories, ...manifestCategories]));
  };

  const filteredManifests = useMemo(() => {
    if (!searchQuery.trim()) return manifests;
    const query = searchQuery.toLowerCase();
    return manifests.filter((m) => {
      const categoryKeys = getManifestCategoryKeys(m);
      const categoryLabels = categoryKeys.map(
        (k) => manifestCategoryKeyToLabel.get(k) || k,
      );
      return (
        ((m as any)?._group as ManifestGroup | undefined)?.metadata?.name.toLowerCase().includes(query) ||
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
      const categoryKeys = getManifestCategoryKeys(m);
      categoryKeys.forEach((k) => {
        const label =
          manifestCategoryKeyToLabel.get(k) ||
          k.replace(/[_-]/g, " ").replace(/\s+/g, " ").trim().replace(/\b\w/g, (c) => c.toUpperCase())
        set.add(label);
      });
    });
    return Array.from(set);
  }, [filteredManifests, manifestCategoryKeyToLabel]);

  const isManifestOrGroupDownloaded = useCallback(
    (manifest: ManifestDocument): boolean => {
      if (manifest.downloaded) return true;
      const group =
        ((manifest as any)?._group as ManifestGroup | undefined) ?? undefined;
      if (!group?.variants?.length) return false;
      return group.variants.some(
        (variant) => !!(variant?.manifest as ManifestDocument | null)?.downloaded,
      );
    },
    [],
  );

  const hasDownloaded = useMemo(() => {
    return filteredManifests.some((m) => isManifestOrGroupDownloaded(m));
  }, [filteredManifests, isManifestOrGroupDownloaded]);

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
        panelSize={panelSize}
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
            manifests={filteredManifests.filter((m) =>
              isManifestOrGroupDownloaded(m),
            )}
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
              const keys = getManifestCategoryKeys(m);
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
  const primaryFetched =
    (useGroupedManifestEndpoint && groupsQuery.isFetched) ||
    (useLegacyManifestListEndpoint && manifestsQuery.isFetched);
  const showEmptyState = primaryFetched && !hasAnyFiltered;

  return (
    <>
      <style>{`
        .carousel-container::-webkit-scrollbar { display: none; }
      `}</style>
      <div className="flex flex-col h-full w-full  mt-2 border-t border-brand-light/5">
        <div className="flex flex-1 min-h-0 w-full">
          <CategorySidebar
            categories={isSearching ? [] : categories}
            activeCategory={activeCategory}
            onCategoryClick={handleCategoryClick}
            title="MODELS"
            persistenceKey="sidebar:model"
            downloadedItem={
              !isSearching && hasDownloaded
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
                          disabled={
                            manifestVersionGateQuery.isFetching ||
                            groupsQuery.isFetching ||
                            manifestsQuery.isFetching
                          }
                          onClick={() => {
                            manifestVersionGateQuery.refetch();
                            if (useGroupedManifestEndpoint) {
                              groupsQuery.refetch();
                            }
                            if (useLegacyManifestListEndpoint) {
                              manifestsQuery.refetch();
                            }
                          }}
                          className="text-[11px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
                        >
                          <LuRefreshCw
                            className={`w-3.5 h-3.5 ${
                              manifestVersionGateQuery.isFetching ||
                              groupsQuery.isFetching ||
                              manifestsQuery.isFetching
                                ? "animate-spin"
                                : ""
                            }`}
                          />
                          <span>Refresh</span>
                        </button>
                      ) : null}
                    </div>
                  </div>
                </div>
              ) : isSearching ? (
                <div className="px-7 pt-3 pb-28">
                  <div
                    className="grid gap-x-2 gap-y-3"
                    style={{
                      gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
                    }}
                  >
                    {filteredManifests.map((manifest) => (
                      <div
                        key={manifest.metadata?.id || ""}
                        className="flex justify-center"
                      >
                        <ModelItem manifest={manifest} />
                      </div>
                    ))}
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
                          const keys = getManifestCategoryKeys(m);
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
