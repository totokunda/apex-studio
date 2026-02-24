import { useManifestStore } from "@/lib/manifest/store";
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { LuChevronLeft, LuPlus, LuRefreshCw } from "react-icons/lu";
import { ScrollArea } from "../ui/scroll-area";
import { useControlsStore } from "@/lib/control";
import {
  useClipStore,
  getTimelineHeightForClip,
  getTimelineTypeForClip,
  isValidTimelineForClip,
} from "@/lib/clip";
import { v4 as uuidv4 } from "uuid";
import { refreshManifest, useManifestQuery } from "@/lib/manifest/queries";
import ComponentCard, { LoraCard } from "./ComponentCard2";
import { useQueryClient } from "@tanstack/react-query";
import { getOffloadDefaultsForManifest } from "@app/preload";
import {
  ensureExternalAssetUrl,
  getLastPathSegment,
  inferExternalFolderFromPath,
} from "@/lib/externalAssets";
import type { ManifestGroup, ManifestGroupVariant, ManifestDocument } from "@/lib/manifest";

interface ModelPageProps {
  manifestId: string;
  scrollCache?: Map<string, number>;
  scrollKey?: string;
  panelSize?: number;
}

const VariantTabPreview: React.FC<{
  src: string | undefined;
  alt: string;
  className?: string;
}> = ({ src, alt, className }) => {
  const [resolvedSrc, setResolvedSrc] = useState<string | undefined>(src);
  const triedFallbackRef = useRef(false);

  useEffect(() => {
    triedFallbackRef.current = false;
    setResolvedSrc(src);
  }, [src]);

  const ensureFallback = async () => {
    if (triedFallbackRef.current) return;
    triedFallbackRef.current = true;
    if (!src) return;
    const folder = inferExternalFolderFromPath(src);
    const seg = getLastPathSegment(src);
    if (!seg) return;
    try {
      const url = await ensureExternalAssetUrl({ folder, filePath: seg });
      if (url) setResolvedSrc(url);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    const raw = String(src || "").trim().toLowerCase();
    if (!raw) return;
    const isAlreadyResolved =
      raw.startsWith("app://") ||
      raw.startsWith("http://") ||
      raw.startsWith("https://") ||
      raw.startsWith("blob:") ||
      raw.startsWith("data:");
    if (!isAlreadyResolved) {
      void ensureFallback();
    }
  }, [src]);

  const isVideo = useMemo(() => {
    const value = String(resolvedSrc || "").toLowerCase();
    if (!value) return false;
    try {
      const url = new URL(value);
      const ext = (url.pathname.split(".").pop() || "").toLowerCase();
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
  }, [resolvedSrc]);

  return (
    <div
      className={`shrink-0 overflow-hidden rounded-[6px] bg-brand-light/10 ${className || ""}`}
    >
      {resolvedSrc ? (
        isVideo ? (
          <video
            src={resolvedSrc}
            className="h-full w-full object-cover"
            autoPlay
            muted
            loop
            playsInline
            onError={() => {
              void ensureFallback();
            }}
          />
        ) : (
          <img
            src={resolvedSrc}
            alt={alt}
            className="h-full w-full object-cover"
            onError={() => {
              void ensureFallback();
            }}
          />
        )
      ) : (
        <div className="h-full w-full" />
      )}
    </div>
  );
};

const ModelPage: React.FC<ModelPageProps> = ({
  manifestId,
  scrollCache,
  scrollKey,
  panelSize = 0,
}) => {
  const { clearSelectedManifestId, setSelectedManifestId } = useManifestStore();
  const ctrlToggleClipSelection = useControlsStore(
    (s) => s.toggleClipSelection,
  );
  const queryClient = useQueryClient();
  const { data: manifest, isFetching } = useManifestQuery(manifestId);
  const [isRefreshingManifest, setIsRefreshingManifest] = React.useState(false);
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);

  // Find the parent group (if any) for the current manifest.
  // This enables variant tabs when the manifest belongs to a multi-variant group.
  const parentGroup: ManifestGroup | null = useMemo(() => {
    const groups = queryClient.getQueryData<ManifestGroup[]>(["manifestGroups"]);
    if (!Array.isArray(groups) || groups.length === 0) return null;
    

    for (const group of groups) {
      const variants = group.variants ?? [];
      for (const variant of variants) {
        // Match by manifest_ref id, variant id, or resolved manifest id
        if (variant.id === manifestId) return group;
        if (variant.manifest?.metadata?.id === manifestId) return group;
        if (variant.manifest?.id === manifestId) return group;
      }
    }
    return null;
  }, [manifestId, queryClient]);


  // Only show tabs when the group has more than one variant
  const groupVariants: ManifestGroupVariant[] = useMemo(() => {
    if (!parentGroup) return [];
    return parentGroup.variants ?? [];
  }, [parentGroup]);

  const showVariantTabs = groupVariants.length > 1;
  const [expandedDescriptions, setExpandedDescriptions] = useState<
    Record<string, boolean>
  >({});

  // Determine which variant is currently active
  const activeVariantId: string | null = useMemo(() => {
    if (!showVariantTabs) return null;
    for (const variant of groupVariants) {
      if (variant.id === manifestId) return variant.id;
      if (variant.manifest?.metadata?.id === manifestId) return variant.id;
      if (variant.manifest?.id === manifestId) return variant.id;
    }
    return null;
  }, [groupVariants, manifestId, showVariantTabs]);

  const activeVariant = useMemo(() => {
    if (!activeVariantId) return null;
    return groupVariants.find((variant) => variant.id === activeVariantId) ?? null;
  }, [activeVariantId, groupVariants]);

  const resolveVariantTargetId = (variant: ManifestGroupVariant) =>
    variant.manifest?.metadata?.id ?? variant.manifest?.id ?? variant.id;

  // Enforce hard bounds for variant rows/text based on current panel width.
  const variantRowMaxWidth = useMemo(() => {
    if (!panelSize || panelSize <= 0) return undefined;
    return Math.max(0, panelSize - 72);
  }, [panelSize]);
  const variantTextMaxWidth = useMemo(() => {
    if (!panelSize || panelSize <= 0) return undefined;
    return Math.max(0, panelSize - 124);
  }, [panelSize]);

  if (!manifest) return null;

  const demoPath = manifest.metadata?.demo_path || manifest.demo_path;

  const [resolvedDemoPath, setResolvedDemoPath] = useState<string | undefined>(
    demoPath,
  );
  const triedDemoFallbackRef = useRef(false);

  useEffect(() => {
    triedDemoFallbackRef.current = false;
    setResolvedDemoPath(demoPath);
  }, [demoPath]);

  const ensureDemoFallback = async () => {
    if (triedDemoFallbackRef.current) return;
    triedDemoFallbackRef.current = true;
    if (!demoPath) return;
    const folder = inferExternalFolderFromPath(demoPath);
    const seg = getLastPathSegment(demoPath);
    if (!seg) return;
    const url = await ensureExternalAssetUrl({ folder, filePath: seg });
    if (url) setResolvedDemoPath(url);
  };



  

  useLayoutEffect(() => {
    if (!scrollCache || !scrollKey) return;
    const root = scrollAreaRef.current;
    if (!root) return;

    const viewport = root.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLDivElement | null;
    if (!viewport) return;

    const onScroll = () => {
      scrollCache.set(scrollKey, viewport.scrollTop);
    };

    const saved = scrollCache.get(scrollKey);
    if (typeof saved === "number") {
      viewport.scrollTop = saved;
    }

    viewport.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      viewport.removeEventListener("scroll", onScroll as EventListener);
    };
  }, [manifestId, scrollCache, scrollKey]);

  const isVideoDemo = React.useMemo(() => {
    const value = (demoPath || "").toLowerCase();
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
  }, [demoPath]);

  const components = manifest?.spec?.components || [];

  const addClip = useClipStore((state) => state.addClip);
  
  return (
    <div className="flex flex-col h-full w-full">
      <ScrollArea className="flex-1" ref={scrollAreaRef}>
        <div className="p-7 pt-3 pb-28">
          <div className="flex items-center gap-x-2">
            <button
              onClick={async () => {
                clearSelectedManifestId();
              }}
              className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded transition-colors cursor-pointer"
            >
              <LuChevronLeft className="w-3 h-3" />
            </button>
            <span className="text-brand-light/90 text-[11px] font-medium">
              Back
            </span>
          </div>
          <div className="mt-4 flex min-w-0 w-full flex-row gap-x-4 overflow-x-hidden">
            <div className="rounded-md overflow-hidden flex items-center w-44 aspect-square justify-start shrink-0">
              {isVideoDemo ? (
                <video
                  src={resolvedDemoPath}
                  className="h-full w-full object-cover rounded-md"
                  autoPlay
                  muted
                  loop
                  playsInline
                  onError={() => {
                    void ensureDemoFallback();
                  }}
                />
              ) : (
                <img
                  src={resolvedDemoPath}
                  alt={manifest.metadata.name}
                  className="h-full object-cover rounded-md"
                  onError={() => {
                    void ensureDemoFallback();
                  }}
                />
              )}
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-y-1 justify-start overflow-x-hidden">
              <h2 className="text-brand-light text-[18px] font-semibold text-start wrap-break-word">
                {parentGroup ? parentGroup.name : manifest.metadata.name}
              </h2>
              {parentGroup && (
                <p className="text-brand-light/50 text-[11px] text-start -mt-0.5">
                  {manifest.metadata.name}
                </p>
              )}
              <p className="text-brand-light/90 text-[12px] text-start wrap-break-word">
                {manifest.metadata.description}
              </p>

              <div className="flex flex-col mt-1 items-start gap-y-0.5">
                <span className="text-brand-light text-[12px] font-medium">
                  {manifest.metadata.license}
                </span>
                <span className="text-brand-light/80 text-[11px]">
                  {manifest.metadata.author}
                </span>

              </div>

              <div className="flex flex-row items-center gap-1.5 mt-2 flex-wrap">
                {manifest.metadata?.tags?.map((tag) => (
                  <span
                    key={tag}
                    className="text-brand-light text-[11px] bg-brand border shadow border-brand-light/10 rounded px-2 py-0.5"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
          {(
            <div className="mt-4">
              <button
                type="button"
                className="text-[11px] font-medium w-full flex items-center transition-all duration-200 justify-center gap-x-1.5 rounded-[6px] px-12 py-2 shrink-0 text-brand-light hover:text-brand-light/90 bg-brand-background-light hover:bg-brand/90 border border-brand-light/5"
                title="Add clip at playhead"
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

                    // Choose an existing compatible timeline with free space
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
                    // If no space found, create a new timeline
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
                    // Build and add clip – prefer the active variant's resolved
                    // manifest when available so the clip tracks the exact variant.
                    const variantManifest =
                      (activeVariant?.manifest as ManifestDocument | null | undefined) ?? manifest;
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
                      manifest: variantManifest,
                      variantId: activeVariant?.id ? String(activeVariant.id) : undefined,
                    };
                    if (parentGroup) {
                      clipBase.group = parentGroup;
                    }
                    try {
                      const mfId = String(variantManifest?.metadata?.id || "").trim();
                      if (mfId) {
                        const defaults = await getOffloadDefaultsForManifest(mfId);
                        if (defaults) {
                          clipBase.offload = defaults;
                        }
                      }
                    } catch {
                      // ignore; defaults are best-effort
                    }
                    addClip(clipBase);
                    //setselected 
                    ctrlToggleClipSelection(newClipId);
                    
                  } catch {}
                }}
              >
                <LuPlus className="w-4 h-4" />
                <span className="">Add Clip</span>
              </button>
            </div>
          )}
          {/* Variant selector - only rendered when the manifest belongs to a group */}
          {showVariantTabs && (
            <div className="mt-5 min-w-0 w-full overflow-x-hidden">
              <div className="w-full min-w-0 max-w-full overflow-hidden">
                <div className="text-brand-light text-[13.5px] font-semibold text-start mb-2">
                  Variants
                </div>
                <div className="w-full min-w-0 max-w-full overflow-hidden rounded-[6px] gap-y-2 flex flex-col">
                  {groupVariants.map((variant) => {
                    const isActive = variant.id === activeVariantId;
                    const targetId = resolveVariantTargetId(variant);
                    const description = variant.description || "";
                    const previewPath =
                      variant.manifest?.metadata?.demo_path ??
                      variant.manifest?.demo_path ??
                      parentGroup?.metadata?.demo_path ??
                      parentGroup?.demo_path;
                    const isExpanded = !!expandedDescriptions[variant.id];
                    const shouldShowToggle = description.length > 90;
                    return (
                      <div
                        key={variant.id}
                        className="flex-1 relative w-full min-w-0 max-w-full overflow-hidden  text-brand-light text-[11px] cursor-pointer bg-brand px-3 py-2 rounded-[6px] data-[state=active]:bg-brand-light/10 text-start hover:bg-brand-light/10 transition-colors duration-200 border border-brand-light/10"
                        style={
                          variantRowMaxWidth
                            ? { maxWidth: `${variantRowMaxWidth}px` }
                            : undefined
                        }
                        data-state={isActive ? "active" : "inactive"}
                        role="button"
                        tabIndex={0}
                        onClick={() => {
                          if (!isActive && targetId) {
                            setSelectedManifestId(targetId);
                          }
                        }}
                        onKeyDown={(e) => {
                          if ((e.key === "Enter" || e.key === " ") && !isActive && targetId) {
                            e.preventDefault();
                            setSelectedManifestId(targetId);
                          }
                        }}
                      >

                        <div className="flex w-full min-w-0 max-w-full items-start gap-2.5">
                          <VariantTabPreview
                            src={previewPath}
                            alt={variant.label || "Variant"}
                            className="h-12 w-12 mt-0.5"
                          />
                          <div className="min-w-0 flex-1">
                            <div
                              className="block w-full min-w-0 max-w-full overflow-hidden text-ellipsis whitespace-nowrap font-medium"
                              style={
                                variantTextMaxWidth
                                  ? { maxWidth: `${variantTextMaxWidth}px` }
                                  : undefined
                              }
                            >
                              {variant.label}
                            </div>
                            <div
                              className="mt-0.5 w-full min-w-0 max-w-full overflow-hidden"
                              style={
                                variantTextMaxWidth
                                  ? { maxWidth: `${variantTextMaxWidth}px` }
                                  : undefined
                              }
                            >
                              <div
                                className={`block w-full min-w-0 max-w-full text-brand-light/60 text-[10px] font-medium ${
                                  isExpanded
                                    ? "overflow-hidden whitespace-normal wrap-break-word"
                                    : "truncate"
                                }`}
                              >
                                {description}
                              </div>
                              {shouldShowToggle && (
                                <button
                                  type="button"
                                  className="mt-0.5 text-[10px] font-medium text-brand-light/80 underline decoration-brand-light/30 underline-offset-2 hover:text-brand-light"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setExpandedDescriptions((prev) => ({
                                      ...prev,
                                      [variant.id]: !prev[variant.id],
                                    }));
                                  }}
                                  onKeyDown={(e) => {
                                    e.stopPropagation();
                                  }}
                                >
                                  {isExpanded ? "See less" : "See more"}
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
          
          
          <div className="mt-3 ">
            <div className="flex items-center justify-between">
              <h3 className="text-brand-light text-[13.5px] font-semibold text-start">
                Model Architecture
              </h3>
              <button
                type="button"
                title="Refresh manifest"
                aria-label="Refresh manifest"
                disabled={isRefreshingManifest}
                onClick={async () => {
                  if (isRefreshingManifest) return;
                  try {
                    setIsRefreshingManifest(true);
                    await refreshManifest(manifestId, queryClient);
                  } catch {
                    // Intentionally no-op; errors are handled by existing query error UX/toasts elsewhere
                  } finally {
                    setIsRefreshingManifest(false);
                  }
                }}
                className="text-[11px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
              >
                <LuRefreshCw
                  className={`w-3.5 h-3.5 ${(isRefreshingManifest || isFetching) ? "animate-spin" : ""}`}
                />
                <span>Refresh</span>
              </button>
            </div>

            <div className="space-y-2 mt-3.5">
              {components.map((component, index) => (
                <div key={index}>
                <ComponentCard
                  key={index}
                  component={component}
                  manifestId={manifestId}
                  index={index}
                />
                </div>
              ))}
              {components.length === 0 && (
                <div className="text-brand-light/60 text-[12px] text-center py-8">
                  No components available
                </div>
              )}
            </div>
          </div>
          {manifest.spec.loras && manifest.spec.loras.length > 0 && (
            <div className="mt-6">
              <div className="flex items-center justify-between">
                <h3 className="text-brand-light text-[13.5px] font-semibold text-start">
                  LoRAs
                </h3>
              </div>
              <div className="space-y-2 mt-3.5">
                {manifest.spec.loras.map((l, idx) => (
                  <LoraCard key={idx} lora={l} manifestId={manifestId} loraIndex={idx} />
                ))}
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default ModelPage;
