import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { ModelInputsPanel } from "./ModelInputsPanel";
import type {
  UIPanel,
  ManifestGroup,
  ManifestDocument,
} from "@/lib/manifest/api";
import { InputControlsProvider } from "@/lib/inputControl";
import { getOffloadDefaultsForManifest } from "@app/preload";
import { LuChevronDown } from "react-icons/lu";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import {
  ensureExternalAssetUrl,
  getLastPathSegment,
  inferExternalFolderFromPath,
} from "@/lib/externalAssets";

const getManifestStorageKey = (
  manifest: ManifestDocument | undefined | null,
): string => {
  const mfId = String(manifest?.metadata?.id || "").trim();
  if (mfId) return mfId;
  const fallbackId = String((manifest as any)?.id || "").trim();
  if (fallbackId) return fallbackId;
  return "__default__";
};

const extractManifestInputValues = (
  manifest: ManifestDocument | undefined | null,
): Record<string, any> => {
  const ui = manifest?.spec?.ui || (manifest as any)?.ui;
  const inputs = Array.isArray(ui?.inputs) ? (ui.inputs as Array<any>) : [];
  const out: Record<string, any> = {};
  for (const inp of inputs) {
    if (!inp || typeof inp.id !== "string") continue;
    if (inp.value !== undefined) out[inp.id] = inp.value;
  }
  return out;
};

const cloneManifestWithInputValues = (
  manifest: ManifestDocument,
  values?: Record<string, any>,
): ManifestDocument => {
  let cloned: any = manifest;
  try {
    cloned = JSON.parse(JSON.stringify(manifest));
  } catch {
    // best effort fallback
    cloned = { ...manifest };
  }

  if (!values || Object.keys(values).length === 0) return cloned as ManifestDocument;

  const ui = cloned?.spec?.ui || cloned?.ui;
  if (!ui || !Array.isArray(ui.inputs)) return cloned as ManifestDocument;

  ui.inputs = ui.inputs.map((inp: any) => {
    if (!inp || typeof inp.id !== "string") return inp;
    if (!Object.prototype.hasOwnProperty.call(values, inp.id)) return inp;
    return { ...inp, value: values[inp.id] };
  });
  return cloned as ManifestDocument;
};

// ── Variant selector ────────────────────────────────────────────────

const VariantPreview: React.FC<{
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

  const ensureFallback = useCallback(async () => {
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
  }, [src]);

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
  }, [src, ensureFallback]);

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
      className={cn(
        "shrink-0 overflow-hidden rounded-[6px] bg-brand-light/10",
        className,
      )}
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

const VariantSelector: React.FC<{
  clipId: string;
  group: ManifestGroup;
  currentManifestId: string | undefined;
}> = ({ clipId, group, currentManifestId }) => {
  const updateClip = useClipStore((s) => s.updateClip);
  const switchingRef = useRef(false);
  const [isVariantDropdownOpen, setIsVariantDropdownOpen] = useState(false);

  const variants = useMemo(() => group.variants ?? [], [group]);

  const activeVariantId = useMemo(() => {
    for (const v of variants) {
      if (v.id === currentManifestId) return v.id;
      if (v.manifest?.metadata?.id === currentManifestId) return v.id;
      if (v.manifest?.id === currentManifestId) return v.id;
    }
    return variants[0]?.id ?? null;
  }, [variants, currentManifestId]);

  const handleChange = useCallback(
    async (selectedId: string) => {
      // Guard: skip if already the active variant or if a switch is in progress
      if (selectedId === activeVariantId || switchingRef.current) return;

      const variant = variants.find((v) => v.id === selectedId);
      if (!variant?.manifest) return;
      const targetManifest = variant.manifest as ManifestDocument;

      switchingRef.current = true;
      try {
        const store = useClipStore.getState();
        const currentClip = store.getClipById(clipId) as
          | ModelClipProps
          | undefined;
        if (!currentClip?.manifest) return;

        const currentKey = getManifestStorageKey(currentClip.manifest);
        const currentValues = extractManifestInputValues(currentClip.manifest);
        const currentSelectedComponents = (currentClip.selectedComponents ||
          {}) as Record<string, any>;
        const nextInputValuesByVariant = {
          ...(currentClip.modelInputValuesByVariant || {}),
          [currentKey]: currentValues,
        };
        const nextSelectedComponentsByVariant = {
          ...(currentClip.selectedComponentsByVariant || {}),
          [currentKey]: currentSelectedComponents,
        };

        const targetKey = getManifestStorageKey(targetManifest);
        const targetValues = nextInputValuesByVariant[targetKey];
        const hydratedTargetManifest = cloneManifestWithInputValues(
          targetManifest,
          targetValues,
        );

        // Swap manifest and clear stale generation/asset state
        const patch: Partial<ModelClipProps> = {
          manifest: hydratedTargetManifest,
          modelStatus: undefined,
          assetId: undefined,
          previewPath: undefined,
          selectedComponents: nextSelectedComponentsByVariant[targetKey],
          modelInputValues: targetValues,
          modelInputValuesByVariant: nextInputValuesByVariant,
          selectedComponentsByVariant: nextSelectedComponentsByVariant,
        } as any;

        // Best-effort: fetch offload defaults for the new manifest
        try {
          const mfId = String(targetManifest.metadata?.id || "").trim();
          if (mfId) {
            const defaults = await getOffloadDefaultsForManifest(mfId);
            if (defaults) (patch as any).offload = defaults;
          }
        } catch {
          // ignore
        }

        updateClip(clipId, patch as any);
      } finally {
        switchingRef.current = false;
      }
    },
    [clipId, updateClip, variants, activeVariantId],
  );

  const activeVariant = useMemo(() => {
    if (!activeVariantId) return null;
    return variants.find((variant) => variant.id === activeVariantId) ?? null;
  }, [activeVariantId, variants]);

  const activeVariantPreviewPath =
    activeVariant?.manifest?.metadata?.demo_path ??
    activeVariant?.manifest?.demo_path ??
    group?.metadata?.demo_path ??
    group?.demo_path;

  if (variants.length < 2) return null;

  return (
    <div className="px-3 pt-3 pb-3 w-full min-w-0 max-w-full overflow-hidden border-b border-brand-light/5 bg-brand-background">
      
      <Popover
        open={isVariantDropdownOpen}
        onOpenChange={setIsVariantDropdownOpen}
      >
        <PopoverTrigger asChild>
          <button
            type="button"
            className="w-full min-h-9 px-3 py-2.5 rounded-[6px] border shadow border-brand-light/5 bg-brand-background-light/80 text-brand-light hover:border-brand-light/20 hover:bg-brand-background-light/90 transition-all grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 overflow-hidden"
          >
            <VariantPreview
              src={activeVariantPreviewPath}
              alt={activeVariant?.label || "Variant"}
              className="h-12 w-12"
            />
            <div className="min-w-0 text-left">
              <p className="text-[12px] font-medium truncate">
                {activeVariant?.label || "Select variant"}
              </p>
              {activeVariant?.description ? (
                <p className="truncate text-[11px] text-brand-light/80 leading-tight mt-0.5">
                  {activeVariant.description}
                </p>
              ) : null}
            </div>
            <LuChevronDown
              className={cn(
                "w-3.5 h-3.5 text-brand-light/70 transition-transform duration-200 shrink-0 ml-2",
                isVariantDropdownOpen && "rotate-180",
              )}
            />
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          sideOffset={8}
          className="w-(--radix-popover-trigger-width) font-poppins p-2 bg-brand border border-brand-light/10 rounded-[8px] shadow-xl z-100"
        >
          <div className="space-y-1.5">
            {variants.map((variant) => {
              const isActive = variant.id === activeVariantId;
              return (
                <button
                  key={variant.id}
                  type="button"
                  title={variant.description || variant.label}
                  onClick={() => {
                    if (!isActive) {
                      void handleChange(variant.id);
                    }
                    setIsVariantDropdownOpen(false);
                  }}
                  className={cn(
                    "w-full rounded-[6px] px-2.5 py-2 transition-colors border border-transparent grid grid-cols-[auto_minmax(0,1fr)] gap-2.5 items-start text-left",
                    isActive
                      ? "bg-brand-light/15 border-brand-light/10"
                      : "hover:bg-brand-light/8 hover:border-brand-light/10",
                  )}
                >
                  <VariantPreview
                    src={
                      variant.manifest?.metadata?.demo_path ??
                      variant.manifest?.demo_path ??
                      group?.metadata?.demo_path ??
                      group?.demo_path
                    }
                    alt={variant.label || "Variant"}
                    className="h-10 w-10 mt-0.5"
                  />
                  <div className="min-w-0">
                    <p
                      className={cn(
                        "text-[11px] font-medium truncate",
                        isActive ? "text-brand-light" : "text-brand-light/90",
                      )}
                    >
                      {variant.label}
                    </p>
                    <p className="mt-0.5 text-[10px] leading-[1.35] text-brand-light/70 line-clamp-2">
                      {variant.description || "No description available"}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
};

// ── Main properties panel ────────────────────────────────────────────

interface ModelInputsPropertiesProps {
  clipId: string;
  panelSize: number;
}

export const ModelInputsProperties: React.FC<ModelInputsPropertiesProps> = ({
  clipId,
  panelSize,
}) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;

  // IMPORTANT: use clip.manifest (which carries per-clip input values) for
  // rendering panels/inputs. Using useManifestQuery here would return the
  // raw manifest WITHOUT input values, causing "set default" calls during
  // render that trigger an infinite update loop.
  const manifest = clip?.manifest;

  const group: ManifestGroup | undefined = clip?.group;
  const showVariantSelector =
    group && Array.isArray(group.variants) && group.variants.length > 1;

  return (
    <InputControlsProvider clipId={clipId}>
      <div className="flex w-full min-w-0 max-w-full flex-col overflow-x-hidden pb-7">
        {/* Variant selector (shown when clip belongs to a multi-variant group) */}
        {showVariantSelector && (
          <VariantSelector
            clipId={clipId}
            group={group}
            currentManifestId={manifest?.metadata?.id}
          />
        )}

        <div className="text-brand-light text-[10px] flex w-full min-w-0 max-w-full flex-col divide-y divide-brand-light/5 overflow-x-hidden">
          {(() => {
            const basePanels = (manifest?.spec?.ui?.panels ||
              []) as UIPanel[];
            const components = manifest?.spec?.components || [];
            const schedulerOptions = components
              .filter((c: any) => String(c?.type) === "scheduler")
              .flatMap((c: any) =>
                Array.isArray(c?.scheduler_options)
                  ? c.scheduler_options
                  : [],
              );
            const hasSchedulerOptions =
              schedulerOptions && schedulerOptions.length > 0;
            const alreadyHasSchedulerPanel = basePanels.some(
              (p) => String(p?.name || "").toLowerCase() === "scheduler",
            );
            let panelsToRender =
              hasSchedulerOptions && !alreadyHasSchedulerPanel
                ? [
                    ...basePanels,
                    {
                      name: "scheduler",
                      label: "Scheduler",
                      collapsible: true,
                      default_open: false,
                      layout: { flow: "column", rows: [] },
                    } as UIPanel,
                  ]
                : basePanels;

            // Append Attention panel at the very end if options exist and panel not present
            const attentionOptions = (manifest?.spec
              ?.attention_types_detail || []) as any[];
            const hasAttentionOptions =
              Array.isArray(attentionOptions) && attentionOptions.length > 0;
            const alreadyHasAttentionPanel = panelsToRender.some(
              (p) => String(p?.name || "").toLowerCase() === "attention",
            );
            if (hasAttentionOptions && !alreadyHasAttentionPanel) {
              panelsToRender = [
                ...panelsToRender,
                {
                  name: "attention",
                  label: "Attention",
                  collapsible: true,
                  default_open: false,
                  layout: { flow: "column", rows: [] },
                } as UIPanel,
              ];
            }
            return panelsToRender.map((panel) => {
              return (
                <ModelInputsPanel
                  key={panel.name}
                  panel={panel}
                  inputs={manifest?.spec?.ui?.inputs || []}
                  clipId={clipId}
                  panelSize={panelSize}
                />
              );
            });
          })()}
        </div>
      </div>
    </InputControlsProvider>
  );
};
