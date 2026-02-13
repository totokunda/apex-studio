import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { cn } from "@/lib/utils";
import { generatePosterCanvas } from "@/lib/media/timeline";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import { pathToFileURLString } from "@app/preload";
import { useControlsStore } from "@/lib/control";
import {
  getVariantScopedValue,
  getVariantStorageKey,
  getVariantStorageLookupKeys,
} from "@/lib/manifest/variantStorageKey";

const extractManifestInputValues = (manifest: any): Record<string, any> => {
  const ui = manifest?.spec?.ui || manifest?.ui;
  const inputs = Array.isArray(ui?.inputs) ? ui.inputs : [];
  const out: Record<string, any> = {};
  const seen = new Set<string>();
  for (const inp of inputs) {
    if (!inp || typeof inp.id !== "string") continue;
    if (seen.has(inp.id)) continue;
    seen.add(inp.id);
    if (inp.value !== undefined) out[inp.id] = inp.value;
  }
  return out;
};

const cloneManifestWithInputValues = (
  manifest: any,
  values?: Record<string, any>,
): any => {
  let cloned = manifest;
  try {
    cloned = JSON.parse(JSON.stringify(manifest));
  } catch {
    cloned = { ...manifest };
  }
  const ui = cloned?.spec?.ui || cloned?.ui;
  if (!ui || !Array.isArray(ui.inputs)) return cloned;
  ui.inputs = ui.inputs.map((inp: any) => {
    if (!inp || typeof inp.id !== "string") return inp;
    const { value: _existingValue, ...rest } = inp;
    if (!values || !Object.prototype.hasOwnProperty.call(values, inp.id)) {
      return rest;
    }
    return { ...rest, value: values[inp.id] };
  });
  return cloned;
};

const resolveVariantForGeneration = (
  clip: ModelClipProps,
  gen: any,
): any | null => {
  const variants = clip?.group?.variants ?? [];
  if (!variants.length) return null;

  const variantId = String(gen?.variantId || "").trim();
  if (variantId) {
    const match = variants.find((v: any) => String(v?.id || "") === variantId);
    if (match) return match;
  }

  const manifestId = String(gen?.manifestId || "").trim();
  if (manifestId) {
    const match = variants.find((v: any) => {
      const byVariantId = String(v?.id || "") === manifestId;
      const byMetaId = String(v?.manifest?.metadata?.id || "") === manifestId;
      const byManifestId = String(v?.manifest?.id || "") === manifestId;
      return byVariantId || byMetaId || byManifestId;
    });
    if (match) return match;
  }

  const valueKeys = Object.keys((gen?.values || {}) as Record<string, any>);
  if (valueKeys.length === 0) return null;

  let best: any | null = null;
  let bestScore = -1;
  for (const variant of variants as any[]) {
    const ui = variant?.manifest?.spec?.ui || variant?.manifest?.ui;
    const inputs = Array.isArray(ui?.inputs) ? ui.inputs : [];
    if (inputs.length === 0) continue;
    const ids = new Set(
      inputs
        .map((inp: any) => (typeof inp?.id === "string" ? inp.id : ""))
        .filter(Boolean),
    );
    let overlap = 0;
    for (const key of valueKeys) {
      if (ids.has(key)) overlap += 1;
    }
    if (overlap > bestScore) {
      best = variant;
      bestScore = overlap;
    }
  }
  return bestScore > 0 ? best : null;
};

interface ModelGenerationPropertiesProps {
  clipId: string;
}

export const ModelGenerationProperties: React.FC<
  ModelGenerationPropertiesProps
> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;
  const updateClip = useClipStore((s) => s.updateClip);
  const updateModelInput = useClipStore((s) => s.updateModelInput);
  const getAssetById = useClipStore((s) => s.getAssetById);

  const fps = useControlsStore((s) => s.fps);
  const generations = useMemo(
    () => clip?.generations ?? [],
    [clip?.generations],
  );

  const visibleGenerations = useMemo(() => {
    return (generations || [])
      .filter(
        (g) =>
          g.assetId !== null && g.assetId !== undefined,
      )
      .sort((a, b) => (b?.createdAt ?? 0) - (a?.createdAt ?? 0));
  }, [generations]);

  const normalizeToFileUrl = useCallback(
    (maybePath: string | undefined | null): string | null => {
      if (!maybePath) return null;
      try {
        // If it already looks like a file URL, return as-is
        if (maybePath.startsWith("file://")) return maybePath;
        return pathToFileURLString(maybePath);
      } catch {
        return null;
      }
    },
    [],
  );

  const selectedAssetId = String(clip?.assetId || "");
  const selectedAsset = useMemo(() => getAssetById(selectedAssetId), [selectedAssetId]);

  const selectedIndex = useMemo(() => {
    if (!visibleGenerations || visibleGenerations.length === 0) return -1;
    const idx = visibleGenerations.findIndex((g) => {
      const asset = getAssetById(g?.assetId);
      const url = normalizeToFileUrl(asset?.path);
      return url && url === selectedAsset?.path;
    });
    return idx;
  }, [visibleGenerations, normalizeToFileUrl, selectedAsset?.path]);

  const onSelectGeneration = useCallback(
    async (index: number) => {
      // Prevent re-selecting the already selected generation
      if (index === selectedIndex) return;
      const gen = visibleGenerations[index];

      if (!gen) return;


      try {
        const store = useClipStore.getState();
        const currentClip = store.getClipById(clipId) as
          | ModelClipProps
          | undefined;
        if (!currentClip) return;

        const currentManifest = currentClip.manifest;
        const currentKey = getVariantStorageKey({
          group: currentClip.group,
          manifest: currentManifest,
          preferredVariantId: currentClip.variantId,
        });
        const currentValues = extractManifestInputValues(currentManifest);
        const nextInputValuesByVariant = {
          ...(currentClip.modelInputValuesByVariant || {}),
          [currentKey]: currentValues,
        };
        const nextSelectedComponentsByVariant = {
          ...(currentClip.selectedComponentsByVariant || {}),
          [currentKey]: (currentClip.selectedComponents || {}) as Record<
            string,
            any
          >,
        };

        const targetVariant = resolveVariantForGeneration(currentClip, gen);
        let targetManifest = currentManifest;
        let targetKey = currentKey;
        let targetValues = getVariantScopedValue(nextInputValuesByVariant, [
          currentKey,
        ]);
        let targetVariantId = currentClip.variantId;
        if (targetVariant?.manifest) {
          targetVariantId = String((targetVariant as any)?.id || "").trim() || undefined;
          targetKey = getVariantStorageKey({
            group: currentClip.group,
            manifest: targetVariant.manifest,
            preferredVariantId: targetVariantId,
          });
          targetValues = getVariantScopedValue(
            nextInputValuesByVariant,
            getVariantStorageLookupKeys({
              group: currentClip.group,
              manifest: targetVariant.manifest,
              preferredVariantId: targetVariantId,
              includeLegacy: false,
            }),
          );
          targetManifest = cloneManifestWithInputValues(
            targetVariant.manifest,
            targetValues,
          );
        }

        // Persist current clip transform into the previously selected generation entry (if any)
        let updates: Partial<ModelClipProps> = {
          assetId: gen.assetId,
          variantId: targetVariantId,
          manifest: targetManifest,
          modelInputValuesByVariant: nextInputValuesByVariant,
          selectedComponentsByVariant: nextSelectedComponentsByVariant,
          modelInputValues: targetValues,
        };
        try {
          const prevVisibleGen = selectedIndex >= 0 ? visibleGenerations[selectedIndex] : null;
          const prevIdx = prevVisibleGen
            ? (currentClip.generations || []).findIndex(
                (g) =>
                  String(g?.jobId || "") === String(prevVisibleGen?.jobId || "") &&
                  Number(g?.createdAt || 0) === Number(prevVisibleGen?.createdAt || 0),
              )
            : -1;
          const currentTransform = currentClip?.transform;
          if (
            typeof prevIdx === "number" &&
            prevIdx >= 0 &&
            currentTransform &&
            Array.isArray(currentClip?.generations)
          ) {
            const gens = (currentClip?.generations || []).map((g: any, i: number) =>
              i === prevIdx ? { ...g, transform: currentTransform } : g,
            );
            updates.generations = gens;
          }
        } catch {}

        // When switching to a generation: derive width/height from the asset's
        // intrinsic dimensions only, and clamp the long side to BASE_LONG_SIDE
        // while preserving aspect ratio (asset is the single source of truth).
        // update endFrame based on the duration of the generation
        if (gen.startFrame && gen.endFrame) {
          updates.startFrame = gen.startFrame;
          updates.endFrame = gen.endFrame;
        } else {
          const mediaInfo = getMediaInfoCached(gen.assetId);
        if (mediaInfo && mediaInfo.duration) {
          let newDuration = Math.floor(mediaInfo.duration * fps);
          updates.endFrame = currentClip.startFrame + newDuration;
        }
        }
        

        if (gen.transform) {
          updates.transform = {...gen.transform};
        } 
        // update the gen.transform to the current clip.transform
        const generations = [...(currentClip?.generations || [])]; 
        const selectedGenerationIndex = generations.findIndex(
          (g) =>
            String(g?.jobId || "") === String(gen?.jobId || "") &&
            Number(g?.createdAt || 0) === Number(gen?.createdAt || 0),
        );
        
        // check if gen has attribute trimStart or trimEnd, if so, update the transform to the current clip.transform
        if (gen.trimStart) {
          updates.trimStart = gen.trimStart;
        }
        if (gen.trimEnd) {
          updates.trimEnd = gen.trimEnd;
        }

        if (selectedGenerationIndex >= 0) {
          generations[selectedGenerationIndex].transform = currentClip?.transform;
          generations[selectedGenerationIndex].trimStart = currentClip?.trimStart;
          generations[selectedGenerationIndex].trimEnd = currentClip?.trimEnd;
          generations[selectedGenerationIndex].startFrame = currentClip?.startFrame;

          generations[selectedGenerationIndex].endFrame = currentClip?.endFrame;
        }
        updates.generations = generations;
        if (gen.selectedComponents) {
          updates.selectedComponents = gen.selectedComponents;
          nextSelectedComponentsByVariant[targetKey] = gen.selectedComponents;
          updates.selectedComponentsByVariant = nextSelectedComponentsByVariant;
        } else {
          const selectedForVariant = getVariantScopedValue(
            nextSelectedComponentsByVariant,
            getVariantStorageLookupKeys({
              group: currentClip.group,
              manifest: targetManifest,
              preferredVariantId: targetVariantId,
              includeLegacy: !targetVariantId,
            }),
          );
          if (selectedForVariant) {
            updates.selectedComponents = selectedForVariant;
          }
        }
        updateClip(clipId, updates);
      } catch {}

      const vals = gen.values || {};
      for (const [inputId, v] of Object.entries(vals)) {
        updateModelInput(clipId, inputId, { value: v } as any);
      }
      // No additional transform work here; applied atomically with updateClip above.
    },
    [
      clipId,
      visibleGenerations,
      updateClip,
      updateModelInput,
      selectedIndex,
      fps,
    ],
  );

  if (!visibleGenerations || visibleGenerations.length === 0) {
    return (
      <div className="flex flex-col gap-y-2.5 p-4">
        <span className="text-brand-light text-[12px] font-medium text-start">
          Generations
        </span>
        <div className="text-[11.5px] text-start text-brand-light/70">
          No generations created.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-y-4 p-5">
      <div className="flex flex-row items-center justify-between">
        <span className="text-brand-light text-[12px] font-medium">
          Generations
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {visibleGenerations.map((g, idx) => {
          return (
            <GenerationCard
              key={`${g.jobId || idx}`}
              generation={g}
              isSelected={idx === selectedIndex}
              onSelect={() => onSelectGeneration(idx)}
            />
          );
        })}
      </div>
    </div>
  );
};

const formatTime = (ts: number | undefined) => {
  if (!ts || !Number.isFinite(ts)) return "";
  try {
    const d = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const sec = Math.max(0, Math.floor(diffMs / 1000));
    const min = Math.floor(sec / 60);
    const hr = Math.floor(min / 60);

    if (sec < 30) return "just now";
    if (sec < 60) return `${sec}s ago`;
    if (min < 60) return `${min}m ago`;
    if (hr < 24) return `${hr}h ago`;

    const sameDay = (a: Date, b: Date) =>
      a.getFullYear() === b.getFullYear() &&
      a.getMonth() === b.getMonth() &&
      a.getDate() === b.getDate();
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);

    const timePart = d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    if (sameDay(d, yesterday)) return `Yesterday ${timePart}`;

    const sameYear = d.getFullYear() === now.getFullYear();
    const dateOptsSameYear: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    };
    const dateOptsWithYear: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    };
    return d.toLocaleString([], sameYear ? dateOptsSameYear : dateOptsWithYear);
  } catch {
    return "";
  }
};

const GenerationCard: React.FC<{
  generation: NonNullable<ModelClipProps["generations"]>[number];
  isSelected: boolean;
  onSelect: () => void;
}> = ({ generation, isSelected, onSelect }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [meta, setMeta] = useState<{ duration?: number } | null>(null);
  const getAssetById = useClipStore((s) => s.getAssetById);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const el = canvasRef.current;
      const asset = getAssetById(generation?.assetId);
      const src = asset?.path;
      if (!el || !src) {
        return;
      }
      setLoading(true);
      try {
        const url = src.startsWith("file://") ? src : pathToFileURLString(src);
        const info = await getMediaInfo(url, { sourceDir: "apex-cache" });
        if (!cancelled) setMeta({ duration: info?.duration });

        const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
        const cssWidth = el.clientWidth || 240;
        const cssHeight = Math.round((cssWidth * 9) / 16);

        // Backing canvas resolution in device pixels
        el.width = cssWidth * dpr;
        el.height = cssHeight * dpr;

        // Generate a poster at intrinsic resolution, then letterbox into 16:9
        const poster = await generatePosterCanvas(url, undefined, undefined, {
          mediaInfo: info,
        });
        if (!poster || cancelled) return;

        const ctx = el.getContext("2d");
        if (!ctx || cancelled) return;

        const sourceWidth =
          // Prefer the actual poster dimensions if available
          (poster as any)?.width ??
          info?.video?.displayWidth ??
          info?.image?.width ??
          asset?.width ??
          el.width;
        const sourceHeight =
          (poster as any)?.height ??
          info?.video?.displayHeight ??
          info?.image?.height ??
          asset?.height ??
          el.height;

        const targetW = el.width;
        const targetH = el.height;

        // Contain-fit into the 16:9 box with black bars (no stretching)
        const scale = Math.min(targetW / sourceWidth, targetH / sourceHeight);
        const drawW = sourceWidth * scale;
        const drawH = sourceHeight * scale;
        const offsetX = (targetW - drawW) / 2;
        const offsetY = (targetH - drawH) / 2;

        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, targetW, targetH);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, targetW, targetH);
        ctx.drawImage(
          poster as CanvasImageSource,
          offsetX,
          offsetY,
          drawW,
          drawH,
        );
        ctx.restore();
      } catch {
        // ignore
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [generation?.assetId]);

  const durationText = useMemo(() => {
    const dur = meta?.duration;
    if (!dur || !Number.isFinite(dur)) return null;
    const total = Math.floor(dur);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }, [meta?.duration]);

  return (
    <button
      onClick={onSelect}
      disabled={isSelected}
      className={cn(
        "w-full flex flex-col items-stretch justify-start rounded-[7px] transition-all duration-150 shadow border border-t-0 border-brand-light/15 bg-brand",
        isSelected ? "cursor-default opacity-90" : "",
      )}
      style={{ textAlign: "left" }}
    >
      <div
        className="relative w-full rounded-t-md overflow-hidden"
        style={{ aspectRatio: "16 / 9" }}
      >
        <canvas ref={canvasRef} className="w-full h-full block" />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-[11px] text-brand-light/70 bg-brand-background/40">
            Loading…
          </div>
        )}

        {durationText && (
          <div className="absolute bottom-1 left-1 text-[10px] px-1.5 py-0.5 rounded bg-brand-background-dark/70 text-brand-light/90">
            {durationText}
          </div>
        )}
      </div>
      <div className=" py-1.5 flex flex-row items-center justify-between gap-y-1 relative px-3">
        <div className="text-[10.5px] py-0.5 text-brand-light/90">
          {formatTime(generation.createdAt)}
        </div>
        {isSelected && (
          <div className="w-fit text-[10px] px-2.5 font-medium py-0.5 rounded bg-brand-accent-two-shade text-white">
            Selected
          </div>
        )}
      </div>
    </button>
  );
};
