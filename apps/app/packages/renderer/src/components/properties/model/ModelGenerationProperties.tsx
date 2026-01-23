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

interface ModelGenerationPropertiesProps {
  clipId: string;
}

export const ModelGenerationProperties: React.FC<
  ModelGenerationPropertiesProps
> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;
  const updateClip = useClipStore((s) => s.updateClip);
  const updateModelInput = useClipStore((s) => s.updateModelInput);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
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
        // Persist current clip transform into the previously selected generation entry (if any)
        let updates: Partial<ModelClipProps> = { assetId: gen.assetId };
        try {
          const prevIdx = selectedIndex;
          const currentTransform = clip?.transform;
          if (
            typeof prevIdx === "number" &&
            prevIdx >= 0 &&
            currentTransform &&
            Array.isArray(clip?.generations)
          ) {
            const gens = (clip?.generations || []).map((g: any, i: number) =>
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
          updates.endFrame = clip.startFrame + newDuration;
        }
        }
        

        if (gen.transform) {
          updates.transform = {...gen.transform};
        } 
        // update the gen.transform to the current clip.transform
        const generations = [...(clip?.generations || [])]; 
        
        // check if gen has attribute trimStart or trimEnd, if so, update the transform to the current clip.transform
        if (gen.trimStart) {
          updates.trimStart = gen.trimStart;
        }
        if (gen.trimEnd) {
          updates.trimEnd = gen.trimEnd;
        }

        generations[index].transform = clip?.transform;
        generations[index].trimStart = clip?.trimStart;
        generations[index].trimEnd = clip?.trimEnd;
        generations[index].startFrame = clip?.startFrame;

        generations[index].endFrame = clip?.endFrame;
        updates.generations = generations;
        if (gen.selectedComponents) {
          updates.selectedComponents = gen.selectedComponents;
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
      normalizeToFileUrl,
      updateClip,
      updateModelInput,
      selectedIndex,
      clip,
      setClipTransform,
      clip.transform,
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
