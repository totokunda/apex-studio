import { useClipStore } from "@/lib/clip";
import {
  ManifestComponent,
  ManifestComponentModelPathItem,
} from "@/lib/manifest/api";
import { ModelClipProps } from "@/lib/types";
import { useMemo, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import { LuCheck } from "react-icons/lu";
import { useManifestQuery } from "@/lib/manifest/queries";
import { getSchedulerComponentKey } from "@/lib/manifest/componentKey";

interface ModelComponentsPropertiesProps {
  clipId: string;
}

const formatComponentName = (name: string): string => {
  return name
    .replace(/\./g, " ")
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const ModelComponentsProperties = ({
  clipId,
}: ModelComponentsPropertiesProps) => {
  const getComponentKey = (comp: ManifestComponent): string => {
    return String((comp as any).name || comp.type || "component");
  };

  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | ModelClipProps
    | undefined;
  if (!clip) return null;

  const { data: manifest } = useManifestQuery(
    String(clip.manifest?.metadata?.id || ""),
  );
  if (!manifest) return null;

  const components = useMemo(() => {
    return (manifest.spec?.components || []) as ManifestComponent[];
  }, [manifest]);

  const selectedMap = (clip.selectedComponents || {}) as Record<string, any>;

  const setSelection = useCallback(
    (key: string, value: any) => {
      const store = useClipStore.getState();
      const currentClip = store.getClipById(clipId) as
        | ModelClipProps
        | undefined;
      const prevSelected = (currentClip?.selectedComponents || {}) as Record<
        string,
        any
      >;
      const next = { ...prevSelected, [key]: value };
      store.updateClip(clipId, { selectedComponents: next } as any);
    },
    [clipId],
  );

  const formatCompLabel = (c: ManifestComponent) =>
    formatComponentName(c.label || c.name || c.base || c.type);

  const normalizeModelPaths = (
    c: ManifestComponent,
  ): ManifestComponentModelPathItem[] => {
    const raw = Array.isArray(c.model_path)
      ? c.model_path
      : c.model_path
        ? [{ path: c.model_path }]
        : [];
    return (raw as any[])
      .map((it) => (typeof it === "string" ? { path: it } : it))
      .filter((it) => it && it.path) as ManifestComponentModelPathItem[];
  };

  const isItemDownloaded = (item: any): boolean => {
    // Backend augments with is_downloaded per item; fall back to false when unknown
    return !!(item && item.is_downloaded === true);
  };

  // Ensure defaults: first scheduler option and first downloaded model_path per component
  useEffect(() => {
    if (!components || components.length === 0) return;

    // Scheduler default
    let schedulerIdx = 0;
    components.forEach((comp) => {
      if (
        comp.type === "scheduler" &&
        comp.scheduler_options &&
        comp.scheduler_options.length > 0
      ) {
        const key = getSchedulerComponentKey(comp, schedulerIdx++);
        const curr = selectedMap[key] as { name?: string } | undefined;
        const names = comp.scheduler_options.map((o) => String(o.name));
        const hasValid = curr && names.includes(String(curr.name));
        if (!hasValid) {
          const first = comp.scheduler_options[0];
          setSelection(key, {
            name: first.name,
            base: first.base,
            config_path: first.config_path,
          });
        }
      }
    });

    // Model path defaults per component (only if downloaded exists)
    components.forEach((comp) => {
      if (!comp.model_path) return;
      const items = normalizeModelPaths(comp).filter((it) =>
        isItemDownloaded(it),
      );
      if (items.length === 0) return;
      const key = getComponentKey(comp);
      const curr = selectedMap[key] as { path?: string } | undefined;
      const paths = items.map((i) => String(i.path));
      const hasValid = curr && curr.path && paths.includes(String(curr.path));
      if (!hasValid) {
        const first = items[0];
        setSelection(key, {
          path: first.path,
          variant: first.variant,
          precision: first.precision,
          type: first.type,
        });
      }
    });
    // selectedMap intentionally not included to avoid unnecessary loops; setSelection ensures idempotency
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [components, clipId]);

  const renderModelPathSection = (comp: ManifestComponent) => {
    const items = normalizeModelPaths(comp);
    const downloaded = items.filter((it) => isItemDownloaded(it));
    const key = getComponentKey(comp);
    const sel = selectedMap[key] as any;
    const selectedPath = sel?.path || "";

    return (
      <div className="flex flex-col gap-2 justify-start items-start w-full p-2 ">
        <div className="text-brand-light text-[11px] font-medium w-full text-left">
          {formatCompLabel(comp)}
        </div>
        {downloaded.length === 0 ? (
          <div className="text-[10.5px] text-brand-light/60 ">
            No component downloaded.
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 w-full min-w-0 justify-start items-start ">
            {downloaded.map((it, idx) => {
              const label = formatComponentName(
                it.variant ||
                  (idx === 0 ? "Default" : it.path.split("/").pop() || it.path),
              );
              const subtitle = [
                it.precision?.toUpperCase(),
                it.type?.toLowerCase() === "gguf"
                  ? "GGUF"
                  : formatComponentName(it.type || ""),
              ]
                .filter(Boolean)
                .join(" • ");
              const isSelected = String(selectedPath) === String(it.path);
              return (
                <button
                  key={`${it.path}-${idx}`}
                  type="button"
                  onClick={() =>
                    setSelection(key, {
                      path: it.path,
                      variant: it.variant,
                      precision: it.precision,
                      type: it.type,
                    })
                  }
                  className={cn(
                    "w-full min-w-0 overflow-hidden text-left bg-brand/50 border duration-200 border-brand-light/5 rounded-[7px] px-3 py-2.5 transition-all",
                    "text-[10.5px] text-brand-light ",
                    isSelected && "bg-brand-light/7.5 border-brand-light/5",
                    !isSelected && "hover:bg-brand",
                  )}
                >
                  <div className="flex items-start justify-between gap-2 w-full min-w-0 overflow-hidden">
                    <div className="min-w-0 w-0 basis-0 flex-1 overflow-hidden">
                      <div className="text-brand-light text-[11px] font-medium truncate">
                        {label}
                      </div>
                      <div className="text-[10px] text-brand-light/70 mt-0.5 font-mono min-w-0 truncate">
                        {it.path}
                      </div>
                    </div>
                    {isSelected && (
                      <div className="shrink-0 p-[2.5px] rounded-full bg-green-500/20 border-green-500/40 border ">
                        <LuCheck className="w-2.5 h-2.5 text-green-400 shrink-0" />
                      </div>
                    )}
                  </div>
                  {subtitle && (
                    <div className="text-[10px] text-brand-light/70  truncate flex flex-row items-center gap-x-1 justify-between mt-1">
                      {it.type && (
                        <span className="text-brand-light/90 font-medium">
                          {it.type?.toLowerCase() === "gguf"
                            ? "GGUF"
                            : formatComponentName(it.type || "")}
                        </span>
                      )}
                      {it.precision && (
                        <span className="text-brand-light/90 font-mono">
                          {it.precision?.toUpperCase()}
                        </span>
                      )}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col p-4 justify-start items-stretch w-full">
      <div className="flex flex-col gap-3 w-full">
        {components.map((comp, index) => {
          if (comp.model_path) {
            return (
              <div key={`${comp.type}-${index}`} className="w-full">
                {renderModelPathSection(comp)}
              </div>
            );
          }
          return null;
        })}
      </div>
    </div>
  );
};

export default ModelComponentsProperties;
