import React, { useMemo, useCallback, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import type { ModelClipProps } from "@/lib/types";
import type { ManifestComponent } from "@/lib/manifest/api";
import { cn } from "@/lib/utils";
import Input from "./Input";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Check, ChevronDown } from "lucide-react";
import { LuInfo } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useManifestQuery } from "@/lib/manifest/queries";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import NumberInputSlider from "@/components/properties/model/inputs/NumberInputSlider";
import { setOffloadDefaultsForManifest } from "@app/preload";

interface OffloadPropertiesProps {
  clipId: string;
}

const formatComponentName = (name: string): string => {
  return String(name || "")
    .replace(/\./g, " ")
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

type OffloadLevel = "leaf" | "block";
type OffloadMode = "group" | "budget";

const OFFLOAD_LEVELS: Array<{
  value: OffloadLevel;
  name: string;
  description: string;
}> = [
  {
    value: "leaf",
    name: "Leaf level",
    description: "Offload at the leaf/module level for finer granularity.",
  },
  {
    value: "block",
    name: "Block level",
    description:
      "Offload by blocks for fewer transfers; tune the number of blocks.",
  },
];

const OffloadLevelDropdown: React.FC<{
  value: OffloadLevel;
  onChange: (next: OffloadLevel) => void;
}> = ({ value, onChange }) => {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const selected = useMemo(() => {
    return OFFLOAD_LEVELS.find((o) => o.value === value) || OFFLOAD_LEVELS[0];
  }, [value]);

  return (
    <div className="w-full min-w-0">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            ref={triggerRef}
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={cn(
              "w-full min-w-0 overflow-hidden justify-between rounded-[6px] dark bg-brand-background-light/60 border-0 shadow-none px-3 text-brand-light text-[11px] h-12 hover:bg-brand-background-light/70",
            )}
            onClick={() => setOpen(true)}
          >
            <div className="flex flex-col text-left min-w-0 w-0 flex-1 gap-y-0.5 overflow-hidden">
              <span className="block truncate font-medium">
                {selected?.name || "Select offload level"}
              </span>
              {selected?.description && (
                <span className="block truncate text-[10px] text-brand-light/70 font-normal">
                  {selected.description}
                </span>
              )}
            </div>
            <div className="flex flex-col items-center shrink-0">
       
              <ChevronDown className="h-4q! w-3! opacity-50" />
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="p-0 bg-brand-background border-brand-light/10 font-poppins"
          style={{ width: triggerRef.current?.offsetWidth || 320 }}
        >
          <Command className="bg-brand-background">
           
            <CommandList>
              <CommandEmpty className="text-brand-light/40 p-3 text-xs">
                No level found.
              </CommandEmpty>
              <CommandGroup className="px-1 py-1">
                {OFFLOAD_LEVELS.map((opt) => {
                  const isSelected = opt.value === value;
                  return (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={(val) => {
                        const picked =
                          OFFLOAD_LEVELS.find(
                            (o) => String(o.value) === String(val),
                          ) || OFFLOAD_LEVELS[0];
                        onChange(picked.value);
                        setOpen(false);
                      }}
                      className="flex items-center gap-2 px-2 py-2 hover:bg-brand-light/5 rounded-sm "
                    >
                      <div className="h-4 w-4 mt-0.5">
                        {isSelected ? (
                          <Check className="h-4 w-4 text-brand-lighter" />
                        ) : (
                          <span className="inline-block h-4 w-4" />
                        )}
                      </div>
                      <div className="flex flex-col min-w-0 max-w-full overflow-hidden">
                        <span className="block text-brand-light text-[11.5px] font-medium truncate">
                          {opt.name}
                        </span>
                        <span className="block text-[10px] text-brand-light/70 truncate">
                          {opt.description}
                        </span>
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
};

interface ToggleProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  ariaLabel?: string;
}

const Toggle: React.FC<ToggleProps> = ({ checked, onChange, disabled, ariaLabel }) => {
  return (
    <button
      type="button"
      aria-pressed={checked}
      aria-label={ariaLabel || (checked ? "Disable" : "Enable")}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none",
        checked ? "bg-blue-600" : "bg-brand-background border border-brand-light/10",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 transform rounded-full bg-brand-light shadow transition-transform",
          checked ? "translate-x-4.5" : "translate-x-0.5",
        )}
      />
    </button>
  );
};

const InfoTip: React.FC<{ text: string; disabled?: boolean }> = ({
  text,
  disabled,
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "text-brand-light/70 hover:text-brand-light focus:outline-none",
            disabled && "opacity-40 cursor-not-allowed hover:text-brand-light/70",
          )}
        >
          <LuInfo className="w-3 h-3" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        sideOffset={6}
        className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
      >
        {text}
      </TooltipContent>
    </Tooltip>
  );
};

const OffloadProperties: React.FC<OffloadPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | ModelClipProps
    | undefined;

  if (!clip || clip.type !== "model") return null;

  const manifestId = String(clip.manifest?.metadata?.id || "");
  const { data: manifest } = useManifestQuery(manifestId);
  if (!manifest) return null;

  const components = useMemo(() => {
    const comps = (manifest?.spec?.components || []) as ManifestComponent[];
    // Show the same “component list” as architecture (model_path-based), but exclude scheduler explicitly.
    return comps.filter((c) => (c as any)?.type !== "scheduler" && !!(c as any)?.model_path);
  }, [manifest]);

  const getComponentKey = useCallback((comp: ManifestComponent): string => {
    return String((comp as any).name || (comp as any).type || "component");
  }, []);

  const getLabel = useCallback((comp: ManifestComponent): string => {
    return formatComponentName(
      (comp as any).label || (comp as any).name || (comp as any).base || (comp as any).type,
    );
  }, []);

  const getOffloadState = useCallback(
    (key: string) => {
      const map = (clip.offload || {}) as NonNullable<ModelClipProps["offload"]>;
      const existing = map[key] || {};
      const enabled = !!existing.enabled;
      // Defaults: prefer Budget mode (new default). If explicitly set to group, respect it.
      const offload_mode: OffloadMode =
        (existing as any)?.offload_mode === "group" ? "group" : "budget";
      // Group mode default: block level (new default).
      const level: OffloadLevel = (existing.level as OffloadLevel) || "block";
      // Defaults: stream + record + low CPU memory ON unless explicitly disabled
      const use_stream = existing.use_stream ?? true;
      const record_stream = existing.record_stream ?? true;
      const low_cpu_mem_usage = existing.low_cpu_mem_usage ?? true;
      const num_blocks_raw = typeof existing.num_blocks === "number" ? existing.num_blocks : 1;
      const num_blocks = Math.max(1, Math.floor(Number.isFinite(num_blocks_raw) ? num_blocks_raw : 1));

      const budget_mb_raw = (existing as any)?.budget_mb;
      let budget_mb_mb: number = 3000;
      if (typeof budget_mb_raw === "number" && Number.isFinite(budget_mb_raw)) {
        budget_mb_mb = budget_mb_raw;
      } else if (typeof budget_mb_raw === "string") {
        const trimmed = budget_mb_raw.trim();
        // Slider UI only supports numeric MB budgets (100MB–5GB). If a legacy value is a percent-string,
        // fall back to a sane numeric default.
        if (!trimmed.endsWith("%")) {
          const parsed = Number(trimmed);
          if (Number.isFinite(parsed)) budget_mb_mb = parsed;
        }
      }
      budget_mb_mb = Math.min(5000, Math.max(100, Math.round(budget_mb_mb)));
      const async_transfers = (existing as any)?.async_transfers ?? true;
      const prefetch = (existing as any)?.prefetch ?? true;
      const pin_cpu_memory = (existing as any)?.pin_cpu_memory ?? false;
      const vram_safety_raw = (existing as any)?.vram_safety_coefficient;
      const vram_safety_coefficient =
        typeof vram_safety_raw === "number" && Number.isFinite(vram_safety_raw)
          ? vram_safety_raw
          : 0.8;
      const vram_safety_percent = Math.min(
        100,
        Math.max(0, Math.round(vram_safety_coefficient * 100)),
      );
      const offload_after_forward = (existing as any)?.offload_after_forward ?? false;

      return {
        enabled,
        offload_mode,
        // group
        level,
        num_blocks,
        use_stream,
        record_stream,
        low_cpu_mem_usage,
        // budget
        budget_mb_mb,
        vram_safety_percent,
        async_transfers,
        prefetch,
        pin_cpu_memory,
        offload_after_forward,
      };
    },
    [clip.offload],
  );

  const patchOffload = useCallback((key: string, patch: Partial<NonNullable<ModelClipProps["offload"]>[string]>) => {
    const store = useClipStore.getState();
    const current = store.getClipById(clipId) as ModelClipProps | undefined;
    if (!current || current.type !== "model") return;

    const prev = (current.offload || {}) as NonNullable<ModelClipProps["offload"]>;
    const prevEntry = prev[key] || {};

    let nextEntry = { ...prevEntry, ...patch };

    // Normalize/clamp
    // Defaults: prefer Budget mode unless explicitly set to group.
    const mode: OffloadMode =
      (nextEntry as any)?.offload_mode === "group" ? "group" : "budget";
    const useStream = nextEntry.use_stream ?? true;
    // Group mode default: block level (new default).
    const level = (nextEntry.level as OffloadLevel) || "block";
    let numBlocks = typeof nextEntry.num_blocks === "number" ? nextEntry.num_blocks : 1;
    numBlocks = Math.max(1, Math.floor(Number.isFinite(numBlocks) ? numBlocks : 1));
    const lowCpuMemUsage = nextEntry.low_cpu_mem_usage ?? true;

    if (useStream) {
      numBlocks = 1;
      // Default record_stream ON when streaming is enabled (unless explicitly disabled)
      nextEntry.record_stream = nextEntry.record_stream ?? true;
    }

    const rawBudget = (nextEntry as any)?.budget_mb;
    let budgetMb: number | undefined;
    if (typeof rawBudget === "number" && Number.isFinite(rawBudget)) {
      budgetMb = rawBudget;
    } else if (typeof rawBudget === "string") {
      const trimmed = rawBudget.trim();
      if (!trimmed.endsWith("%")) {
        const parsed = Number(trimmed);
        if (Number.isFinite(parsed)) budgetMb = parsed;
      }
    }
    if (mode === "budget") {
      // Slider UX: always store a numeric MB budget between 100MB and 5GB.
      const fallback = 3000;
      const chosen = budgetMb ?? fallback;
      (nextEntry as any).budget_mb = Math.min(5000, Math.max(100, Math.round(chosen)));
    }

    const vramSafetyRaw = (nextEntry as any)?.vram_safety_coefficient;
    const vramSafety =
      typeof vramSafetyRaw === "number" && Number.isFinite(vramSafetyRaw)
        ? vramSafetyRaw
        : 0.8;
    const clampedVramSafety = Math.min(0.99, Math.max(0.01, vramSafety));

    (nextEntry as any).offload_mode = mode;
    nextEntry.level = level;
    nextEntry.num_blocks = numBlocks;
    nextEntry.low_cpu_mem_usage = lowCpuMemUsage;
    (nextEntry as any).async_transfers =
      typeof (nextEntry as any).async_transfers === "boolean"
        ? (nextEntry as any).async_transfers
        : true;
    (nextEntry as any).prefetch =
      typeof (nextEntry as any).prefetch === "boolean" ? (nextEntry as any).prefetch : true;
    (nextEntry as any).pin_cpu_memory =
      typeof (nextEntry as any).pin_cpu_memory === "boolean"
        ? (nextEntry as any).pin_cpu_memory
        : false;
    (nextEntry as any).vram_safety_coefficient = clampedVramSafety;
    (nextEntry as any).offload_after_forward =
      typeof (nextEntry as any).offload_after_forward === "boolean"
        ? (nextEntry as any).offload_after_forward
        : false;

    // If stream is disabled, keep record_stream but it will be ignored by the engine unless stream is on
    nextEntry.use_stream = useStream;

    const next = { ...prev, [key]: nextEntry };
    store.updateClip(clipId, { offload: next } as any);

    // Persist global defaults for this manifest id (scoped by active backend URL in main process).
    // Fire-and-forget: UI updates are driven by clip state above.
    if (manifestId) {
      void setOffloadDefaultsForManifest(manifestId, next as any).catch(() => undefined);
    }
  }, [clipId, manifestId]);

  if (!components.length) return null;

  return (
    <div>
        <h3 className="text-brand-light text-[12px] font-medium text-start ml-4 mt-4">
            Offload
        </h3>
        <p className="text-brand-light/60 text-[10px] font-normal text-start ml-4 mt-1">
            Configure offloading for each component in the model.
        </p>
    <div className="flex flex-col p-4 justify-start items-stretch w-full">
      <div className="flex flex-col gap-3 w-full">
        {components.map((comp, idx) => {
          const key = getComponentKey(comp);
          const label = getLabel(comp);
          const state = getOffloadState(key);

          return (
            <div
              key={`${key}-${idx}`}
              className="flex flex-col gap-y-2 p-3 rounded-[6px] bg-brand/70 border border-brand-light/5"
            >
              <div className="flex flex-row items-center justify-between w-full gap-x-3 min-w-0">

                  <div className="text-brand-light text-[11px] font-medium truncate">
                    {label}
                  </div>
                <Toggle
                  checked={state.enabled}
                  onChange={(next) => patchOffload(key, { enabled: next })}
                  ariaLabel={state.enabled ? "Disable offloading" : "Enable offloading"}
                />
              </div>

              {state.enabled && (
                <div className="flex flex-col gap-y-3">
                  <Tabs
                    value={state.offload_mode}
                    onValueChange={(v) => {
                      const nextMode = v as OffloadMode;
                      if (nextMode === "budget") {
                        patchOffload(key, {
                          offload_mode: "budget",
                          budget_mb: state.budget_mb_mb,
                          vram_safety_coefficient: Math.min(
                            0.99,
                            Math.max(0.01, state.vram_safety_percent / 100),
                          ),
                        } as any);
                        return;
                      }
                      patchOffload(key, { offload_mode: "group" } as any);
                    }}
                    className="w-full"
                  >
                    <TabsList className="w-full bg-brand-background-light/60 border border-brand-light/10 rounded-[6px] px-1 py-1 gap-1 ">
                      <TabsTrigger
                        value="group"
                        className="flex-1 text-[10.5px] text-brand-light/70 data-[state=active]:text-brand-light rounded-[6px] transition-all"
                      >
                        Group
                      </TabsTrigger>
                      <TabsTrigger
                        value="budget"
                        className="flex-1 text-[10.5px] text-brand-light/70 data-[state=active]:text-brand-light rounded-[6px] transition-all"
                      >
                        Budget
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="group" className="mt-3">
                      <div className="flex flex-col gap-y-3">
                        <div className="flex flex-col gap-y-1 w-full min-w-0">
                          <OffloadLevelDropdown
                            value={state.level}
                            onChange={(next) => patchOffload(key, { level: next, offload_mode: "group" } as any)}
                          />
                        </div>

                        {state.level === "block" && (
                          <div className={cn("flex flex-col gap-1", state.use_stream && "opacity-70")}>
                            <Input
                              className="bg-brand-background h-7"
                              stepClassName="h-7 bg-brand-background-light/60"
                              label="Number of blocks"
                              description="How many blocks to split the component into for block-level offloading. More blocks can reduce peak memory per block, but may increase overhead."
                              startLogo="N"
                              value={String(state.num_blocks)}
                              onChange={(v) => {
                                if (state.use_stream) {
                                  // Hard constraint: streaming requires a single block
                                  patchOffload(key, { num_blocks: 1, offload_mode: "group" } as any);
                                  return;
                                }
                                const parsed = Math.max(1, Math.floor(Number(v)));
                                patchOffload(key, { num_blocks: Number.isFinite(parsed) ? parsed : 1, offload_mode: "group" } as any);
                              }}
                              canStep
                              step={1}
                              min={1}
                            />
                            {state.use_stream && (
                              <div className="text-[10px] text-brand-light/60 text-start">
                                When <span className="text-brand-light/80 font-medium">Use stream</span> is enabled,{" "}
                                <span className="text-brand-light/80 font-medium">Number of blocks</span> is fixed to 1.
                              </div>
                            )}
                          </div>
                        )}

                        <div className="flex flex-col gap-y-2">
                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div className="text-brand-light text-[11px] font-medium">
                                Use stream
                              </div>
                              <InfoTip text="Enable streaming mode for group offloading. When enabled, the engine streams data instead of staging multiple blocks; this requires Number of blocks to be 1." />
                            </div>
                            <Toggle
                              checked={state.use_stream}
                              onChange={(next) =>
                                patchOffload(key, {
                                  offload_mode: "group",
                                  use_stream: next,
                                  num_blocks: next ? 1 : state.num_blocks,
                                  record_stream: next ? (state.record_stream ?? true) : state.record_stream,
                                } as any)
                              }
                              ariaLabel={state.use_stream ? "Disable use stream" : "Enable use stream"}
                            />
                          </div>

                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div
                                className={cn(
                                  "text-[11px] font-medium",
                                  state.use_stream ? "text-brand-light" : "text-brand-light/40",
                                )}
                              >
                                Record stream
                              </div>
                              <InfoTip
                                disabled={!state.use_stream}
                                text="Record stream events for debugging/inspection. Requires Use stream to be enabled."
                              />
                            </div>
                            <Toggle
                              checked={state.record_stream}
                              disabled={!state.use_stream}
                              onChange={(next) => patchOffload(key, { record_stream: next, offload_mode: "group" } as any)}
                              ariaLabel={
                                state.record_stream
                                  ? "Disable record stream"
                                  : "Enable record stream"
                              }
                            />
                          </div>

                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div className="text-brand-light text-[11px] font-medium">
                                Low CPU Memory
                              </div>
                              <InfoTip text="Reduce CPU RAM usage during offloading by using a lower-memory pathway. This can be slightly slower, but helps avoid CPU OOMs on large models." />
                            </div>
                            <Toggle
                              checked={state.low_cpu_mem_usage}
                              onChange={(next) => patchOffload(key, { low_cpu_mem_usage: next, offload_mode: "group" } as any)}
                              ariaLabel={
                                state.low_cpu_mem_usage
                                  ? "Disable low CPU memory"
                                  : "Enable low CPU memory"
                              }
                            />
                          </div>
                        </div>
                      </div>
                    </TabsContent>

                    <TabsContent value="budget" className="mt-3">
                      <div className="flex flex-col gap-y-3">
                        <NumberInputSlider
                          label="Budget"
                          description="VRAM budget for this component (stored in MB)."
                          value={state.budget_mb_mb}
                          onChange={(v) =>
                            patchOffload(key, {
                              offload_mode: "budget",
                              budget_mb: Math.min(5000, Math.max(100, Math.round(v))),
                            } as any)
                          }
                          renderInput={(raw) => {
                            const s = String(raw ?? "");
                            const num = Number(s.replace(/[^0-9.]/g, ""));
                            if (!Number.isFinite(num)) return s;
                            if (num >= 1000) return `${(num / 1000).toFixed(1)} GB`;
                            return `${Math.round(num)} MB`;
                          }}
                          min={100}
                          max={5000}
                          step={100}
                          toFixed={0}
                          suffix=""
                        />

                        <NumberInputSlider
                          label="VRAM safety"
                          description="How much VRAM to keep as headroom for transient allocations. Stored normalized between 0 and 1."
                          value={state.vram_safety_percent}
                          onChange={(pct) => {
                            const clampedPct = Math.min(100, Math.max(0, Math.round(pct)));
                            const normalized = Math.min(0.99, Math.max(0.01, clampedPct / 100));
                            patchOffload(key, { offload_mode: "budget", vram_safety_coefficient: normalized } as any);
                          }}
                          min={0}
                          max={100}
                          step={1}
                          toFixed={0}
                          suffix="%"
                        />

                        <div className="flex flex-col gap-y-2">
                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div className="text-brand-light text-[11px] font-medium">
                                Async transfers
                              </div>
                              <InfoTip text="Allow asynchronous CPU↔GPU transfers when supported. Usually faster, but can increase concurrency/peaks on some systems." />
                            </div>
                            <Toggle
                              checked={state.async_transfers}
                              onChange={(next) => patchOffload(key, { offload_mode: "budget", async_transfers: next } as any)}
                              ariaLabel={state.async_transfers ? "Disable async transfers" : "Enable async transfers"}
                            />
                          </div>

                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div className="text-brand-light text-[11px] font-medium">
                                Prefetch
                              </div>
                              <InfoTip text="Prefetch the next blocks ahead of time (when async transfers are enabled) to overlap transfer and compute." />
                            </div>
                            <Toggle
                              checked={state.prefetch}
                              onChange={(next) => patchOffload(key, { offload_mode: "budget", prefetch: next } as any)}
                              ariaLabel={state.prefetch ? "Disable prefetch" : "Enable prefetch"}
                            />
                          </div>

                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div className="text-brand-light text-[11px] font-medium">
                                Pin CPU memory
                              </div>
                              <InfoTip text="Reserve pinned host memory for faster transfers. Helps performance but uses extra RAM." />
                            </div>
                            <Toggle
                              checked={state.pin_cpu_memory}
                              onChange={(next) => patchOffload(key, { offload_mode: "budget", pin_cpu_memory: next } as any)}
                              ariaLabel={state.pin_cpu_memory ? "Disable pin CPU memory" : "Enable pin CPU memory"}
                            />
                          </div>

                          <div className="flex flex-row items-center justify-between w-full gap-x-3">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <div className="text-brand-light text-[11px] font-medium">
                                Offload after forward
                              </div>
                              <InfoTip text="Aggressively offload blocks after they run. Can reduce VRAM peaks but may add transfer overhead." />
                            </div>
                            <Toggle
                              checked={state.offload_after_forward}
                              onChange={(next) => patchOffload(key, { offload_mode: "budget", offload_after_forward: next } as any)}
                              ariaLabel={state.offload_after_forward ? "Disable offload after forward" : "Enable offload after forward"}
                            />
                          </div>
                        </div>

                        <div className="text-[10px] text-brand-light/60 text-start">
                          Budget offloading cannot be combined with group offloading on the same component.
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
    </div>
  );
};

export default OffloadProperties;

