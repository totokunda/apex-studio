import React, { useEffect, useMemo, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ManifestComponent } from "@/lib/manifest/api";
import { getSchedulerComponentKey } from "@/lib/manifest/componentKey";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import SelectInput from "./inputs/SelectInput";
import BooleanInput from "./inputs/BooleanInput";
import NumberInput from "./inputs/NumberInput";
import NumberInputSlider from "./inputs/NumberInputSlider";
import { useManifestQuery } from "@/lib/manifest/queries";

interface SchedulerPanelProps {
  clipId: string;
  component: ManifestComponent;
  schedulerIndex: number;
}

type SchedulerOption = {
  name: string;
  label?: string;
  description?: string;
  base?: string;
  config_path?: string;
  config_id?: string;
  config?: Record<string, any>;
};

type SchedulerField = {
  label?: string;
  description?: string;
  type?: string;
  value_type?: string;
  min?: number;
  max?: number;
  step?: number;
  default?: any;
  options?: Array<{ name: string; value: any }>;
};

const RESERVED_KEYS = new Set(["name", "base", "config_path", "config_id", "config"]);

const normalizeConfig = (value: any): Record<string, any> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, any>;
};

const SchedulerPanel: React.FC<SchedulerPanelProps> = ({
  clipId,
  component,
  schedulerIndex,
}) => {
  const getClipById = useClipStore((s) => s.getClipById);
  const updateClip = useClipStore((s) => s.updateClip);
  const clip: any = getClipById(clipId);
  const manifestId = String(clip?.manifest?.metadata?.id || "").trim();
  const { data: latestManifest } = useManifestQuery(manifestId || null, true);

  const componentKey = useMemo(() => {
    return getSchedulerComponentKey(component, schedulerIndex);
  }, [component, schedulerIndex]);

  const latestResolvedComponent = useMemo(() => {
    const latestComponents = (latestManifest?.spec?.components || []) as ManifestComponent[];
    const latestSchedulers = latestComponents.filter(
      (c: any) => String(c?.type) === "scheduler",
    );
    if (!latestSchedulers.length) return component;

    for (let idx = 0; idx < latestSchedulers.length; idx += 1) {
      const candidate = latestSchedulers[idx];
      const candidateKey = getSchedulerComponentKey(candidate, idx);
      if (candidateKey === componentKey) return candidate;
    }
    return component;
  }, [latestManifest?.spec?.components, component, componentKey]);

  const options = useMemo(() => {
    const all = Array.isArray((latestResolvedComponent as any)?.scheduler_options)
      ? ((latestResolvedComponent as any).scheduler_options as any[])
      : [];
    return all.filter((opt) => String(opt?.name || "").trim().length > 0) as SchedulerOption[];
  }, [latestResolvedComponent]);

  const schedulerFields = useMemo(() => {
    const raw = (latestResolvedComponent as any)?.scheduler_fields;
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {} as Record<string, SchedulerField>;
    return raw as Record<string, SchedulerField>;
  }, [latestResolvedComponent]);

  const selectedSpec: Record<string, any> | undefined = useMemo(() => {
    const fromKey = clip?.selectedComponents?.[componentKey];
    return fromKey && typeof fromKey === "object" ? fromKey : undefined;
  }, [clip?.selectedComponents, componentKey]);

  const selectedName: string | undefined = useMemo(() => {
    return String(selectedSpec?.name ?? "") || undefined;
  }, [selectedSpec]);

  const selectedOption = useMemo(
    () => options.find((o) => String(o.name) === String(selectedName)),
    [options, selectedName],
  );
  const defaultSchedulerName = useMemo(() => {
    const v = String((latestResolvedComponent as any)?.default || "").trim();
    return v || undefined;
  }, [latestResolvedComponent]);

  const makeSelectionPayload = useMemo(() => {
    return (
      option: SchedulerOption,
      configOverride?: Record<string, any>,
      previous?: Record<string, any>,
    ) => {
      const baseConfig = normalizeConfig(option?.config);
      const mergedConfig = {
        ...baseConfig,
        ...(configOverride || {}),
      };

      const next: Record<string, any> = {
        ...(previous || {}),
        name: option?.name,
        base: option?.base,
      };

      if (option?.config_path != null) next.config_path = option.config_path;
      else delete next.config_path;
      if (option?.config_id != null) next.config_id = option.config_id;
      else delete next.config_id;

      next.config = mergedConfig;

      // Back-compat for older API handlers that read flat override keys.
      Object.keys(next).forEach((k) => {
        if (!RESERVED_KEYS.has(k)) delete next[k];
      });
      Object.entries(mergedConfig).forEach(([k, v]) => {
        if (!RESERVED_KEYS.has(k)) {
          next[k] = v;
        }
      });

      return next;
    };
  }, []);

  // Ensure a default selection (and backfill config for old persisted shape).
  useEffect(() => {
    if (!clip || !options || options.length === 0) return;

    const names = options.map((o) => String(o?.name));
    const curr = selectedSpec;
    const currName = curr?.name;
    const hasValidSelection = currName && names.includes(String(currName));

    if (!hasValidSelection) {
      const preferred = defaultSchedulerName
        ? options.find((opt) => String(opt.name) === defaultSchedulerName)
        : undefined;
      const first = preferred || options[0];
      const nextSelected = {
        ...(clip.selectedComponents || {}),
        [componentKey]: makeSelectionPayload(first),
      };
      updateClip(clipId, { selectedComponents: nextSelected } as any);
      return;
    }

    const selected = options.find((o) => String(o.name) === String(currName));
    if (!selected) return;

    const hasConfigObject =
      curr &&
      typeof curr === "object" &&
      curr.config &&
      typeof curr.config === "object" &&
      !Array.isArray(curr.config);

    if (!hasConfigObject) {
      const nextSelected = {
        ...(clip.selectedComponents || {}),
        [componentKey]: makeSelectionPayload(selected, undefined, curr),
      };
      updateClip(clipId, { selectedComponents: nextSelected } as any);
    }
  }, [
    clip,
    clipId,
    options,
    selectedSpec,
    updateClip,
    componentKey,
    makeSelectionPayload,
    defaultSchedulerName,
  ]);

  const effectiveConfig = useMemo(() => {
    const out: Record<string, any> = {
      ...normalizeConfig(selectedOption?.config),
      ...normalizeConfig(selectedSpec?.config),
    };

    // Recover legacy flat config keys from selected component object.
    if (selectedSpec && typeof selectedSpec === "object") {
      Object.entries(selectedSpec).forEach(([k, v]) => {
        if (RESERVED_KEYS.has(k)) return;
        const hasFieldDef = Object.prototype.hasOwnProperty.call(schedulerFields, k);
        const hasDefault = Object.prototype.hasOwnProperty.call(normalizeConfig(selectedOption?.config), k);
        if (hasFieldDef || hasDefault) out[k] = v;
      });
    }

    return out;
  }, [selectedOption, selectedSpec, schedulerFields]);

  const configKeys = useMemo(() => {
    const ordered: string[] = [];
    const seen = new Set<string>();
    const selectedBaseConfig = normalizeConfig(selectedOption?.config);
    const selectedConfig = normalizeConfig(selectedSpec?.config);

    const add = (k: string) => {
      const key = String(k || "").trim();
      if (!key || seen.has(key)) return;
      seen.add(key);
      ordered.push(key);
    };

    // Show only params relevant to the currently selected scheduler.
    Object.keys(selectedBaseConfig).forEach(add);
    Object.keys(selectedConfig).forEach(add);

    // Back-compat: include flattened legacy keys persisted on selectedSpec.
    if (selectedSpec && typeof selectedSpec === "object") {
      Object.keys(selectedSpec).forEach((k) => {
        if (RESERVED_KEYS.has(k)) return;
        if (
          Object.prototype.hasOwnProperty.call(selectedBaseConfig, k) ||
          Object.prototype.hasOwnProperty.call(selectedConfig, k) ||
          Object.prototype.hasOwnProperty.call(schedulerFields, k)
        ) {
          add(k);
        }
      });
    }

    return ordered;
  }, [schedulerFields, selectedOption, selectedSpec]);

  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const updateSchedulerSelection = (next: Record<string, any>) => {
    const nextSelected = {
      ...(clip?.selectedComponents || {}),
      [componentKey]: next,
    };
    updateClip(clipId, { selectedComponents: nextSelected } as any);
  };

  const onSchedulerPick = (name: string) => {
    const picked = options.find((o) => String(o.name) === String(name));
    if (!picked) return;
    updateSchedulerSelection(makeSelectionPayload(picked));
    setOpen(false);
  };

  const onConfigFieldChange = (key: string, rawValue: any) => {
    if (!selectedOption) return;

    const field = schedulerFields[key];
    let nextValue: any = rawValue;
    const fieldType = String(field?.type || "").toLowerCase();

    if (fieldType === "number" || fieldType === "number+slider") {
      const numeric = Number(rawValue);
      if (Number.isFinite(numeric)) {
        if ((field?.value_type || "").toLowerCase() === "integer") {
          nextValue = Math.round(numeric);
        } else {
          nextValue = numeric;
        }
      }
    }

    if (fieldType === "boolean") {
      nextValue = Boolean(rawValue);
    }

    const nextConfig = {
      ...effectiveConfig,
      [key]: nextValue,
    };

    updateSchedulerSelection(
      makeSelectionPayload(selectedOption, nextConfig, selectedSpec),
    );
  };

  const renderField = (key: string) => {
    const field = schedulerFields[key] || {};
    const fieldType = String(field.type || "").toLowerCase();
    const value = effectiveConfig[key];
    const fallbackLabel = String(key)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (m) => m.toUpperCase());
    const label = String(field.label || fallbackLabel);
    const description = field.description;
    const min = typeof field.min === "number" ? field.min : undefined;
    const max = typeof field.max === "number" ? field.max : undefined;
    const step = typeof field.step === "number" ? field.step : undefined;
    const isInteger = String(field.value_type || "").toLowerCase() === "integer";
    const isSliderType = fieldType === "number+slider";
    const isNumberType = fieldType === "number";
    const defaultValue = field.default;
    const numericBaseValue =
      value ?? (typeof defaultValue === "number" ? defaultValue : undefined);
    const numericStep = step ?? (isInteger ? 1 : 0.01);
    const shouldUseSlider =
      (isSliderType || isNumberType) &&
      typeof min === "number" &&
      typeof max === "number" &&
      Number.isFinite(min) &&
      Number.isFinite(max) &&
      max > min;

    return (
      <div key={key} className="w-full min-w-0 ">
        {fieldType === "select" && Array.isArray(field.options) ? (
          <SelectInput
            label={label}
            description={description}
            value={String(value ?? "")}
            options={(field.options || []).map((opt) => ({
              name: String(opt.name ?? opt.value),
              value: String(opt.value),
            }))}
            onChange={(next) => {
              const picked = (field.options || []).find(
                (opt) => String(opt.value) === String(next),
              );
              onConfigFieldChange(key, picked ? picked.value : next);
            }}
            useDropdown
          />
        ) : null}

        {fieldType === "boolean" ? (
          <BooleanInput
            label={label}
            description={description}
            value={Boolean(value)}
            onChange={(next) => onConfigFieldChange(key, next)}
          />
        ) : null}

        {(isNumberType || isSliderType) && shouldUseSlider ? (
          <NumberInputSlider
            label={label}
            description={description}
            value={Number(numericBaseValue ?? min ?? 0)}
            min={field.min}
            max={field.max}
            step={numericStep}
            toFixed={isInteger ? 0 : 3}
            onChange={(next) => onConfigFieldChange(key, next)}
            inputClass="bg-brand"
          />
        ) : null}

        {(isNumberType || isSliderType) && !shouldUseSlider ? (
          <NumberInput
            label={label}
            description={description}
            value={String(value ?? defaultValue ?? "")}
            min={min}
            max={max}
            step={numericStep}
            toFixed={isInteger ? 0 : 3}
            onChange={(next) => onConfigFieldChange(key, next)}
          />
        ) : null}

        {fieldType !== "select" && fieldType !== "boolean" && !isNumberType && !isSliderType ? (
          <div className="flex flex-col items-start w-full gap-y-1 min-w-0">
            <div className="text-brand-light text-[10px] font-medium text-start">
              {label}
            </div>
            {description ? (
              <div className="text-brand-light/70 text-[9.5px] text-start leading-snug">
                {description}
              </div>
            ) : null}
            <input
              type="text"
              className="w-full h-7 px-2 text-brand-light text-[11px] outline-none rounded font-normal items-center border border-brand-light/5 bg-brand"
              value={String(value ?? "")}
              onChange={(e) => onConfigFieldChange(key, e.target.value)}
            />
          </div>
        ) : null}

        
      </div>
    );
  };

  return (
    <div className="w-full min-w-0">
      <div className="text-[10px] text-brand-light/70 mb-1.5 px-0.5 text-start">
        {String((component as any)?.label || (component as any)?.name || "").trim() ||
          (schedulerIndex > 0 ? `Scheduler ${schedulerIndex + 1}` : "Scheduler")}
      </div>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            ref={triggerRef}
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={cn(
              "w-full min-w-0 overflow-hidden justify-between rounded-[6px] dark bg-brand-background-light border border-brand-light/5 shadow-none px-3 text-brand-light text-[11px] h-13 hover:bg-brand-background-light/70",
            )}
            onClick={() => setOpen(true)}
          >
            <div className="flex flex-col min-w-0 w-0 flex-1 gap-y-0.5 overflow-hidden text-start">
              <span className="block truncate font-medium">
                {selectedOption?.label ||
                  selectedOption?.name ||
                  "Select scheduler"}
              </span>
              {selectedOption?.description && (
                <span className="block truncate text-[10px] text-brand-light/70 font-normal">
                  {selectedOption.description}
                </span>
              )}
            </div>
            <div className="flex flex-col items-center shrink-0">
              <ChevronUp className="h-3! w-3!  opacity-50" />
              <ChevronDown className="h-3! w-3!  opacity-50" />
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="p-0 bg-brand-background border-brand-light/10 font-inter"
          style={{ width: triggerRef.current?.offsetWidth || 320 }}
        >
          <Command className="bg-brand-background">
            <CommandInput
              placeholder="Search scheduler"
              className="text-brand-light placeholder:text-brand-light/40 text-[11px]"
            />
            <CommandList>
              <CommandEmpty className="text-brand-light/40 p-3 text-xs">
                No scheduler found.
              </CommandEmpty>
              <CommandGroup className="px-1 py-1">
                {options.map((opt) => {
                  const isSelected = String(opt.name) === String(selectedName);
                  return (
                    <CommandItem
                      key={opt.name}
                      value={opt.name}
                      onSelect={(val) => onSchedulerPick(val)}
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
                          {opt.label || opt.name}
                        </span>
                        {opt.description && (
                          <span className="block text-[10px] text-brand-light/70 truncate">
                            {opt.description}
                          </span>
                        )}
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedOption && configKeys.length > 0 ? (
        <div className="mt-4 flex w-full flex-col gap-2.5 px-1.5 ">{configKeys.map(renderField)}</div>
      ) : null}
    </div>
  );
};

export default SchedulerPanel;
